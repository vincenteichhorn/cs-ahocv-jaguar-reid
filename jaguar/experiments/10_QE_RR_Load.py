import os
import torch
import torch.nn as nn
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import wandb

from jaguar.components import DINOv3, VielleichtguarModel, EmbeddingProjection
from jaguar.criteria import ArcFaceCriterion
from jaguar.datasets import get_dataloaders
from jaguar.submission import build_submission

PROJECT = "jaguar-reid-josefandvincent"
GROUP = "10_QE_RR_Load"
RUN_NAME = f"{GROUP}-dinov3-projection-arcface"

# Path to the checkpoint to load
CHECKPOINT_TO_LOAD = Path("checkpoints/09_augmentation-elastic_rotation_perspective_best.pth")

BASE_CONFIG = {
    "random_seed": 42,
    "data_dir": Path("./data"),  # Path("/kaggle/input/jaguar-re-id"),
    "checkpoint_dir": Path("checkpoints"),
    "cache_dir": Path("./cache"),
    "embeddings_dir": Path("./embeddings"),
    "validation_split_size": 0.2,
}

EXPERIMENT_CONFIGS = [
    {
        "batch_size": 64,
        "image_size": 256,
        "hidden_dim": 512,
        "output_dim": 256,
        "dropout": 0.3,
        "train_backbone": False,
        "QE_enabled": False,
        "RR_enabled": True,
        "TTA_enabled": False,
    },
]

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(BASE_CONFIG["random_seed"])
np.random.seed(BASE_CONFIG["random_seed"])

load_dotenv()

for EXPERIMENT_CONFIG in EXPERIMENT_CONFIGS:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=PROJECT, config={**EXPERIMENT_CONFIG, **BASE_CONFIG}, group=GROUP, name=RUN_NAME)

    BASE_CONFIG["checkpoint_dir"].mkdir(exist_ok=True)
    submission_path = (
        BASE_CONFIG["checkpoint_dir"]
        / f"{RUN_NAME}_QE{EXPERIMENT_CONFIG['QE_enabled']}_RR{EXPERIMENT_CONFIG['RR_enabled']}_TTA{EXPERIMENT_CONFIG['TTA_enabled']}_submission.csv"
    )

    backbone = DINOv3(freeze=not EXPERIMENT_CONFIG["train_backbone"], cache_folder=BASE_CONFIG["embeddings_dir"])
    base_transforms = backbone.get_transforms()

    train_dataloader, validation_dataloader, test_dataloader, num_classes, label_encoder = get_dataloaders(
        data_dir=BASE_CONFIG["data_dir"],
        validation_split_size=BASE_CONFIG["validation_split_size"],
        seed=BASE_CONFIG["random_seed"],
        batch_size=EXPERIMENT_CONFIG["batch_size"],
        image_size=EXPERIMENT_CONFIG["image_size"],
        cache_dir=BASE_CONFIG["cache_dir"],
        train_process_fn=base_transforms,
        val_process_fn=base_transforms,
        mode="background",
    )

    model = VielleichtguarModel(
        backbone=backbone,
        layers=nn.Sequential(
            EmbeddingProjection(
                input_dim=backbone.out_dim(),
                hidden_dim=EXPERIMENT_CONFIG["hidden_dim"],
                output_dim=EXPERIMENT_CONFIG["output_dim"],
                dropout=EXPERIMENT_CONFIG["dropout"],
            ),
        ),
        criterion=ArcFaceCriterion(
            embedding_dim=EXPERIMENT_CONFIG["output_dim"],
            num_classes=num_classes,
            margin=0.5,  # These values don't matter for inference
            scale=64.0,
        ),
    ).to(device)

    # Load the checkpoint
    print(f"Loading checkpoint from {CHECKPOINT_TO_LOAD}")
    model.load_model(CHECKPOINT_TO_LOAD)
    model.eval()

    # Generate submission
    print(f"Generating submission with QE={EXPERIMENT_CONFIG['QE_enabled']}, RR={EXPERIMENT_CONFIG['RR_enabled']}, TTA={EXPERIMENT_CONFIG['TTA_enabled']}")
    build_submission(
        submission_path,
        model,
        test_dataloader,
        device,
        query_expansion_enabled=EXPERIMENT_CONFIG["QE_enabled"],
        k_reciprocal_reranking_enabled=EXPERIMENT_CONFIG["RR_enabled"],
        tta_enabled=EXPERIMENT_CONFIG["TTA_enabled"],
    )

    print(f"Submission saved to {submission_path}")

    submission_artifact = wandb.Artifact(
        name=f"{RUN_NAME}_submission_QE{EXPERIMENT_CONFIG['QE_enabled']}_RR{EXPERIMENT_CONFIG['RR_enabled']}_TTA{EXPERIMENT_CONFIG['TTA_enabled']}",
        type="submission",
    )
    submission_artifact.add_file(str(submission_path))
    wandb.log_artifact(submission_artifact)
    wandb.finish()
