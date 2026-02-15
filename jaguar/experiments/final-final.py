import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.v2 as v2
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import wandb

from jaguar.components import DINOv3, VielleichtguarModel, EmbeddingProjection
from jaguar.criteria import ArcFaceCriterion, FocalArcFaceCriterion
from jaguar.datasets import get_dataloaders
from jaguar.submission import build_submission
from jaguar.train import train_epoch, validate_epoch

PROJECT = "jaguar-reid-josefandvincent"
GROUP = "_test"
RUN_NAME = f"{GROUP}-baba"

BASE_CONFIG = {
    "random_seed": 456,
    "data_dir": Path("./data"),  # Path("/kaggle/input/jaguar-re-id"),
    "checkpoint_dir": Path("checkpoints"),
    "cache_dir": Path("./cache"),
    "embeddings_dir": Path("./embeddings"),
    "validation_split_size": 0.2,
}

EXPERIMENT_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "image_size": 256,
    "hidden_dim": 1024,
    "n_layers": 3,
    "output_dim": 256,
    "dropout": 0.1,
    "weight_decay": 7e-3,
    "learning_rate": 4e-4,
    "arcface_margin": 0.47,
    "arcface_scale": 40.0,
    "patience": 10,
    "backbone_lr_multiplier": 0.2,
    "background_intervention": "segmented",
    "focal_arcface_gamma": 2.0,
    "max_LR": 1e-3,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(BASE_CONFIG["random_seed"])
np.random.seed(BASE_CONFIG["random_seed"])

load_dotenv()
# user_secrets = UserSecretsClient()
# os.environ["HF_TOKEN"]= user_secrets.get_secret("hf_token")
# os.environ["WANDB_API_KEY"] = user_secrets.get_secret("wandb_token")

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project=PROJECT, config=EXPERIMENT_CONFIG, group=GROUP, name=RUN_NAME)

BASE_CONFIG["checkpoint_dir"].mkdir(exist_ok=True)
checkpoint_path = BASE_CONFIG["checkpoint_dir"] / f"{RUN_NAME}_best.pth"
submission_path = BASE_CONFIG["checkpoint_dir"] / f"{RUN_NAME}_submission.csv"

backbone = DINOv3(freeze=False, cache_folder=BASE_CONFIG["embeddings_dir"], use_caching=False)
base_transforms = backbone.get_transforms()
augmentation_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.25), value=0),
        *base_transforms.transforms,
    ]
)

train_dataloader, validation_dataloader, test_dataloader, num_classes, label_encoder = get_dataloaders(
    data_dir=BASE_CONFIG["data_dir"],
    validation_split_size=BASE_CONFIG["validation_split_size"],
    seed=BASE_CONFIG["random_seed"],
    batch_size=EXPERIMENT_CONFIG["batch_size"],
    image_size=EXPERIMENT_CONFIG["image_size"],
    cache_dir=BASE_CONFIG["cache_dir"],
    train_process_fn=augmentation_transforms,
    val_process_fn=base_transforms,
    mode=EXPERIMENT_CONFIG["background_intervention"],
)

model = VielleichtguarModel(
    backbone=backbone,
    layers=nn.Sequential(
        EmbeddingProjection(
            input_dim=backbone.out_dim(),
            hidden_dim=EXPERIMENT_CONFIG["hidden_dim"],
            output_dim=EXPERIMENT_CONFIG["output_dim"],
            dropout=EXPERIMENT_CONFIG["dropout"],
            n_layers=EXPERIMENT_CONFIG["n_layers"],
        ),
    ),
    criterion=FocalArcFaceCriterion(
        embedding_dim=EXPERIMENT_CONFIG["output_dim"],
        num_classes=num_classes,
        margin=EXPERIMENT_CONFIG["arcface_margin"],
        scale=EXPERIMENT_CONFIG["arcface_scale"],
        gamma=EXPERIMENT_CONFIG["focal_arcface_gamma"],
    ),
).to(device)

wandb.log(
    {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "total_parameters": sum(p.numel() for p in model.parameters())}
)


optimizer = AdamW(model.parameters(), lr=EXPERIMENT_CONFIG["learning_rate"], weight_decay=EXPERIMENT_CONFIG["weight_decay"])
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=EXPERIMENT_CONFIG["max_LR"], steps_per_epoch=len(train_dataloader), epochs=EXPERIMENT_CONFIG["epochs"]
)

best_epoch, best_map, patience_counter, total_duration, eta = 0, 0.0, 0, 0.0, 0.0
for epoch in range(EXPERIMENT_CONFIG["epochs"]):

    start = time.time()
    train_loss = train_epoch(model, train_dataloader, optimizer, device, lr_scheduler)
    validation_loss, validation_map = validate_epoch(model, validation_dataloader, device)
    end = time.time()
    epoch_time = end - start
    total_duration += epoch_time
    eta = (epoch_time if epoch == 0 else 0.9 * (eta / (EXPERIMENT_CONFIG["epochs"] - epoch)) + 0.1 * epoch_time) * (EXPERIMENT_CONFIG["epochs"] - epoch - 1)

    print(
        f"epoch: {epoch+1:>2}/{EXPERIMENT_CONFIG['epochs']} | ",
        f"train/loss: {train_loss:>8.4f} | ",
        f"val/loss: {validation_loss:>8.4f} | ",
        f"val/mAP: {validation_map:>7.4f} | ",
        f"lr: {optimizer.param_groups[0]['lr']:>7.1e} | ",
        f"eta: {max(0,eta)/60:.1f} min | " if max(0, eta) > 60 else f"eta: {max(0,eta):.1f} sec | ",
        f"patience: {patience_counter}/{EXPERIMENT_CONFIG['patience']} | ",
        sep="",
    )

    wandb.log(
        {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": validation_loss,
            "val/mAP": validation_map,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )

    if validation_map > best_map:
        best_map = validation_map
        best_epoch = epoch + 1
        patience_counter = 0
        model.save_model(
            checkpoint_path,
            with_backbone=True,
            with_criterion=False,
        )
    else:
        patience_counter += 1

    if patience_counter >= EXPERIMENT_CONFIG["patience"]:
        break

model.load_model(checkpoint_path)
build_submission(submission_path, model, test_dataloader, device, query_expansion_enabled=True, tta_enabled=True, k_reciprocal_reranking_enabled=True)

wandb.run.summary["best_val_mAP"] = best_map
wandb.run.summary["best_epoch"] = best_epoch
wandb.run.summary["total_epochs"] = epoch + 1
wandb.run.summary["total_training_time"] = total_duration
model_artifact = wandb.Artifact(name=f"{RUN_NAME}_model", type="model")
model_artifact.add_file(str(checkpoint_path))
wandb.log_artifact(model_artifact)
submission_artifact = wandb.Artifact(name=f"{RUN_NAME}_submission", type="submission")
submission_artifact.add_file(str(submission_path))
wandb.log_artifact(submission_artifact)
wandb.finish()
