import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import wandb

from jaguar.datasets import EmbeddingDataset, PKSampler, get_data
from jaguar.embedding_models import DINOv3
from jaguar.models import ArcFaceCriterion, EmbeddingProjection, CosFaceCriterion, SubCenterArcFaceCriterion, TripletCriterion
from jaguar.train import train_epoch, validate_epoch
from jaguar.validation import build_submission, compute_validation_map


project = "jaguar-reid-josefandvincent"
group = "02-loss-functions"
config = {
    "random_seed": 42,
    "data_dir": Path("./data"),  # Path("/kaggle/input/jaguar-re-id"),
    "checkpoint_dir": Path("checkpoints"),
    "embeddings_dir": Path("embeddings"),
    "input_size": 384,
    "embedding_dim": 256,
    "hidden_dim": 512,
    # arc face
    "arcface_margin": 0.5,
    "arcface_scale": 64.0,
    # subcenter arc face
    "num_subcenters": 3,
    # triplet
    "triplet_margin": 0.5,
    # cos face
    "cosface_margin": 0.35,
    "cosface_scale": 64.0,
    # training
    "dropout": 0.3,
    # "batch_size": 32,
    "batch_P": 16,  # number of identities per batch
    "batch_K": 4,  # number of samples per identity
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "patience": 10,
    "validation_split_size": 0.2,
}


torch.manual_seed(config["random_seed"])
np.random.seed(config["random_seed"])
load_dotenv()
# user_secrets = UserSecretsClient()
# os.environ["HF_TOKEN"]= user_secrets.get_secret("hf_token")
# os.environ["WANDB_API_KEY"] = user_secrets.get_secret("wandb_token")

config["checkpoint_dir"].mkdir(exist_ok=True)
config["embeddings_dir"].mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = DINOv3(device=device)
embedding_dim = embedding_model.get_embedding_dim()
criterions = [
    lambda _: TripletCriterion(margin=config["triplet_margin"]),
    lambda num_classes: ArcFaceCriterion(
        embedding_dim=config["embedding_dim"], num_classes=num_classes, margin=config["arcface_margin"], scale=config["arcface_scale"]
    ),
    lambda num_classes: SubCenterArcFaceCriterion(
        embedding_dim=config["embedding_dim"],
        num_classes=num_classes,
        margin=config["arcface_margin"],
        scale=config["arcface_scale"],
        num_subcenters=config["num_subcenters"],
    ),
    lambda num_classes: CosFaceCriterion(
        embedding_dim=config["embedding_dim"], num_classes=num_classes, margin=config["cosface_margin"], scale=config["cosface_scale"]
    ),
]

for crit_fn in criterions:
    run_name = f"{group}-{crit_fn(1).__class__.__name__.lower()}"
    checkpoint_path = config["checkpoint_dir"] / f"{run_name}_best.pth"
    submission_path = config["checkpoint_dir"] / f"{run_name}_submission.csv"

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=project, config=config, group=group, name=run_name)

    train_data, val_data, test_query, test_gallery, num_classes, label_encoder = get_data(
        config["data_dir"],
        validation_split_size=config["validation_split_size"],
        seed=config["random_seed"],
    )

    train_dataset = EmbeddingDataset(train_data, embedding_model=embedding_model, key="train", cache_folder=config["embeddings_dir"])
    pk_sampler = PKSampler(train_dataset.labels, p=config["batch_P"], k=config["batch_K"])
    train_loader = DataLoader(train_dataset, batch_sampler=pk_sampler, num_workers=4)

    val_dataset = EmbeddingDataset(val_data, embedding_model=embedding_model, key="val", cache_folder=config["embeddings_dir"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_K"] * config["batch_P"], shuffle=False, num_workers=4)
    validation_embeddings = val_dataset.get_embeddings().numpy()

    test_query_dataset = EmbeddingDataset(test_query, embedding_model=embedding_model, key="test_query", cache_folder=config["embeddings_dir"])
    test_gallery_dataset = EmbeddingDataset(test_gallery, embedding_model=embedding_model, key="test_gallery", cache_folder=config["embeddings_dir"])

    model = EmbeddingProjection(
        input_dim=embedding_model.get_embedding_dim(),
        hidden_dim=config["hidden_dim"],
        output_dim=config["embedding_dim"],
        dropout=config["dropout"],
    ).to(device)

    criterion = crit_fn(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    wandb.log(
        {
            "num_parameters": sum(p.numel() for p in model.parameters())
            + sum(p.numel() for p in embedding_model.model.parameters())
            + sum(p.numel() for p in criterion.parameters())
        }
    )

    best_val_loss = float("inf")
    best_map = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(config["num_epochs"]):

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_map = compute_validation_map(model, validation_embeddings, val_data["ground_truth"].values, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/map": val_map,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch+1:>2}/{config['num_epochs']}, train/loss: {train_loss:>6.4f}, val/loss: {val_loss:>6.4f}, val/mAP: {val_map:>6.4f}, lr: {current_lr:>8.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_map = val_map
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val/loss": val_loss,
                    "val/map": val_map,
                    "config": config,
                    "label_encoder_classes": label_encoder.classes_.tolist(),
                    "num_classes": num_classes,
                },
                checkpoint_path,
            )

        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            break

    print(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val mAP: {best_map:.4f})")

    wandb.run.summary["best_val_mAP"] = best_map
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["total_epochs"] = epoch + 1

    build_submission(submission_path, model, test_query_dataset, test_gallery_dataset, device)

    model_artifact = wandb.Artifact(
        name=f"{run_name}_model",
        type="model",
    )
    model_artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(model_artifact)

    submission_artifact = wandb.Artifact(name="submission", type="submission")
    submission_artifact.add_file(str(submission_path))
    wandb.log_artifact(submission_artifact)
    wandb.finish()
