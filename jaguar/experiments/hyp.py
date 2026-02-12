import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.transforms.v2 as v2
from pathlib import Path
import numpy as np
import wandb

# Assuming your jaguar module is in the python path
from jaguar.components import DINOv3, EmbeddingProjection, VielleichtguarModel
from jaguar.criteria import ArcFaceCriterion
from jaguar.datasets import get_dataloaders
from jaguar.train import train_epoch, validate_epoch

# --- 1. SWEEP CONFIGURATION ---
SWEEP_CONFIG = {
    "method": "bayes",  # Bayesian optimization
    "metric": {"name": "val/mAP", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-4},
        "backbone_lr_mult": {"values": [0.1, 0.01]},  # Backbone trains slower than head
        "arcface_margin": {"distribution": "uniform", "min": 0.3, "max": 0.5},
        "arcface_scale": {"values": [32.0, 64.0]},
        "weight_decay": {"values": [1e-4, 1e-3]},
        "dropout": {"values": [0.2, 0.3, 0.4]},
        "batch_size": {"values": [16, 32]},
    },
}

# --- 2. GLOBAL SETTINGS ---
PROJECT = "jaguar-reid-josefandvincent"
BASE_CONFIG = {
    "random_seed": 456,
    "data_dir": Path("./data"),
    "checkpoint_dir": Path("checkpoints"),
    "cache_dir": Path("./cache"),
    "embeddings_dir": Path("./embeddings"),
    "validation_split_size": 0.2,
    "image_size": 256,
    "output_dim": 256,
    "hidden_dim": 512,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    # Initialize W&B for this specific trial
    with wandb.init() as run:
        config = run.config

        # Set seeds for reproducibility within the trial
        torch.manual_seed(BASE_CONFIG["random_seed"])
        np.random.seed(BASE_CONFIG["random_seed"])

        # --- Data Setup ---
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

        train_loader, val_loader, _, num_classes, _ = get_dataloaders(
            data_dir=BASE_CONFIG["data_dir"],
            validation_split_size=BASE_CONFIG["validation_split_size"],
            seed=BASE_CONFIG["random_seed"],
            batch_size=config.batch_size,
            image_size=BASE_CONFIG["image_size"],
            cache_dir=BASE_CONFIG["cache_dir"],
            train_process_fn=augmentation_transforms,
            val_process_fn=base_transforms,
            mode="background",
        )

        # --- Model Setup ---
        model = VielleichtguarModel(
            backbone=backbone,
            layers=nn.Sequential(
                EmbeddingProjection(
                    input_dim=backbone.out_dim(),
                    hidden_dim=BASE_CONFIG["hidden_dim"],
                    output_dim=BASE_CONFIG["output_dim"],
                    dropout=config.dropout,
                ),
            ),
            criterion=ArcFaceCriterion(
                embedding_dim=BASE_CONFIG["output_dim"],
                num_classes=num_classes,
                margin=config.arcface_margin,
                scale=config.arcface_scale,
            ),
        ).to(device)

        # --- Optimizer with Differential Learning Rates ---
        # We apply the multiplier to the backbone parameters only
        params = [
            {"params": model.backbone.parameters(), "lr": config.learning_rate * config.backbone_lr_mult},
            {"params": model.layers.parameters(), "lr": config.learning_rate},
            {"params": model.criterion.parameters(), "lr": config.learning_rate},
        ]
        optimizer = AdamW(params, weight_decay=config.weight_decay)

        # --- Training Loop ---
        best_map = 0.0
        # For sweeps, we often run fewer epochs to explore more combinations
        max_epochs = 30
        patience = 7
        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_map = validate_epoch(model, val_loader, device)

            print(f"Epoch {epoch+1}/{max_epochs} | " f"Train Loss: {train_loss:.4f} | " f"Val Loss: {val_loss:.4f} | " f"Val mAP: {val_map:.4f}")

            wandb.log({"epoch": epoch + 1, "train/loss": train_loss, "val/loss": val_loss, "val/mAP": val_map, "lr_head": optimizer.param_groups[1]["lr"]})

            if val_map > best_map:
                best_map = val_map
                patience_counter = 0
                # In a sweep, we usually don't save every "best" to disk to save space,
                # but we keep track of the performance for the Bayesian agent.
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break


# --- 3. EXECUTION ---
if __name__ == "__main__":
    BASE_CONFIG["checkpoint_dir"].mkdir(exist_ok=True)

    # 1. Login
    wandb.login()

    # 2. Create the sweep ID
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT)

    # 3. Run the agent (set count to however many trials you want to run)
    # If you have multiple GPUs/machines, run this same script on all of them
    # using the same sweep_id to parallelize the search.
    wandb.agent(sweep_id, function=train, count=30)
