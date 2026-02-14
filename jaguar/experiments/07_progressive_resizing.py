"""
Experiment 07 – Progressive Resizing
=====================================
Research question: Does progressive resizing improve identity-balanced mAP?

Ablation schedules (all share the same total epoch budget of 100):
  - fixed_128:              train at 128 for 100 epochs
  - fixed_256:              train at 256 for 100 epochs
  - fixed_384:              train at 384 for 100 epochs  (≈ baseline)
  - progressive_128_256_384: 128 for 30 → 256 for 30 → 384 for 40 epochs
  - progressive_256_384:    256 for 40 → 384 for 60 epochs

Each schedule is an independent W&B run with identical model, optimizer, and
hyperparameters. Only the image resolution schedule differs.

Logged per stage: resolution, stage duration (seconds), stage-end mAP.
Logged globally: best mAP, best epoch, total training time.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import wandb

from jaguar.components import MegaDescriptor, VielleichtguarModel, EmbeddingProjection
from jaguar.criteria import ArcFaceCriterion
from jaguar.datasets import get_dataloaders
from jaguar.submission import build_submission
from jaguar.train import train_epoch, validate_epoch

# ── W&B identifiers ──────────────────────────────────────────────────────────
PROJECT = "jaguar-reid-josefandvincent"
GROUP = "07_progressive_resizing"

# ── Shared configs ────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "random_seed": 42,
    "data_dir": Path("./data"),
    "checkpoint_dir": Path("checkpoints"),
    "cache_dir": Path("./cache"),
    "embeddings_dir": Path("./embeddings"),
    "validation_split_size": 0.2,
}

EXPERIMENT_CONFIG = {
    "batch_size": 64,
    "hidden_dim": 512,
    "output_dim": 256,
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "learning_rate": 5e-4,
    "arcface_margin": 0.5,
    "arcface_scale": 64.0,
    "patience": 10,
    "train_backbone": False,
}

# ── Progressive resizing schedules ───────────────────────────────────────────
# Each schedule is a list of (image_size, max_epochs_in_stage) tuples.
# The total epoch budget across all stages is kept at 100.
SCHEDULES = {
    "fixed_128": [(128, 100)],
    "fixed_256": [(256, 100)],
    "fixed_384": [(384, 100)],
    "progressive_128_256_384": [(128, 30), (256, 30), (384, 40)],
    "progressive_256_384": [(256, 40), (384, 60)],
}

device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()


def run_schedule(schedule_name: str, stages: list[tuple[int, int]]):
    """Train one full schedule (a sequence of resolution stages)."""

    torch.manual_seed(BASE_CONFIG["random_seed"])
    np.random.seed(BASE_CONFIG["random_seed"])

    run_name = f"{GROUP}-{schedule_name}"

    total_stage_epochs = sum(ep for _, ep in stages)
    schedule_config = {
        **EXPERIMENT_CONFIG,
        **BASE_CONFIG,
        "schedule_name": schedule_name,
        "stages": str(stages),
        "total_epoch_budget": total_stage_epochs,
    }

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project=PROJECT, config=schedule_config, group=GROUP, name=run_name, reinit=True)

    BASE_CONFIG["checkpoint_dir"].mkdir(exist_ok=True)
    checkpoint_path = BASE_CONFIG["checkpoint_dir"] / f"{run_name}_best.pth"
    submission_path = BASE_CONFIG["checkpoint_dir"] / f"{run_name}_submission.csv"

    # ── Build model (resolution-agnostic) ─────────────────────────────────
    backbone = MegaDescriptor(
        freeze=not EXPERIMENT_CONFIG["train_backbone"],
        cache_folder=BASE_CONFIG["embeddings_dir"],
    )
    base_transforms = backbone.get_transforms()

    # We need num_classes for ArcFace — get it from an initial dataloader call.
    # Use the first stage's resolution; num_classes is resolution-independent.
    first_size = stages[0][0]
    _, _, _, num_classes, label_encoder = get_dataloaders(
        data_dir=BASE_CONFIG["data_dir"],
        validation_split_size=BASE_CONFIG["validation_split_size"],
        seed=BASE_CONFIG["random_seed"],
        batch_size=EXPERIMENT_CONFIG["batch_size"],
        image_size=first_size,
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
            margin=EXPERIMENT_CONFIG["arcface_margin"],
            scale=EXPERIMENT_CONFIG["arcface_scale"],
        ),
    ).to(device)

    wandb.log({
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    })

    optimizer = AdamW(
        model.parameters(),
        lr=EXPERIMENT_CONFIG["learning_rate"],
        weight_decay=EXPERIMENT_CONFIG["weight_decay"],
    )
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # ── Training loop across stages ───────────────────────────────────────
    global_epoch = 0
    best_epoch, best_map = 0, 0.0
    patience_counter = 0
    total_duration = 0.0
    early_stopped = False

    for stage_idx, (image_size, stage_max_epochs) in enumerate(stages):
        stage_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"  Stage {stage_idx + 1}/{len(stages)}: resolution {image_size}×{image_size}, up to {stage_max_epochs} epochs")
        print(f"{'='*60}\n")

        wandb.log({"stage": stage_idx + 1, "stage_image_size": image_size})

        # Create fresh dataloaders at the new resolution
        train_dl, val_dl, test_dl, _, _ = get_dataloaders(
            data_dir=BASE_CONFIG["data_dir"],
            validation_split_size=BASE_CONFIG["validation_split_size"],
            seed=BASE_CONFIG["random_seed"],
            batch_size=EXPERIMENT_CONFIG["batch_size"],
            image_size=image_size,
            cache_dir=BASE_CONFIG["cache_dir"],
            train_process_fn=base_transforms,
            val_process_fn=base_transforms,
            mode="background",
        )

        # Reset patience at stage transitions so the model can adapt to the
        # new resolution before being penalised for stagnation.
        if stage_idx > 0:
            patience_counter = 0

        for stage_epoch in range(stage_max_epochs):
            global_epoch += 1

            start = time.time()
            train_loss = train_epoch(model, train_dl, optimizer, device)
            val_loss, val_map = validate_epoch(model, val_dl, device)
            end = time.time()

            epoch_time = end - start
            total_duration += epoch_time
            remaining = total_stage_epochs - global_epoch
            eta = epoch_time * remaining

            lr_scheduler.step(val_loss)

            print(
                f"epoch: {global_epoch:>3}/{total_stage_epochs} "
                f"[stage {stage_idx+1} @ {image_size}px] | "
                f"train/loss: {train_loss:>8.4f} | "
                f"val/loss: {val_loss:>8.4f} | "
                f"val/mAP: {val_map:>7.4f} | "
                f"lr: {optimizer.param_groups[0]['lr']:>7.1e} | "
                + (f"eta: {max(0, eta)/60:.1f} min" if eta > 60 else f"eta: {max(0, eta):.1f} sec"),
            )

            wandb.log({
                "epoch": global_epoch,
                "stage": stage_idx + 1,
                "stage_image_size": image_size,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mAP": val_map,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if val_map > best_map:
                best_map = val_map
                best_epoch = global_epoch
                patience_counter = 0
                model.save_model(
                    checkpoint_path,
                    with_backbone=EXPERIMENT_CONFIG["train_backbone"],
                    with_criterion=False,
                )
            else:
                patience_counter += 1

            if patience_counter >= EXPERIMENT_CONFIG["patience"]:
                print(f"  ⏹ Early stopping triggered at global epoch {global_epoch}")
                early_stopped = True
                break

        # Log per-stage summary
        stage_duration = time.time() - stage_start_time
        wandb.log({
            f"stage_{stage_idx+1}_resolution": image_size,
            f"stage_{stage_idx+1}_duration_sec": stage_duration,
            f"stage_{stage_idx+1}_end_val_map": val_map,
        })
        print(f"  Stage {stage_idx+1} done — {stage_duration:.1f}s, val/mAP={val_map:.4f}")

        if early_stopped:
            break

    # ── Submission & logging ──────────────────────────────────────────────
    # Always evaluate at the highest resolution (384) for a fair submission
    _, _, test_dl_final, _, _ = get_dataloaders(
        data_dir=BASE_CONFIG["data_dir"],
        validation_split_size=BASE_CONFIG["validation_split_size"],
        seed=BASE_CONFIG["random_seed"],
        batch_size=EXPERIMENT_CONFIG["batch_size"],
        image_size=384,
        cache_dir=BASE_CONFIG["cache_dir"],
        train_process_fn=base_transforms,
        val_process_fn=base_transforms,
        mode="background",
    )

    model.load_model(checkpoint_path)
    build_submission(submission_path, model, test_dl_final, device)

    wandb.run.summary["best_val_mAP"] = best_map
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["total_epochs"] = global_epoch
    wandb.run.summary["total_training_time"] = total_duration
    wandb.run.summary["schedule_name"] = schedule_name

    model_artifact = wandb.Artifact(name=f"{run_name}_model", type="model")
    model_artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(model_artifact)

    submission_artifact = wandb.Artifact(name=f"{run_name}_submission", type="submission")
    submission_artifact.add_file(str(submission_path))
    wandb.log_artifact(submission_artifact)

    wandb.finish()

    return best_map, best_epoch, global_epoch, total_duration


# ── Main: run all ablation schedules ──────────────────────────────────────────
if __name__ == "__main__":
    results = {}
    for schedule_name, stages in SCHEDULES.items():
        print(f"\n{'#'*70}")
        print(f"  Schedule: {schedule_name}  |  stages: {stages}")
        print(f"{'#'*70}")
        best_map, best_epoch, total_epochs, duration = run_schedule(schedule_name, stages)
        results[schedule_name] = {
            "best_mAP": best_map,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "duration_sec": duration,
        }

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Progressive Resizing – Ablation Summary")
    print(f"{'='*80}")
    print(f"{'Schedule':<30} {'Best mAP':>10} {'Best Ep':>10} {'Epochs':>10} {'Time (min)':>12}")
    print(f"{'-'*80}")
    for name, r in results.items():
        print(f"{name:<30} {r['best_mAP']:>10.4f} {r['best_epoch']:>10} {r['total_epochs']:>10} {r['duration_sec']/60:>12.1f}")
    print(f"{'='*80}")
