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

from jaguar.components import DINOv3, MegaDescriptor, VielleichtguarModel, EmbeddingProjection
from jaguar.criteria import ArcFaceCriterion
from jaguar.datasets import get_dataloaders
from jaguar.submission import build_submission
from jaguar.train import train_epoch, validate_epoch

PROJECT = "jaguar-reid-josefandvincent"
GROUP = "13_optimizer"

BASE_CONFIG = {
    "random_seeds": [42, 123, 456, 789, 2024],  # Multiple seeds for stability analysis
    "data_dir": Path("./data"),  # Path("/kaggle/input/jaguar-re-id"),
    "checkpoint_dir": Path("checkpoints"),
    "cache_dir": Path("./cache"),
    "embeddings_dir": Path("./embeddings"),
    "validation_split_size": 0.2,
}

EXPERIMENT_CONFIGS = [
    {
        "epochs": 100,
        "batch_size": 64,
        "image_size": 256,
        "hidden_dim": 512,
        "output_dim": 256,
        "dropout": 0.3,
        "weight_decay": 1e-4,
        "learning_rate": 5e-4,
        "arcface_margin": 0.5,
        "arcface_scale": 64.0,
        "patience": 10,
        "train_backbone": False,
        "optimizer": "Adam",
    },
    {
        "epochs": 100,
        "batch_size": 64,
        "image_size": 256,
        "hidden_dim": 512,
        "output_dim": 256,
        "dropout": 0.3,
        "weight_decay": 1e-4,
        "learning_rate": 5e-4,
        "arcface_margin": 0.5,
        "arcface_scale": 64.0,
        "patience": 10,
        "train_backbone": False,
        "optimizer": "AdamW",
    },
    {
        "epochs": 100,
        "batch_size": 64,
        "image_size": 256,
        "hidden_dim": 512,
        "output_dim": 256,
        "dropout": 0.3,
        "weight_decay": 1e-4,
        "learning_rate": 5e-4,
        "arcface_margin": 0.5,
        "arcface_scale": 64.0,
        "patience": 10,
        "train_backbone": False,
        "optimizer": "SGD",
        "momentum": 0.9,
    },
    {
        "epochs": 100,
        "batch_size": 64,
        "image_size": 256,
        "hidden_dim": 512,
        "output_dim": 256,
        "dropout": 0.3,
        "weight_decay": 1e-4,
        "learning_rate": 5e-4,
        "arcface_margin": 0.5,
        "arcface_scale": 64.0,
        "patience": 10,
        "train_backbone": False,
        "optimizer": "SGD_Nesterov",
        "momentum": 0.9,
    },
    {
        "epochs": 100,
        "batch_size": 64,
        "image_size": 256,
        "hidden_dim": 512,
        "output_dim": 256,
        "dropout": 0.3,
        "weight_decay": 1e-4,
        "learning_rate": 5e-4,
        "arcface_margin": 0.5,
        "arcface_scale": 64.0,
        "patience": 10,
        "train_backbone": False,
        "optimizer": "RMSprop",
        "momentum": 0.9,
    },
]

device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()

# Store results across seeds for each optimizer
optimizer_results = {}  # {optimizer_name: {seed: {mAP, diverged, etc.}}}

for EXPERIMENT_CONFIG in EXPERIMENT_CONFIGS:
    optimizer_name = EXPERIMENT_CONFIG["optimizer"]
    optimizer_results[optimizer_name] = {}

    for random_seed in BASE_CONFIG["random_seeds"]:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        run_name = f"{GROUP}-{optimizer_name}-seed{random_seed}"
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        config_with_seed = {**EXPERIMENT_CONFIG, **BASE_CONFIG, "current_seed": random_seed}
        wandb.init(project=PROJECT, config=config_with_seed, group=GROUP, name=run_name, reinit=True)

        BASE_CONFIG["checkpoint_dir"].mkdir(exist_ok=True)
        checkpoint_path = BASE_CONFIG["checkpoint_dir"] / f"{run_name}_best.pth"
        submission_path = BASE_CONFIG["checkpoint_dir"] / f"{run_name}_submission.csv"

        backbone = DINOv3(freeze=not EXPERIMENT_CONFIG["train_backbone"], cache_folder=BASE_CONFIG["embeddings_dir"])
        base_transforms = backbone.get_transforms()

        train_dataloader, validation_dataloader, test_dataloader, num_classes, label_encoder = get_dataloaders(
            data_dir=BASE_CONFIG["data_dir"],
            validation_split_size=BASE_CONFIG["validation_split_size"],
            seed=random_seed,
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
                margin=EXPERIMENT_CONFIG["arcface_margin"],
                scale=EXPERIMENT_CONFIG["arcface_scale"],
            ),
        ).to(device)

        wandb.log(
            {
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in model.parameters()),
            }
        )

        if EXPERIMENT_CONFIG["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=EXPERIMENT_CONFIG["learning_rate"],
                weight_decay=EXPERIMENT_CONFIG["weight_decay"],
            )
        elif EXPERIMENT_CONFIG["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=EXPERIMENT_CONFIG["learning_rate"],
                weight_decay=EXPERIMENT_CONFIG["weight_decay"],
            )
        elif EXPERIMENT_CONFIG["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=EXPERIMENT_CONFIG["learning_rate"],
                weight_decay=EXPERIMENT_CONFIG["weight_decay"],
                momentum=EXPERIMENT_CONFIG.get("momentum", 0.9),
            )
        elif EXPERIMENT_CONFIG["optimizer"] == "SGD_Nesterov":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=EXPERIMENT_CONFIG["learning_rate"],
                weight_decay=EXPERIMENT_CONFIG["weight_decay"],
                momentum=EXPERIMENT_CONFIG.get("momentum", 0.9),
                nesterov=True,
            )
        elif EXPERIMENT_CONFIG["optimizer"] == "RMSprop":
            optimizer = torch.optim.RMSprop(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=EXPERIMENT_CONFIG["learning_rate"],
                weight_decay=EXPERIMENT_CONFIG["weight_decay"],
                momentum=EXPERIMENT_CONFIG.get("momentum", 0.9),
            )
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Stability tracking metrics
        best_epoch, best_map, patience_counter, total_duration, eta = 0, 0.0, 0, 0.0, 0.0
        diverged = False
        train_losses, val_losses, val_maps = [], [], []
        gradient_norms = []

        for epoch in range(EXPERIMENT_CONFIG["epochs"]):

            start = time.time()
            train_loss = train_epoch(model, train_dataloader, optimizer, device)
            validation_loss, validation_map = validate_epoch(model, validation_dataloader, device)

            # Check for divergence
            if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
                print(f"Training diverged at epoch {epoch+1} (NaN/Inf loss)")
                diverged = True
                break

            # Compute gradient norm for stability analysis
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm**0.5
            gradient_norms.append(total_grad_norm)

            train_losses.append(train_loss)
            val_losses.append(validation_loss)
            val_maps.append(validation_map)

            end = time.time()
            epoch_time = end - start
            total_duration += epoch_time
            eta = (epoch_time if epoch == 0 else 0.9 * (eta / (EXPERIMENT_CONFIG["epochs"] - epoch)) + 0.1 * epoch_time) * (
                EXPERIMENT_CONFIG["epochs"] - epoch - 1
            )
            lr_scheduler.step(validation_loss)

            print(
                f"epoch: {epoch+1:>2}/{EXPERIMENT_CONFIG['epochs']} | ",
                f"train/loss: {train_loss:>8.4f} | ",
                f"val/loss: {validation_loss:>8.4f} | ",
                f"val/mAP: {validation_map:>7.4f} | ",
                f"lr: {optimizer.param_groups[0]['lr']:>7.1e} | ",
                f"grad_norm: {total_grad_norm:>7.2f} | ",
                f"eta: {max(0,eta)/60:.1f} min" if max(0, eta) > 60 else f"eta: {max(0,eta):.1f} sec",
                sep="",
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": validation_loss,
                    "val/mAP": validation_map,
                    "lr": optimizer.param_groups[0]["lr"],
                    "gradient_norm": total_grad_norm,
                }
            )

            if validation_map > best_map:
                best_map = validation_map
                best_epoch = epoch + 1
                patience_counter = 0
                model.save_model(
                    checkpoint_path,
                    with_backbone=EXPERIMENT_CONFIG["train_backbone"],
                    with_criterion=False,
                )
            else:
                patience_counter += 1

            if patience_counter >= EXPERIMENT_CONFIG["patience"]:
                break

        # Compute stability metrics
        if not diverged:
            model.load_model(checkpoint_path)
            build_submission(submission_path, model, test_dataloader, device)

            loss_variance = np.var(train_losses)
            map_variance = np.var(val_maps)
            grad_norm_mean = np.mean(gradient_norms)
            grad_norm_std = np.std(gradient_norms)
        else:
            loss_variance = float("inf")
            map_variance = float("inf")
            grad_norm_mean = float("inf")
            grad_norm_std = float("inf")

        # Store results for this seed
        optimizer_results[optimizer_name][random_seed] = {
            "mAP": best_map if not diverged else 0.0,
            "diverged": diverged,
            "total_epochs": epoch + 1,
            "loss_variance": loss_variance,
            "map_variance": map_variance,
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_std": grad_norm_std,
        }

        wandb.run.summary["best_val_mAP"] = best_map
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["total_epochs"] = epoch + 1
        wandb.run.summary["total_training_time"] = total_duration
        wandb.run.summary["diverged"] = diverged
        wandb.run.summary["loss_variance"] = loss_variance
        wandb.run.summary["map_variance"] = map_variance
        wandb.run.summary["grad_norm_mean"] = grad_norm_mean
        wandb.run.summary["grad_norm_std"] = grad_norm_std

        if not diverged:
            model_artifact = wandb.Artifact(name=f"{run_name}_model", type="model")
            model_artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(model_artifact)
            submission_artifact = wandb.Artifact(name=f"{run_name}_submission", type="submission")
            submission_artifact.add_file(str(submission_path))
            wandb.log_artifact(submission_artifact)

        wandb.finish()

# Aggregate results across seeds for each optimizer
print("\n" + "=" * 80)
print("OPTIMIZER COMPARISON RESULTS")
print("=" * 80)
for optimizer_name, seed_results in optimizer_results.items():
    maps = [r["mAP"] for r in seed_results.values() if not r["diverged"]]
    divergence_rate = sum(1 for r in seed_results.values() if r["diverged"]) / len(seed_results)

    if maps:
        mean_map = np.mean(maps)
        std_map = np.std(maps)
        print(f"\n{optimizer_name}:")
        print(f"  Mean mAP: {mean_map:.4f} ± {std_map:.4f}")
        print(f"  Divergence rate: {divergence_rate:.2%}")
        print(f"  Successful runs: {len(maps)}/{len(seed_results)}")
    else:
        print(f"\n{optimizer_name}:")
        print(f"  All runs diverged!")
        print(f"  Divergence rate: 100%")
