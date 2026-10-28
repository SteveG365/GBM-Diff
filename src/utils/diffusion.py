import csv
from pathlib import Path
import torch

def append_history(
    history_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    learning_rate: float
):

    file_exists = history_path.exists()

    with open(history_path, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "learning_rate"
                ]
            )

        writer.writerow(
            [
                epoch,
                train_loss,
                val_loss,
                learning_rate
            ]
        )

def save_checkpoint(
    path: Path,
    model,
    optimiser,
    scaler,
    epoch: int,
    best_val_loss: float,
    latent_scale: float,
    config: dict
):

    model_to_save = (
        model._orig_mod
        if hasattr(model, "_orig_mod")
        else model
    )

    torch.save(
        {
            "model": model_to_save.state_dict(),
            "optimizer": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "latent_scale": latent_scale,
            "config": config
        },
        path
    )

def load_diffusion_from_checkpoint(
    path,
    model,
    device
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model