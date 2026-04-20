"""Train regression model for car coupling detection."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from train.regressor.dataset import IMAGE_WIDTH, create_dataloaders
from train.regressor.model import CouplingRegressor


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, avg_mae_pixels)."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            mae_pixels = torch.abs(predictions - targets).mean().item() * IMAGE_WIDTH
            total_mae += mae_pixels

        num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model. Returns (avg_loss, avg_mae_pixels)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            mae_pixels = torch.abs(predictions - targets).mean().item() * IMAGE_WIDTH
            total_mae += mae_pixels

            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def train(
    checkpoint_dir: Path = Path("checkpoints"),
    batch_size: int = 8,
    epochs: int = 100,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    patience: int = 15,
    input_size: tuple[int, int] = (256, 192),
) -> None:
    """Main training function."""
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        input_size=input_size,
    )

    model = CouplingRegressor()
    model = model.to(device)

    criterion = nn.MSELoss()
    param_groups = model.get_param_groups(backbone_lr, head_lr)
    optimizer = AdamW(param_groups)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f}, MAE: {train_mae:.1f}px | "
            f"Val Loss: {val_loss:.6f}, MAE: {val_mae:.1f}px"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / "regressor_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "input_size": input_size,
                },
                checkpoint_path,
            )
            print(f"  -> Saved best model (val_mae: {val_mae:.1f}px)")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

    print(f"\nBest validation MAE: {best_val_mae:.1f}px")
    print(f"Best model saved to: {checkpoint_dir / 'regressor_best.pt'}")
