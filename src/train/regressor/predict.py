"""Predict car coupling x-coordinates using regression model."""

from pathlib import Path

import torch
from PIL import Image

from train.regressor.dataset import IMAGE_WIDTH, CouplingDataset
from train.regressor.model import CouplingRegressor


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[CouplingRegressor, tuple[int, int]]:
    """Load trained model. Returns (model, input_size)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = CouplingRegressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["input_size"]


def predict(
    model: CouplingRegressor,
    image_path: Path,
    device: torch.device,
    input_size: tuple[int, int] = (256, 192),
) -> float:
    """Predict x-coordinate using regression model."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = CouplingDataset.preprocess(image, input_size).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_coord = model(image_tensor).item()

    return pred_coord * IMAGE_WIDTH
