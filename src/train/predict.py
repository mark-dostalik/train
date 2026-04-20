"""Predict car coupling x-coordinates using YOLO with regressor fallback."""

import argparse
from pathlib import Path

import torch
import ultralytics

from train.regressor.predict import load_model as load_regressor_model
from train.regressor.predict import predict as predict_regressor
from train.yolo import predict as predict_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict car coupling x-coordinates")
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="Image files to process",
    )
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=Path("checkpoints/yolo_best.pt"),
        help="Path to YOLO model checkpoint",
    )
    parser.add_argument(
        "--regressor-checkpoint",
        type=Path,
        default=Path("checkpoints/regressor_best.pt"),
        help="Path to regressor model checkpoint",
    )
    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    yolo_model = ultralytics.YOLO(str(args.yolo_checkpoint))
    regressor_model, input_size = load_regressor_model(
        args.regressor_checkpoint, device
    )

    for image_path in args.images:
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        x = predict_yolo(yolo_model, image_path)
        if x is None:
            x = predict_regressor(regressor_model, image_path, device, input_size)

        print(int(round(x)))


if __name__ == "__main__":
    main()
