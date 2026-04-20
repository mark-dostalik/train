"""Train both YOLO and regressor models for car coupling detection."""

import argparse
from pathlib import Path

from train.yolo import train as train_yolo
from train.regressor.train import train as train_regressor


def train(
    data_dir: Path = Path("car_coupling_train"),
    checkpoint_dir: Path = Path("checkpoints"),
    yolo_epochs: int = 100,
    regressor_epochs: int = 100,
    seed: int = 42,
) -> None:
    """Train both YOLO and regressor models."""
    print("=" * 60)
    print("Training YOLO model")
    print("=" * 60)
    train_yolo(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=yolo_epochs,
        seed=seed,
    )

    print()
    print("=" * 60)
    print("Training regressor model")
    print("=" * 60)
    train_regressor(
        checkpoint_dir=checkpoint_dir,
        epochs=regressor_epochs,
    )

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"YOLO model: {checkpoint_dir / 'yolo_best.pt'}")
    print(f"Regressor model: {checkpoint_dir / 'regressor_best.pt'}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train coupling detector models")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("car_coupling_train"),
        help="Path to training data directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--yolo-epochs",
        type=int,
        default=100,
        help="Maximum epochs for YOLO training",
    )
    parser.add_argument(
        "--regressor-epochs",
        type=int,
        default=100,
        help="Maximum epochs for regressor training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        yolo_epochs=args.yolo_epochs,
        regressor_epochs=args.regressor_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
