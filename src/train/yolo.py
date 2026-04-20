"""YOLO-based car coupling detection."""

import shutil
from pathlib import Path

import ultralytics

from train.dataset import create_dataset


def train(
    data_dir: Path = Path("car_coupling_train"),
    checkpoint_dir: Path = Path("checkpoints"),
    model_size: str = "n",
    epochs: int = 100,
    patience: int = 20,
    imgsz: int = 640,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Train YOLO model for coupling detection."""
    yaml_path = create_dataset(data_dir, val_ratio, seed)

    model_name = f"yolo26{model_size}.pt"
    print(f"Loading pretrained model: {model_name}")
    model = ultralytics.YOLO(model_name)

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch_size,
        exist_ok=True,
        seed=seed,
        verbose=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        fliplr=0.5,
    )

    best_model_path = Path("runs/detect/train/weights/best.pt")
    if best_model_path.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target_path = checkpoint_dir / "yolo_best.pt"
        shutil.copy(best_model_path, target_path)
        print(f"Best model saved to: {target_path}")


def predict(
    model: ultralytics.YOLO, image_path: Path, conf: float = 0.2
) -> float | None:
    """Predict x-coordinate of coupling from image.

    Returns the center x-coordinate of the detected bounding box,
    or None if no coupling is detected.
    """
    results = model(str(image_path), conf=conf, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    best_idx = boxes.conf.argmax()
    box = boxes.xyxy[best_idx].cpu().numpy()

    return float((box[0] + box[2]) / 2)
