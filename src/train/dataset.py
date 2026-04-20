"""Dataset preparation utilities for YOLO training."""

import json
import random
import shutil
from pathlib import Path

import yaml


DATASET_DIR = Path("dataset_yolo")


def polygon_to_bbox(points: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert polygon points to axis-aligned bounding box.

    Returns: (x_min, y_min, x_max, y_max)
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def bbox_to_yolo(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert bbox to YOLO format (normalized x_center, y_center, width, height)."""
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def process_annotations(
    json_files: list[Path],
    data_dir: Path,
    image_dir: Path,
    label_dir: Path,
) -> None:
    """Process multiple annotation files."""
    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        image_path = data_dir / data["imagePath"]
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")

        symlink_path = image_dir / image_path.name
        symlink_path.parent.mkdir(parents=True, exist_ok=True)
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(image_path.absolute())

        label_path = label_dir / (image_path.stem + ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w") as f:
            for shape in data["shapes"]:
                if shape["shape_type"] == "polygon":
                    x_min, y_min, x_max, y_max = polygon_to_bbox(shape["points"])
                    x_c, y_c, w, h = bbox_to_yolo(
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        data["imageWidth"],
                        data["imageHeight"],
                    )
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def create_dataset(
    data_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Create YOLO dataset from JSON annotations.

    Images are symlinked to save disk space. Labels are generated in YOLO format.
    Dataset is created in DATASET_DIR with train/val split.

    Returns path to dataset.yaml config file.
    """
    random.seed(seed)

    # Clean up existing dataset
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    json_files = list(data_dir.glob("*.json"))
    random.shuffle(json_files)
    split_idx = int(len(json_files) * (1 - val_ratio))
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    print(f"Creating dataset: {len(train_files)} train, {len(val_files)} val samples")

    process_annotations(
        train_files,
        data_dir,
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "labels" / "train",
    )
    process_annotations(
        val_files,
        data_dir,
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "labels" / "val",
    )

    config = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "coupling"},
    }

    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Dataset created: {DATASET_DIR}")
    return yaml_path
