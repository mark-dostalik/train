"""Dataset utilities for regression-based car coupling detection."""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATASET_DIR = Path("dataset_yolo")
IMAGE_WIDTH = 2048

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CouplingDataset(Dataset):
    """Dataset for car coupling x-coordinate regression."""

    def __init__(
        self,
        annotations: list[dict],
        augment: bool = False,
        input_size: tuple[int, int] = (256, 192),
    ):
        self.annotations = annotations
        self.augment = augment
        self.input_size = input_size

        self.color_aug = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ann = self.annotations[idx]

        image = Image.open(ann["image_path"]).convert("RGB")
        target_x = ann["center_x"]

        if self.augment:
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                target_x = 1.0 - target_x

            image = self.color_aug(image)

        image = self.preprocess(image, self.input_size)

        return image, torch.tensor(target_x, dtype=torch.float32)

    @classmethod
    def preprocess(
        cls,
        image: Image.Image,
        input_size: tuple[int, int] = (256, 192),
    ) -> torch.Tensor:
        """Preprocess image for inference."""
        return transforms.Compose(
            [
                transforms.Resize((input_size[1], input_size[0])),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )(image)


def load_annotations(split: str) -> list[dict]:
    """Load annotations from YOLO dataset for a given split (train/val)."""
    annotations = []

    labels_dir = DATASET_DIR / "labels" / split
    images_dir = DATASET_DIR / "images" / split

    for label_path in labels_dir.glob("*.txt"):
        for suffix in [".jpg", ".jpeg"]:
            image_path = images_dir / label_path.with_suffix(suffix).name
            if image_path.exists():
                break
        else:
            print(f"Warning: Image not found for {label_path}")
            continue

        with open(label_path) as f:
            line = f.readline().strip()

        parts = line.split()
        center_x = float(parts[1])

        annotations.append({"image_path": image_path, "center_x": center_x})

    return annotations


def create_dataloaders(
    batch_size: int = 8,
    input_size: tuple[int, int] = (256, 192),
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    train_ann = load_annotations("train")
    val_ann = load_annotations("val")

    print(f"Training samples: {len(train_ann)}")
    print(f"Validation samples: {len(val_ann)}")

    train_loader = DataLoader(
        CouplingDataset(train_ann, augment=True, input_size=input_size),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        CouplingDataset(val_ann, augment=False, input_size=input_size),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader
