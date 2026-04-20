"""ResNet-18 regression model for car coupling detection."""

import torch
import torch.nn as nn
from torchvision import models


class CouplingRegressor(nn.Module):
    """ResNet-18 based model for x-coordinate regression."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Get parameter groups with different learning rates."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]
