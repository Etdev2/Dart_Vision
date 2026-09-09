"""The Brain: a shared backbone feeding the two heads from #15.

A small top-down feature pyramid brings the backbone's multi-scale features to
one stride, because both heads need spatial precision that a stride-32 map
cannot carry. #15's precision budget puts the scoring rings at 17-24 px in the
image, so at stride 4 a ring spans 4-6 cells -- enough for a peak plus its
neighbourhood, which is what soft-argmax needs.

``Brain`` is a protocol rather than a base class, so #15's YOLOX challenger can
be dropped in behind the same interface without inheriting anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from dartvision.model.heads import LandmarkHead, TipHead

__all__ = ["BrainOutputs", "Brain", "DartVisionBrain"]


@dataclass
class BrainOutputs:
    """What every Brain implementation must produce."""

    landmark_heat: Tensor  # (B, L, H, W) logits
    tip_heat: Tensor       # (B, 1, H, W) logits
    tip_offset: Tensor     # (B, 2, H, W) cells, within +/-0.5

    @property
    def grid(self) -> tuple[int, int]:
        return (self.landmark_heat.shape[-2], self.landmark_heat.shape[-1])


@runtime_checkable
class Brain(Protocol):
    """The interface the training harness depends on, not a base class."""

    stride: int

    def __call__(self, images: Tensor) -> BrainOutputs: ...


class DartVisionBrain(nn.Module):
    """Decomposed two-head model on a `timm` backbone.

    ``backbone`` must be a permissively licensed model (#16 rules out anything
    AGPL, which excludes the Ultralytics family entirely).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        landmarks: int = 8,
        width: int = 64,
        stride: int = 4,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        import timm

        if stride not in (2, 4, 8):
            raise ValueError("stride must be 2, 4 or 8")
        self.stride = stride
        self.landmarks = landmarks

        self.backbone = timm.create_model(
            backbone, features_only=True, pretrained=pretrained
        )
        channels = self.backbone.feature_info.channels()
        reductions = self.backbone.feature_info.reduction()
        if stride not in reductions:
            raise ValueError(
                f"backbone {backbone!r} has no stride-{stride} feature map; "
                f"available strides are {reductions}"
            )
        self.target_level = reductions.index(stride)

        self.lateral = nn.ModuleList(
            nn.Conv2d(c, width, 1) for c in channels[self.target_level:]
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.landmark_head = LandmarkHead(width, landmarks=landmarks, width=width)
        self.tip_head = TipHead(width, width=width)

    def forward(self, images: Tensor) -> BrainOutputs:
        features = self.backbone(images)[self.target_level:]
        merged = self.lateral[-1](features[-1])
        for level in range(len(features) - 2, -1, -1):
            lateral = self.lateral[level](features[level])
            merged = lateral + nn.functional.interpolate(
                merged, size=lateral.shape[-2:], mode="nearest"
            )
        merged = self.smooth(merged)

        tips = self.tip_head(merged)
        return BrainOutputs(
            landmark_heat=self.landmark_head(merged),
            tip_heat=tips.heat,
            tip_offset=tips.offset,
        )

    @torch.no_grad()
    def grid_for(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Output grid for a given input size, without running the model."""
        height, width = image_size
        return (height // self.stride, width // self.stride)
