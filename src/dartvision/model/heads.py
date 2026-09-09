"""The two prediction heads from #15.

Landmarks and dart tips have opposite structure, so they get different
representations rather than one compromise:

* **Landmarks** -- exactly four (or eight) semantically distinct points, one
  instance each. A heatmap channel per landmark fits perfectly, and soft-argmax
  recovers sub-cell position without a second output.
* **Tips** -- up to three instances of *one* class, sometimes clustered within
  millimetres. A per-channel heatmap provably cannot separate those, so tips get
  a single centre map plus explicit sub-cell offsets.

The offset head is bounded by ``tanh`` scaled to ``±0.5``, which is exactly the
range the encoder produces: an offset is ``cx - round(cx)``. Letting it predict
outside that range would only ever be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["LandmarkHead", "TipHead", "TipPrediction"]


def _trunk(in_channels: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    )


class LandmarkHead(nn.Module):
    """One heatmap logit per landmark."""

    def __init__(self, in_channels: int, landmarks: int = 8, width: int = 64) -> None:
        super().__init__()
        if landmarks not in (4, 8):
            raise ValueError("landmarks must be 4 or 8")
        self.landmarks = landmarks
        self.trunk = _trunk(in_channels, width)
        self.out = nn.Conv2d(width, landmarks, 1)
        # A heatmap is almost entirely background, so start pessimistic rather
        # than letting the first steps be dominated by suppressing false peaks.
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, features: Tensor) -> Tensor:
        return self.out(self.trunk(features))


@dataclass
class TipPrediction:
    """Raw tip outputs: centre logits and bounded sub-cell offsets."""

    heat: Tensor    # (B, 1, H, W) logits
    offset: Tensor  # (B, 2, H, W) in cells, x then y, within +/-0.5


class TipHead(nn.Module):
    """A centre heatmap plus sub-cell offsets, sharing one trunk."""

    def __init__(self, in_channels: int, width: int = 64) -> None:
        super().__init__()
        self.trunk = _trunk(in_channels, width)
        self.heat = nn.Conv2d(width, 1, 1)
        self.offset = nn.Conv2d(width, 2, 1)
        nn.init.constant_(self.heat.bias, -4.0)
        nn.init.zeros_(self.offset.bias)

    def forward(self, features: Tensor) -> TipPrediction:
        shared = self.trunk(features)
        # The encoder's offsets are cx - round(cx), so they live in [-0.5, 0.5].
        # Bounding the head to that range removes predictions that cannot be right.
        return TipPrediction(
            heat=self.heat(shared),
            offset=0.5 * torch.tanh(self.offset(shared)),
        )
