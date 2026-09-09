"""A PyTorch dataset over a scene manifest.

Reads annotations and their images, resizes to a fixed input, and encodes the
targets both heads need. Kept deliberately thin: augmentation beyond
photometric jitter is #18's question, and guessing at it now would bake in
choices that ticket exists to measure.

**Geometric augmentation is deliberately absent**, and that is a decision
rather than an omission. Any transform that moves pixels must move the labels
with them, and two of the obvious candidates are subtler than they look:

* A horizontal flip produces a *mirrored* dartboard -- a board whose sector
  order runs the wrong way and which does not exist in reality. The reference
  implementation used flips anyway (#13); whether that trades realism for
  invariance profitably is exactly an #18 experiment.
* Rotation is valid only in 90-degree steps, because our landmarks sit 90
  degrees apart: such a rotation permutes the landmark *identities* cyclically,
  so the channels must be permuted too. Rotating by anything else leaves the
  landmark labels meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from dartvision.data.labels import Annotation, image_filename
from dartvision.model.targets import encode_landmark_heatmaps, encode_tip_targets

__all__ = ["Sample", "PhotometricJitter", "SceneDataset", "collate"]


@dataclass
class Sample:
    """One training example: an image and the targets for both heads."""

    image: torch.Tensor          # (3, H, W) float32 in [0, 1]
    landmark_heat: torch.Tensor  # (L, h, w)
    tip_heat: torch.Tensor       # (1, h, w)
    tip_offset: torch.Tensor     # (2, h, w)
    tip_mask: torch.Tensor       # (1, h, w)
    image_id: str
    session_id: str


@dataclass(frozen=True)
class PhotometricJitter:
    """Label-safe augmentation: changes pixels, never geometry.

    Applied to the whole image, so no coordinate can move and no label can go
    stale. That is the entire reason this is separated from geometric
    augmentation rather than bundled with it.
    """

    brightness: float = 0.25
    contrast: float = 0.25
    noise: float = 0.02

    def __call__(self, image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if self.brightness:
            image = image + float(rng.uniform(-self.brightness, self.brightness))
        if self.contrast:
            factor = 1.0 + float(rng.uniform(-self.contrast, self.contrast))
            image = (image - 0.5) * factor + 0.5
        if self.noise:
            image = image + torch.from_numpy(
                rng.normal(0.0, self.noise, tuple(image.shape)).astype("float32")
            )
        return image.clamp(0.0, 1.0)


class SceneDataset(Dataset):
    """Annotations plus their rendered or captured images."""

    def __init__(
        self,
        annotations: Sequence[Annotation],
        image_root: str | Path,
        input_size: tuple[int, int] = (512, 512),
        stride: int = 4,
        sigma_cells: float = 1.5,
        jitter: PhotometricJitter | None = None,
        seed: int = 0,
    ) -> None:
        if not annotations:
            raise ValueError("no annotations")
        if any(s % stride for s in input_size):
            raise ValueError(f"input_size {input_size} must be divisible by stride {stride}")

        self.annotations = list(annotations)
        self.image_root = Path(image_root)
        self.input_size = input_size
        self.stride = stride
        self.sigma_cells = sigma_cells
        self.jitter = jitter
        self.seed = seed
        self.landmark_count = len(self.annotations[0].landmarks)
        if any(len(a.landmarks) != self.landmark_count for a in self.annotations):
            raise ValueError(
                "every annotation must carry the same number of landmarks; a mixed "
                "manifest would silently change the head's channel count"
            )

    @property
    def grid(self) -> tuple[int, int]:
        return (self.input_size[0] // self.stride, self.input_size[1] // self.stride)

    def __len__(self) -> int:
        return len(self.annotations)

    def path_for(self, annotation: Annotation) -> Path:
        return self.image_root / image_filename(annotation.image_id)

    def _load(self, annotation: Annotation) -> torch.Tensor:
        from PIL import Image

        path = self.path_for(annotation)
        if not path.exists():
            raise FileNotFoundError(
                f"no image for {annotation.image_id!r} at {path}; render the manifest first"
            )
        with Image.open(path) as handle:
            # Labels are normalized, so a resize needs no label change at all.
            resized = handle.convert("RGB").resize(
                (self.input_size[1], self.input_size[0]), Image.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index: int) -> Sample:
        annotation = self.annotations[index]
        image = self._load(annotation)
        if self.jitter is not None:
            image = self.jitter(image, np.random.default_rng(self.seed + index))

        landmark_heat = encode_landmark_heatmaps(
            annotation.landmarks, self.grid, self.sigma_cells
        )
        tips = encode_tip_targets(annotation.tips, self.grid, self.sigma_cells)
        return Sample(
            image=image,
            landmark_heat=torch.from_numpy(landmark_heat),
            tip_heat=torch.from_numpy(tips.heat).unsqueeze(0),
            tip_offset=torch.from_numpy(tips.offset),
            tip_mask=torch.from_numpy(tips.mask).unsqueeze(0),
            image_id=annotation.image_id,
            session_id=annotation.session_id,
        )


def collate(samples: Sequence[Sample]) -> dict[str, object]:
    """Batch samples, keeping ids alongside the tensors for traceability."""
    return {
        "image": torch.stack([s.image for s in samples]),
        "landmark_heat": torch.stack([s.landmark_heat for s in samples]),
        "tip_heat": torch.stack([s.tip_heat for s in samples]),
        "tip_offset": torch.stack([s.tip_offset for s in samples]),
        "tip_mask": torch.stack([s.tip_mask for s in samples]),
        "image_id": [s.image_id for s in samples],
        "session_id": [s.session_id for s in samples],
    }
