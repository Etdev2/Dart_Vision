"""Training: configuration, provenance, and the loop."""

from dartvision.train.config import RunManifest, TrainConfig, resolve_device
from dartvision.train.runner import EpochResult, build_dataset, seed_everything, train

__all__ = [
    "EpochResult",
    "RunManifest",
    "TrainConfig",
    "build_dataset",
    "resolve_device",
    "seed_everything",
    "train",
]
