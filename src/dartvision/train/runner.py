"""The training loop.

Deliberately small. Its job is to be correct and reproducible, not clever: the
interesting decisions live in the target encoding (#15), the benchmark (#14) and
the ship gates (#21), and a loop that hides bugs behind abstraction would
undermine all three.

Every run writes ``run.json`` before the first step -- provenance first, so a
crashed run is still identifiable -- and appends one JSON object per epoch to
``metrics.jsonl``, which is the full metric history #16's contract requires
rather than a final number.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dartvision.data.labels import Annotation, read_jsonl
from dartvision.data.splits import split_for_tier
from dartvision.data.torch_dataset import PhotometricJitter, SceneDataset, collate
from dartvision.model.losses import channel_peak_mask, heatmap_focal_loss, offset_l1_loss
from dartvision.model.net import DartVisionBrain
from dartvision.train.config import RunManifest, TrainConfig, resolve_device

__all__ = ["EpochResult", "seed_everything", "build_dataset", "apply_split", "train"]


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    landmark_loss: float
    tip_loss: float
    offset_loss: float
    val_loss: float | None
    seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "landmark_loss": round(self.landmark_loss, 6),
            "tip_loss": round(self.tip_loss, 6),
            "offset_loss": round(self.offset_loss, 6),
            "val_loss": None if self.val_loss is None else round(self.val_loss, 6),
            "seconds": round(self.seconds, 2),
        }


def seed_everything(seed: int) -> None:
    """Seed every generator a run touches. Recorded in the manifest."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_split(
    config: TrainConfig, annotations: Sequence[Annotation]
) -> tuple[list[Annotation], list[Annotation], str | None]:
    """Partition annotations for the configured tier.

    Returns ``(train, val, split_digest)``. The test partition is deliberately
    *not* returned: training must never see it, and handing it back here would
    make that a matter of discipline rather than of construction. Evaluation
    rebuilds the same split from the same tier and seed and reads test itself.
    """
    if config.split == "none":
        return list(annotations), [], None

    assignment = split_for_tier(
        annotations, config.split, config.holdout_setup, config.val_fraction, config.seed
    )
    train_ids = {i for i, p in assignment.partitions.items() if p == "train"}
    val_ids = {i for i, p in assignment.partitions.items() if p == "val"}
    return (
        [a for a in annotations if a.image_id in train_ids],
        [a for a in annotations if a.image_id in val_ids],
        assignment.digest,
    )


def build_dataset(config: TrainConfig, annotations: Sequence[Annotation]) -> SceneDataset:
    return SceneDataset(
        annotations,
        image_root=config.image_root,
        input_size=config.input_size,
        stride=config.stride,
        sigma_cells=config.sigma_cells,
        jitter=PhotometricJitter() if config.jitter else None,
        seed=config.seed,
    )


def _losses(outputs, batch, offset_weight: float):
    landmark_target = batch["landmark_heat"]
    landmark = heatmap_focal_loss(
        outputs.landmark_heat, landmark_target, channel_peak_mask(landmark_target)
    )
    # Tips share one channel, so positives come from the encoder's mask, never
    # from the target's per-channel maximum -- see dartvision.model.losses.
    tip = heatmap_focal_loss(outputs.tip_heat, batch["tip_heat"], batch["tip_mask"])
    offset = offset_l1_loss(outputs.tip_offset, batch["tip_offset"], batch["tip_mask"])
    return landmark + tip + offset_weight * offset, landmark, tip, offset


def _to_device(batch: dict[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train(
    config: TrainConfig,
    train_annotations: Sequence[Annotation] | None = None,
    val_annotations: Sequence[Annotation] | None = None,
    split_digest: str | None = None,
) -> list[EpochResult]:
    """Run training. Returns the per-epoch history, also written to disk."""
    device = resolve_device(config.device, config.allow_mps)
    seed_everything(config.seed)

    if train_annotations is None:
        loaded = list(read_jsonl(config.manifest))
        if config.limit:
            loaded = loaded[: config.limit]
        train_annotations, split_val, digest = apply_split(config, loaded)
        if val_annotations is None:
            val_annotations = split_val
        if split_digest is None:
            split_digest = digest
    elif config.limit:
        train_annotations = list(train_annotations)[: config.limit]
    if not train_annotations:
        raise ValueError("no training annotations")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest.capture(
        config, device, config.manifest, split_digest,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    manifest.write(out_dir)

    loader = DataLoader(
        build_dataset(config, train_annotations),
        batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
        collate_fn=collate, drop_last=False,
    )
    val_loader = None
    if val_annotations:
        val_loader = DataLoader(
            SceneDataset(
                val_annotations, image_root=config.image_root,
                input_size=config.input_size, stride=config.stride,
                sigma_cells=config.sigma_cells, jitter=None,
            ),
            batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, collate_fn=collate,
        )

    model = DartVisionBrain(
        backbone=config.backbone, landmarks=config.landmarks,
        width=config.width, stride=config.stride, pretrained=config.pretrained,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

    metrics_path = out_dir / "metrics.jsonl"
    history: list[EpochResult] = []

    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        model.train()
        totals = np.zeros(4)
        batches = 0

        for batch in loader:
            batch = _to_device(batch, device)
            optimiser.zero_grad()
            loss, landmark, tip, offset = _losses(model(batch["image"]), batch, config.offset_weight)
            loss.backward()
            optimiser.step()
            totals += [loss.item(), landmark.item(), tip.item(), offset.item()]
            batches += 1

        val_loss = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                seen, accumulated = 0, 0.0
                for batch in val_loader:
                    batch = _to_device(batch, device)
                    loss, *_ = _losses(model(batch["image"]), batch, config.offset_weight)
                    accumulated += loss.item()
                    seen += 1
                val_loss = accumulated / max(seen, 1)

        schedule.step()
        result = EpochResult(
            epoch=epoch,
            train_loss=totals[0] / batches,
            landmark_loss=totals[1] / batches,
            tip_loss=totals[2] / batches,
            offset_loss=totals[3] / batches,
            val_loss=val_loss,
            seconds=time.monotonic() - started,
        )
        history.append(result)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict()) + "\n")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "config": config.__dict__,
                "config_digest": config.digest,
            },
            out_dir / "last.pt",
        )
        print(json.dumps(result.to_dict()), flush=True)

    return history
