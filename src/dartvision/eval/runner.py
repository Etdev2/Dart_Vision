"""Evaluating a checkpoint against a benchmark tier.

Rebuilds the split from its tier and seed rather than accepting a set of
annotations to score. That is deliberate: the test partition is defined by the
benchmark, not by whoever calls this, so it cannot drift or be quietly widened
between runs. Training never receives it (see ``train.runner.apply_split``) and
evaluation derives it independently -- the two agree because the split is a pure
function of the manifest, tier and seed.

Reading the test set is recorded in the holdout ledger, which refuses a repeat
on the same checkpoint and split without a logged override. #14 asked for that
to be mechanical rather than a matter of intent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from dartvision.data.labels import Annotation, read_jsonl
from dartvision.data.splits import HoldoutLedger, split_for_tier
from dartvision.data.torch_dataset import SceneDataset, collate
from dartvision.eval.evaluate import FrameResult, evaluate_all, evaluate_frame
from dartvision.model.targets import (
    decode_landmark_heatmaps,
    decode_tip_targets,
    recommended_window,
)

__all__ = ["checkpoint_digest", "load_model", "predict_frames", "evaluate_checkpoint"]


def checkpoint_digest(path: str | Path) -> str:
    """Short content hash, so a report names the exact weights it scored."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_model(checkpoint: dict, device: str):
    """Rebuild the model exactly as trained, from the checkpoint's own config."""
    from dartvision.model.net import DartVisionBrain

    config = checkpoint.get("config") or {}
    model = DartVisionBrain(
        backbone=config.get("backbone", "resnet18"),
        landmarks=config.get("landmarks", 8),
        width=config.get("width", 64),
        stride=config.get("stride", 4),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def predict_frames(
    model,
    annotations: Sequence[Annotation],
    image_root: str | Path,
    input_size: tuple[int, int],
    stride: int,
    sigma_cells: float,
    device: str,
    batch_size: int = 8,
    landmark_threshold: float = 0.15,
    tip_threshold: float = 0.25,
) -> list[FrameResult]:
    """Run the model over annotations and score each frame against its labels."""
    dataset = SceneDataset(
        annotations, image_root=image_root, input_size=input_size,
        stride=stride, sigma_cells=sigma_cells, jitter=None,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    by_id = {a.image_id: a for a in annotations}
    window = recommended_window(sigma_cells)

    results: list[FrameResult] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device))
            landmark_probs = torch.sigmoid(outputs.landmark_heat).cpu().numpy()
            tip_probs = torch.sigmoid(outputs.tip_heat).cpu().numpy()
            offsets = outputs.tip_offset.cpu().numpy()

            for index, image_id in enumerate(batch["image_id"]):
                landmarks = [
                    point
                    for point, _ in decode_landmark_heatmaps(
                        landmark_probs[index], window=window,
                        threshold=landmark_threshold,
                    )
                ]
                tips = decode_tip_targets(
                    tip_probs[index, 0], offsets[index],
                    threshold=tip_threshold, max_detections=3,
                )
                results.append(evaluate_frame(by_id[image_id], landmarks, tips))
    return results


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest: str | Path,
    image_root: str | Path,
    *,
    split: str = "none",
    holdout_setup: str = "",
    val_fraction: float = 0.15,
    seed: int = 0,
    partition: str = "test",
    device: str = "cpu",
    batch_size: int = 8,
    out_dir: str | Path | None = None,
    ledger_path: str | Path | None = None,
    override: bool = False,
    override_reason: str | None = None,
) -> dict[str, object]:
    """Score a checkpoint and return #21's gate report."""
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = loaded.get("config") or {}
    annotations = list(read_jsonl(manifest))

    split_digest: str | None = None
    if split == "none":
        selected = annotations
    else:
        assignment = split_for_tier(annotations, split, holdout_setup, val_fraction, seed)
        split_digest = assignment.digest
        wanted = {i for i, p in assignment.partitions.items() if p == partition}
        selected = [a for a in annotations if a.image_id in wanted]

    if not selected:
        raise ValueError(f"the {partition!r} partition is empty")

    digest = checkpoint_digest(checkpoint_path)
    if partition == "test" and ledger_path is not None:
        HoldoutLedger(Path(ledger_path)).record(
            digest, split_digest or "no-split",
            override=override, reason=override_reason,
        )

    model = load_model(loaded, device)
    frames = predict_frames(
        model, selected, image_root,
        input_size=(config.get("input_height", 512), config.get("input_width", 512)),
        stride=config.get("stride", 4),
        sigma_cells=config.get("sigma_cells", 1.5),
        device=device, batch_size=batch_size,
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_digest": digest,
        "split": split,
        "split_digest": split_digest,
        "partition": partition,
        "epoch": loaded.get("epoch"),
        "config_digest": loaded.get("config_digest"),
        **evaluate_all(frames),
    }
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
