"""Run configuration and the provenance every run must record.

#16 set a reproducibility contract: a run that informs a #20 or #21 decision
records its git SHA, resolved config, seeds, dataset manifest hash, split
digest, and full metric history. This module makes that a data structure rather
than a convention, so a run that omits it cannot be produced by accident.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["TrainConfig", "RunManifest", "resolve_device"]


@dataclass
class TrainConfig:
    """Everything needed to reproduce a run."""

    manifest: str
    image_root: str
    out_dir: str = "runs/dev"

    backbone: str = "resnet18"
    landmarks: int = 8
    width: int = 64
    stride: int = 4
    pretrained: bool = False

    input_height: int = 512
    input_width: int = 512
    sigma_cells: float = 1.5

    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    offset_weight: float = 5.0
    workers: int = 2

    split: str = "none"       # none | session | cross_setup | sim_to_real
    holdout_setup: str = ""
    val_fraction: float = 0.15

    jitter: bool = True
    seed: int = 0
    device: str = "auto"
    allow_mps: bool = False
    limit: int = 0  # 0 = use everything; otherwise a smoke-test subset

    def __post_init__(self) -> None:
        if self.landmarks not in (4, 8):
            raise ValueError("landmarks must be 4 or 8")
        if any(s % self.stride for s in self.input_size):
            raise ValueError("input size must be divisible by stride")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.split not in ("none", "session", "cross_setup", "sim_to_real"):
            raise ValueError(f"unknown split {self.split!r}")
        if self.split == "cross_setup" and not self.holdout_setup:
            raise ValueError("cross_setup requires holdout_setup")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")

    @property
    def input_size(self) -> tuple[int, int]:
        return (self.input_height, self.input_width)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_device(requested: str, allow_mps: bool) -> str:
    """Pick a device, refusing to train on MPS by accident.

    #16 is explicit that Apple's backend must not be used for runs of record:
    its numerics differ from CUDA and some operators fall back to CPU, so a
    result produced there is not comparable to one produced on the rented box.
    ``auto`` therefore never selects MPS -- it prefers CUDA and falls back to
    **CPU**, which is slow but correct. Choosing MPS requires saying so.
    """
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps" and not allow_mps:
        raise ValueError(
            "refusing to use MPS without allow_mps=True. Its numerics differ from "
            "CUDA and some operators silently fall back to CPU, so results are not "
            "comparable to a rented-GPU run (#16). Set allow_mps for a smoke test "
            "only, and never for a run that informs a decision."
        )
    return requested


@dataclass
class RunManifest:
    """Provenance for one run. Written before training starts, not after."""

    config: dict[str, object]
    config_digest: str
    git_sha: str | None
    git_dirty: bool
    split_digest: str | None
    manifest_digest: str
    device: str
    package_versions: dict[str, str]
    platform: str
    started_at: str
    metrics_path: str
    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        config: TrainConfig,
        device: str,
        manifest_path: str | Path,
        split_digest: str | None,
        started_at: str,
    ) -> "RunManifest":
        import torch

        versions = {"torch": torch.__version__, "python": platform.python_version()}
        try:
            import timm
            versions["timm"] = timm.__version__
        except ImportError:
            pass

        payload = Path(manifest_path).read_bytes()
        status = _git("status", "--porcelain")
        return cls(
            config=asdict(config),
            config_digest=config.digest,
            git_sha=_git("rev-parse", "HEAD"),
            git_dirty=bool(status),
            split_digest=split_digest,
            manifest_digest=hashlib.sha256(payload).hexdigest()[:16],
            device=device,
            package_versions=versions,
            platform=f"{platform.system()} {platform.release()} {platform.machine()}",
            started_at=started_at,
            metrics_path="metrics.jsonl",
        )

    def write(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
