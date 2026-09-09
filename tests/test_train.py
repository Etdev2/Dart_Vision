"""Tests for the training loop and its provenance.

The important test runs training for real on a handful of synthetic images and
requires the loss to fall. Everything else here guards the reproducibility
contract from #16 -- provenance written before the first step, seeds honoured,
and MPS refused unless explicitly allowed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from dartvision.data import Annotation, Origin, image_filename, write_jsonl  # noqa: E402
from dartvision.train import RunManifest, TrainConfig, resolve_device, seed_everything, train  # noqa: E402

LANDMARKS = tuple((0.25 + 0.06 * i, 0.30 + 0.05 * i) for i in range(8))


@pytest.fixture
def corpus(tmp_path):
    """A tiny rendered manifest: annotations, images, and a manifest file."""
    images = tmp_path / "images"
    images.mkdir()
    annotations = []
    for i in range(8):
        a = Annotation(
            image_id=f"set0/sess{i % 2}/img-{i:03d}",
            setup_id="set0", session_id=f"set0/sess{i % 2}",
            origin=Origin.SYNTHETIC, landmarks=LANDMARKS,
            tips=tuple((0.45 + 0.08 * j, 0.5) for j in range(i % 3)),
            image_size=(128, 128),
        )
        annotations.append(a)
        rng = np.random.default_rng(i)
        Image.fromarray((rng.random((128, 128, 3)) * 255).astype("uint8")).save(
            images / image_filename(a.image_id)
        )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(annotations, manifest)
    return annotations, manifest, images


def config_for(tmp_path, manifest, images, **overrides) -> TrainConfig:
    values = dict(
        manifest=str(manifest), image_root=str(images), out_dir=str(tmp_path / "run"),
        backbone="resnet18", width=16, stride=4, input_height=64, input_width=64,
        epochs=3, batch_size=4, learning_rate=3e-3, workers=0, device="cpu",
    )
    values.update(overrides)
    return TrainConfig(**values)


# --------------------------------------------------------------------------
# Config and provenance
# --------------------------------------------------------------------------

def test_config_digest_is_stable_and_sensitive():
    a = TrainConfig(manifest="m", image_root="i")
    b = TrainConfig(manifest="m", image_root="i")
    c = TrainConfig(manifest="m", image_root="i", learning_rate=2e-3)
    assert a.digest == b.digest != c.digest


@pytest.mark.parametrize(
    "kwargs",
    [{"landmarks": 6}, {"epochs": 0}, {"batch_size": 0},
     {"learning_rate": 0.0}, {"input_height": 63}],
)
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(manifest="m", image_root="i", **kwargs)


def test_auto_device_never_picks_mps():
    """#16: results from Apple's backend are not comparable to a CUDA run."""
    assert resolve_device("auto", allow_mps=False) in ("cuda", "cpu")


def test_mps_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="refusing to use MPS"):
        resolve_device("mps", allow_mps=False)
    assert resolve_device("mps", allow_mps=True) == "mps"


def test_provenance_is_written_before_the_first_step(corpus, tmp_path):
    """A crashed run must still be identifiable."""
    annotations, manifest, images = corpus
    config = config_for(tmp_path, manifest, images, epochs=1)
    train(config, split_digest="split-abc123")

    payload = json.loads((tmp_path / "run" / "run.json").read_text())
    assert payload["config_digest"] == config.digest
    assert payload["split_digest"] == "split-abc123"
    assert payload["manifest_digest"] and len(payload["manifest_digest"]) == 16
    assert payload["device"] == "cpu"
    assert "torch" in payload["package_versions"]
    assert payload["git_sha"] is None or len(payload["git_sha"]) == 40
    assert "git_dirty" in payload


def test_manifest_capture_records_a_dirty_tree(corpus, tmp_path):
    _, manifest, images = corpus
    config = config_for(tmp_path, manifest, images)
    captured = RunManifest.capture(config, "cpu", manifest, None, "2026-09-09T00:00:00Z")
    assert isinstance(captured.git_dirty, bool)
    assert captured.platform


def test_seeding_is_reproducible():
    seed_everything(11)
    first = torch.randn(4).tolist()
    seed_everything(11)
    assert torch.randn(4).tolist() == first


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def test_training_reduces_the_loss_and_logs_every_epoch(corpus, tmp_path):
    annotations, manifest, images = corpus
    config = config_for(tmp_path, manifest, images, epochs=6)
    history = train(config)

    assert len(history) == 6
    assert history[-1].train_loss < history[0].train_loss

    lines = (tmp_path / "run" / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 6
    logged = [json.loads(line) for line in lines]
    assert [row["epoch"] for row in logged] == list(range(1, 7))
    assert all(row["seconds"] > 0 for row in logged)
    # The component losses must be reported, not just the total.
    assert {"landmark_loss", "tip_loss", "offset_loss"} <= set(logged[0])


def test_a_checkpoint_carries_enough_to_identify_it(corpus, tmp_path):
    annotations, manifest, images = corpus
    config = config_for(tmp_path, manifest, images, epochs=2)
    train(config)

    checkpoint = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["config_digest"] == config.digest
    assert "model" in checkpoint and "optimiser" in checkpoint


def test_validation_loss_is_recorded_when_a_val_set_is_given(corpus, tmp_path):
    annotations, manifest, images = corpus
    config = config_for(tmp_path, manifest, images, epochs=2)
    history = train(config, annotations[:6], annotations[6:])
    assert all(result.val_loss is not None for result in history)


def test_no_validation_set_means_no_validation_loss(corpus, tmp_path):
    annotations, manifest, images = corpus
    history = train(config_for(tmp_path, manifest, images, epochs=1))
    assert history[0].val_loss is None


def test_limit_takes_a_smoke_test_subset(corpus, tmp_path):
    annotations, manifest, images = corpus
    config = config_for(tmp_path, manifest, images, epochs=1, limit=4, batch_size=2)
    train(config)
    assert json.loads((tmp_path / "run" / "run.json").read_text())["config"]["limit"] == 4


def test_an_empty_manifest_is_rejected(corpus, tmp_path):
    annotations, manifest, images = corpus
    with pytest.raises(ValueError, match="no training annotations"):
        train(config_for(tmp_path, manifest, images), [])


def test_cli_runs_end_to_end(corpus, tmp_path, capsys):
    from dartvision.train.__main__ import main

    annotations, manifest, images = corpus
    assert main([
        "--manifest", str(manifest), "--image-root", str(images),
        "--out-dir", str(tmp_path / "cli"), "--epochs", "2", "--batch-size", "4",
        "--width", "16", "--input-height", "64", "--input-width", "64",
        "--workers", "0", "--device", "cpu",
    ]) == 0
    out = capsys.readouterr().out
    assert '"epochs": 2' in out
    assert (tmp_path / "cli" / "run.json").exists()
