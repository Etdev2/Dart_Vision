"""Tests for the checkpoint evaluation runner and its CLI.

Two things here matter more than the mechanics. The first is that the test
partition is *derived* from the tier and seed rather than handed in, so a
report cannot quietly be produced against a widened or reshuffled set. The
second is that reading it is recorded, and a second read of the same
(checkpoint, split) pair is refused -- #14 wanted holdout discipline to be
mechanical rather than a matter of intent.

The prediction path is exercised against a stub Brain emitting known targets.
That is deliberate: an untrained network detects nothing (the heads carry a
-4.0 prior bias), so an end-to-end run over real weights proves the wiring but
can never prove the decode, homography and scoring chain is joined up
correctly. The stub does that; the real checkpoint proves the loading,
splitting and ledger path around it.
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
from dartvision.data.splits import LeakageError, split_for_tier  # noqa: E402
from dartvision.eval.runner import (  # noqa: E402
    checkpoint_digest,
    evaluate_checkpoint,
    load_model,
    predict_frames,
)
from dartvision.model.net import BrainOutputs, DartVisionBrain  # noqa: E402
from dartvision.model.targets import encode_landmark_heatmaps, encode_tip_targets  # noqa: E402

LANDMARKS = (
    (0.30, 0.50), (0.50, 0.30), (0.70, 0.50), (0.50, 0.70),
    (0.36, 0.36), (0.64, 0.36), (0.64, 0.64), (0.36, 0.64),
)
TIPS = ((0.42, 0.45), (0.58, 0.60))
GRID = (32, 32)
SIGMA = 1.5
INPUT = (128, 128)
STRIDE = 4


def _logit(probability: np.ndarray) -> torch.Tensor:
    clamped = np.clip(probability.astype(np.float64), 1e-6, 1 - 1e-6)
    return torch.from_numpy(np.log(clamped / (1 - clamped))).float()


class PerfectBrain:
    """A stand-in Brain that always emits the same known targets.

    Every annotation in the fixture carries identical labels, so one set of
    targets is the correct answer for the whole batch.
    """

    def __init__(self, landmarks=LANDMARKS, tips=TIPS, grid=GRID, sigma=SIGMA):
        self.landmark_heat = _logit(encode_landmark_heatmaps(landmarks, grid, sigma))
        targets = encode_tip_targets(tips, grid, sigma)
        self.tip_heat = _logit(targets.heat)[None]
        self.tip_offset = torch.from_numpy(targets.offset).float()

    def __call__(self, images: torch.Tensor) -> BrainOutputs:
        batch = images.shape[0]
        return BrainOutputs(
            landmark_heat=self.landmark_heat.expand(batch, -1, -1, -1),
            tip_heat=self.tip_heat.expand(batch, -1, -1, -1),
            tip_offset=self.tip_offset.expand(batch, -1, -1, -1),
        )


def _annotations(sessions: int = 6, per_session: int = 2) -> list[Annotation]:
    return [
        Annotation(
            image_id=f"set{s % 3}/sess{s}/img-{s:02d}{i:02d}",
            setup_id=f"set{s % 3}",
            session_id=f"set{s % 3}/sess{s}",
            origin=Origin.SYNTHETIC,
            landmarks=LANDMARKS,
            tips=TIPS,
            image_size=(256, 256),
        )
        for s in range(sessions)
        for i in range(per_session)
    ]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Twelve labelled images across six sessions and three setups."""
    root = tmp_path_factory.mktemp("corpus")
    images = root / "images"
    images.mkdir()
    annotations = _annotations()
    for index, annotation in enumerate(annotations):
        rng = np.random.default_rng(index)
        Image.fromarray((rng.random((256, 256, 3)) * 255).astype("uint8")).save(
            images / image_filename(annotation.image_id)
        )
    manifest = root / "manifest.jsonl"
    write_jsonl(annotations, manifest)
    return annotations, manifest, images


@pytest.fixture(scope="module")
def checkpoint(corpus, tmp_path_factory):
    """An untrained checkpoint saved in exactly the shape ``train`` writes."""
    torch.manual_seed(0)
    model = DartVisionBrain(
        backbone="resnet18", landmarks=8, width=16, stride=STRIDE, pretrained=False
    )
    path = tmp_path_factory.mktemp("run") / "last.pt"
    torch.save(
        {
            "epoch": 3,
            "model": model.state_dict(),
            "config": {
                "backbone": "resnet18", "landmarks": 8, "width": 16, "stride": STRIDE,
                "input_height": INPUT[0], "input_width": INPUT[1], "sigma_cells": SIGMA,
            },
            "config_digest": "cfg0123456789ab",
        },
        path,
    )
    return path


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_digest_is_stable_and_content_sensitive(tmp_path):
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    third = tmp_path / "c.pt"
    first.write_bytes(b"weights")
    second.write_bytes(b"weights")
    third.write_bytes(b"weights ")

    assert checkpoint_digest(first) == checkpoint_digest(second)
    assert checkpoint_digest(first) != checkpoint_digest(third)
    assert len(checkpoint_digest(first)) == 16


def test_report_names_the_weights_and_config_it_scored(corpus, checkpoint):
    _, manifest, images = corpus
    report = evaluate_checkpoint(checkpoint, manifest, images, batch_size=4)

    assert report["checkpoint_digest"] == checkpoint_digest(checkpoint)
    assert report["config_digest"] == "cfg0123456789ab"
    assert report["epoch"] == 3


# --------------------------------------------------------------------------
# Rebuilding the model
# --------------------------------------------------------------------------

def test_model_is_rebuilt_from_the_checkpoints_own_config(checkpoint):
    """The caller's defaults must not leak in: a width-64 rebuild of a width-16
    checkpoint would fail to load, and silently substituting defaults for a
    missing key would score a different model than the one named."""
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_model(loaded, "cpu")

    assert model.landmarks == 8
    assert model.stride == STRIDE
    assert not model.training
    widths = {p.shape[0] for n, p in model.named_parameters() if "lateral" in n}
    assert widths == {16}


def test_a_checkpoint_from_a_different_shape_is_refused(checkpoint, tmp_path):
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    loaded["config"] = dict(loaded["config"], width=32)
    path = tmp_path / "mismatched.pt"
    torch.save(loaded, path)

    with pytest.raises(RuntimeError):
        load_model(torch.load(path, map_location="cpu", weights_only=False), "cpu")


# --------------------------------------------------------------------------
# The prediction and scoring chain
# --------------------------------------------------------------------------

def test_a_perfect_prediction_scores_perfectly(corpus):
    annotations, _, images = corpus
    frames = predict_frames(
        PerfectBrain(), annotations, images,
        input_size=INPUT, stride=STRIDE, sigma_cells=SIGMA, device="cpu", batch_size=4,
    )

    assert len(frames) == len(annotations)
    assert all(f.calibratable for f in frames)
    assert all(f.landmark_error_mm < 1.0 for f in frames)
    assert all(len(f.records) == len(TIPS) for f in frames)
    for frame in frames:
        for record in frame.records:
            assert record.tip_error_mm < 2.0
            assert record.predicted == record.truth


def test_predictions_are_returned_in_manifest_order(corpus):
    annotations, _, images = corpus
    frames = predict_frames(
        PerfectBrain(), annotations, images,
        input_size=INPUT, stride=STRIDE, sigma_cells=SIGMA, device="cpu", batch_size=4,
    )
    assert [f.image_id for f in frames] == [a.image_id for a in annotations]


def test_a_silent_model_reports_the_calibration_cliff(corpus, checkpoint):
    """An untrained head fires nowhere. That is not a scoring error -- it is
    #13's cliff, and the report must say so rather than produce a number."""
    _, manifest, images = corpus
    report = evaluate_checkpoint(checkpoint, manifest, images, batch_size=4)

    assert report["calibratable_rate"] == 0.0
    assert report["darts"] == 0
    assert "per_dart_accuracy" not in report


def test_gate_keys_are_present_when_the_model_detects(corpus, monkeypatch):
    """#21 reads these keys by name; a rename here is a silent gate failure."""
    annotations, manifest, images = corpus
    monkeypatch.setattr(
        "dartvision.eval.runner.load_model", lambda checkpoint, device: PerfectBrain()
    )
    report = evaluate_checkpoint(
        _fake_checkpoint(manifest.parent), manifest, images, batch_size=4
    )

    for key in (
        "frames", "calibratable_rate", "darts", "per_dart_accuracy",
        "legs_error_free_ungated", "legs_error_free_at_15pct_flagged",
        "error_concentration_at_15pct", "coverage_at_0.3pct_error",
        "landmark_error_mm_median", "tip_error_mm_p95", "margin_diagnosis",
    ):
        assert key in report, key
    assert report["calibratable_rate"] == 1.0
    assert report["per_dart_accuracy"] == 1.0


def _fake_checkpoint(directory):
    """A checkpoint carrying only the config; the weights are never loaded
    because ``load_model`` is stubbed out."""
    path = directory / "stub.pt"
    torch.save(
        {
            "epoch": 1,
            "model": {},
            "config": {
                "input_height": INPUT[0], "input_width": INPUT[1],
                "stride": STRIDE, "sigma_cells": SIGMA,
            },
            "config_digest": "stub",
        },
        path,
    )
    return path


# --------------------------------------------------------------------------
# The split is derived, not supplied
# --------------------------------------------------------------------------

def test_the_test_partition_comes_from_the_tier_not_the_caller(corpus, checkpoint):
    """There is no argument for 'which images to score'. The tier and seed
    decide, so two runs of the same tier score the same images."""
    annotations, manifest, images = corpus
    expected = sum(
        1
        for partition in split_for_tier(annotations, "session", seed=0).partitions.values()
        if partition == "test"
    )
    assert 0 < expected < len(annotations)

    report = evaluate_checkpoint(
        checkpoint, manifest, images, split="session", partition="test", batch_size=4
    )
    assert report["frames"] == expected
    assert report["split_digest"] == split_for_tier(annotations, "session", seed=0).digest


def test_partitions_are_disjoint_across_a_tier(corpus, checkpoint):
    _, manifest, images = corpus
    counts = {
        partition: evaluate_checkpoint(
            checkpoint, manifest, images, split="session",
            partition=partition, batch_size=4,
        )["frames"]
        for partition in ("train", "val", "test")
    }
    assert sum(counts.values()) == 12


def test_an_empty_partition_is_an_error_not_an_empty_report(corpus, checkpoint):
    """A tier that holds nothing back must fail loudly. An empty report reads
    as a clean run to anything downstream, which is the worst outcome."""
    _, manifest, images = corpus
    with pytest.raises(ValueError, match="empty"):
        evaluate_checkpoint(
            checkpoint, manifest, images, split="session",
            val_fraction=0.0, partition="test", batch_size=4,
        )


def test_no_split_scores_the_whole_manifest(corpus, checkpoint):
    annotations, manifest, images = corpus
    report = evaluate_checkpoint(checkpoint, manifest, images, batch_size=4)
    assert report["frames"] == len(annotations)
    assert report["split_digest"] is None


# --------------------------------------------------------------------------
# Holdout discipline
# --------------------------------------------------------------------------

def test_a_second_read_of_the_test_set_is_refused(corpus, checkpoint, tmp_path):
    _, manifest, images = corpus
    ledger = tmp_path / "holdout.json"
    kwargs = dict(
        split="session", partition="test", batch_size=4, ledger_path=ledger
    )
    evaluate_checkpoint(checkpoint, manifest, images, **kwargs)

    with pytest.raises(LeakageError, match="already been read"):
        evaluate_checkpoint(checkpoint, manifest, images, **kwargs)


def test_an_override_is_allowed_but_must_state_a_reason(corpus, checkpoint, tmp_path):
    _, manifest, images = corpus
    ledger = tmp_path / "holdout.json"
    kwargs = dict(split="session", partition="test", batch_size=4, ledger_path=ledger)
    evaluate_checkpoint(checkpoint, manifest, images, **kwargs)

    with pytest.raises(ValueError, match="reason"):
        evaluate_checkpoint(checkpoint, manifest, images, override=True, **kwargs)

    evaluate_checkpoint(
        checkpoint, manifest, images, override=True,
        override_reason="re-scored after fixing the decoder", **kwargs,
    )
    entries = json.loads(ledger.read_text(encoding="utf-8"))
    assert [e["override"] for e in entries] == [False, True]
    assert entries[-1]["reason"] == "re-scored after fixing the decoder"


def test_the_ledger_is_keyed_on_checkpoint_and_split(corpus, checkpoint, tmp_path):
    """A different tier is a different measurement, not a repeat read."""
    _, manifest, images = corpus
    ledger = tmp_path / "holdout.json"
    evaluate_checkpoint(
        checkpoint, manifest, images, split="session", partition="test",
        batch_size=4, ledger_path=ledger,
    )
    evaluate_checkpoint(
        checkpoint, manifest, images, split="cross_setup", holdout_setup="set0",
        partition="test", batch_size=4, ledger_path=ledger,
    )
    assert len(json.loads(ledger.read_text(encoding="utf-8"))) == 2


def test_reading_validation_does_not_consume_the_holdout(corpus, checkpoint, tmp_path):
    _, manifest, images = corpus
    ledger = tmp_path / "holdout.json"
    for _ in range(3):
        evaluate_checkpoint(
            checkpoint, manifest, images, split="session", partition="val",
            batch_size=4, ledger_path=ledger,
        )
    assert not ledger.exists()


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

def test_cli_writes_a_report(corpus, checkpoint, tmp_path, capsys):
    from dartvision.eval.__main__ import main

    _, manifest, images = corpus
    out = tmp_path / "eval"
    code = main([
        "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--image-root", str(images), "--split", "session", "--partition", "test",
        "--batch-size", "4", "--out-dir", str(out),
        "--ledger", str(tmp_path / "holdout.json"),
    ])

    assert code == 0
    written = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert written["split"] == "session"
    assert json.loads(capsys.readouterr().out) == written


def test_cli_refuses_an_unknown_tier(checkpoint):
    from dartvision.eval.__main__ import main

    with pytest.raises(SystemExit):
        main(["--checkpoint", str(checkpoint), "--manifest", "m",
              "--image-root", "i", "--split", "whatever"])
