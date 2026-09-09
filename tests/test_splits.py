"""Tests for leakage-safe splits.

Leakage is the failure that invalidates every downstream number while looking
perfect, so these tests are mostly about what the builders *refuse* to do.
"""

from __future__ import annotations

import json

import pytest

from dartvision.data.labels import Annotation, Origin
from dartvision.data.splits import (
    HoldoutLedger,
    LeakageError,
    SplitAssignment,
    build_cross_setup_split,
    build_session_split,
    build_sim_to_real_split,
    validate_split,
)

LANDMARKS = tuple((0.1 * i, 0.1 * i) for i in range(4))


def corpus(setups=3, sessions=2, images=3, origin=Origin.REAL_THROWN,
           prefix="setup") -> list[Annotation]:
    out = []
    for s in range(setups):
        for n in range(sessions):
            for i in range(images):
                out.append(Annotation(
                    image_id=f"{prefix}{s}/sess{n}/img{i}",
                    setup_id=f"{prefix}{s}",
                    session_id=f"{prefix}{s}/sess{n}",
                    origin=origin,
                    landmarks=LANDMARKS,
                ))
    return out


# --------------------------------------------------------------------------
# Session-level grouping
# --------------------------------------------------------------------------

def test_a_session_never_spans_partitions():
    annotations = corpus()
    split = build_session_split(annotations, ["setup0/sess1"], ["setup2/sess0"])

    by_session: dict[str, set[str]] = {}
    for a in annotations:
        by_session.setdefault(a.session_id, set()).add(split.partitions[a.image_id])
    assert all(len(p) == 1 for p in by_session.values())


def test_every_image_is_assigned():
    annotations = corpus()
    split = build_session_split(annotations, ["setup0/sess1"], ["setup2/sess0"])
    assert set(split.partitions) == {a.image_id for a in annotations}
    assert sum(split.counts().values()) == len(annotations)


def test_unknown_sessions_are_rejected():
    with pytest.raises(LeakageError, match="unknown sessions"):
        build_session_split(corpus(), ["nope"], [])


def test_a_session_reused_across_setups_is_rejected():
    """Ambiguous session ids would make grouping silently meaningless."""
    annotations = corpus(setups=1)
    annotations.append(Annotation(
        image_id="other/img", setup_id="setupX", session_id="setup0/sess0",
        origin=Origin.REAL_THROWN, landmarks=LANDMARKS,
    ))
    with pytest.raises(LeakageError, match="two setups"):
        build_session_split(annotations, [], [])


def test_validate_catches_a_hand_edited_split_that_leaks():
    annotations = corpus()
    bad = SplitAssignment(
        tier="synthetic",
        partitions={a.image_id: "train" for a in annotations},
        sessions={"train": ("setup0/sess0",), "test": ("setup0/sess0",)},
        setups={"train": ("setup0",), "test": ("setup0",)},
    )
    with pytest.raises(LeakageError, match="both"):
        validate_split(annotations, bad)


# --------------------------------------------------------------------------
# Cross-setup
# --------------------------------------------------------------------------

def test_cross_setup_holds_out_a_whole_setup():
    annotations = corpus(setups=3)
    split = build_cross_setup_split(annotations, "setup2")

    assert split.setups["test"] == ("setup2",)
    assert "setup2" not in split.setups["train"]
    for a in annotations:
        expected = "test" if a.setup_id == "setup2" else "train"
        assert split.partitions[a.image_id] == expected


def test_cross_setup_needs_three_setups():
    """Two setups give a point estimate with no notion of variance (#14)."""
    with pytest.raises(LeakageError, match="at least 3 setups"):
        build_cross_setup_split(corpus(setups=2), "setup1")


def test_cross_setup_rejects_validation_drawn_from_the_holdout():
    with pytest.raises(LeakageError, match="cannot come from the held-out setup"):
        build_cross_setup_split(corpus(), "setup2", val_sessions=["setup2/sess0"])


def test_cross_setup_rejects_an_unknown_setup():
    with pytest.raises(LeakageError, match="unknown setup"):
        build_cross_setup_split(corpus(), "nope")


# --------------------------------------------------------------------------
# Sim-to-real
# --------------------------------------------------------------------------

def test_sim_to_real_trains_on_synthetic_and_tests_on_real():
    annotations = (
        corpus(setups=1, origin=Origin.SYNTHETIC, prefix="synth")
        + corpus(setups=1, prefix="real")
    )
    split = build_sim_to_real_split(annotations)

    for a in annotations:
        expected = "test" if a.origin.is_real else "train"
        assert split.partitions[a.image_id] == expected


def test_sim_to_real_needs_both_kinds():
    with pytest.raises(LeakageError, match="both synthetic and real"):
        build_sim_to_real_split(corpus(origin=Origin.SYNTHETIC, prefix="synth"))


def test_a_session_mixing_synthetic_and_real_is_rejected():
    annotations = corpus(setups=1, sessions=1, images=2, origin=Origin.SYNTHETIC,
                         prefix="synth")
    annotations.append(Annotation(
        image_id="synth0/sess0/real", setup_id="synth0", session_id="synth0/sess0",
        origin=Origin.REAL_THROWN, landmarks=LANDMARKS,
    ))
    annotations += corpus(setups=1, sessions=1, prefix="real")
    with pytest.raises(LeakageError, match="mixes synthetic and real"):
        build_sim_to_real_split(annotations)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_digest_is_stable_and_sensitive():
    annotations = corpus()
    a = build_session_split(annotations, ["setup0/sess1"], ["setup2/sess0"])
    b = build_session_split(annotations, ["setup0/sess1"], ["setup2/sess0"])
    c = build_session_split(annotations, ["setup1/sess1"], ["setup2/sess0"])

    assert a.digest == b.digest
    assert a.digest != c.digest
    assert len(a.digest) == 16


def test_split_writes_a_reviewable_file(tmp_path):
    split = build_cross_setup_split(corpus(), "setup1")
    path = tmp_path / "split.json"
    digest = split.write(path)

    payload = json.loads(path.read_text())
    assert payload["digest"] == digest
    assert payload["tier"] == "cross_setup"
    assert payload["setups"]["test"] == ["setup1"]


# --------------------------------------------------------------------------
# Holdout discipline
# --------------------------------------------------------------------------

def test_ledger_allows_a_first_read_and_refuses_a_second(tmp_path):
    ledger = HoldoutLedger(tmp_path / "ledger.json")
    ledger.record("ckpt-abc", "split-123")
    assert ledger.reads("ckpt-abc", "split-123") == 1

    with pytest.raises(LeakageError, match="already been read"):
        ledger.record("ckpt-abc", "split-123")


def test_ledger_permits_a_logged_override(tmp_path):
    ledger = HoldoutLedger(tmp_path / "ledger.json")
    ledger.record("ckpt-abc", "split-123")
    ledger.record("ckpt-abc", "split-123", override=True, reason="rerun after disk fault")

    entries = json.loads((tmp_path / "ledger.json").read_text())
    assert entries[-1]["override"] is True
    assert "disk fault" in entries[-1]["reason"]


def test_an_override_must_state_a_reason(tmp_path):
    ledger = HoldoutLedger(tmp_path / "ledger.json")
    ledger.record("ckpt", "split")
    with pytest.raises(ValueError, match="must state a reason"):
        ledger.record("ckpt", "split", override=True)


def test_different_checkpoints_and_splits_are_independent(tmp_path):
    ledger = HoldoutLedger(tmp_path / "ledger.json")
    ledger.record("ckpt-a", "split-1")
    ledger.record("ckpt-b", "split-1")
    ledger.record("ckpt-a", "split-2")
    assert ledger.reads("ckpt-a", "split-1") == 1
    assert ledger.reads("ckpt-b", "split-1") == 1
