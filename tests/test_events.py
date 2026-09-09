"""Tests for the canonical throw event.

The contract's job is to keep three things apart: what vision saw, what the
geometry deterministically implies, and what the rules make of it. Most of
these tests are about that separation holding.
"""

from __future__ import annotations

import math

import pytest

from dartvision.events import (
    CalibrationQuality,
    CalibrationStatus,
    ConfirmationPolicy,
    Detection,
    Source,
    THROW_EVENT_FIELDS,
    ThrowEvent,
)
from dartvision.geometry.board import BDO_BOARD


def polar(r_mm: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return r_mm * math.cos(a), r_mm * math.sin(a)


def event(x_mm: float = 0.0, y_mm: float = 102.0, confidence: float = 0.95, **kwargs):
    return ThrowEvent.from_board_point(
        id=kwargs.pop("id", "e1"), session_id="s1", calibration_id="c1",
        x_mm=x_mm, y_mm=y_mm, confidence=confidence, **kwargs,
    )


# --------------------------------------------------------------------------
# The score is derived, not asserted
# --------------------------------------------------------------------------

def test_score_follows_from_geometry_not_from_the_caller():
    treble = event(*polar(102.0, 90.0))
    assert treble.detected.notation == "T20"
    assert treble.detected.segment == 20 and treble.detected.multiplier == 3
    assert treble.detected.score == 60


def test_a_detection_whose_score_contradicts_its_segment_is_rejected():
    with pytest.raises(ValueError, match="does not follow from"):
        Detection(segment=20, multiplier=3, score=20, notation="T20",
                  confidence=0.9, margin_mm=5.0)


def test_the_event_carries_the_margin_and_names_the_boundary():
    near_wire = event(*polar(107.0, 90.0))
    assert near_wire.detected.margin_mm == pytest.approx(0.4, abs=0.01)
    assert near_wire.detected.boundary == "outer treble wire"


def test_board_millimetres_are_canonical_and_image_coordinates_optional():
    e = event(image_x=0.51, image_y=0.32)
    payload = e.to_dict()
    assert payload["board_x_mm"] is not None and payload["board_y_mm"] is not None
    assert payload["image_x"] == 0.51
    assert event().to_dict()["image_x"] is None


def test_a_miss_scores_zero():
    e = event(*polar(200.0, 45.0))
    assert (e.detected.notation, e.detected.score, e.detected.multiplier) == ("MISS", 0, 0)


# --------------------------------------------------------------------------
# Provenance and corrections
# --------------------------------------------------------------------------

def test_a_correction_must_declare_itself_as_one():
    with pytest.raises(ValueError, match="source 'correction'"):
        event(corrects="e1")
    corrected = event(id="e2", corrects="e1", source=Source.CORRECTION)
    assert corrected.corrects == "e1"


def test_identity_fields_are_required():
    for missing in ("id", "session_id", "calibration_id"):
        kwargs = {"id": "e", "session_id": "s", "calibration_id": "c"}
        kwargs[missing] = ""
        with pytest.raises(ValueError, match="required"):
            ThrowEvent.from_board_point(x_mm=0.0, y_mm=0.0, confidence=0.9, **kwargs)


def test_sequence_is_bounded_by_the_darts_in_a_turn():
    assert event(sequence=3).sequence == 3
    with pytest.raises(ValueError, match="1, 2 or 3"):
        event(sequence=4)


def test_serialized_field_set_matches_the_documented_contract():
    """Guards the TypeScript definition in docs/architecture/throw-event.md."""
    payload = event().to_dict()
    assert set(payload) == set(THROW_EVENT_FIELDS)
    assert set(payload["detected"]) == {
        "segment", "multiplier", "score", "notation",
        "confidence", "margin_mm", "boundary",
    }


# --------------------------------------------------------------------------
# Calibration is a separate concern
# --------------------------------------------------------------------------

def status(found=8, coverage=0.2, expected=8) -> CalibrationStatus:
    return CalibrationStatus(
        calibration_id="c1", session_id="s1", established_at="2026-09-09T12:00:00Z",
        landmarks_found=found, landmarks_expected=expected, board_coverage=coverage,
    )


def test_fewer_than_four_landmarks_means_nothing_scores():
    assert not status(found=3).scoring_possible
    assert status(found=3).quality is CalibrationQuality.UNUSABLE
    assert status(found=4).scoring_possible


def test_low_coverage_is_unusable_however_many_landmarks_are_found():
    assert status(found=8, coverage=0.05).quality is CalibrationQuality.UNUSABLE
    assert status(found=8, coverage=0.10).quality is CalibrationQuality.MARGINAL
    assert status(found=8, coverage=0.20).quality is CalibrationQuality.GOOD


def test_partial_landmarks_are_marginal_not_good():
    assert status(found=5, coverage=0.3).quality is CalibrationQuality.MARGINAL


def test_calibration_validates_its_inputs():
    with pytest.raises(ValueError, match="4 or 8"):
        status(expected=6)
    with pytest.raises(ValueError, match="landmarks_found"):
        status(found=9)
    with pytest.raises(ValueError, match="board_coverage"):
        status(coverage=1.4)


# --------------------------------------------------------------------------
# The confirmation policy: margin, not confidence alone
# --------------------------------------------------------------------------

def test_a_confident_throw_far_from_any_boundary_is_auto_scored():
    policy = ConfirmationPolicy()
    middle_of_the_bed = event(*polar(140.0, 90.0), confidence=0.95)
    assert middle_of_the_bed.detected.margin_mm > 5.0
    assert not policy.should_confirm(middle_of_the_bed)


def test_an_unconfident_throw_is_always_confirmed():
    assert ConfirmationPolicy().should_confirm(event(*polar(140.0, 90.0), confidence=0.5))


def test_the_same_confidence_is_confirmed_when_it_sits_on_a_wire():
    """The central claim: confidence alone cannot make this call."""
    policy = ConfirmationPolicy()
    safe = event(*polar(140.0, 90.0), confidence=0.95)
    knife_edge = event(*polar(107.2, 90.0), confidence=0.95)

    assert safe.detected.confidence == knife_edge.detected.confidence
    assert not policy.should_confirm(safe)
    assert policy.should_confirm(knife_edge)


def test_a_very_confident_call_stands_even_near_a_boundary():
    policy = ConfirmationPolicy()
    assert not policy.should_confirm(event(*polar(107.2, 90.0), confidence=0.995))


def test_partition_splits_a_turn_into_auto_and_confirm():
    policy = ConfirmationPolicy()
    events = [
        event(*polar(140.0, 90.0), confidence=0.97, id="a"),
        event(*polar(107.1, 90.0), confidence=0.93, id="b"),
        event(*polar(60.0, 200.0), confidence=0.40, id="c"),
    ]
    auto, confirm = policy.partition(events)
    assert [e.id for e in auto] == ["a"]
    assert [e.id for e in confirm] == ["b", "c"]


@pytest.mark.parametrize(
    "kwargs",
    [{"min_confidence": 1.5}, {"min_margin_mm": -1.0}, {"min_confidence": 0.9, "high_confidence": 0.5}],
)
def test_invalid_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ConfirmationPolicy(**kwargs)


# --------------------------------------------------------------------------
# The boundary with the game engine
# --------------------------------------------------------------------------

def test_the_event_scores_the_dart_but_says_nothing_about_the_rules():
    """A double 20 is worth 40 whether or not it checked out or busted."""
    payload = event(*polar(165.0, 90.0)).to_dict()
    assert payload["detected"]["score"] == 40
    rule_words = {"bust", "checkout", "remaining", "leg", "win", "turn_over"}
    assert not rule_words & set(payload) | (rule_words & set(payload["detected"]))
