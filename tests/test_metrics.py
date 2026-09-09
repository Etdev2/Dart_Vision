"""Tests for the scoring metrics.

Several tests check the metrics against the tables published in
``docs/research/21-ship-gates.md``, so the documented gates and the code that
evaluates them cannot drift apart.
"""

from __future__ import annotations

import pytest

from dartvision.geometry.board import Hit
from dartvision.metrics import (
    FailureKind,
    ThrowRecord,
    classify,
    coverage_at_error_rate,
    diagnose,
    effective_legs_error_free,
    error_concentration,
    legs_error_free,
    per_dart_accuracy,
    required_per_dart_accuracy,
    risk_coverage_curve,
)

T20 = Hit(segment=20, multiplier=3, score=60, notation="T20")
S20 = Hit(segment=20, multiplier=1, score=20, notation="S20")
S5 = Hit(segment=5, multiplier=1, score=5, notation="S5")
D5 = Hit(segment=5, multiplier=2, score=10, notation="D5")


def record(correct: bool, confidence: float, margin_mm: float | None = None) -> ThrowRecord:
    return ThrowRecord(
        predicted=T20 if correct else S20,
        truth=T20,
        confidence=confidence,
        margin_mm=margin_mm,
    )


def ranked(n: int, error_ranks) -> list[ThrowRecord]:
    """``n`` records with confidence spread evenly over (0, 1).

    ``error_ranks`` are positions counting from *least* confident, so rank 0 is
    the throw the model is least sure about. Spreading the correct records
    across the range too is what makes "uninformative confidence" actually
    uninformative -- pinning them all to one high value would smuggle in the
    very signal the test is meant to withhold.
    """
    errors = set(error_ranks)
    return [record(i not in errors, (i + 1) / (n + 1)) for i in range(n)]


# --------------------------------------------------------------------------
# Compounding: the #21 table
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("target_legs", "visits", "expected_dart_accuracy"),
    [
        (0.50, 12, 0.98093),
        (0.80, 12, 0.99382),
        (0.90, 12, 0.99708),
        (0.95, 12, 0.99858),
        (0.50, 9, 0.97465),
        (0.90, 17, 0.99794),
    ],
)
def test_matches_the_published_compounding_table(target_legs, visits, expected_dart_accuracy):
    assert required_per_dart_accuracy(target_legs, visits=visits) == pytest.approx(
        expected_dart_accuracy, abs=5e-6
    )


def test_legs_error_free_inverts_required_accuracy():
    for target in (0.5, 0.8, 0.9, 0.99):
        for visits in (9, 12, 17):
            p = required_per_dart_accuracy(target, visits=visits)
            assert legs_error_free(p, visits=visits) == pytest.approx(target)


def test_published_state_of_the_art_lands_on_the_fifty_percent_row():
    """94.7% per image implies ~1.8% per dart, spoiling about half of all legs."""
    per_dart = 0.947 ** (1 / 3)
    assert per_dart == pytest.approx(0.982, abs=0.001)
    assert 0.50 <= legs_error_free(per_dart, visits=12) <= 0.55


def test_perfect_and_hopeless_extremes():
    assert legs_error_free(1.0) == 1.0
    assert legs_error_free(0.0) == 0.0
    assert required_per_dart_accuracy(1.0) == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_rejects_out_of_range_accuracy(bad):
    with pytest.raises(ValueError):
        legs_error_free(bad)


# --------------------------------------------------------------------------
# Confidence: the gate that matters
# --------------------------------------------------------------------------

def test_reproduces_the_published_confidence_gating_row():
    """2% raw error, flag 15% capturing 80% of errors -> ~84% of legs clean."""
    # 1000 darts, 20 wrong: 16 among the least confident 150, 4 above.
    records = (
        [record(False, 0.10 + 0.001 * i) for i in range(16)]
        + [record(True, 0.10 + 0.001 * (16 + i)) for i in range(134)]
        + [record(False, 0.90 + 0.0001 * i) for i in range(4)]
        + [record(True, 0.95 + 0.00001 * i) for i in range(846)]
    )
    assert len(records) == 1000
    assert per_dart_accuracy(records) == pytest.approx(0.98)
    assert error_concentration(records, 0.15) == pytest.approx(0.80)
    assert effective_legs_error_free(records, flag_fraction=0.15) == pytest.approx(
        0.844, abs=0.005
    )
    # Without gating the same model leaves less than half of legs clean.
    assert legs_error_free(0.98, visits=12) == pytest.approx(0.483, abs=0.005)


def test_concentration_is_one_when_confidence_ranks_every_error_last():
    assert error_concentration(ranked(100, range(5)), 0.10) == pytest.approx(1.0)


def test_concentration_is_low_when_confidence_is_uninformative():
    """Errors spread evenly across the confidence range are barely caught."""
    records = ranked(100, range(0, 100, 10))  # 10 errors, one per decile
    assert error_concentration(records, 0.10) == pytest.approx(0.1)


def test_concentration_is_one_when_there_are_no_errors():
    assert error_concentration([record(True, 0.9) for _ in range(10)], 0.2) == 1.0


def test_honest_confidence_beats_higher_raw_accuracy():
    """The central claim of #21, as an executable comparison.

    The honest model is twice as wrong in raw terms, but every mistake it makes
    is one it was unsure about, so a confirmation flow removes them all. The
    overconfident model is more accurate and unusable.
    """
    honest = ranked(1000, range(40))                 # 4% error, all least-confident
    overconfident = ranked(1000, range(980, 1000))   # 2% error, all most-confident

    assert per_dart_accuracy(honest) == pytest.approx(0.96)
    assert per_dart_accuracy(overconfident) == pytest.approx(0.98)
    assert per_dart_accuracy(honest) < per_dart_accuracy(overconfident)

    assert effective_legs_error_free(honest, 0.15) == pytest.approx(1.0)
    assert effective_legs_error_free(overconfident, 0.15) < 0.5


# --------------------------------------------------------------------------
# Risk-coverage
# --------------------------------------------------------------------------

def test_risk_coverage_curve_is_monotone_in_coverage():
    curve = risk_coverage_curve(ranked(50, range(5)))
    assert [c for c, _ in curve] == sorted(c for c, _ in curve)
    assert curve[-1][0] == pytest.approx(1.0)
    assert curve[-1][1] == pytest.approx(0.1)


def test_coverage_at_error_rate_finds_the_usable_operating_point():
    records = ranked(100, range(10))
    assert coverage_at_error_rate(records, 0.0) == pytest.approx(0.90)
    assert coverage_at_error_rate(records, 1.0) == pytest.approx(1.0)


def test_coverage_is_zero_when_the_most_confident_throw_is_wrong():
    records = [record(False, 0.99)] + [record(True, 0.5) for _ in range(9)]
    assert coverage_at_error_rate(records, 0.0) == 0.0


# --------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("predicted", "truth", "expected"),
    [
        (T20, T20, FailureKind.CORRECT),
        (None, T20, FailureKind.MISSED_DART),
        (T20, None, FailureKind.PHANTOM_DART),
        (S5, S20, FailureKind.WRONG_SECTOR),
        (S20, T20, FailureKind.WRONG_MULTIPLIER),
        (D5, T20, FailureKind.WRONG_BOTH),
        (None, None, FailureKind.CORRECT),
    ],
)
def test_failure_taxonomy_separates_angular_from_radial(predicted, truth, expected):
    assert classify(ThrowRecord(predicted=predicted, truth=truth, confidence=0.5)) == expected


# --------------------------------------------------------------------------
# Margin diagnosis
# --------------------------------------------------------------------------

def test_diagnosis_flags_a_model_problem_when_errors_have_room_to_spare():
    records = [record(False, 0.5, margin_mm=25.0) for _ in range(8)] + [
        record(True, 0.9, margin_mm=25.0) for _ in range(92)
    ]
    d = diagnose(records, margin_threshold_mm=3.0)
    assert d.wrong_with_room == 8 and d.wrong_near_boundary == 0
    assert d.model_problem_share == pytest.approx(1.0)


def test_diagnosis_flags_a_precision_limit_when_errors_sit_on_boundaries():
    records = [record(False, 0.5, margin_mm=0.4) for _ in range(8)] + [
        record(True, 0.9, margin_mm=25.0) for _ in range(92)
    ]
    d = diagnose(records, margin_threshold_mm=3.0)
    assert d.wrong_near_boundary == 8 and d.wrong_with_room == 0
    assert d.model_problem_share == pytest.approx(0.0)


def test_diagnosis_skips_records_without_a_margin():
    records = [record(False, 0.5), record(True, 0.9, margin_mm=10.0)]
    assert diagnose(records, margin_threshold_mm=3.0).total == 1


def test_diagnosis_share_is_zero_when_nothing_is_wrong():
    d = diagnose([record(True, 0.9, margin_mm=10.0)], margin_threshold_mm=3.0)
    assert d.model_problem_share == 0.0


def test_empty_inputs_are_rejected():
    for fn in (per_dart_accuracy, risk_coverage_curve):
        with pytest.raises(ValueError, match="no records"):
            fn([])
