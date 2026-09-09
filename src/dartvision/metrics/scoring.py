"""Scoring metrics: #14's definitions and #21's gates, as code.

Standard library only, so the same metrics run in the training harness, in a
notebook, and in the application.

The metrics that matter here are not the obvious ones. Per-image accuracy
flatters a scorer because players experience *legs*, and a single wrong dart
spoils one. And a model that routes its least-confident throws to a one-tap
confirmation converts raw accuracy into something far better -- so the
operationally decisive quantities are the risk-coverage curve and the share of
errors that actually land in the flagged set.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from dartvision.geometry.board import Hit

__all__ = [
    "ThrowRecord",
    "FailureKind",
    "per_dart_accuracy",
    "legs_error_free",
    "required_per_dart_accuracy",
    "risk_coverage_curve",
    "coverage_at_error_rate",
    "error_concentration",
    "effective_legs_error_free",
    "Diagnosis",
    "diagnose",
]

DARTS_PER_VISIT = 3
TYPICAL_VISITS_PER_LEG = 12


@dataclass(frozen=True)
class ThrowRecord:
    """One predicted dart, paired with truth and the geometry around it."""

    predicted: Hit | None
    truth: Hit | None
    confidence: float
    tip_error_mm: float | None = None
    margin_mm: float | None = None

    @property
    def correct(self) -> bool:
        """Both present and scoring identically. A miss on either side counts."""
        if self.predicted is None or self.truth is None:
            return False
        return (
            self.predicted.segment == self.truth.segment
            and self.predicted.multiplier == self.truth.multiplier
        )


class FailureKind(str, Enum):
    """#14's failure taxonomy. Angular and radial errors stay separate: they
    have different causes and different fixes."""

    CORRECT = "correct"
    MISSED_DART = "missed-dart"          # truth present, nothing predicted
    PHANTOM_DART = "phantom-dart"        # predicted, no truth
    WRONG_SECTOR = "wrong-sector"        # angular error
    WRONG_MULTIPLIER = "wrong-multiplier"  # radial error
    WRONG_BOTH = "wrong-both"


def classify(record: ThrowRecord) -> FailureKind:
    if record.truth is None and record.predicted is None:
        return FailureKind.CORRECT
    if record.predicted is None:
        return FailureKind.MISSED_DART
    if record.truth is None:
        return FailureKind.PHANTOM_DART
    if record.correct:
        return FailureKind.CORRECT
    sector_wrong = record.predicted.segment != record.truth.segment
    multiplier_wrong = record.predicted.multiplier != record.truth.multiplier
    if sector_wrong and multiplier_wrong:
        return FailureKind.WRONG_BOTH
    return FailureKind.WRONG_SECTOR if sector_wrong else FailureKind.WRONG_MULTIPLIER


def per_dart_accuracy(records: Sequence[ThrowRecord]) -> float:
    if not records:
        raise ValueError("no records")
    return sum(1 for r in records if r.correct) / len(records)


def legs_error_free(
    dart_accuracy: float,
    visits: int = TYPICAL_VISITS_PER_LEG,
    darts_per_visit: int = DARTS_PER_VISIT,
) -> float:
    """Fraction of legs with no scoring error, given per-dart accuracy.

    Assumes independent per-dart errors. That is deliberately conservative:
    real errors correlate -- a poor homography spoils all three darts in one
    image -- which concentrates errors into fewer legs and makes true
    cleanliness somewhat better than this returns.
    """
    if not 0.0 <= dart_accuracy <= 1.0:
        raise ValueError("dart_accuracy must be in [0, 1]")
    if visits < 1 or darts_per_visit < 1:
        raise ValueError("visits and darts_per_visit must be positive")
    return dart_accuracy ** (visits * darts_per_visit)


def required_per_dart_accuracy(
    target_legs_error_free: float,
    visits: int = TYPICAL_VISITS_PER_LEG,
    darts_per_visit: int = DARTS_PER_VISIT,
) -> float:
    """Inverse of :func:`legs_error_free` -- what a per-leg target costs."""
    if not 0.0 < target_legs_error_free <= 1.0:
        raise ValueError("target must be in (0, 1]")
    if visits < 1 or darts_per_visit < 1:
        raise ValueError("visits and darts_per_visit must be positive")
    return target_legs_error_free ** (1.0 / (visits * darts_per_visit))


def _by_confidence(records: Sequence[ThrowRecord]) -> list[ThrowRecord]:
    return sorted(records, key=lambda r: r.confidence, reverse=True)


def risk_coverage_curve(
    records: Sequence[ThrowRecord],
) -> list[tuple[float, float]]:
    """``(coverage, error_rate)`` for auto-scoring the most confident throws.

    This curve is the answer to "should uncertain detections auto-score or
    ask?" -- it says what error rate each level of automation costs.
    """
    if not records:
        raise ValueError("no records")
    ordered = _by_confidence(records)
    curve: list[tuple[float, float]] = []
    errors = 0
    for i, record in enumerate(ordered, start=1):
        if not record.correct:
            errors += 1
        curve.append((i / len(ordered), errors / i))
    return curve


def coverage_at_error_rate(
    records: Sequence[ThrowRecord], max_error_rate: float
) -> float:
    """Largest share of throws auto-scorable while holding error at or below a bound.

    Returns 0.0 when even the single most confident throw exceeds the bound.
    """
    if not 0.0 <= max_error_rate <= 1.0:
        raise ValueError("max_error_rate must be in [0, 1]")
    best = 0.0
    for coverage, error_rate in risk_coverage_curve(records):
        if error_rate <= max_error_rate:
            best = coverage
    return best


def error_concentration(
    records: Sequence[ThrowRecord], flag_fraction: float
) -> float:
    """Share of all errors falling inside the least-confident ``flag_fraction``.

    #21 treats this as a first-class gate. It is what makes a confirmation flow
    worth having: a model whose mistakes cluster where it is unsure can be made
    trustworthy by asking, while one whose confidence is uninformative cannot.
    Returns 1.0 when there are no errors to concentrate.
    """
    if not 0.0 < flag_fraction <= 1.0:
        raise ValueError("flag_fraction must be in (0, 1]")
    if not records:
        raise ValueError("no records")

    total_errors = sum(1 for r in records if not r.correct)
    if total_errors == 0:
        return 1.0
    ordered = _by_confidence(records)
    flagged_count = max(1, round(len(ordered) * flag_fraction))
    flagged = ordered[len(ordered) - flagged_count:]
    return sum(1 for r in flagged if not r.correct) / total_errors


def effective_legs_error_free(
    records: Sequence[ThrowRecord],
    flag_fraction: float = 0.15,
    visits: int = TYPICAL_VISITS_PER_LEG,
) -> float:
    """Leg cleanliness once flagged throws are confirmed by the player.

    Flagged throws are treated as corrected, which is what a confirmation tap
    achieves. This is the headline product number in #21.
    """
    if not records:
        raise ValueError("no records")
    ordered = _by_confidence(records)
    flagged_count = max(1, round(len(ordered) * flag_fraction))
    auto = ordered[: len(ordered) - flagged_count]
    if not auto:
        return 1.0
    return legs_error_free(per_dart_accuracy(auto), visits=visits)


@dataclass(frozen=True)
class Diagnosis:
    """#14's margin cross-tab: is the model imprecise, or was this unscoreable?

    A localization error only matters if it crosses a boundary. Splitting
    misscored darts by how close they were to one separates "keep training"
    from "the information is not in the image", which #21 requires as evidence
    before any single-camera-physics claim.
    """

    wrong_with_room: int      # misscored despite a comfortable margin
    wrong_near_boundary: int  # misscored while close to a boundary
    right_with_room: int
    right_near_boundary: int  # correct, but only just

    @property
    def total(self) -> int:
        return (
            self.wrong_with_room
            + self.wrong_near_boundary
            + self.right_with_room
            + self.right_near_boundary
        )

    @property
    def model_problem_share(self) -> float:
        """Of the misscored darts, the share that had room to spare.

        High means the model is genuinely imprecise and more training should
        help. Low means errors sit almost entirely on knife-edge darts, which
        is the signature of a precision limit rather than a learning failure.
        """
        wrong = self.wrong_with_room + self.wrong_near_boundary
        return self.wrong_with_room / wrong if wrong else 0.0


def diagnose(
    records: Sequence[ThrowRecord], margin_threshold_mm: float
) -> Diagnosis:
    """Build the margin cross-tab. Records lacking a margin are skipped."""
    if margin_threshold_mm <= 0:
        raise ValueError("margin_threshold_mm must be positive")
    cells = {"ww": 0, "wn": 0, "rw": 0, "rn": 0}
    for record in records:
        if record.margin_mm is None:
            continue
        near = record.margin_mm < margin_threshold_mm
        if record.correct:
            cells["rn" if near else "rw"] += 1
        else:
            cells["wn" if near else "ww"] += 1
    return Diagnosis(
        wrong_with_room=cells["ww"],
        wrong_near_boundary=cells["wn"],
        right_with_room=cells["rw"],
        right_near_boundary=cells["rn"],
    )
