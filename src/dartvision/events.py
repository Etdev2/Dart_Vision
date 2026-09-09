"""The canonical throw event: the contract between the Brain and everything else.

`AGENTS.md` states the guardrail this implements -- computer vision emits a
normalized observation, and downstream systems consume it, so third-party
adapters never depend on model internals and game rules never leak into vision.

The boundary, precisely:

* **Vision** answers *where the tip is* and *how sure it is*.
* **Geometry** -- deterministic, tested, no ML -- turns that into a board
  coordinate and a segment.
* **This event** carries both, plus the provenance needed to correct it.
* **The game engine** applies rules: bust, double-out, turn order, checkout.

So an event says "this dart is a treble 20, worth 60". It never says whether
that busted the leg. That is the engine's job, and keeping it out is what makes
scoring testable without a rulebook and rules testable without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from dartvision.geometry.board import BDO_BOARD, BoardSpec, Hit, nearest_boundary, score_at

__all__ = [
    "Source",
    "CalibrationQuality",
    "CalibrationStatus",
    "Detection",
    "ThrowEvent",
    "ConfirmationPolicy",
    "THROW_EVENT_FIELDS",
]


class Source(str, Enum):
    """Where a throw event came from."""

    CAMERA = "camera"
    MANUAL = "manual"              # entered by hand, no detection at all
    CORRECTION = "correction"      # replaces an earlier event


class CalibrationQuality(str, Enum):
    GOOD = "good"
    MARGINAL = "marginal"
    UNUSABLE = "unusable"


@dataclass(frozen=True)
class CalibrationStatus:
    """Whether the board can be scored at all, kept separate from throws.

    This is not a property of a throw and must not be modelled as one. Below
    four landmarks there is **no score for any dart** -- a cliff, not a slope --
    so the application needs to distinguish "the board is not calibrated" from
    "no darts were detected". Collapsing the two produces a frozen scoreboard
    with no explanation, which is the worst failure the UI can present.
    """

    calibration_id: str
    session_id: str
    established_at: str
    landmarks_found: int
    landmarks_expected: int
    board_coverage: float
    landmark_confidence: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.landmarks_expected not in (4, 8):
            raise ValueError("landmarks_expected must be 4 or 8")
        if not 0 <= self.landmarks_found <= self.landmarks_expected:
            raise ValueError("landmarks_found must be between 0 and landmarks_expected")
        if not 0.0 <= self.board_coverage <= 1.0:
            raise ValueError("board_coverage must be in [0, 1]")

    @property
    def scoring_possible(self) -> bool:
        """Four landmarks are the minimum for a homography. Below it, nothing scores."""
        return self.landmarks_found >= 4

    @property
    def quality(self) -> CalibrationQuality:
        """Coverage below ~12% carries no recoverable precision (see #25)."""
        if not self.scoring_possible or self.board_coverage < 0.08:
            return CalibrationQuality.UNUSABLE
        if self.board_coverage < 0.12 or self.landmarks_found < self.landmarks_expected:
            return CalibrationQuality.MARGINAL
        return CalibrationQuality.GOOD

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_id": self.calibration_id,
            "session_id": self.session_id,
            "established_at": self.established_at,
            "landmarks_found": self.landmarks_found,
            "landmarks_expected": self.landmarks_expected,
            "landmark_confidence": list(self.landmark_confidence),
            "board_coverage": round(self.board_coverage, 4),
            "scoring_possible": self.scoring_possible,
            "quality": self.quality.value,
        }


@dataclass(frozen=True)
class Detection:
    """What the dart scored, and how much to trust it."""

    segment: int
    multiplier: int
    score: int
    notation: str
    confidence: float
    margin_mm: float
    boundary: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.margin_mm < 0.0:
            raise ValueError("margin_mm cannot be negative")
        if self.multiplier not in (0, 1, 2, 3):
            raise ValueError("multiplier must be 0, 1, 2 or 3")
        if self.score != (0 if self.multiplier == 0 else self.segment * self.multiplier):
            raise ValueError(
                f"score {self.score} does not follow from segment {self.segment} "
                f"and multiplier {self.multiplier}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "segment": self.segment,
            "multiplier": self.multiplier,
            "score": self.score,
            "notation": self.notation,
            "confidence": round(self.confidence, 4),
            "margin_mm": round(self.margin_mm, 3),
            "boundary": self.boundary,
        }


THROW_EVENT_FIELDS = (
    "id", "session_id", "captured_at", "calibration_id", "source",
    "board_x_mm", "board_y_mm", "image_x", "image_y",
    "detected", "player_id", "turn_id", "sequence", "corrects", "image_ref",
)


@dataclass(frozen=True)
class ThrowEvent:
    """One dart, as observed.

    ``board_x_mm``/``board_y_mm`` are canonical: millimetres in the rectified
    board plane, origin at the bull. Image coordinates are carried only for
    drawing an overlay and mean nothing without the calibration that produced
    them -- which is why the earlier draft's image-normalized impact was the
    wrong choice for the canonical field.
    """

    id: str
    session_id: str
    captured_at: str
    calibration_id: str
    source: Source
    board_x_mm: float
    board_y_mm: float
    detected: Detection
    image_x: float | None = None
    image_y: float | None = None
    player_id: str | None = None
    turn_id: str | None = None
    sequence: int | None = None
    corrects: str | None = None
    image_ref: str | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.session_id or not self.calibration_id:
            raise ValueError("id, session_id and calibration_id are required")
        if self.sequence is not None and not 1 <= self.sequence <= 3:
            raise ValueError("sequence must be 1, 2 or 3 within a turn")
        if self.corrects and self.source is not Source.CORRECTION:
            raise ValueError("an event that corrects another must have source 'correction'")

    @classmethod
    def from_board_point(
        cls,
        *,
        id: str,
        session_id: str,
        calibration_id: str,
        x_mm: float,
        y_mm: float,
        confidence: float,
        source: Source = Source.CAMERA,
        board: BoardSpec = BDO_BOARD,
        captured_at: str | None = None,
        **kwargs: object,
    ) -> "ThrowEvent":
        """Build an event from a board coordinate, scoring it deterministically.

        The score is *derived*, never supplied: a caller cannot assert a
        segment that disagrees with the geometry.
        """
        hit: Hit = score_at(x_mm, y_mm, board)
        boundary = nearest_boundary(x_mm, y_mm, board)
        return cls(
            id=id,
            session_id=session_id,
            calibration_id=calibration_id,
            captured_at=captured_at or datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            source=source,
            board_x_mm=float(x_mm),
            board_y_mm=float(y_mm),
            detected=Detection(
                segment=hit.segment,
                multiplier=hit.multiplier,
                score=hit.score,
                notation=hit.notation,
                confidence=confidence,
                margin_mm=boundary.distance_mm,
                boundary=boundary.name,
            ),
            **kwargs,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "session_id": self.session_id,
            "captured_at": self.captured_at,
            "calibration_id": self.calibration_id,
            "source": self.source.value,
            "board_x_mm": round(self.board_x_mm, 3),
            "board_y_mm": round(self.board_y_mm, 3),
            "image_x": self.image_x,
            "image_y": self.image_y,
            "detected": self.detected.to_dict(),
            "player_id": self.player_id,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "corrects": self.corrects,
            "image_ref": self.image_ref,
        }
        if self.meta:
            payload["meta"] = self.meta
        return payload


@dataclass(frozen=True)
class ConfirmationPolicy:
    """Decides which throws are auto-scored and which are put to the player.

    #5 set the budget: flag roughly the least-confident 15% of darts, about two
    confirmations a leg. But **model confidence alone is the wrong signal.**
    Confidence answers "how sure am I where the tip is"; it does not answer
    "does that uncertainty change the score". Those come apart constantly:

    * 2 mm of uncertainty, 20 mm from any boundary -- certain, whatever the
      model says.
    * The same 2 mm, 0.5 mm from a treble wire -- a coin flip.

    So the policy combines both. A throw is auto-scored only when the model is
    confident *and* the dart is far enough from a boundary that the residual
    error cannot cross it. This is the routing decision #10's correction flow
    is built around, and it is cheap: the margin comes free from geometry.
    """

    min_confidence: float = 0.90
    min_margin_mm: float = 2.0
    high_confidence: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.min_margin_mm < 0.0:
            raise ValueError("min_margin_mm cannot be negative")
        if not self.min_confidence <= self.high_confidence <= 1.0:
            raise ValueError("high_confidence must be at least min_confidence")

    def should_confirm(self, event: ThrowEvent) -> bool:
        """True when the player should be asked rather than told."""
        detected = event.detected
        if detected.confidence < self.min_confidence:
            return True
        if detected.margin_mm < self.min_margin_mm:
            # Near a boundary, only a very confident call stands on its own.
            return detected.confidence < self.high_confidence
        return False

    def partition(
        self, events: Sequence[ThrowEvent]
    ) -> tuple[list[ThrowEvent], list[ThrowEvent]]:
        """``(auto_scored, needs_confirmation)``."""
        auto, confirm = [], []
        for event in events:
            (confirm if self.should_confirm(event) else auto).append(event)
        return auto, confirm
