"""Turning a checkpoint into #21's gate numbers.

The chain, in full: image -> model -> decoded landmarks and tips -> homography
from the *predicted* landmarks -> board coordinates -> deterministic score ->
comparison against truth. Every stage already exists and is tested; this wires
them into the numbers a decision can be made on.

One thing here is easy to get subtly wrong and worth stating. The homography
must be estimated from the **predicted** landmarks, not the true ones. Using
truth would measure tip localization in isolation and quietly hide the largest
error source in the system: #15 established that landmark error propagates
through the homography and amplifies toward the board edge, where the doubles
are. An evaluation that rectifies with ground-truth landmarks reports a number
the product will never see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dartvision.data.labels import Annotation
from dartvision.eval.matching import match_tips
from dartvision.geometry.board import BDO_BOARD, BoardSpec, nearest_boundary, score_at
from dartvision.geometry.calibration import estimate_homography
from dartvision.metrics.scoring import ThrowRecord

__all__ = ["FrameResult", "evaluate_frame", "evaluate_all"]


@dataclass
class FrameResult:
    """What one image produced."""

    image_id: str
    session_id: str
    calibratable: bool
    landmark_error_mm: float | None
    records: list[ThrowRecord]

    @property
    def scored(self) -> bool:
        return self.calibratable


def _to_board(landmarks, points, board: BoardSpec):
    homography = estimate_homography(landmarks, board=board)
    if not points:
        return np.zeros((0, 2))
    return homography.to_board(points)


def evaluate_frame(
    annotation: Annotation,
    predicted_landmarks: Sequence[tuple[float, float] | None],
    predicted_tips: Sequence[tuple[tuple[float, float], float]],
    board: BoardSpec = BDO_BOARD,
    max_match_mm: float = 25.0,
) -> FrameResult:
    """Score one frame's predictions against its labels.

    ``predicted_tips`` are ``(normalized point, confidence)`` pairs as the
    decoder returns them.
    """
    visible = [p for p in predicted_landmarks if p is not None]
    truth_visible = [p for p in annotation.landmarks if p is not None]

    # Below four landmarks nothing in the frame can be scored -- the cliff from
    # #13. Report that as its own outcome rather than as a scoring error.
    if len(visible) < 4 or len(truth_visible) < 4:
        return FrameResult(
            image_id=annotation.image_id, session_id=annotation.session_id,
            calibratable=False, landmark_error_mm=None, records=[],
        )

    truth_board = _to_board(annotation.landmarks, annotation.tips, board)
    predicted_board = _to_board(
        predicted_landmarks, [p for p, _ in predicted_tips], board
    )

    # Landmark quality, measured where it matters: in board millimetres after
    # rectification, not in image pixels.
    truth_landmarks = _to_board(annotation.landmarks, annotation.landmarks, board)
    predicted_as_board = _to_board(predicted_landmarks, predicted_landmarks, board)
    landmark_error = float(
        np.linalg.norm(predicted_as_board - truth_landmarks, axis=1).mean()
    )

    records: list[ThrowRecord] = []
    for match in match_tips(
        [tuple(p) for p in truth_board],
        [tuple(p) for p in predicted_board],
        max_distance_mm=max_match_mm,
    ):
        truth_hit = (
            score_at(*truth_board[match.truth_index], board)
            if match.truth_index is not None else None
        )
        predicted_hit = (
            score_at(*predicted_board[match.prediction_index], board)
            if match.prediction_index is not None else None
        )
        confidence = (
            predicted_tips[match.prediction_index][1]
            if match.prediction_index is not None else 0.0
        )
        margin = (
            nearest_boundary(*truth_board[match.truth_index], board).distance_mm
            if match.truth_index is not None else None
        )
        records.append(
            ThrowRecord(
                predicted=predicted_hit,
                truth=truth_hit,
                confidence=confidence,
                tip_error_mm=match.distance_mm,
                margin_mm=margin,
            )
        )

    return FrameResult(
        image_id=annotation.image_id, session_id=annotation.session_id,
        calibratable=True, landmark_error_mm=landmark_error, records=records,
    )


def evaluate_all(frames: Sequence[FrameResult]) -> dict[str, object]:
    """#21's gate numbers, from a set of evaluated frames."""
    from dartvision.metrics.scoring import (
        coverage_at_error_rate,
        diagnose,
        effective_legs_error_free,
        error_concentration,
        legs_error_free,
        per_dart_accuracy,
    )

    if not frames:
        raise ValueError("no frames")

    records = [r for frame in frames for r in frame.records]
    scored = [f for f in frames if f.scored]
    landmark_errors = [
        f.landmark_error_mm for f in scored if f.landmark_error_mm is not None
    ]
    tip_errors = [
        r.tip_error_mm for r in records if r.tip_error_mm is not None
    ]

    report: dict[str, object] = {
        "frames": len(frames),
        # The cliff from #13: #21 gates this directly at 99.5%.
        "calibratable_rate": round(len(scored) / len(frames), 5),
        "darts": len(records),
    }
    if landmark_errors:
        report["landmark_error_mm_median"] = round(float(np.median(landmark_errors)), 3)
        report["landmark_error_mm_p95"] = round(
            float(np.percentile(landmark_errors, 95)), 3
        )
    if tip_errors:
        report["tip_error_mm_median"] = round(float(np.median(tip_errors)), 3)
        report["tip_error_mm_p95"] = round(float(np.percentile(tip_errors, 95)), 3)

    if records:
        accuracy = per_dart_accuracy(records)
        report.update({
            "per_dart_accuracy": round(accuracy, 5),
            "legs_error_free_ungated": round(legs_error_free(accuracy), 5),
            "legs_error_free_at_15pct_flagged": round(
                effective_legs_error_free(records, flag_fraction=0.15), 5
            ),
            "error_concentration_at_15pct": round(error_concentration(records, 0.15), 5),
            "coverage_at_0.3pct_error": round(coverage_at_error_rate(records, 0.003), 5),
        })
        # #14's cross-tab: is the model imprecise, or was the dart unscoreable?
        diagnosis = diagnose(records, margin_threshold_mm=3.0)
        report["margin_diagnosis"] = {
            "wrong_with_room": diagnosis.wrong_with_room,
            "wrong_near_boundary": diagnosis.wrong_near_boundary,
            "model_problem_share": round(diagnosis.model_problem_share, 4),
        }
    return report
