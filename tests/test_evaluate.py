"""Tests for the evaluation pass.

Built on the synthetic generator so truth is exact: a scene whose predictions
are its own labels must evaluate as perfect, and any degradation must show up in
the right metric rather than somewhere plausible-looking.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dartvision.data.labels import Annotation, Origin
from dartvision.eval import FrameResult, evaluate_all, evaluate_frame
from dartvision.geometry.camera import CameraPose
from dartvision.synthetic import build_scene, sample_tip_uniform


def scene_and_labels(tips_mm=None, seed=0):
    rng = np.random.default_rng(seed)
    tips = tips_mm if tips_mm is not None else [sample_tip_uniform(rng) for _ in range(3)]
    scene = build_scene(
        CameraPose(elevation_deg=55.0, azimuth_deg=20.0, distance_mm=1800.0, focal_px=4200.0),
        tips, image_id="f0", setup_id="s0", session_id="s0/sess0",
    )
    return scene


def perfect(scene) -> FrameResult:
    """Predictions that are exactly the labels."""
    return evaluate_frame(
        scene.annotation,
        scene.annotation.landmarks,
        [(tip, 0.99) for tip in scene.annotation.tips],
    )


# --------------------------------------------------------------------------
# The identity case
# --------------------------------------------------------------------------

def test_predicting_the_labels_scores_perfectly():
    scene = scene_and_labels()
    result = perfect(scene)

    assert result.calibratable
    assert len(result.records) == 3
    assert all(r.correct for r in result.records)
    assert result.landmark_error_mm == pytest.approx(0.0, abs=1e-6)
    assert all(r.tip_error_mm == pytest.approx(0.0, abs=1e-6) for r in result.records)


def test_every_record_carries_a_margin_for_the_cross_tab():
    """#14's diagnosis needs the margin on every scored dart."""
    for record in perfect(scene_and_labels()).records:
        assert record.margin_mm is not None and record.margin_mm >= 0.0


def test_an_empty_board_produces_no_records_but_still_calibrates():
    scene = scene_and_labels(tips_mm=[])
    result = perfect(scene)
    assert result.calibratable and result.records == []


# --------------------------------------------------------------------------
# The calibration cliff
# --------------------------------------------------------------------------

def test_too_few_predicted_landmarks_means_the_frame_is_unscored():
    """Below four landmarks nothing scores -- reported as its own outcome."""
    scene = scene_and_labels()
    degraded = list(scene.annotation.landmarks)
    for i in range(5):
        degraded[i] = None

    result = evaluate_frame(
        scene.annotation, degraded, [(t, 0.9) for t in scene.annotation.tips]
    )
    assert not result.calibratable
    assert result.records == []
    assert result.landmark_error_mm is None


def test_the_report_separates_calibration_failure_from_scoring_error():
    scene = scene_and_labels()
    good = perfect(scene)
    bad = evaluate_frame(scene.annotation, [None] * 8, [])

    report = evaluate_all([good, bad])
    assert report["frames"] == 2
    assert report["calibratable_rate"] == pytest.approx(0.5)
    # The unscoreable frame must not contribute phantom or missed darts.
    assert report["darts"] == 3


# --------------------------------------------------------------------------
# Degradation shows up in the right place
# --------------------------------------------------------------------------

def test_a_shifted_tip_becomes_a_localization_error_not_a_miss():
    scene = scene_and_labels(tips_mm=[(0.0, 140.0)])
    annotation = scene.annotation
    shifted = (annotation.tips[0][0] + 0.004, annotation.tips[0][1])

    result = evaluate_frame(annotation, annotation.landmarks, [(shifted, 0.9)])
    assert len(result.records) == 1
    assert result.records[0].tip_error_mm > 0.5
    assert result.records[0].predicted is not None


def test_a_wildly_wrong_tip_is_a_phantom_and_a_miss():
    scene = scene_and_labels(tips_mm=[(0.0, 140.0)])
    annotation = scene.annotation
    far = (annotation.tips[0][0] + 0.35, annotation.tips[0][1] + 0.2)

    records = evaluate_frame(annotation, annotation.landmarks, [(far, 0.9)]).records
    assert len(records) == 2
    assert sum(r.predicted is None for r in records) == 1   # the miss
    assert sum(r.truth is None for r in records) == 1        # the phantom


def test_landmark_error_is_measured_in_board_millimetres():
    """Not image pixels -- #15 showed landmark error amplifies through the homography."""
    scene = scene_and_labels()
    nudged = [(x + 0.002, y) for x, y in scene.annotation.landmarks]

    result = evaluate_frame(
        scene.annotation, nudged, [(t, 0.9) for t in scene.annotation.tips]
    )
    assert result.landmark_error_mm is not None and result.landmark_error_mm > 0.0


def test_the_homography_comes_from_predicted_landmarks_not_truth():
    """Using truth would hide the system's largest error source.

    Rectifying with ground-truth landmarks measures tip localization in
    isolation and reports a number the product never sees.
    """
    scene = scene_and_labels(tips_mm=[(0.0, 165.0)])
    annotation = scene.annotation
    skewed = [(x + 0.01, y + 0.006) for x, y in annotation.landmarks]

    result = evaluate_frame(
        annotation, skewed, [(t, 0.95) for t in annotation.tips]
    )
    # Tips were predicted exactly, so any tip error must come from the
    # landmarks' effect on rectification.
    assert result.records[0].tip_error_mm is not None
    assert result.records[0].tip_error_mm > 0.5


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def test_report_carries_the_gate_numbers_21_asks_for():
    frames = [perfect(scene_and_labels(seed=s)) for s in range(6)]
    report = evaluate_all(frames)

    for key in (
        "calibratable_rate", "per_dart_accuracy",
        "legs_error_free_ungated", "legs_error_free_at_15pct_flagged",
        "error_concentration_at_15pct", "coverage_at_0.3pct_error",
        "landmark_error_mm_median", "tip_error_mm_p95", "margin_diagnosis",
    ):
        assert key in report, key
    assert report["per_dart_accuracy"] == pytest.approx(1.0)
    assert report["legs_error_free_ungated"] == pytest.approx(1.0)


def test_report_rejects_an_empty_evaluation():
    with pytest.raises(ValueError, match="no frames"):
        evaluate_all([])
