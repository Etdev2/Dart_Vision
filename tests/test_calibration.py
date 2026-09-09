"""Tests for calibration landmarks and homography estimation.

Homographies are generated from a synthetic pinhole camera viewing the board
plane, so the tests exercise the same projective geometry a real phone produces
rather than an arbitrary matrix.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dartvision.geometry.board import BDO_BOARD, score_at
from dartvision.geometry.calibration import (
    CALIBRATION_ANGLES_DEG,
    assess_landmarks,
    calibration_points_4,
    calibration_points_8,
    estimate_homography,
)


def board_to_image_homography(
    elevation_deg: float = 20.0,
    azimuth_deg: float = 0.0,
    distance_mm: float = 2400.0,
    focal_px: float = 1400.0,
    principal_point: tuple[float, float] = (600.0, 800.0),
) -> np.ndarray:
    """Homography mapping board millimetres to pixels, via a pinhole camera."""
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    centre = distance_mm * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    forward = -centre / np.linalg.norm(centre)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    rotation = np.vstack([right, -up, forward])
    translation = -rotation @ centre
    intrinsics = np.array(
        [[focal_px, 0.0, principal_point[0]],
         [0.0, focal_px, principal_point[1]],
         [0.0, 0.0, 1.0]]
    )
    h = intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    return h / h[2, 2]


def project(h: np.ndarray, points) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return hom[:, :2] / hom[:, 2:3]


# --------------------------------------------------------------------------
# Landmark definition
# --------------------------------------------------------------------------

def test_landmark_angles_are_mutually_perpendicular():
    diffs = [b - a for a, b in zip(CALIBRATION_ANGLES_DEG, CALIBRATION_ANGLES_DEG[1:])]
    assert diffs == [90.0, 90.0, 90.0]


@pytest.mark.parametrize(
    ("angle", "beds"),
    [(9.0, {"D6", "D13"}), (99.0, {"D20", "D5"}), (189.0, {"D11", "D8"}), (279.0, {"D3", "D17"})],
)
def test_landmarks_sit_exactly_on_sector_wires(angle, beds):
    """Straddling each landmark angle must land in the two beds it separates."""
    found = set()
    for delta in (-0.5, 0.5):
        a = math.radians(angle + delta)
        r = BDO_BOARD.r_double - 1.0
        found.add(score_at(r * math.cos(a), r * math.sin(a)).notation)
    assert found == beds


def test_landmarks_lie_on_the_expected_rings():
    for x, y in calibration_points_4():
        assert math.hypot(x, y) == pytest.approx(BDO_BOARD.r_double)
    eight = calibration_points_8()
    assert len(eight) == 8
    for x, y in eight[4:]:
        assert math.hypot(x, y) == pytest.approx(BDO_BOARD.r_treble)


# --------------------------------------------------------------------------
# Homography estimation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("elevation", [5.0, 20.0, 45.0, 75.0])
@pytest.mark.parametrize("azimuth", [0.0, 37.0, 180.0])
def test_recovers_board_coordinates_exactly_from_four_landmarks(elevation, azimuth):
    h = board_to_image_homography(elevation_deg=elevation, azimuth_deg=azimuth)
    board = np.asarray(calibration_points_4())
    recovered = estimate_homography(project(h, board)).to_board(project(h, board))
    assert recovered == pytest.approx(board, abs=1e-6)


def test_recovers_arbitrary_board_points_not_just_the_landmarks():
    h = board_to_image_homography(elevation_deg=30.0, azimuth_deg=15.0)
    estimated = estimate_homography(project(h, calibration_points_4()))

    probes = [(0.0, 0.0), (0.0, 165.0), (100.0, 0.0), (-60.0, 60.0), (0.0, 102.0)]
    assert estimated.to_board(project(h, probes)) == pytest.approx(np.asarray(probes), abs=1e-6)


def test_round_trip_board_to_image_and_back():
    h = board_to_image_homography()
    estimated = estimate_homography(project(h, calibration_points_4()))
    probes = np.asarray([(0.0, 0.0), (40.0, -120.0), (150.0, 20.0)])
    assert estimated.to_board(estimated.to_image(probes)) == pytest.approx(probes, abs=1e-6)


def test_treble_ring_lands_where_the_scorer_expects_it():
    """An end-to-end check that calibration and scoring agree."""
    h = board_to_image_homography(elevation_deg=35.0, azimuth_deg=22.0)
    estimated = estimate_homography(project(h, calibration_points_4()))
    # A point in the middle of the treble 20 ring, in board space.
    r = (BDO_BOARD.r_treble_inner + BDO_BOARD.r_treble) / 2.0
    board_pt = (r * math.cos(math.radians(90.0)), r * math.sin(math.radians(90.0)))
    x, y = estimated.to_board(project(h, [board_pt]))[0]
    assert score_at(x, y).notation == "T20"


def test_rejects_too_few_correspondences():
    with pytest.raises(ValueError, match="at least 4 correspondences"):
        estimate_homography([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])


def test_rejects_a_landmark_count_with_no_canonical_board_points():
    """Five landmarks must not silently be matched against the 8-point set."""
    with pytest.raises(ValueError, match="no canonical board points"):
        estimate_homography([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 2.0)])


def test_a_custom_landmark_count_works_when_board_points_are_given():
    h = board_to_image_homography(elevation_deg=25.0)
    board = calibration_points_4() + [(0.0, 0.0), (0.0, 102.0)]
    estimated = estimate_homography(project(h, board), board_points=board)
    assert estimated.to_board(project(h, board)) == pytest.approx(np.asarray(board), abs=1e-6)


def test_rejects_coincident_points():
    with pytest.raises(ValueError, match="degenerate"):
        estimate_homography([(5.0, 5.0)] * 4)


# --------------------------------------------------------------------------
# Eight landmarks: why over-determination is worth having
# --------------------------------------------------------------------------

def test_four_landmarks_have_no_residual_by_construction():
    """Four correspondences determine a homography exactly, noise or not."""
    h = board_to_image_homography(elevation_deg=25.0)
    rng = np.random.default_rng(0)
    noisy = project(h, calibration_points_4()) + rng.normal(0.0, 2.0, (4, 2))
    quality = assess_landmarks(noisy)
    assert quality.residual_mm is None
    assert quality.usable


def test_eight_landmarks_expose_noise_as_a_residual():
    h = board_to_image_homography(elevation_deg=25.0)
    clean = project(h, calibration_points_8())
    rng = np.random.default_rng(1)

    assert assess_landmarks(clean).residual_mm == pytest.approx(0.0, abs=1e-6)
    noisy = assess_landmarks(clean + rng.normal(0.0, 2.0, (8, 2))).residual_mm
    assert noisy is not None and noisy > 1.0


def test_eight_landmarks_beat_four_under_noise():
    """Least squares over more landmarks averages down per-point error."""
    h = board_to_image_homography(elevation_deg=25.0, azimuth_deg=10.0)
    probes = np.asarray([(0.0, 0.0), (0.0, 150.0), (120.0, 40.0), (-90.0, -90.0)])
    truth = project(h, probes)
    rng = np.random.default_rng(7)

    err4, err8 = [], []
    for _ in range(200):
        noise4 = rng.normal(0.0, 1.5, (4, 2))
        noise8 = rng.normal(0.0, 1.5, (8, 2))
        h4 = estimate_homography(project(h, calibration_points_4()) + noise4)
        h8 = estimate_homography(project(h, calibration_points_8()) + noise8)
        err4.append(np.linalg.norm(h4.to_board(truth) - probes, axis=1).mean())
        err8.append(np.linalg.norm(h8.to_board(truth) - probes, axis=1).mean())

    assert float(np.median(err8)) < float(np.median(err4))


# --------------------------------------------------------------------------
# What four landmarks can and cannot tell us
# --------------------------------------------------------------------------

def test_detects_a_crossed_landmark_ordering():
    h = board_to_image_homography()
    pts = project(h, calibration_points_4())
    swapped = pts[[0, 2, 1, 3]]  # crossed quadrilateral
    assert not assess_landmarks(swapped).convex_and_ordered
    assert not assess_landmarks(swapped).usable


def test_accepts_a_correctly_ordered_projection_at_steep_angles():
    for elevation in (8.0, 20.0, 40.0, 70.0):
        pts = project(board_to_image_homography(elevation_deg=elevation), calibration_points_4())
        assert assess_landmarks(pts).usable


def test_collinear_landmarks_are_not_usable():
    collinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
    assert not assess_landmarks(collinear).usable
