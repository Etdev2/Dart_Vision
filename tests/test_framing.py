"""Tests for setup-time framing guidance.

Built on the synthetic camera, so the true elevation and framing of every test
case are known exactly and the assessment can be checked against them rather
than against itself.

The claim worth testing hardest is that elevation is recoverable from the
landmarks alone. If that holds, the setup screen never has to ask the player
for an angle -- and a browser cannot be trusted to report focal length, sensor
size or distance, so any method needing those would not work at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dartvision.calibrate import (
    MARGINAL,
    READY,
    UNUSABLE,
    FramingLimits,
    FramingReport,
    assess_framing,
)
from dartvision.geometry.calibration import calibration_points_8
from dartvision.geometry.camera import CameraPose, board_to_image_matrix
from dartvision.synthetic.scene import focal_for_board_fill

IMAGE = (1600, 1200)          # (height, width)


def landmarks_for(elevation_deg, azimuth_deg=25.0, board_fill=0.85, distance_mm=2000.0,
                  noise_px=0.0, seed=0):
    """The 8 landmarks a detector would report for a given camera pose."""
    pose = CameraPose(
        elevation_deg=elevation_deg, azimuth_deg=azimuth_deg, distance_mm=distance_mm,
        focal_px=focal_for_board_fill(distance_mm, board_fill, IMAGE),
        image_size=(IMAGE[1], IMAGE[0]),
    )
    matrix = board_to_image_matrix(pose)
    rng = np.random.default_rng(seed)

    points = []
    for bx, by in calibration_points_8():
        projected = matrix @ np.array([bx, by, 1.0])
        x, y = projected[:2] / projected[2]
        if noise_px:
            x, y = x + rng.normal(0, noise_px), y + rng.normal(0, noise_px)
        points.append((x / IMAGE[1], y / IMAGE[0]))
    return points


# --------------------------------------------------------------------------
# Recovering the angle
# --------------------------------------------------------------------------

@pytest.mark.parametrize("elevation", [20, 30, 40, 50, 60, 70, 80])
def test_elevation_is_recovered_without_any_camera_intrinsics(elevation):
    """Nothing about the lens, sensor or distance is supplied -- only the eight
    detected points. A browser cannot report those reliably, so a method that
    needed them would be unusable in the product."""
    report = assess_framing(landmarks_for(elevation), IMAGE)
    assert report.elevation_deg == pytest.approx(elevation, abs=0.5)


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 187.0, 300.0])
def test_the_estimate_does_not_depend_on_which_side_the_phone_is_on(azimuth):
    report = assess_framing(landmarks_for(50.0, azimuth_deg=azimuth), IMAGE)
    assert report.elevation_deg == pytest.approx(50.0, abs=0.5)


@pytest.mark.parametrize("distance", [1400.0, 2000.0, 3000.0])
def test_the_estimate_does_not_depend_on_distance(distance):
    report = assess_framing(
        landmarks_for(55.0, distance_mm=distance, board_fill=0.85), IMAGE
    )
    assert report.elevation_deg == pytest.approx(55.0, abs=0.5)


def test_the_estimate_survives_realistic_landmark_noise():
    """Three pixels of landmark error is worse than #21 gates for, and the
    guidance band is twenty degrees wide, so a degree or two costs nothing."""
    errors = [
        assess_framing(landmarks_for(50.0, noise_px=3.0, seed=s), IMAGE).elevation_deg - 50.0
        for s in range(40)
    ]
    assert max(abs(e) for e in errors) < 3.0
    assert abs(float(np.mean(errors))) < 1.0


# --------------------------------------------------------------------------
# The verdicts
# --------------------------------------------------------------------------

def test_a_well_mounted_phone_is_ready():
    report = assess_framing(landmarks_for(55.0, board_fill=0.92), IMAGE)
    assert report.verdict == READY
    assert report.ready
    assert report.guidance == "Ready to throw."


def test_a_side_on_view_is_refused_with_the_reason():
    """#2: below about 30 degrees, #21's gates need sub-pixel precision that
    the representation cannot deliver."""
    report = assess_framing(landmarks_for(18.0), IMAGE)
    assert report.verdict == UNUSABLE
    assert "toward the front" in report.guidance


def test_being_in_the_throwing_line_outranks_everything_else():
    """Face-on is the most accurate view #2 measured. It is also where the
    darts are, and that costs a phone rather than a score."""
    report = assess_framing(landmarks_for(84.0, board_fill=0.3), IMAGE)

    assert report.verdict == UNUSABLE
    assert "where the darts fly" in report.guidance
    # The board is also far too small, but that is the less urgent problem.
    assert report.ring_px < 10.0


def test_a_distant_board_is_refused_for_resolution():
    report = assess_framing(landmarks_for(55.0, board_fill=0.25), IMAGE)
    assert report.verdict == UNUSABLE
    assert "too small in the frame" in report.guidance


def test_a_workable_but_imperfect_angle_is_marginal_not_refused():
    report = assess_framing(landmarks_for(40.0, board_fill=0.92), IMAGE)
    assert report.verdict == MARGINAL
    assert "Good enough to score" in report.guidance


def test_an_oblique_view_is_told_to_square_up_not_to_move_closer():
    """The same short ring measurement has two causes, and only one of them is
    fixed by moving closer. A player already filling the frame at 36 degrees
    cannot move close enough, and telling them to is the worst advice available.
    """
    report = assess_framing(landmarks_for(36.0, board_fill=0.92), IMAGE)

    assert report.verdict == UNUSABLE
    assert report.board_fill > 0.85           # already as close as it gets
    assert "Too side-on for this distance" in report.guidance


def test_a_slightly_small_board_warns_about_the_wires():
    """The failure this produces is not random -- it is specifically scores
    near a boundary, which is what #14 makes a first-class metric."""
    report = assess_framing(landmarks_for(55.0, board_fill=0.85), IMAGE)

    assert report.verdict == MARGINAL
    assert "near the wires" in report.guidance


# --------------------------------------------------------------------------
# Resolution is measured where it is lost
# --------------------------------------------------------------------------

def test_ring_pixels_are_measured_at_the_model_input_not_the_camera():
    """A 12-megapixel frame resized to 640 keeps 640 pixels' worth of detail.
    Judging the framing on the camera's resolution would pass framings the
    model cannot actually use (#4)."""
    big = assess_framing(landmarks_for(55.0), IMAGE, FramingLimits(model_input_px=768))
    small = assess_framing(landmarks_for(55.0), IMAGE, FramingLimits(model_input_px=320))

    assert big.ring_px > small.ring_px
    assert small.ring_px == pytest.approx(big.ring_px * 320 / 768, rel=1e-6)


def test_perspective_costs_ring_pixels_the_face_on_arithmetic_does_not_see():
    """The easy calculation -- ring width times focal over distance -- is the
    face-on figure. A board tilted out of the image plane is compressed along
    the tilt by roughly sin(elevation), and believing the face-on number is how
    a model input gets chosen a size too small (#4).
    """
    from dartvision.geometry.board import BDO_BOARD

    fill, model_input = 0.85, 768
    face_on = BDO_BOARD.ring_width * fill * model_input / (2 * BDO_BOARD.r_board)
    report = assess_framing(landmarks_for(55.0, board_fill=fill), IMAGE,
                            FramingLimits(model_input_px=model_input))

    assert face_on == pytest.approx(14.5, abs=0.2)
    assert report.ring_px == pytest.approx(face_on * math.sin(math.radians(55.0)), rel=0.05)
    assert report.ring_px < face_on


def test_filling_more_of_the_frame_gives_more_ring_pixels():
    near = assess_framing(landmarks_for(55.0, board_fill=0.9), IMAGE)
    far = assess_framing(landmarks_for(55.0, board_fill=0.5), IMAGE)

    assert near.ring_px > far.ring_px
    assert near.board_fill == pytest.approx(0.9, abs=0.03)
    assert far.board_fill == pytest.approx(0.5, abs=0.03)


def test_an_oblique_view_costs_precision_at_the_same_framing():
    """The worst case is what matters: one compressed direction is enough to
    lose a score, however good the other one looks."""
    square = assess_framing(landmarks_for(70.0, board_fill=0.85), IMAGE)
    oblique = assess_framing(landmarks_for(35.0, board_fill=0.85), IMAGE)

    assert oblique.mm_per_px_worst > square.mm_per_px_worst


# --------------------------------------------------------------------------
# Not enough board
# --------------------------------------------------------------------------

def test_no_landmarks_asks_the_player_to_aim_at_the_board():
    report = assess_framing([None] * 8, IMAGE)
    assert report.verdict == UNUSABLE
    assert report.landmarks_found == 0
    assert "Point the phone at the board" in report.guidance
    assert report.elevation_deg is None


def test_a_partly_visible_board_says_so_specifically():
    partial = list(landmarks_for(55.0))
    for i in range(5):
        partial[i] = None

    report = assess_framing(partial, IMAGE)
    assert report.verdict == UNUSABLE
    assert report.landmarks_found == 3
    assert "out of frame or hidden" in report.guidance


def test_four_landmarks_are_enough_to_assess():
    """The same cliff as everywhere else: four is the minimum, and above it the
    screen should give real guidance rather than refusing."""
    # The four outer-double-wire points. Dropping the *alternate* indices
    # instead would leave four points on one diameter -- collinear, and a
    # homography through them means nothing.
    partial = list(landmarks_for(55.0, board_fill=0.92))
    for i in (4, 5, 6, 7):
        partial[i] = None

    report = assess_framing(partial, IMAGE)
    assert report.landmarks_found == 4
    assert report.elevation_deg == pytest.approx(55.0, abs=1.0)
    assert report.verdict == READY


def test_a_nonsense_landmark_set_is_refused_rather_than_crashing():
    report = assess_framing([(0.5, 0.5)] * 8, IMAGE)
    assert report.verdict == UNUSABLE
    assert report.elevation_deg is None


def test_collinear_landmarks_are_refused_rather_than_answered():
    """Any four points in general position admit an exact homography, so a
    detector that reports four points along one wire produces confident
    nonsense rather than an error. It has to be caught here."""
    collinear = list(landmarks_for(55.0))
    for i in (1, 3, 5, 7):        # leaves both wires at 9 and 189 degrees: one line
        collinear[i] = None

    report = assess_framing(collinear, IMAGE)
    assert report.verdict == UNUSABLE
    assert report.elevation_deg is None
    assert "does not look right" in report.guidance


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

def test_the_report_serializes_for_the_ui():
    payload = assess_framing(landmarks_for(55.0, board_fill=0.92), IMAGE).to_dict()
    assert payload["ready"] is True
    assert set(payload) == {
        "verdict", "guidance", "ready", "landmarks_found", "elevation_deg",
        "board_fill", "ring_px", "ring_px_worst", "mm_per_px_worst",
    }


def test_limits_must_be_ordered():
    for kwargs in (
        {"min_elevation_deg": 50.0},                       # above the ideal band
        {"ideal_elevation_deg": (65.0, 45.0)},             # inverted
        {"max_elevation_deg": 95.0},                       # past face-on
        {"min_ring_px": 0.0},
        {"model_input_px": 0},
    ):
        with pytest.raises(ValueError):
            FramingLimits(**kwargs)
