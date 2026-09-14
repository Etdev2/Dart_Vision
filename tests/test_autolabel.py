"""Suggesting dart tips from what changed between two stills.

Tested against scenes whose answer is known by construction: the dart is drawn,
so its tip is not estimated but chosen. That matters more here than usual —
the thing being checked is an estimate of a point, and a fixture that also
estimated it could only prove the two agreed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dartvision.autolabel.tips import (
    SuggestSettings,
    changed_mask,
    choose_tip,
    endpoints,
    largest_component,
    suggest_pair,
)
from dartvision.geometry.board import BDO_BOARD
from dartvision.geometry.calibration import calibration_points_8, estimate_homography
from dartvision.geometry.camera import CameraPose, board_to_image_matrix, project

WIDTH, HEIGHT = 900, 900
POSE = CameraPose(
    elevation_deg=55.0, azimuth_deg=20.0, distance_mm=1800.0, focal_px=3000.0,
    image_size=(WIDTH, HEIGHT),
)
MATRIX = board_to_image_matrix(POSE)


def to_image(points):
    return project(MATRIX, np.asarray(points, dtype=float))


def homography():
    """The homography the annotator's eight landmarks would produce."""
    board = calibration_points_8(BDO_BOARD)
    return estimate_homography(to_image(board).tolist(), board)


def board_frame() -> np.ndarray:
    """A plain board: dark disc, lighter surround, a little noise."""
    rng = np.random.default_rng(7)
    frame = np.full((HEIGHT, WIDTH), 210.0)
    edge = to_image([(BDO_BOARD.r_board * math.cos(a), BDO_BOARD.r_board * math.sin(a))
                     for a in np.linspace(0, 2 * math.pi, 240)])
    ys, xs = np.mgrid[:HEIGHT, :WIDTH]
    centre = to_image([(0.0, 0.0)])[0]
    radius = np.hypot(edge[:, 0] - centre[0], edge[:, 1] - centre[1]).mean()
    frame[np.hypot(xs - centre[0], ys - centre[1]) <= radius] = 70.0
    return np.clip(frame + rng.normal(0, 1.0, frame.shape), 0, 255).astype(np.uint8)


def with_dart(frame: np.ndarray, tip_mm, height_mm: float = 90.0,
              lean=(0.35, 0.25)) -> tuple[np.ndarray, tuple[float, float]]:
    """Draw a dart standing off the board, and return its true tip in pixels.

    The tip sits on the board plane; the flight stands ``height_mm`` off it
    toward the camera, which is what makes one end recoverable and the other
    not. Height enters the projection properly rather than as a pixel nudge —
    the whole disambiguation rests on that displacement being real.
    """
    tip_px = to_image([tip_mm])[0]
    flight_board = (tip_mm[0] + lean[0] * height_mm, tip_mm[1] + lean[1] * height_mm)
    flat = to_image([flight_board])[0]
    # A point standing off the plane projects further from the board centre
    # than its base, along the line from the principal point.
    principal = np.array(POSE.principal_point, dtype=float)
    flight_px = flat + (flat - principal) * (height_mm / POSE.distance_mm) * 3.0

    drawn = frame.astype(float).copy()
    steps = np.linspace(0, 1, 400)
    for t in steps:
        x = tip_px[0] + t * (flight_px[0] - tip_px[0])
        y = tip_px[1] + t * (flight_px[1] - tip_px[1])
        radius = 1 if t < 0.55 else 3            # thin shaft, fatter flight
        xi, yi = int(round(x)), int(round(y))
        drawn[max(0, yi - radius):yi + radius + 1,
              max(0, xi - radius):xi + radius + 1] = 245.0
    return np.clip(drawn, 0, 255).astype(np.uint8), (float(tip_px[0]), float(tip_px[1]))


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------

def test_only_the_new_dart_registers_as_changed():
    """The premise: with a fixed camera, everything but the dart is identical."""
    empty = board_frame()
    one, _ = with_dart(empty, (40.0, 60.0))

    changed = changed_mask(empty, one, 25.0)

    assert 0 < changed.sum() < 0.01 * changed.size


def test_endpoints_run_along_the_dart_not_the_bounding_box():
    """A diagonal streak's extremes in x and y are corners of a box, not ends."""
    points = np.array([(100 + i, 200 + i) for i in range(60)], dtype=np.int64)
    low, high = endpoints(points)

    assert math.hypot(high[0] - low[0], high[1] - low[1]) == pytest.approx(
        math.hypot(59, 59), rel=0.05)


def test_the_largest_component_ignores_scattered_noise():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:12, 5:40] = True                      # a streak
    mask[50, 50] = mask[55, 20] = True            # specks

    assert len(largest_component(mask)) == 70


def test_no_component_at_all_is_not_a_crash():
    assert len(largest_component(np.zeros((20, 20), dtype=bool))) == 0


# --------------------------------------------------------------------------
# The point the whole thing exists to produce
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tip_mm", [(40.0, 60.0), (-90.0, 20.0), (10.0, -120.0),
                                    (0.0, 0.0), (130.0, 40.0)])
def test_the_suggested_point_is_the_tip_and_not_the_flight(tip_mm):
    """The disambiguation, which is the only part pixels cannot do."""
    empty = board_frame()
    landed, true_tip = with_dart(empty, tip_mm)

    suggestion = suggest_pair(empty, landed)

    assert suggestion.point is not None, suggestion.reason
    error = math.hypot(suggestion.point[0] - true_tip[0],
                       suggestion.point[1] - true_tip[1])
    assert error < 12, f"{error:.1f} px from the tip that was drawn"


def test_a_cleared_board_is_refused_rather_than_guessed_at():
    """Three darts leaving at once is not one dart arriving."""
    empty = board_frame()
    full = empty
    for tip in ((40.0, 60.0), (-50.0, 30.0), (20.0, -70.0)):
        full, _ = with_dart(full, tip)

    suggestion = suggest_pair(full, empty, SuggestSettings(max_area=0.0015))

    assert suggestion.point is None
    assert "cleared board" in suggestion.reason


def test_an_unchanged_pair_suggests_nothing():
    empty = board_frame()
    suggestion = suggest_pair(empty, empty)

    assert suggestion.point is None
    assert "nothing changed" in suggestion.reason


def test_the_thinner_end_is_the_one_chosen():
    """The rule, on a shape built to have one fat end and one thin one."""
    shaft = [(200, 100 + i) for i in range(40)]
    flight = [(200 + dy, 140 + dx) for dy in range(-6, 7) for dx in range(0, 10)]
    points = np.array(shaft + flight, dtype=np.int64)

    tip, ratio = choose_tip(points, endpoints(points))

    assert tip[0] == pytest.approx(100, abs=3), "the thin end, not the flight"
    assert ratio > 1.6


def test_a_dart_pointing_at_the_camera_is_flagged_as_unconfident():
    """Equally wide at both ends is a blob, and the rule has nothing to work
    with — the reader correcting a hundred points should be told which few."""
    from dartvision.autolabel.tips import Suggestion

    blob = np.array([(200 + dy, 300 + dx) for dy in range(-5, 6)
                     for dx in range(-5, 6)], dtype=np.int64)
    _, ratio = choose_tip(blob, endpoints(blob))

    assert not Suggestion((300.0, 200.0), len(blob), ratio).confident


def test_the_lean_dominates_the_parallax_the_old_rule_relied_on():
    """Why the tip is not simply the end nearer the bull. Standing 90 mm off
    the plane moves a flight a few millimetres; the dart's own lean moves it
    ten times as far, in whatever direction the dart happens to point."""
    height, radius = 90.0, 90.0
    parallax = height * radius / (POSE.distance_mm * math.sin(math.radians(55)))
    lean = 90.0 * math.hypot(0.35, 0.25)

    assert lean > 5 * parallax


@pytest.mark.parametrize("kwargs", [{"level": 0.0}, {"level": 300.0},
                                    {"min_area": 0.5}, {"analysis_width": 8}])
def test_impossible_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        SuggestSettings(**kwargs)


# --------------------------------------------------------------------------
# A whole session, end to end
# --------------------------------------------------------------------------

def write_jpg(frame: np.ndarray, path) -> None:
    import subprocess
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-i", "-",
         "-frames:v", "1", "-q:v", "2", str(path)],
        input=frame.tobytes(), capture_output=True,
    )
    if result.returncode != 0 or not path.exists():
        pytest.skip("this ffmpeg cannot write a test image")


def visit_frames(tips_mm):
    """empty, then one dart, two, three — a visit as the extractor emits it."""
    frames, truths, frame = [board_frame()], [], board_frame()
    for tip in tips_mm:
        frame, true_tip = with_dart(frame, tip)
        frames.append(frame)
        truths.append(true_tip)
    return frames, truths


def test_a_visit_accumulates_its_darts(tmp_path):
    from dartvision.autolabel import suggest_session

    tips_mm = [(40.0, 60.0), (-70.0, 30.0), (15.0, -95.0)]
    frames, truths = visit_frames(tips_mm)
    paths = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"frame-{index:04d}.jpg"
        write_jpg(frame, path)
        paths.append(path)

    found = suggest_session(paths)

    assert [len(f.tips) for f in found] == [0, 1, 2, 3], (
        "each still holds the darts standing in it at that moment"
    )
    assert [f.event for f in found] == ["first", "dart", "dart", "dart"]

    # And they are the darts that were drawn, not merely three points.
    last = found[-1]
    for (nx, ny), (tx, ty) in zip(last.tips, truths):
        error = math.hypot(nx * WIDTH - tx, ny * HEIGHT - ty)
        assert error < 20, f"{error:.0f} px from the tip that was drawn"


def test_tips_come_back_normalized(tmp_path):
    """A manifest stores fractions, and the analysis ran at a reduced width —
    a pixel here is not a pixel there."""
    from dartvision.autolabel import suggest_session

    frames, _ = visit_frames([(40.0, 60.0)])
    paths = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"frame-{index:04d}.jpg"
        write_jpg(frame, path)
        paths.append(path)

    found = suggest_session(paths)

    assert found[-1].size == (WIDTH, HEIGHT)
    for x, y in found[-1].tips:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_a_cleared_board_resets_the_count(tmp_path):
    """Two visits in one session: the darts must not carry over."""
    from dartvision.autolabel import suggest_session
    from dartvision.autolabel.tips import SuggestSettings

    first, _ = visit_frames([(40.0, 60.0), (-70.0, 30.0)])
    second, _ = visit_frames([(20.0, -80.0)])
    paths = []
    for index, frame in enumerate(first + second):
        path = tmp_path / f"frame-{index:04d}.jpg"
        write_jpg(frame, path)
        paths.append(path)

    found = suggest_session(paths)

    assert any(f.event == "cleared" for f in found), "the empty board must register"
    assert found[-1].tips and len(found[-1].tips) < 3, (
        "the second visit starts from an empty board, not from the first one's darts"
    )


def test_disagreement_with_a_hand_label_is_reported(tmp_path):
    from dartvision.autolabel import compare_to_labels, suggest_session

    frames, truths = visit_frames([(40.0, 60.0)])
    paths = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"frame-{index:04d}.jpg"
        write_jpg(frame, path)
        paths.append(path)
    found = suggest_session(paths)

    agreeing = {"frame-0001.jpg": [[truths[0][0] / WIDTH, truths[0][1] / HEIGHT]]}
    assert compare_to_labels(found, agreeing, (WIDTH, HEIGHT)) == []

    misplaced = {"frame-0001.jpg": [[0.9, 0.9]]}
    assert compare_to_labels(found, misplaced, (WIDTH, HEIGHT))

    miscounted = {"frame-0001.jpg": [[0.4, 0.4], [0.5, 0.5]]}
    reasons = compare_to_labels(found, miscounted, (WIDTH, HEIGHT))
    assert reasons and "labelled 2 dart(s)" in reasons[0][1]
