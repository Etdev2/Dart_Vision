"""Tests for the report that explains an extraction.

The point of this module is that it tells the truth about footage nobody has
seen yet, so what is tested is the reporting itself: that a camera move is
counted once rather than once per frame, that the sweep reflects what the real
chooser would do, and that asking for a diagnosis never writes anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from dartvision.capture import ExtractionSettings
from dartvision.capture.diagnose import (
    DISTURBANCE_FRACTION,
    render,
    scan,
    sweep,
    viewpoints,
)

from tests.test_capture_frames import unattended_board, unattended_visit

SETTINGS = ExtractionSettings()


def test_a_scan_measures_every_step_and_the_first_frame_is_still():
    frames = unattended_visit()
    measured = scan(frames, SETTINGS)

    assert measured.frames_read == len(frames)
    assert measured.means[0] == 0 and measured.changed[0] == 0
    assert measured.seconds == pytest.approx(len(frames) / SETTINGS.analysis_fps)


def test_settled_needs_both_tests_to_pass():
    """The whole point of the rewrite: either measure alone is not enough."""
    measured = scan(unattended_visit(), SETTINGS)

    blind = ExtractionSettings(still_pixels=10_000_000)
    assert measured.settled(blind).sum() > measured.settled(SETTINGS).sum(), (
        "ignoring the pixel test must call strictly more frames still"
    )


def test_a_disturbance_is_counted_once_not_once_per_frame():
    """A phone being re-propped is one event that happens to span a second."""
    still = [unattended_board(1, seed=i) for i in range(6)]
    moved = [np.roll(unattended_board(1, seed=100 + i), 60, axis=1) for i in range(6)]
    # Rolling back and forth: three frames of travel, then a settled new view.
    frames = still + [np.roll(f, 30, axis=1) for f in moved[:3]] + moved

    measured = scan(frames, SETTINGS)
    moves = measured.disturbances()

    assert len(moves) <= 2, f"expected one or two travel spans, got {moves}"
    assert all(end >= start for start, end in moves)


def test_a_static_recording_reports_no_disturbance():
    assert scan(unattended_visit(), SETTINGS).disturbances() == []


def test_a_move_has_to_be_big_to_count_as_one():
    """A dart is not a camera move, however bright it is."""
    measured = scan(unattended_visit(), SETTINGS)
    dart = measured.changed.max()

    assert 0 < dart < measured.pixels * DISTURBANCE_FRACTION


def test_the_sweep_reports_the_current_setting_and_agrees_with_it():
    frames = unattended_visit()
    rows = sweep(frames, SETTINGS)

    current = [row for row in rows if row["current"]]
    assert len(current) == 1, "exactly one row is the setting actually in force"

    from dartvision.capture import choose_frames
    kept, runs, _, _ = choose_frames(frames, SETTINGS)
    assert current[0]["runs"] == runs and current[0]["kept"] == len(kept)


def test_the_sweep_covers_the_requested_grid():
    rows = sweep(unattended_visit(), SETTINGS, still_pixels=(10, 25), min_still_frames=(3,))
    assert [(r["still_pixels"], r["min_still_frames"]) for r in rows] == [(10, 3), (25, 3)]


def test_the_report_says_what_a_person_needs_to_change():
    frames = unattended_visit()
    text = render(scan(frames, SETTINGS), sweep(frames, SETTINGS), SETTINGS)

    assert "still_pixels" in text
    assert "<- current" in text
    assert str(SETTINGS.still_pixels) in text


def test_diagnosing_writes_nothing(tmp_path, monkeypatch):
    """--diagnose is for deciding whether to run the extraction at all, so it
    must not leave a half-made corpus behind if the answer is no."""
    from dartvision.capture import __main__ as cli

    monkeypatch.setattr(
        cli, "scan_video", lambda video, settings: (scan(unattended_visit(), settings),
                                                    unattended_visit()),
    )
    assert cli.main(["--video", str(tmp_path / "anything.mp4"), "--diagnose"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_an_extraction_without_a_destination_is_refused(tmp_path):
    from dartvision.capture import __main__ as cli

    with pytest.raises(SystemExit):
        cli.main(["--video", str(tmp_path / "anything.mp4")])


# --------------------------------------------------------------------------
# Viewpoints: how many sessions is this recording?
# --------------------------------------------------------------------------

def test_a_steady_recording_is_one_session():
    frames = unattended_visit()
    found = viewpoints(frames, SETTINGS, SETTINGS.analysis_fps)

    assert len(found) == 1
    assert found[0].states == 4


def test_a_repositioned_camera_splits_the_recording():
    """The number that decides how footage may be used: landmarks are annotated
    once per session, so a recording with two viewpoints is two sessions."""
    def shifted(darts, seed):
        return np.roll(unattended_board(darts, seed=seed), 55, axis=1)

    frames = (
        [unattended_board(1, seed=i) for i in range(10)]
        + [unattended_board(2, seed=50 + i) for i in range(10)]
        + [shifted(0, seed=500 + i) for i in range(10)]
        + [shifted(1, seed=600 + i) for i in range(10)]
    )
    found = viewpoints(frames, SETTINGS, SETTINGS.analysis_fps)

    assert len(found) == 2
    assert [view.states for view in found] == [2, 2]
    assert found[0].end < found[1].start


def test_someone_crossing_the_shot_is_not_a_new_viewpoint():
    """The failure this replaced: a body entering the frame changes as much of
    the picture as a camera move does, and for half a second looks identical."""
    from tests.test_capture_frames import obstruction

    frames = (
        [unattended_board(1, seed=i) for i in range(10)]
        + [unattended_board(2, seed=40 + i) for i in range(10)]
        + [obstruction(seed=300 + i) for i in range(10)]
        + [unattended_board(0, seed=400 + i) for i in range(10)]
        + [unattended_board(1, seed=700 + i) for i in range(10)]
    )
    found = viewpoints(frames, SETTINGS, SETTINGS.analysis_fps)

    assert len(found) == 1, "the camera never moved; someone walked past it"


def test_a_recording_that_ends_with_someone_at_the_board():
    """No neighbour on the far side to close the parenthesis, so the end rule
    decides it: a state cannot differ from the one before it by that much."""
    from tests.test_capture_frames import obstruction
    from dartvision.capture import choose_frames

    frames = (
        [unattended_board(1, seed=i) for i in range(10)]
        + [unattended_board(2, seed=40 + i) for i in range(10)]
        + [obstruction(seed=300 + i) for i in range(10)]
    )
    kept, _, _, blocked = choose_frames(frames, SETTINGS)

    assert blocked == 1 and len(kept) == 2
    assert len(viewpoints(frames, SETTINGS, SETTINGS.analysis_fps)) == 1, (
        "a body at the end of a recording is not a second session"
    )


def test_the_report_names_the_sessions_when_the_camera_moved():
    def shifted(darts, seed):
        return np.roll(unattended_board(darts, seed=seed), 55, axis=1)

    frames = (
        [unattended_board(1, seed=i) for i in range(10)]
        + [unattended_board(2, seed=50 + i) for i in range(10)]
        + [shifted(0, seed=500 + i) for i in range(10)]
        + [shifted(1, seed=600 + i) for i in range(10)]
    )
    measured = scan(frames, SETTINGS)
    text = render(measured, sweep(frames, SETTINGS), SETTINGS,
                  viewpoints(frames, SETTINGS, SETTINGS.analysis_fps))

    assert "2 viewpoints" in text and "2 session(s)" in text
    assert "the camera moved 1 time(s)" in text


def test_a_split_records_how_far_past_the_line_it_was():
    """Twenty viewpoints is either a phone picked up twenty times or a
    threshold sitting too close to darts being pulled, and a count cannot tell
    them apart. A margin can: a real reposition clears the line many times."""
    def shifted(darts, seed):
        return np.roll(unattended_board(darts, seed=seed), 55, axis=1)

    frames = (
        [unattended_board(1, seed=i) for i in range(10)]
        + [unattended_board(2, seed=50 + i) for i in range(10)]
        + [shifted(0, seed=500 + i) for i in range(10)]
        + [shifted(1, seed=600 + i) for i in range(10)]
    )
    found = viewpoints(frames, SETTINGS, SETTINGS.analysis_fps)

    assert found[0].split_margin == 0.0, "nothing opened the first viewpoint"
    assert found[1].split_margin > 1.0, "a split happens only past the line"
    assert found[1].split_margin > 2.0, (
        "a whole board displaced sideways is not a borderline call — it clears "
        "the line by more than the margin the report calls borderline"
    )


def test_the_report_flags_borderline_splits():
    from dartvision.capture.diagnose import Viewpoint

    views = [
        Viewpoint(0.0, 10.0, 8, 0.0),
        Viewpoint(12.0, 20.0, 6, 1.2),        # only just over
        Viewpoint(22.0, 30.0, 6, 14.0),       # unmistakable
    ]
    text = render(scan(unattended_visit(), SETTINGS), [], SETTINGS, views)

    assert "1 of these splits sit under 2x the line" in text
    assert "obstruction-fraction" in text


def test_nothing_is_flagged_when_every_split_is_decisive():
    from dartvision.capture.diagnose import Viewpoint

    views = [Viewpoint(0.0, 10.0, 8, 0.0), Viewpoint(12.0, 20.0, 6, 9.0)]
    text = render(scan(unattended_visit(), SETTINGS), [], SETTINGS, views)

    assert "under 2x the line" not in text


def test_the_report_shows_how_dart_sized_changes_compare_to_the_line():
    """The measurement that decides whether a split is real. A dart landing
    should sit far below the line and a camera move far above it; a recording
    whose largest jump is barely over contains no camera moves at all."""
    from dartvision.capture.frames import choose_frames, state_jumps

    frames = unattended_visit()
    kept, _, _, _ = choose_frames(frames, SETTINGS)
    jumps = state_jumps(frames, kept, SETTINGS)
    text = render(scan(frames, SETTINGS), [], SETTINGS, [], jumps)

    assert "change between one board state and the next" in text
    assert "largest" in text


def test_a_dart_sized_jump_is_far_below_the_line():
    from dartvision.capture.frames import choose_frames, state_jumps

    frames = unattended_visit()
    kept, _, _, _ = choose_frames(frames, SETTINGS)
    jumps = state_jumps(frames, kept, SETTINGS)
    budget = SETTINGS.obstruction_fraction * frames[0].size

    assert len(jumps) == len(kept) - 1
    assert jumps.max() < budget / 2, (
        "darts landing must not approach the line that splits a session"
    )


def test_the_report_works_without_any_jumps():
    """A recording yielding one state has nothing to compare."""
    text = render(scan(unattended_visit(), SETTINGS), [], SETTINGS, [], None)
    assert "change between one board state and the next" not in text
