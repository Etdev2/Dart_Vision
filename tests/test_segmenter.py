"""Tests for the temporal layer.

The cases that matter are the ones where a single frame is misleading: a dart
that disappears behind the next dart, a hand over the board, a bumped camera, a
detection that flickers for one frame. Each of those looks, to a naive reader
of one frame, exactly like something else.

The occlusion case is the reason this module exists at all. #2 measured that
roughly a fifth of tightly grouped darts have their tip hidden behind another
dart, and that the rate does not improve at any camera angle. If absence were
read as removal, a tight treble-20 grouping would lose a dart from the score
every time -- which is the throw the product exists to get right.
"""

from __future__ import annotations

import pytest

from dartvision.stream import (
    BOARD_CLEARED,
    BOUNCE_OUT,
    CALIBRATION_LOST,
    CALIBRATION_RESTORED,
    THROW,
    VISIT_COMPLETE,
    FrameObservation,
    SegmenterConfig,
    ThrowSegmenter,
    Tip,
)

T20 = Tip(9.0, 103.0, 0.95)          # in the treble 20 bed
T20_B = Tip(-4.0, 106.0, 0.93)       # 13 mm away -- a real grouping
T20_C = Tip(2.0, 118.0, 0.91)
BULL = Tip(0.0, 0.0, 0.99)


def frame(tips=(), at="t", settled=True, scoring=True, shift=0.0):
    return FrameObservation(
        at=at, tips=tuple(tips), settled=settled,
        scoring_possible=scoring, landmark_shift_mm=shift,
    )


def steady(segmenter, tips, count=2, at="t"):
    """Feed the same observation repeatedly, as a settled board produces."""
    events = []
    for i in range(count):
        events.extend(segmenter.observe(frame(tips, at=f"{at}{i}")))
    return events


def kinds(events):
    return [e.kind for e in events]


# --------------------------------------------------------------------------
# Committing a dart
# --------------------------------------------------------------------------

def test_a_dart_is_committed_only_after_repeated_sightings():
    """One frame is noise. The model will occasionally fire on a shadow."""
    s = ThrowSegmenter()
    assert s.observe(frame([T20])) == []
    assert kinds(s.observe(frame([T20]))) == [THROW]
    assert s.committed == (T20,)


def test_a_one_frame_flicker_never_becomes_a_throw():
    s = ThrowSegmenter()
    s.observe(frame([T20]))
    s.observe(frame([]))
    s.observe(frame([]))
    assert s.committed == ()


def test_three_darts_complete_a_visit_in_order():
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])
    events = steady(s, [T20, T20_B, T20_C])

    assert kinds(events)[-2:] == [THROW, VISIT_COMPLETE]
    assert s.committed == (T20, T20_B, T20_C)
    assert [e.sequence for e in events if e.kind == THROW] == [3]


def test_a_low_confidence_detection_is_ignored():
    s = ThrowSegmenter(SegmenterConfig(min_confidence=0.5))
    steady(s, [Tip(9.0, 103.0, 0.1)], count=4)
    assert s.committed == ()


# --------------------------------------------------------------------------
# Occlusion -- the reason the module exists
# --------------------------------------------------------------------------

def test_a_dart_hidden_by_the_next_dart_stays_scored():
    """#2: about a fifth of tightly grouped darts are occluded, at every camera
    angle. The dart was seen when it landed; forgetting it loses the treble."""
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])

    # The third dart lands and hides the first. Many frames pass.
    events = steady(s, [T20_B, T20_C], count=8)

    assert BOUNCE_OUT not in kinds(events)
    assert T20 in s.committed
    assert len(s.committed) == 3


def test_a_reappearing_dart_is_not_counted_twice():
    """Occlusion is rarely total -- the tip drifts in and out of view."""
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])
    for _ in range(3):
        s.observe(frame([T20_B]))
        s.observe(frame([T20, T20_B]))

    assert len(s.committed) == 2


def test_a_wobbling_estimate_is_the_same_dart():
    """The tip estimate moves between frames; #21 gates p95 error at 12 mm.
    Re-reporting the same dart as a new one would double every score."""
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [Tip(T20.x_mm + 9.0, T20.y_mm - 6.0, 0.9)], count=4)

    assert len(s.committed) == 1


# --------------------------------------------------------------------------
# Things that look like a dart leaving
# --------------------------------------------------------------------------

def test_darts_vanishing_together_is_retrieval_not_three_bounce_outs():
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])
    steady(s, [T20, T20_B, T20_C])

    events = steady(s, [], count=3)
    assert kinds(events) == [BOARD_CLEARED]
    assert s.committed == ()


def test_one_dart_leaving_while_others_remain_is_a_bounce_out():
    """Nothing plausible hides a dart while leaving its neighbours visible."""
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])

    events = steady(s, [T20_B], count=3)
    bounces = [e for e in events if e.kind == BOUNCE_OUT]
    assert len(bounces) == 1
    assert bounces[0].tip == T20
    assert bounces[0].sequence == 1
    assert s.committed == (T20_B,)


def test_a_single_missed_frame_is_not_a_bounce_out():
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])

    events = s.observe(frame([T20_B]))
    assert events == []
    assert len(s.committed) == 2


# --------------------------------------------------------------------------
# The scene moving
# --------------------------------------------------------------------------

def test_an_unsettled_frame_is_dropped_entirely():
    """A hand over the board must not read as three darts leaving at once.

    This is why `settled` gates absences as well as detections -- the naive
    version fires a bounce-out every time someone reaches in.
    """
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])

    events = []
    for i in range(6):
        events.extend(s.observe(frame([], at=f"m{i}", settled=False)))

    assert events == []
    assert len(s.committed) == 2


def test_detections_during_motion_do_not_commit():
    s = ThrowSegmenter()
    for _ in range(5):
        s.observe(frame([T20, BULL], settled=False))
    assert s.committed == ()


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_losing_the_landmarks_is_announced_once():
    s = ThrowSegmenter()
    steady(s, [T20])

    first = s.observe(frame([T20], scoring=False))
    again = s.observe(frame([T20], scoring=False))

    assert kinds(first) == [CALIBRATION_LOST]
    assert again == []
    assert not s.calibrated


def test_a_bumped_camera_stops_scoring_rather_than_scoring_wrongly():
    """A stale homography produces confident, wrong scores -- worse than none."""
    s = ThrowSegmenter(SegmenterConfig(max_landmark_shift_mm=3.0))
    steady(s, [T20])

    events = s.observe(frame([T20, T20_B], shift=9.0))
    assert kinds(events) == [CALIBRATION_LOST]
    assert "bumped" in events[0].detail
    # The new dart must not be committed through a homography we distrust.
    assert len(s.committed) == 1


def test_calibration_returning_is_announced_and_scoring_resumes():
    s = ThrowSegmenter()
    steady(s, [T20])
    s.observe(frame([], scoring=False))

    events = steady(s, [T20, T20_B], count=3)
    assert kinds(events)[0] == CALIBRATION_RESTORED
    assert THROW in kinds(events)


def test_a_small_shift_is_tolerated():
    """Every real mount creeps. Re-calibrating on a millimetre would thrash."""
    s = ThrowSegmenter(SegmenterConfig(max_landmark_shift_mm=3.0))
    events = steady(s, [T20], count=3, at="x")
    assert s.calibrated
    assert THROW in kinds(events)


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------

def test_a_fourth_dart_is_reported_but_not_scored():
    """Almost always the previous visit was never cleared. Scoring it silently
    would corrupt the leg; the application has to decide."""
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])
    steady(s, [T20, T20_B, T20_C])

    events = steady(s, [T20, T20_B, T20_C, BULL], count=3)
    fourth = [e for e in events if e.kind == THROW and e.sequence is None]
    assert len(fourth) == 1
    assert "fourth dart" in fourth[0].detail
    assert len(s.committed) == 3


def test_reset_starts_a_clean_visit():
    s = ThrowSegmenter()
    steady(s, [T20])
    s.reset()
    assert s.committed == ()

    events = steady(s, [T20_C])
    assert [e.sequence for e in events if e.kind == THROW] == [1]


def test_a_full_visit_replays_to_the_expected_events():
    """End to end: empty board, three darts, a hand, retrieval."""
    s = ThrowSegmenter()
    script = (
        [frame([], at="a")] * 2
        + [frame([T20], at="b")] * 2
        + [frame([], at="c", settled=False)] * 3      # the next throw in flight
        + [frame([T20, T20_B], at="d")] * 2
        + [frame([T20_B, T20_C], at="e")] * 4         # third lands, first hides
        + [frame([], at="f", settled=False)] * 2      # reaching in
        + [frame([], at="g")] * 3                     # collected
    )
    events = s.observe_all(script)

    assert kinds(events) == [
        THROW, THROW, THROW, VISIT_COMPLETE, BOARD_CLEARED
    ]
    assert [e.sequence for e in events if e.kind == THROW] == [1, 2, 3]


def test_invalid_configuration_is_refused():
    for kwargs in (
        {"match_mm": 0.0}, {"confirm_frames": 0},
        {"min_confidence": 1.5}, {"max_landmark_shift_mm": -1.0},
    ):
        with pytest.raises(ValueError):
            SegmenterConfig(**kwargs)


# --------------------------------------------------------------------------
# Known limits, pinned deliberately
# --------------------------------------------------------------------------

def test_a_bounce_out_from_behind_a_later_dart_is_undetectable():
    """Pinned because it is a decision, not an oversight.

    Once a later dart stands in front of dart 1, dart 1's absence carries no
    information: hidden and gone look identical from a single viewpoint. #2
    established that a second viewpoint is the only thing that separates them,
    and that no camera angle substitutes for one.

    The alternative -- believing the absence -- loses a genuinely scored dart
    from roughly a fifth of tight groupings, which is far more common than a
    bounce-out from behind another dart. This is the cheaper error.
    """
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])
    steady(s, [T20_B, T20_C], count=3)      # third lands, first goes out of sight

    events = steady(s, [T20_B, T20_C], count=10)

    assert BOUNCE_OUT not in kinds(events)
    assert len(s.committed) == 3


def test_two_darts_appearing_at_once_are_both_committed_in_order():
    """A frame can be dropped for motion, so the next settled frame sometimes
    carries two new darts rather than one."""
    s = ThrowSegmenter()
    events = steady(s, [T20, T20_B], count=2)

    throws = [e for e in events if e.kind == THROW]
    assert [t.sequence for t in throws] == [1, 2]
    assert len(s.committed) == 2


def test_the_board_clearing_forgets_the_occlusion_history():
    """A stuck 'explained' flag would survive into the next visit and make the
    first dart of it unremovable."""
    s = ThrowSegmenter()
    steady(s, [T20])
    steady(s, [T20, T20_B])
    steady(s, [T20_B, T20_C], count=3)      # dart 1 hidden behind dart 3
    steady(s, [], count=3)                  # collected
    assert s.committed == ()

    steady(s, [T20])
    events = steady(s, [], count=3)
    assert kinds(events) == [BOARD_CLEARED]
