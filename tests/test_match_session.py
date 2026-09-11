"""Tests for the match loop.

Two claims carry the design and both are tested directly: that an uncertain
dart is scored immediately and asked about later, and that a correction never
rewrites what was recorded.

The rest are the states #10 lists, plus the combinations that a hand-maintained
state field gets wrong -- losing calibration mid-visit, a checkout of two
darts, a question still queued when the darts come out.
"""

from __future__ import annotations

import itertools

import pytest

from dartvision.events import CalibrationStatus, ConfirmationPolicy, Source
from dartvision.geometry.board import BDO_BOARD
from dartvision.match import MatchSession, MatchState

# Board coordinates with known scores and known margins.
T20_SAFE = (9.0, 103.0)        # middle of the treble 20 bed
S20_SAFE = (5.0, 140.0)        # single 20, clear of every wire
BULL = (0.0, 0.0)


def on_wire():
    """A point sitting almost exactly on the outer treble wire."""
    return (0.0, BDO_BOARD.r_treble - 0.2)


def calibrated(quality_landmarks: int = 8) -> CalibrationStatus:
    return CalibrationStatus(
        calibration_id="cal-1", session_id="s1", established_at="2026-01-01T00:00:00Z",
        landmarks_found=quality_landmarks, landmarks_expected=8, board_coverage=0.6,
    )


def lost() -> CalibrationStatus:
    return CalibrationStatus(
        calibration_id="cal-1", session_id="s1", established_at="2026-01-01T00:00:00Z",
        landmarks_found=2, landmarks_expected=8, board_coverage=0.6,
    )


@pytest.fixture
def session():
    counter = itertools.count(1)
    s = MatchSession(session_id="s1", new_id=lambda: f"t{next(counter)}")
    s.set_calibration(calibrated())
    return s


def throw_three(session, points=(T20_SAFE, T20_SAFE, S20_SAFE), confidence=0.99):
    return [session.record_throw(*p, confidence=confidence) for p in points]


# --------------------------------------------------------------------------
# The confirmation design
# --------------------------------------------------------------------------

def test_an_uncertain_dart_is_scored_immediately_and_asked_about_later():
    """The core of #5's budget. Three darts take ten seconds, so a prompt at
    impact would arrive while the player is already mid-throw. The dart scores
    at once and the question waits for the pause."""
    session = MatchSession(session_id="s1", new_id=lambda c=itertools.count(1): f"t{next(c)}")
    session.set_calibration(calibrated())

    uncertain = session.record_throw(*on_wire(), confidence=0.93)

    # Scored, visible, counted -- immediately.
    assert session.visit_score == uncertain.detected.score
    assert session.state is MatchState.SCORING
    assert "Was that" not in session.prompt.headline
    # And queued.
    assert session.pending == (uncertain,)


def test_the_question_arrives_when_the_visit_ends():
    session = MatchSession(session_id="s1", new_id=lambda c=itertools.count(1): f"t{next(c)}")
    session.set_calibration(calibrated())
    session.record_throw(*on_wire(), confidence=0.93)
    session.record_throw(*T20_SAFE, confidence=0.99)
    assert session.state is MatchState.SCORING

    session.record_throw(*T20_SAFE, confidence=0.99)

    assert session.state is MatchState.CONFIRMING
    assert session.prompt.headline.startswith("Was that")
    assert session.prompt.actions == ("confirm", "correct")


def test_a_confident_dart_clear_of_the_wires_is_never_asked_about(session):
    throw_three(session)
    assert session.pending == ()
    assert session.state is MatchState.AWAITING_REMOVAL


def test_high_confidence_on_a_wire_is_still_asked(session):
    """Confidence answers 'where is the tip', not 'does that change the score'.
    A dart 0.2 mm from the treble wire is a coin flip however sure the model is."""
    session.record_throw(*on_wire(), confidence=0.97)
    assert len(session.pending) == 1


def test_the_prompt_says_why_it_is_asking(session):
    session.record_throw(*on_wire(), confidence=0.93)
    session.record_throw(*T20_SAFE, confidence=0.99)
    session.record_throw(*T20_SAFE, confidence=0.99)

    detail = session.prompt.detail
    assert "mm from the" in detail
    assert "treble" in detail


def test_confirming_every_question_returns_to_the_board(session):
    session.record_throw(*on_wire(), confidence=0.93)
    session.record_throw(*on_wire(), confidence=0.93)
    session.record_throw(*T20_SAFE, confidence=0.99)

    assert len(session.pending) == 2
    session.confirm(session.pending[0].id)
    assert session.state is MatchState.CONFIRMING       # one still queued
    session.confirm(session.pending[0].id)

    assert session.state is MatchState.AWAITING_REMOVAL
    assert session.pending == ()


# --------------------------------------------------------------------------
# Corrections never rewrite history
# --------------------------------------------------------------------------

def test_a_correction_is_a_new_event_pointing_at_the_old_one(session):
    original = session.record_throw(*on_wire(), confidence=0.93)
    replacement = session.correct(original.id, *S20_SAFE)

    assert replacement.corrects == original.id
    assert replacement.source is Source.CORRECTION
    assert session.find(original.id) is original            # still there
    assert original in session.history


def test_the_effective_history_drops_what_was_superseded(session):
    first = session.record_throw(*T20_SAFE, confidence=0.99)
    session.correct(first.id, *S20_SAFE)

    effective = session.effective_history()
    assert len(effective) == 1
    assert effective[0].detected.notation == "S20"
    assert len(session.history) == 2


def test_a_correction_can_itself_be_corrected(session):
    first = session.record_throw(*T20_SAFE, confidence=0.99)
    second = session.correct(first.id, *S20_SAFE)
    session.correct(second.id, *BULL)

    effective = session.effective_history()
    assert [t.detected.notation for t in effective] == ["DB"]
    assert len(session.history) == 3


def test_a_corrected_score_is_derived_not_asserted(session):
    """The caller supplies a position, never a score. A UI cannot claim a
    treble for a dart that is visibly in the single bed."""
    original = session.record_throw(*T20_SAFE, confidence=0.99)
    replacement = session.correct(original.id, *S20_SAFE)

    assert replacement.detected.notation == "S20"
    assert replacement.detected.score == 20


def test_correcting_answers_the_question_too(session):
    session.record_throw(*on_wire(), confidence=0.93)
    session.record_throw(*T20_SAFE, confidence=0.99)
    session.record_throw(*T20_SAFE, confidence=0.99)
    assert session.state is MatchState.CONFIRMING

    session.correct(session.pending[0].id, *S20_SAFE)
    assert session.state is MatchState.AWAITING_REMOVAL


def test_the_visit_reflects_the_correction(session):
    throws = throw_three(session)
    session.correct(throws[0].id, *BULL)

    assert [t.detected.notation for t in session.visit] == ["DB", "T20", "S20"]
    assert session.visit_score == 50 + 60 + 20


def test_correcting_an_unknown_throw_is_refused(session):
    with pytest.raises(KeyError):
        session.correct("nope", *BULL)


# --------------------------------------------------------------------------
# The states #10 asks for
# --------------------------------------------------------------------------

def test_an_uncalibrated_session_is_its_own_state_not_a_blank_score():
    """#13's cliff. A frozen scoreboard with no explanation is the worst
    failure available -- the player cannot tell whether to keep throwing."""
    session = MatchSession(session_id="s1")
    assert session.state is MatchState.NOT_CALIBRATED
    assert session.prompt.headline == "Board not visible"
    assert "Point the phone at the board" in session.prompt.detail


def test_losing_calibration_mid_visit_keeps_the_darts_already_scored(session):
    first = session.record_throw(*T20_SAFE, confidence=0.99)
    session.set_calibration(lost())

    assert session.state is MatchState.NOT_CALIBRATED
    assert "already thrown this visit are safe" in session.prompt.detail
    assert session.visit == [first]
    assert session.visit_score == 60


def test_a_throw_cannot_be_invented_without_a_calibration(session):
    session.set_calibration(lost())
    with pytest.raises(RuntimeError, match="no valid calibration"):
        session.record_throw(*T20_SAFE, confidence=0.99)


def test_calibration_returning_resumes_the_same_visit(session):
    session.record_throw(*T20_SAFE, confidence=0.99)
    session.set_calibration(lost())
    session.set_calibration(calibrated())

    assert session.state is MatchState.SCORING
    assert session.visit_score == 60


def test_the_scoring_state_shows_the_running_visit(session):
    session.record_throw(*T20_SAFE, confidence=0.99)
    session.record_throw(*T20_SAFE, confidence=0.99)

    prompt = session.prompt
    assert prompt.state is MatchState.SCORING
    assert prompt.headline == "T20"
    assert "120 so far" in prompt.detail
    assert "dart 2 of 3" in prompt.detail


def test_clearing_the_board_starts_the_next_visit(session):
    throw_three(session)
    assert session.state is MatchState.AWAITING_REMOVAL

    session.board_cleared()
    assert session.state is MatchState.READY
    assert session.visit == []
    assert len(session.history) == 3


def test_a_two_dart_checkout_ends_the_visit_when_the_darts_come_out(session):
    """Nobody throws the third dart after checking out."""
    session.record_throw(*T20_SAFE, confidence=0.99)
    session.record_throw(*BULL, confidence=0.99)
    assert session.state is MatchState.SCORING

    session.board_cleared()
    assert session.state is MatchState.READY


def test_a_queued_question_survives_the_darts_coming_out(session):
    """The question is about the score, not about what is stuck in the board."""
    session.record_throw(*on_wire(), confidence=0.93)
    session.board_cleared()

    assert session.state is MatchState.CONFIRMING
    session.confirm(session.pending[0].id)
    assert session.state is MatchState.READY


def test_a_fourth_dart_is_refused_rather_than_silently_scored(session):
    throw_three(session)
    with pytest.raises(RuntimeError, match="at most 3 darts"):
        session.record_throw(*T20_SAFE, confidence=0.99)


# --------------------------------------------------------------------------
# A dart the camera never saw
# --------------------------------------------------------------------------

def test_a_miss_off_the_board_is_entered_by_hand(session):
    """#3: a dart that leaves the board produces no tip and no event. The
    pipeline cannot see it at all, so the player enters it."""
    miss = session.record_miss()

    assert miss.source is Source.MANUAL
    assert miss.detected.notation == "MISS"
    assert miss.detected.score == 0
    assert session.visit_score == 0


def test_a_miss_is_stored_somewhere_that_re_derives_as_a_miss(session):
    """Storing it at the origin would make anything recomputing the score from
    the coordinate read a double bull."""
    from dartvision.geometry.board import score_at

    miss = session.record_miss()
    assert score_at(miss.board_x_mm, miss.board_y_mm).notation == "MISS"


def test_a_miss_counts_toward_the_three_darts(session):
    session.record_throw(*T20_SAFE, confidence=0.99)
    session.record_miss()
    session.record_throw(*T20_SAFE, confidence=0.99)

    assert session.state is MatchState.AWAITING_REMOVAL
    assert session.visit_score == 120


def test_a_miss_is_never_queued_for_confirmation(session):
    """The player entered it. Asking them to confirm their own entry is noise."""
    session.record_miss()
    assert session.pending == ()


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

def test_the_prompt_serializes_for_the_ui(session):
    session.record_throw(*T20_SAFE, confidence=0.99)
    payload = session.prompt.to_dict()

    assert payload["state"] == "scoring"
    assert set(payload) == {"state", "headline", "detail", "actions", "throw_id"}


def test_a_stricter_policy_asks_more_often(session):
    """The budget is a dial, and #5 set it at roughly two taps a leg."""
    lenient = MatchSession(session_id="s1",
                           policy=ConfirmationPolicy(min_confidence=0.5, min_margin_mm=0.5))
    strict = MatchSession(session_id="s1",
                          policy=ConfirmationPolicy(min_confidence=0.99, min_margin_mm=8.0))
    for s in (lenient, strict):
        s.set_calibration(calibrated())
        s.record_throw(*T20_SAFE, confidence=0.95)

    assert lenient.pending == ()
    assert len(strict.pending) == 1


def test_the_session_never_decides_a_rule(session):
    """Vision never decides game rules (`AGENTS.md`). No bust, no checkout, no
    turn order anywhere in what this produces."""
    throw_three(session)
    text = " ".join(
        [session.prompt.headline, session.prompt.detail, *session.prompt.actions]
    ).lower()
    for word in ("bust", "checkout", "remaining", "leg", "win"):
        assert word not in text
