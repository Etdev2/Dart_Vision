"""Tests for the X01 rules.

The interesting cases are all busts. A bust is not "that dart did not count" --
it voids the whole visit and returns the score to what it was when the player
stepped up, which is the rule most naive scorers get wrong. Two of the three
ways to bust are also non-obvious: finishing on a single, and leaving exactly
one.

Everything here is paper darts. No camera, no confidence, no calibration --
which is the point of the boundary.
"""

from __future__ import annotations

import pytest

from dataclasses import replace

from dartvision.game import Dart, Outcome, X01Rules, play, replay

T20, T19, T17 = Dart(20, 3), Dart(19, 3), Dart(17, 3)
S20, S1 = Dart(20, 1), Dart(1, 1)
D12, D20, D1 = Dart(12, 2), Dart(20, 2), Dart(1, 2)
BULL, OUTER_BULL, MISS = Dart(25, 2), Dart(25, 1), Dart(0, 0)

TON_FORTY = [T20, T20, S20]


def outcomes(state):
    return [r.outcome for r in state.results]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_a_visit_subtracts_what_was_thrown():
    state = replay(TON_FORTY)
    assert state.remaining == 501 - 140
    assert state.visit_score == 140
    assert outcomes(state) == [Outcome.SCORED] * 3


def test_a_nine_dart_finish():
    """180, 180, then 141 as T20 T19 D12."""
    state = replay([T20] * 3 + [T20] * 3 + [T20, T19, D12])
    assert state.winner
    assert state.remaining == 0
    assert state.darts_thrown == 9
    assert state.three_dart_average == pytest.approx(167.0)


def test_the_double_bull_finishes_a_leg():
    """50 is a double, which is why it checks out where 25 does not."""
    assert continue_from(leg_on(50), [BULL]).winner
    assert not continue_from(leg_on(50), [OUTER_BULL]).winner


def test_a_miss_scores_nothing_but_uses_a_dart():
    state = replay([T20, MISS, S20])
    assert state.remaining == 501 - 80
    assert state.darts_thrown == 3


def test_notation_matches_the_scorers_vocabulary():
    assert [d.notation for d in (T20, D20, S20, BULL, OUTER_BULL, MISS)] == [
        "T20", "D20", "S20", "DB", "SB", "MISS",
    ]


# --------------------------------------------------------------------------
# The three ways to bust
# --------------------------------------------------------------------------

def leg_on(remaining: int, rules: X01Rules | None = None):
    """A leg standing at `remaining`, at the start of a visit.

    Built by starting the leg there rather than by throwing down to it: the
    rules are identical, and it puts the visit boundary exactly where a bust
    test needs it without a fixture that has to solve for a dart sequence.
    """
    base = rules or X01Rules()
    return replay([], replace(base, start=remaining))


def continue_from(state, darts):
    """Fold more darts onto a leg, respecting visit boundaries."""
    return replay([r.dart for r in state.results] + list(darts), state.rules)


def test_going_below_zero_busts():
    state = continue_from(leg_on(40), [T20])
    assert state.results[-1].outcome is Outcome.BUST
    assert "below zero" in state.results[-1].reason


def test_leaving_exactly_one_busts():
    """No double is worth 1, so the leg would be unfinishable."""
    state = continue_from(leg_on(40), [Dart(13, 3)])   # 40 - 39
    assert state.results[-1].outcome is Outcome.BUST
    assert "left 1" in state.results[-1].reason


def test_finishing_on_a_single_busts_under_double_out():
    state = continue_from(leg_on(40), [Dart(20, 1), Dart(20, 1)])
    assert state.results[-1].outcome is Outcome.BUST
    assert "on a single" in state.results[-1].reason
    assert not state.winner


def test_a_bust_returns_the_score_to_the_start_of_the_visit():
    """Not to before the offending dart. This is the rule naive scorers miss."""
    state = leg_on(100)
    after = continue_from(state, [T20, Dart(19, 1), T20])   # 100 -> 40 -> 21 -> bust

    assert after.results[-1].outcome is Outcome.BUST
    assert after.remaining == 100
    assert after.visit_score == 0


def test_a_bust_ends_the_visit_even_with_darts_left():
    """The player does not throw on after busting with their first dart.

    Asserted through `play`, which takes visits: a flat list of darts cannot
    express "these two were never thrown", and inventing them is exactly the
    ambiguity the visit-shaped input exists to remove.
    """
    game = play([[T20, T20, T20]], rules=X01Rules(start=40), players=("A",))
    leg = game.leg_for("A")

    assert leg.darts_thrown == 1
    assert [r.outcome for r in leg.results] == [Outcome.BUST]
    assert game.remaining["A"] == 40


def test_a_legitimate_double_finish_is_not_a_bust():
    state = continue_from(leg_on(40), [D20])
    assert state.winner
    assert state.remaining == 0


def test_double_out_can_be_turned_off():
    rules = X01Rules(double_out=False)
    state = continue_from(leg_on(40, rules), [Dart(20, 1), Dart(20, 1)])
    assert state.winner
    assert state.remaining == 0


def test_one_is_only_a_bust_when_a_double_is_required():
    rules = X01Rules(double_out=False)
    state = continue_from(leg_on(40, rules), [Dart(13, 3)])
    assert state.results[-1].outcome is Outcome.SCORED
    assert state.remaining == 1


# --------------------------------------------------------------------------
# Double in
# --------------------------------------------------------------------------

def test_nothing_counts_until_a_double_lands():
    rules = X01Rules(double_in=True)
    state = replay([T20, S20, T19, D20], rules)

    assert [r.outcome for r in state.results] == [
        Outcome.IGNORED, Outcome.IGNORED, Outcome.IGNORED, Outcome.SCORED,
    ]
    assert state.remaining == 501 - 40
    assert state.darts_thrown == 4


def test_after_opening_everything_counts():
    rules = X01Rules(double_in=True)
    state = replay([D20, T20, T20], rules)
    assert state.remaining == 501 - 40 - 120


# --------------------------------------------------------------------------
# After the leg
# --------------------------------------------------------------------------

def test_darts_after_the_win_are_ignored_not_scored():
    state = replay([T20] * 3 + [T20] * 3 + [T20, T19, D12] + [T20, T20])
    assert state.remaining == 0
    assert outcomes(state)[-2:] == [Outcome.IGNORED, Outcome.IGNORED]
    assert "already won" in state.results[-1].reason


# --------------------------------------------------------------------------
# A correction is a refold
# --------------------------------------------------------------------------

def test_correcting_a_dart_replays_the_leg():
    """The whole reason the engine is a fold. #10's corrections never mutate
    history, so applying one means folding the corrected sequence again --
    and a bust in the middle unwinds correctly for free."""
    thrown = [T20, T20, T20, T20, T20, T20, T20, T19, D12]
    assert replay(thrown).winner

    # The scorer actually hit S20, not T20, with the third dart.
    corrected = list(thrown)
    corrected[2] = S20
    state = replay(corrected)

    assert not state.winner
    assert state.remaining == 501 - 120 - 20 - 180 - 60 - 57 - 24


def test_a_correction_that_turns_a_score_into_a_bust_unwinds_the_visit():
    """The hardest thing for a mutable scoreboard to undo, and free here."""
    start = leg_on(100)
    scored = continue_from(start, [T20, Dart(19, 1), Dart(5, 2)])   # 100 -> 40 -> 21 -> 11
    assert scored.remaining == 11 and scored.results[-1].outcome is Outcome.SCORED

    # That last dart was really a treble, which takes it below zero.
    state = continue_from(start, [T20, Dart(19, 1), T20])
    assert state.results[-1].outcome is Outcome.BUST
    assert state.remaining == 100


# --------------------------------------------------------------------------
# Matches
# --------------------------------------------------------------------------

def nine_darter():
    return [[T20] * 3, [T20] * 3, [T20, T19, D12]]


def filler():
    return [S1] * 3


def leg_where_a_wins(a_throws_first: bool):
    """A nine-darter for A, interleaved with B's fillers, ending on the win.

    Two things this has to get right, both of which caught the fixture out
    before they caught the engine. Who throws first alternates between legs, so
    a list that ignores that hands A's visits to B in leg two. And the last
    visit must be the winning one -- a trailing filler is thrown into the
    *next* leg, which shifts everything after it.
    """
    visits = []
    for scoring_visit in nine_darter():
        if not a_throws_first:
            visits.append(filler())
        visits.append(scoring_visit)
        if a_throws_first:
            visits.append(filler())
    if a_throws_first:
        visits.pop()          # the win ends the leg; B never answers it
    return visits


def test_visits_alternate_between_players():
    game = play([[T20] * 3, filler(), [T20] * 3], players=("A", "B"))
    assert game.remaining == {"A": 501 - 360, "B": 501 - 3}
    assert game.current_player == "B"


def test_winning_a_leg_resets_the_scores_and_counts_the_leg():
    game = play(leg_where_a_wins(True), players=("A", "B"))

    assert game.won_legs == {"A": 1, "B": 0}
    assert game.remaining == {"A": 501, "B": 501}


def test_the_throw_alternates_between_legs():
    """Otherwise one player has the advantage for the whole match."""
    game = play(leg_where_a_wins(True), players=("A", "B"), legs_to_win=2)

    assert game.won_legs["A"] == 1
    assert game.current_player == "B", "the loser of the last leg throws first"


def test_a_match_ends_when_someone_reaches_the_leg_target():
    game = play(
        leg_where_a_wins(True) + leg_where_a_wins(False),
        players=("A", "B"), legs_to_win=2,
    )
    assert game.winner == "A"
    assert game.won_legs["A"] == 2


def test_visits_after_the_match_is_won_are_not_played():
    won = leg_where_a_wins(True) + leg_where_a_wins(False)
    game = play([*won, [T20] * 3], players=("A", "B"), legs_to_win=2)

    assert game.winner == "A"
    assert game.remaining["A"] == 501


def test_a_short_visit_still_passes_the_turn():
    """A checkout takes two darts; so does giving up on a bad position."""
    game = play([[T20, T20], filler()], players=("A", "B"))
    assert game.remaining["A"] == 501 - 120
    assert game.current_player == "A"


def test_a_solo_game_is_allowed():
    game = play([[T20] * 3], players=("Practice",))
    assert game.current_player == "Practice"
    assert game.remaining["Practice"] == 501 - 180


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"segment": 25, "multiplier": 3},    # no treble bull
    {"segment": 21, "multiplier": 1},
    {"segment": 20, "multiplier": 4},
    {"segment": 5, "multiplier": 0},     # a miss has segment 0
])
def test_impossible_darts_are_refused(kwargs):
    with pytest.raises(ValueError):
        Dart(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"players": ()}, {"players": ("A", "A")}, {"legs_to_win": 0},
])
def test_impossible_games_are_refused(kwargs):
    with pytest.raises(ValueError):
        play([], **kwargs)


def test_the_engine_does_not_depend_on_vision():
    """The mirror of the test on the event side: vision never decides rules,
    and rules never depend on vision (`AGENTS.md`).

    Checked structurally rather than by grepping the text -- the prose
    legitimately explains why a `Dart` is not a `ThrowEvent`, and a substring
    search cannot tell that apart from an actual dependency.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/dartvision/game/x01.py").read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not [m for m in imported if m.startswith("dartvision")], imported

    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            identifiers.add(node.arg)

    forbidden = ("confidence", "calibration", "homography", "landmark", "pixel", "image")
    leaked = sorted(
        name for name in identifiers
        if any(word in name.lower() for word in forbidden)
    )
    assert not leaked, f"the rules reference vision concepts: {leaked}"
