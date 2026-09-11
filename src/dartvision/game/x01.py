"""X01 rules: 501, 301, double-out.

The other half of the boundary `AGENTS.md` draws. Vision says *this dart is a
treble 20*; this says whether that won the leg, busted the visit, or neither.
Nothing here knows what a camera is, and a test asserts the vision vocabulary
never appears in what it produces -- the mirror of the test on the event side.

**The engine is a fold, not a mutable scoreboard.**

`replay()` is the whole rule set for one leg: darts in, a state out, and
`play()` folds a whole match out of visits. That shape is not an
aesthetic preference, it is what makes corrections work. #10 established that a
correction never mutates history -- it appends an event superseding an earlier
one -- so applying one means folding the corrected sequence again from the
start. A mutable scoreboard would have to *undo* a dart, and undoing a bust
correctly -- restoring the pre-visit total, un-ending the turn, un-switching
the player -- is the kind of thing that is wrong for a year before anyone
notices.

Folding is also what lets #14's per-leg metric move from arithmetic to
measurement: replay a real sequence, replay it again with the model's mistakes
in place, and compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Sequence

__all__ = [
    "X01Rules",
    "Dart",
    "Outcome",
    "VisitResult",
    "LegState",
    "GameState",
    "replay",
    "play",
    "DARTS_PER_VISIT",
]

DARTS_PER_VISIT = 3


@dataclass(frozen=True)
class X01Rules:
    """Which variant is being played.

    ``double_out`` is the standard 501 finish: the winning dart must be a double
    (the double bull counts). ``double_in`` requires a double to start scoring
    at all, which is common in leagues and rare in casual play, so it is off by
    default.
    """

    start: int = 501
    double_out: bool = True
    double_in: bool = False

    def __post_init__(self) -> None:
        if self.start < 2:
            raise ValueError("start must be at least 2")
        if self.double_out and self.start % 2 and self.start < 4:
            raise ValueError("an odd start below 4 cannot be finished on a double")


@dataclass(frozen=True)
class Dart:
    """One dart, as the rules see it.

    Deliberately not a ``ThrowEvent``: the engine must be usable without a
    camera, and a rule that depended on a confidence or a calibration id would
    be a rule that could not be tested on paper.
    """

    segment: int        # 1-20, 25 for either bull, 0 for a miss
    multiplier: int     # 0 miss, 1 single, 2 double, 3 treble

    def __post_init__(self) -> None:
        if self.multiplier not in (0, 1, 2, 3):
            raise ValueError("multiplier must be 0, 1, 2 or 3")
        if self.multiplier == 0 and self.segment != 0:
            raise ValueError("a miss has segment 0")
        if self.segment == 25 and self.multiplier == 3:
            raise ValueError("there is no treble bull")
        if self.segment not in (0, 25) and not 1 <= self.segment <= 20:
            raise ValueError(f"no such segment: {self.segment}")

    @property
    def score(self) -> int:
        return self.segment * self.multiplier

    @property
    def is_double(self) -> bool:
        """The double bull counts as a double, which is why 50 finishes a leg."""
        return self.multiplier == 2

    @property
    def notation(self) -> str:
        if self.multiplier == 0:
            return "MISS"
        if self.segment == 25:
            return "DB" if self.multiplier == 2 else "SB"
        return f"{'SDT'[self.multiplier - 1]}{self.segment}"


class Outcome(str, Enum):
    """What a dart did to the leg."""

    SCORED = "scored"
    BUST = "bust"
    WON = "won"
    IGNORED = "ignored"     # before doubling in, or after the leg is over


@dataclass(frozen=True)
class VisitResult:
    """One dart's effect, and enough context to explain it."""

    dart: Dart
    outcome: Outcome
    remaining: int
    reason: str = ""

    @property
    def ends_visit(self) -> bool:
        return self.outcome in (Outcome.BUST, Outcome.WON)


@dataclass(frozen=True)
class LegState:
    """Where a leg stands after some number of darts."""

    rules: X01Rules
    remaining: int
    darts_thrown: int = 0
    visit_darts: int = 0
    visit_start_remaining: int = 0
    opened: bool = False            # has doubled in, if double_in is required
    winner: bool = False
    results: tuple[VisitResult, ...] = field(default_factory=tuple)

    @property
    def finished(self) -> bool:
        return self.winner

    @property
    def visit_score(self) -> int:
        """Points scored in the current visit. Zero after a bust, by definition."""
        return self.visit_start_remaining - self.remaining

    @property
    def three_dart_average(self) -> float:
        if not self.darts_thrown:
            return 0.0
        scored = self.rules.start - self.remaining
        return scored / self.darts_thrown * DARTS_PER_VISIT


def _start(rules: X01Rules) -> LegState:
    return LegState(
        rules=rules, remaining=rules.start,
        visit_start_remaining=rules.start, opened=not rules.double_in,
    )


def _apply(state: LegState, dart: Dart) -> LegState:
    """Fold one dart into a leg. The entire rule set lives here."""
    record = lambda outcome, remaining, reason="": replace(  # noqa: E731
        state,
        remaining=remaining,
        darts_thrown=state.darts_thrown + 1,
        visit_darts=state.visit_darts + 1,
        results=(*state.results, VisitResult(dart, outcome, remaining, reason)),
    )

    if state.winner:
        return replace(
            state,
            results=(*state.results, VisitResult(
                dart, Outcome.IGNORED, state.remaining, "the leg is already won")),
        )

    if not state.opened:
        # Double-in: nothing counts until a double lands.
        if not dart.is_double:
            return record(Outcome.IGNORED, state.remaining, "not opened yet")
        opened = record(Outcome.SCORED, state.remaining - dart.score)
        return replace(opened, opened=True)

    remaining = state.remaining - dart.score

    if remaining == 0 and (dart.is_double or not state.rules.double_out):
        return replace(record(Outcome.WON, 0), winner=True)

    # Three ways to bust, and they are one rule: the leg must remain finishable.
    # Below zero is unreachable; exactly zero without a double did not finish
    # it; and one cannot be finished at all, since no double is worth 1.
    bust: str | None = None
    if remaining < 0:
        bust = "went below zero"
    elif remaining == 0:
        bust = "finished on a single"
    elif remaining == 1 and state.rules.double_out:
        bust = "left 1, which no double can finish"

    if bust:
        # A bust voids the *whole visit*, not just the offending dart: the
        # score returns to what it was when the player stepped up.
        return replace(
            record(Outcome.BUST, state.visit_start_remaining, bust),
            visit_darts=DARTS_PER_VISIT,
        )

    return record(Outcome.SCORED, remaining)


def _end_visit(state: LegState) -> LegState:
    return replace(state, visit_darts=0, visit_start_remaining=state.remaining)


def replay(darts: Iterable[Dart], rules: X01Rules | None = None) -> LegState:
    """Fold a whole sequence of darts into a leg state.

    This is the rules. Everything else is convenience, and a correction is
    simply this function called again on the corrected sequence.
    """
    state = _start(rules or X01Rules())
    for dart in darts:
        if state.winner:
            state = _apply(state, dart)
            continue
        if state.visit_darts >= DARTS_PER_VISIT:
            state = _end_visit(state)
        state = _apply(state, dart)
    return state


@dataclass(frozen=True)
class GameState:
    """A match, folded from the visits thrown so far."""

    rules: X01Rules
    players: tuple[str, ...]
    legs_to_win: int
    legs: tuple[LegState, ...]                 # the current leg, per player
    legs_won: tuple[int, ...]
    to_throw: int                              # index into players
    leg_number: int = 1

    @property
    def current_player(self) -> str:
        return self.players[self.to_throw]

    @property
    def remaining(self) -> dict[str, int]:
        return {p: leg.remaining for p, leg in zip(self.players, self.legs)}

    @property
    def won_legs(self) -> dict[str, int]:
        return dict(zip(self.players, self.legs_won))

    @property
    def winner(self) -> str | None:
        return next(
            (p for p, won in zip(self.players, self.legs_won) if won >= self.legs_to_win),
            None,
        )

    def leg_for(self, player: str) -> LegState:
        return self.legs[self.players.index(player)]


def play(
    visits: Sequence[Sequence[Dart]],
    rules: X01Rules | None = None,
    players: Sequence[str] = ("Player 1", "Player 2"),
    legs_to_win: int = 1,
) -> GameState:
    """Fold a match from its visits.

    A **visit** is one player's darts before they collect them -- one to three,
    fewer than three when they check out or stand down. Taking visits rather
    than a flat list of darts removes the one ambiguity a flat list has: a
    player who throws two darts and walks up is indistinguishable from one
    mid-visit. The match session already knows where that boundary is, because
    the board clearing is what tells it.
    """
    rules = rules or X01Rules()
    names = tuple(players)
    if len(names) < 1:
        raise ValueError("a game needs at least one player")
    if len(set(names)) != len(names):
        raise ValueError("player names must be distinct")
    if legs_to_win < 1:
        raise ValueError("legs_to_win must be at least 1")

    legs = [_start(rules) for _ in names]
    won = [0 for _ in names]
    # Who throws first alternates leg by leg, so the advantage does not sit
    # with one player for the whole match.
    leg_starter = 0
    to_throw = 0
    leg_number = 1

    for visit in visits:
        if any(count >= legs_to_win for count in won):
            break

        index = to_throw
        state = legs[index]
        for dart in visit:
            if state.visit_darts >= DARTS_PER_VISIT or state.winner:
                break
            state = _apply(state, dart)
        legs[index] = state

        if state.winner:
            won[index] += 1
            legs = [_start(rules) for _ in names]
            leg_number += 1
            leg_starter = (leg_starter + 1) % len(names)
            to_throw = leg_starter
            continue

        legs[index] = _end_visit(state)
        to_throw = (to_throw + 1) % len(names)

    return GameState(
        rules=rules, players=names, legs_to_win=legs_to_win,
        legs=tuple(legs), legs_won=tuple(won), to_throw=to_throw,
        leg_number=leg_number,
    )
