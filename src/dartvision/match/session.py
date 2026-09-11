"""The match loop: what the screen says, and what a tap does.

#10 asks for a one-screen flow from a calibrated board through three darts,
confirmation, correction and the next player. Two decisions carry most of it,
and both are departures from the obvious design.

**Confirmations wait for the retrieval pause.**
#5 budgets about two confirmation taps a leg and 300 ms from impact to score.
The obvious reading is a prompt the moment an uncertain dart lands. That cannot
work: three darts take around ten seconds, and the player is already mid-throw
when the question would appear. Interrupting them is worse than guessing.

But the pause already exists. Walking to the board and pulling three darts is
ten to fifteen seconds in which the player's hands are free and nothing is
being scored. So uncertain darts are scored *provisionally* as they land, and
the queued questions are asked at the end of the visit. Two taps a leg then
cost nothing at all, because they are spent in time the game was wasting
anyway. That is what makes #5's budget affordable rather than merely small.

**Correcting is not a state.**
#10's suggested state list includes "correcting", and a modal correction screen
is exactly the interruption the ticket asks to design out. Correction is an
*action*, available on any throw of the current or previous visit, at any time.
It emits a new event whose ``corrects`` points at the one replaced; history is
never mutated, so the engine can still replay the leg deterministically.

Everything else here is bookkeeping in service of those two.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from dartvision.events import (
    CalibrationStatus,
    ConfirmationPolicy,
    Detection,
    Source,
    ThrowEvent,
)
from dartvision.geometry.board import BDO_BOARD, BoardSpec

__all__ = ["MatchState", "Prompt", "MatchSession", "DARTS_PER_VISIT"]

DARTS_PER_VISIT = 3


class MatchState(str, Enum):
    """The one thing the screen is doing right now."""

    NOT_CALIBRATED = "not_calibrated"
    READY = "ready"
    SCORING = "scoring"
    CONFIRMING = "confirming"
    AWAITING_REMOVAL = "awaiting_removal"


@dataclass(frozen=True)
class Prompt:
    """One screen state, one headline, and the taps available from it."""

    state: MatchState
    headline: str
    detail: str = ""
    actions: tuple[str, ...] = ()
    throw_id: str | None = None      # the throw a confirm/correct action acts on

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "headline": self.headline,
            "detail": self.detail,
            "actions": list(self.actions),
            "throw_id": self.throw_id,
        }


@dataclass
class MatchSession:
    """Turns detections into the state of one screen.

    Deliberately knows nothing about rules. It never decides a bust, a
    checkout or whose turn it is -- that is the game engine's, and the
    separation is what lets scoring be tested without a rulebook (`AGENTS.md`).
    """

    session_id: str
    policy: ConfirmationPolicy = field(default_factory=ConfirmationPolicy)
    board: BoardSpec = BDO_BOARD
    new_id: Callable[[], str] = field(default=lambda: uuid.uuid4().hex[:12])

    calibration: CalibrationStatus | None = None
    visit: list[ThrowEvent] = field(default_factory=list)
    history: list[ThrowEvent] = field(default_factory=list)
    _unconfirmed: list[str] = field(default_factory=list)
    _darts_in_board: bool = False
    _visit_closed: bool = False

    # ------------------------------------------------------------- inputs
    def set_calibration(self, status: CalibrationStatus) -> Prompt:
        """Accept a calibration update, including losing it mid-visit.

        Losing calibration does **not** discard the visit. Darts already
        scored were scored under a calibration that was valid at the time;
        throwing them away would be a worse failure than the one being
        reported.
        """
        self.calibration = status
        return self.prompt

    def record_throw(
        self, x_mm: float, y_mm: float, confidence: float,
        sequence: int | None = None, **kwargs: object,
    ) -> ThrowEvent:
        """A detected dart. Scored deterministically from its board position."""
        if self.calibration is None or not self.calibration.scoring_possible:
            raise RuntimeError(
                "no valid calibration; a throw cannot be scored and must not be "
                "invented. The screen should be in NOT_CALIBRATED."
            )
        if len(self.visit) >= DARTS_PER_VISIT:
            raise RuntimeError(
                f"a visit holds at most {DARTS_PER_VISIT} darts; the previous one "
                "was never closed"
            )

        event = ThrowEvent.from_board_point(
            id=self.new_id(), session_id=self.session_id,
            calibration_id=self.calibration.calibration_id,
            x_mm=x_mm, y_mm=y_mm, confidence=confidence,
            board=self.board, sequence=sequence or len(self.visit) + 1,
            **kwargs,
        )
        self._accept(event)
        self._darts_in_board = True
        if len(self.visit) == DARTS_PER_VISIT:
            self._visit_closed = True
        return event

    def record_miss(self, **kwargs: object) -> ThrowEvent:
        """A dart that left the board and was never detected (#3).

        The vision pipeline cannot see this one at all -- there is no tip to
        find -- so it is entered by hand. It is placed at a board coordinate
        that genuinely is off the board, rather than at the origin, so that
        anything re-deriving the score from the coordinate still gets MISS
        instead of a double bull.
        """
        if self.calibration is None:
            raise RuntimeError("a session needs a calibration before any throw")
        event = ThrowEvent.from_board_point(
            id=self.new_id(), session_id=self.session_id,
            calibration_id=self.calibration.calibration_id,
            x_mm=0.0, y_mm=self.board.r_board + 10.0, confidence=1.0,
            source=Source.MANUAL, board=self.board,
            sequence=len(self.visit) + 1, **kwargs,
        )
        self.visit.append(event)
        self.history.append(event)
        if len(self.visit) == DARTS_PER_VISIT:
            self._visit_closed = True
        return event

    def board_cleared(self) -> Prompt:
        """The darts came out. Ends the visit however many were thrown.

        A checkout is two darts, and the player walks up without a third.
        Queued questions survive this: they are about the score, not about
        what is still stuck in the board.
        """
        self._darts_in_board = False
        self._visit_closed = True
        if not self._unconfirmed:
            self._start_visit()
        return self.prompt

    # -------------------------------------------------------------- taps
    def confirm(self, throw_id: str) -> Prompt:
        """The player agreed with what was shown."""
        self._resolve(throw_id)
        return self.prompt

    def correct(self, throw_id: str, x_mm: float, y_mm: float) -> ThrowEvent:
        """Replace a throw's score, without rewriting what was recorded.

        The correction is a new event pointing at the one it replaces. The
        original stays in history, so a leg replays to the same numbers however
        many times it is recomputed -- and so a correction can itself be
        corrected.
        """
        original = self.find(throw_id)
        if original is None:
            raise KeyError(f"no throw {throw_id!r} in this session")

        replacement = ThrowEvent.from_board_point(
            id=self.new_id(), session_id=self.session_id,
            calibration_id=original.calibration_id,
            x_mm=x_mm, y_mm=y_mm, confidence=1.0,
            source=Source.CORRECTION, board=self.board,
            corrects=original.id, sequence=original.sequence,
            player_id=original.player_id, turn_id=original.turn_id,
        )
        self.history.append(replacement)
        self.visit = [replacement if t.id == original.id else t for t in self.visit]
        self._resolve(throw_id)
        return replacement

    # ------------------------------------------------------------- state
    @property
    def pending(self) -> tuple[ThrowEvent, ...]:
        """Throws queued for a question, in the order they were thrown."""
        return tuple(t for t in self.visit if t.id in self._unconfirmed)

    @property
    def visit_score(self) -> int:
        return sum(t.detected.score for t in self.visit)

    @property
    def state(self) -> MatchState:
        """Derived, never stored.

        Storing this was the first design and it drifted immediately: every
        input had to remember to update it, and the combinations that matter
        (calibration lost while a question is queued) are exactly the ones a
        hand-maintained field gets wrong.
        """
        if self.calibration is None or not self.calibration.scoring_possible:
            return MatchState.NOT_CALIBRATED
        if self._visit_closed and self._unconfirmed:
            return MatchState.CONFIRMING
        if self._visit_closed and self._darts_in_board:
            return MatchState.AWAITING_REMOVAL
        if self.visit:
            return MatchState.SCORING
        return MatchState.READY

    @property
    def prompt(self) -> Prompt:
        """The single thing the screen should say, and what can be tapped."""
        state = self.state

        if state is MatchState.NOT_CALIBRATED:
            # #13's cliff. Never show this as a frozen or blank scoreboard --
            # the player cannot tell whether to keep throwing.
            return Prompt(
                state, "Board not visible",
                detail=(
                    "Scoring is paused. Darts already thrown this visit are safe."
                    if self.visit else
                    "Point the phone at the board to start scoring."
                ),
                actions=("recalibrate", "enter_manually"),
            )

        if state is MatchState.CONFIRMING:
            throw = self.pending[0]
            detected = throw.detected
            return Prompt(
                state, f"Was that {detected.notation}?",
                detail=(
                    f"{detected.margin_mm:.1f} mm from the {detected.boundary}"
                    if detected.boundary else "Close to a boundary."
                ),
                actions=("confirm", "correct"),
                throw_id=throw.id,
            )

        if state is MatchState.AWAITING_REMOVAL:
            return Prompt(
                state, f"{self.visit_score}",
                detail="Collect your darts.",
                actions=("correct", "undo_visit"),
            )

        if state is MatchState.SCORING:
            last = self.visit[-1]
            return Prompt(
                state, last.detected.notation,
                detail=f"{self.visit_score} so far · dart {len(self.visit)} of 3",
                actions=("correct", "record_miss"),
                throw_id=last.id,
            )

        return Prompt(
            MatchState.READY, "Ready",
            detail="Throw when you are.",
            actions=("record_miss",),
        )

    def find(self, throw_id: str) -> ThrowEvent | None:
        return next((t for t in self.history if t.id == throw_id), None)

    def effective_history(self) -> list[ThrowEvent]:
        """History with corrections applied, for the game engine to score.

        Superseded events stay in ``history``; this is the view that survives
        them. A correction of a correction resolves through the chain.
        """
        superseded = {t.corrects for t in self.history if t.corrects}
        return [t for t in self.history if t.id not in superseded]

    # ---------------------------------------------------------- internals
    def _accept(self, event: ThrowEvent) -> None:
        self.visit.append(event)
        self.history.append(event)
        # #5: uncertain darts confirm rather than auto-score -- but the asking
        # waits for the pause. Scoring provisionally is what keeps the 300 ms
        # impact-to-score budget while still routing the dart for a question.
        if self.policy.should_confirm(event):
            self._unconfirmed.append(event.id)

    def _resolve(self, throw_id: str) -> None:
        if throw_id in self._unconfirmed:
            self._unconfirmed.remove(throw_id)
        if self._visit_closed and not self._unconfirmed and not self._darts_in_board:
            self._start_visit()

    def _start_visit(self) -> None:
        self.visit = []
        self._visit_closed = False
