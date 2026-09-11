"""Turning a stream of per-frame detections into throws.

The Brain answers one question about one frame: *where are the tips right now?*
It does not answer the question the product asks, which is *did a dart just
land, and which one is new?* This module is that layer.

Why it exists at all, and why it is not frame differencing
----------------------------------------------------------
The obvious design -- subtract consecutive frames, find the blob that appeared,
call it the dart -- fails in three ways that matter here. Camera shake,
lighting change and a hand reaching into frame each produce a difference blob
indistinguishable from a dart. A missed frame corrupts the state permanently,
because the difference is relative to a history that is now wrong. And a dart
that lands while partly hidden is simply never detected: there is no blob to
find.

Detecting the whole board every frame inverts all three. Every frame
re-establishes the full truth independently, so one bad frame costs one frame
rather than the rest of the visit, and the state cannot drift.

But the decisive reason is #2's occlusion finding. Roughly a fifth of tightly
grouped darts have their tip hidden behind another dart, and that rate is flat
across every camera angle -- it is set by the barrel's width, not the mount.
A single frame therefore *cannot* see all three darts in a tight group. What
saves the score is that the hidden dart was visible when it landed, before the
dart that now covers it arrived.

So the temporal layer's job is **memory, not detection**: a dart, once seen,
stays seen. That single rule recovers most of what occlusion takes away, and it
is the reason this module commits darts rather than re-deriving them.

Frame differencing still earns its place -- as a *gate*, never as a detector.
``FrameObservation.settled`` says the scene is not moving, and the segmenter
ignores everything else. A dart in flight or a hand over the board produces
nonsense detections, and the cheapest way to know that is pixel difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "Tip",
    "FrameObservation",
    "SegmentEvent",
    "SegmenterConfig",
    "ThrowSegmenter",
    "MAX_DARTS_PER_VISIT",
]

MAX_DARTS_PER_VISIT = 3

THROW = "throw"
BOUNCE_OUT = "bounce_out"
VISIT_COMPLETE = "visit_complete"
BOARD_CLEARED = "board_cleared"
CALIBRATION_LOST = "calibration_lost"
CALIBRATION_RESTORED = "calibration_restored"


@dataclass(frozen=True)
class Tip:
    """One detected dart tip, in board millimetres."""

    x_mm: float
    y_mm: float
    confidence: float = 1.0

    def distance_to(self, other: "Tip") -> float:
        return math.hypot(self.x_mm - other.x_mm, self.y_mm - other.y_mm)


@dataclass(frozen=True)
class FrameObservation:
    """What one frame saw, after the model and the homography have run.

    ``settled`` comes from the caller, not from this module -- a pixel-level
    motion check is cheap and belongs next to the camera. ``landmark_shift_mm``
    is how far the detected landmarks have moved from the ones the current
    calibration was built on, which is how camera shake announces itself.
    """

    at: str
    tips: tuple[Tip, ...] = ()
    scoring_possible: bool = True
    settled: bool = True
    landmark_shift_mm: float = 0.0


@dataclass(frozen=True)
class SegmentEvent:
    """Something the stream decided happened."""

    kind: str
    at: str
    tip: Tip | None = None
    sequence: int | None = None     # 1-3 within the visit
    detail: str = ""


@dataclass(frozen=True)
class SegmenterConfig:
    """Thresholds, each with a reason rather than a round number.

    ``match_mm`` is the radius within which a detected tip is taken to be an
    already-committed dart rather than a new one. It must exceed the model's
    own p95 tip error (#21 gates that at 12 mm) or a committed dart will be
    re-reported as a second dart every time the estimate wobbles. It must stay
    well under the ~20 mm at which two darts are genuinely distinguishable in a
    grouping, or a real second dart is swallowed.

    ``confirm_frames`` debounces in both directions. A dart must be seen in
    that many consecutive settled frames before it is committed, and must be
    absent for that many before its absence is believed. One frame of either is
    noise; several in a row is an event.
    """

    match_mm: float = 15.0
    confirm_frames: int = 2
    min_confidence: float = 0.25
    max_landmark_shift_mm: float = 3.0

    def __post_init__(self) -> None:
        if self.match_mm <= 0:
            raise ValueError("match_mm must be positive")
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.max_landmark_shift_mm <= 0:
            raise ValueError("max_landmark_shift_mm must be positive")


@dataclass
class _Committed:
    """A dart the segmenter has accepted as really being in the board."""

    tip: Tip
    sequence: int
    missing_frames: int = 0
    # Set when a later dart arrived while this one was out of sight. That later
    # dart is a standing explanation for the absence -- it does not stop
    # hiding this one after a couple of frames -- so the absence stops being
    # evidence of removal for as long as it lasts.
    explained_by_later_dart: bool = False


class ThrowSegmenter:
    """Consumes frame observations, emits throws.

    Stateful by necessity: the whole point is remembering what was seen. Feed
    it every frame; it decides which ones mean anything.
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()
        self._committed: list[_Committed] = []
        self._pending: list[tuple[Tip, int]] = []
        self._calibrated = True
        self._visit_complete = False

    # ----------------------------------------------------------------- state
    @property
    def committed(self) -> tuple[Tip, ...]:
        """The darts currently believed to be in the board, in throw order."""
        return tuple(c.tip for c in sorted(self._committed, key=lambda c: c.sequence))

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def reset(self) -> None:
        """Start a new visit. The board is assumed empty."""
        self._committed.clear()
        self._pending.clear()
        self._visit_complete = False

    # ------------------------------------------------------------------ main
    def observe(self, frame: FrameObservation) -> list[SegmentEvent]:
        """Fold one frame into the state and return whatever it decided."""
        events: list[SegmentEvent] = []

        # Calibration first: without it nothing in the frame can be scored, and
        # scoring through a stale homography is worse than not scoring at all.
        lost = not frame.scoring_possible or (
            frame.landmark_shift_mm > self.config.max_landmark_shift_mm
        )
        if lost:
            self._pending.clear()
            if self._calibrated:
                self._calibrated = False
                events.append(SegmentEvent(
                    CALIBRATION_LOST, frame.at,
                    detail=(
                        "landmarks not found"
                        if not frame.scoring_possible
                        else f"landmarks moved {frame.landmark_shift_mm:.1f} mm; "
                             "the camera was bumped"
                    ),
                ))
            return events

        if not self._calibrated:
            self._calibrated = True
            events.append(SegmentEvent(CALIBRATION_RESTORED, frame.at))

        # A moving scene is a dart in flight or a hand over the board. Neither
        # produces detections worth reasoning about, so the frame is dropped
        # entirely -- including its absences, which would otherwise read as a
        # board full of bounce-outs every time someone reached in.
        if not frame.settled:
            return events

        observed = [t for t in frame.tips if t.confidence >= self.config.min_confidence]
        matched, claimed = self._match(observed)

        established = len(self._committed)
        events.extend(self._handle_new(observed, claimed, frame))
        events.extend(self._handle_absent(matched, frame, established))
        return events

    def observe_all(self, frames: Iterable[FrameObservation]) -> list[SegmentEvent]:
        """Fold a whole sequence. Convenient for tests and replay."""
        events: list[SegmentEvent] = []
        for frame in frames:
            events.extend(self.observe(frame))
        return events

    # ------------------------------------------------------------- internals
    def _match(self, observed: Sequence[Tip]) -> tuple[dict[int, Tip], set[int]]:
        """Nearest-first assignment of observed tips to committed darts.

        Greedy on distance rather than in index order: with two darts close
        together, taking them in order can assign the first observation to the
        wrong dart and then strand the second.
        """
        pairs = sorted(
            (
                (c.tip.distance_to(tip), index, position)
                for index, c in enumerate(self._committed)
                for position, tip in enumerate(observed)
            ),
            key=lambda item: item[0],
        )
        matched: dict[int, Tip] = {}
        taken: set[int] = set()
        for distance, index, position in pairs:
            if distance > self.config.match_mm:
                break
            if index in matched or position in taken:
                continue
            matched[index] = observed[position]
            taken.add(position)
        return matched, taken

    def _handle_new(
        self, observed: Sequence[Tip], claimed: set[int], frame: FrameObservation
    ) -> list[SegmentEvent]:
        """Tips matching nothing committed are candidates for a new dart."""
        fresh = [t for i, t in enumerate(observed) if i not in claimed]

        # Age the candidates: a tip must appear in consecutive settled frames
        # before it is believed.
        survivors: list[tuple[Tip, int]] = []
        for tip in fresh:
            previous = next(
                (
                    (seen, count)
                    for seen, count in self._pending
                    if seen.distance_to(tip) <= self.config.match_mm
                ),
                None,
            )
            survivors.append((tip, (previous[1] if previous else 0) + 1))
        self._pending = [s for s in survivors if s[1] < self.config.confirm_frames]

        events: list[SegmentEvent] = []
        for tip, seen in sorted(survivors, key=lambda s: -s[1]):
            if seen < self.config.confirm_frames:
                continue
            if len(self._committed) >= MAX_DARTS_PER_VISIT:
                # A fourth dart is not a throw. Most likely the previous visit
                # was never cleared, and silently scoring it would corrupt the
                # leg; say so and let the application decide.
                events.append(SegmentEvent(
                    THROW, frame.at, tip=tip,
                    detail="a fourth dart appeared before the board was cleared",
                    sequence=None,
                ))
                continue
            sequence = len(self._committed) + 1
            self._committed.append(_Committed(tip=tip, sequence=sequence))
            events.append(SegmentEvent(THROW, frame.at, tip=tip, sequence=sequence))
            if sequence == MAX_DARTS_PER_VISIT and not self._visit_complete:
                self._visit_complete = True
                events.append(SegmentEvent(VISIT_COMPLETE, frame.at))
        return events

    def _handle_absent(
        self, matched: dict[int, Tip], frame: FrameObservation, established: int
    ) -> list[SegmentEvent]:
        """Committed darts nobody saw this frame.

        Absence is weak evidence, and reading it as removal is how a tight
        grouping loses a dart. Two things guard against that.

        A dart committed *in this frame* is not absent -- ``matched`` was
        computed before it existed, so without this it would be marked missing
        on its own arrival.

        And a dart that vanished when a later dart landed has a standing
        explanation. #2 measured that about a fifth of tightly grouped darts
        are hidden behind another dart, at every camera angle, so this is the
        common case rather than the exotic one. Such a dart still ages -- the
        board must be able to empty -- but its absence alone never removes it.
        """
        arrived = len(self._committed) > established
        for index, entry in enumerate(self._committed[:established]):
            if index in matched:
                entry.missing_frames = 0
                entry.explained_by_later_dart = False
                continue
            entry.missing_frames += 1
            if arrived:
                entry.explained_by_later_dart = True

        threshold = self.config.confirm_frames
        gone = [c for c in self._committed if c.missing_frames >= threshold]
        if not gone:
            return []

        if len(gone) == len(self._committed):
            # Everything went at once: the player collected their darts. That
            # ends the visit; it is not three simultaneous bounce-outs.
            self.reset()
            return [SegmentEvent(BOARD_CLEARED, frame.at)]

        # Some went while others stayed. Nothing hides a dart while leaving its
        # neighbours visible -- unless a later dart is standing in front of it,
        # which is the one case already accounted for.
        events: list[SegmentEvent] = []
        for entry in sorted(gone, key=lambda c: c.sequence):
            if entry.explained_by_later_dart:
                continue
            events.append(SegmentEvent(
                BOUNCE_OUT, frame.at, tip=entry.tip, sequence=entry.sequence,
                detail="committed dart is no longer in the board",
            ))
            self._committed.remove(entry)
        return events
