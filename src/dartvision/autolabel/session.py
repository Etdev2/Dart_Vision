"""Walking a session of stills and proposing a tip for each dart that arrives.

The stills are already in the order the darts landed -- that is what the
extractor produced -- so a session is a sequence of differences, each one a
dart or a board being cleared. Accumulating them gives every frame the darts
standing in it at that moment, which is the same shape the annotator's burst
expansion produces and loads back through the same door.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from dartvision.autolabel.tips import (
    Suggestion,
    SuggestSettings,
    changed_mask,
    suggest_pair,
)

__all__ = ["FrameSuggestion", "suggest_session", "compare_to_labels"]


@dataclass(frozen=True)
class FrameSuggestion:
    """What was proposed for one still, and what happened to get there."""

    path: Path
    tips: tuple[tuple[float, float], ...]
    """Normalized to the frame, 0 to 1, which is what a manifest stores."""
    event: str
    size: tuple[int, int] = (0, 0)
    note: str = ""
    confident: bool = True


def _decode(path: Path, width: int) -> tuple[np.ndarray, tuple[int, int]]:
    from dartvision.capture.framing import _decode as decode_still
    from dartvision.capture.frames import find_ffmpeg

    return decode_still(path, width, find_ffmpeg())


def suggest_session(
    stills: Sequence[Path], settings: SuggestSettings | None = None
) -> list[FrameSuggestion]:
    """Propose the darts standing in each still of a session.

    Board states arrive in order, so the darts accumulate and a cleared board
    resets them.

    Clearing is recognised by the board looking like it did when it was empty,
    not by the size of the change. Size seemed like the signal -- three darts
    leaving at once is a big difference -- and it is not: two darts leaving is
    twice one dart arriving, which no threshold separates from a dart arriving
    into a busy frame. Whereas an emptied board resembles the last empty board
    almost exactly, which is a question with an answer.

    The reference is refreshed each time the board empties, so a session that
    drifts in daylight over fourteen minutes is compared against how it looked
    a visit ago rather than how it looked at the start.
    """
    settings = settings or SuggestSettings()
    results: list[FrameSuggestion] = []
    standing: list[tuple[float, float]] = []

    previous: np.ndarray | None = None
    empty_like: np.ndarray | None = None
    for path in stills:
        frame, source = _decode(path, settings.analysis_width)
        scale = source[0] / frame.shape[1]

        if previous is None:
            results.append(FrameSuggestion(
                path, (), "first", source, "taken as the opening state"))
            previous = empty_like = frame
            continue

        if standing and empty_like is not None:
            back_to_empty = changed_mask(empty_like, frame, settings.level).sum()
            if back_to_empty < settings.min_area * frame.size:
                standing = []
                empty_like = frame
                results.append(FrameSuggestion(
                    path, (), "cleared", source,
                    "the board is back to how it looked empty"))
                previous = frame
                continue

        found: Suggestion = suggest_pair(previous, frame, settings)
        if found.point is not None:
            # Stored normalized, because that is what a manifest holds and what
            # the annotator reads back — and because the analysis ran at a
            # reduced width, so a pixel here is not a pixel there.
            standing.append((found.point[0] * scale / source[0],
                             found.point[1] * scale / source[1]))
            results.append(FrameSuggestion(
                path, tuple(standing), "dart", source,
                "" if found.confident else "ends are equally wide — check this one",
                found.confident,
            ))
        elif "one dart" in found.reason:
            standing = []
            empty_like = frame
            results.append(FrameSuggestion(path, (), "cleared", source, found.reason))
        else:
            if not standing:
                empty_like = frame
            results.append(FrameSuggestion(
                path, tuple(standing), "unchanged", source, found.reason))
        previous = frame
    return results


def compare_to_labels(
    suggestions: Sequence[FrameSuggestion],
    labelled: dict[str, Sequence[Sequence[float]]],
    image_size: tuple[int, int],
    tolerance_px: float = 30.0,
) -> list[tuple[str, str]]:
    """Where a proposal and a hand label disagree, as ``(filename, reason)``.

    Not a correction. Two independent readings of the same frame agreeing is
    weak evidence that both are right; disagreeing is strong evidence that one
    is wrong, and which one is a question for the person who threw the darts.
    A different count is the disagreement worth reading first -- a point in the
    wrong place costs one dart, a miscounted visit costs the burst it expanded
    into.
    """
    found: list[tuple[str, str]] = []
    width, height = image_size
    for frame in suggestions:
        human = labelled.get(frame.path.name)
        if human is None:
            continue
        # Both readings are normalized; the comparison happens in pixels
        # because a tolerance in pixels is a thing a person can picture.
        theirs = [(p[0] * width, p[1] * height) for p in human]
        if len(theirs) != len(frame.tips):
            found.append((frame.path.name,
                          f"labelled {len(theirs)} dart(s), the pixels show "
                          f"{len(frame.tips)}"))
            continue
        ours = [(p[0] * width, p[1] * height) for p in frame.tips]
        for index, (mine, yours) in enumerate(zip(ours, theirs), start=1):
            gap = float(np.hypot(mine[0] - yours[0], mine[1] - yours[1]))
            if gap > tolerance_px:
                found.append((frame.path.name,
                              f"dart {index} is {gap:.0f} px from where the "
                              "pixels put it"))
    return found
