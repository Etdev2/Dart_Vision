"""Finding where a dart landed by looking at what changed.

The insight is the user's, and it is a good one: between two stills of a fixed
camera, the *only* thing that moved is the dart that just arrived. Everything
else -- board, wall, lighting, the darts already in -- is identical pixel for
pixel. So the new dart is not something to detect in a picture; it is the
difference between two pictures, which is a far easier question and needs no
model at all.

That only works because of how this corpus is made. A handheld camera, or one
still-frame per session, and the difference is the whole scene. A phone that
did not move for fourteen minutes makes the difference a dart and nothing else.

What differencing cannot do on its own is say which *end* of the dart is the
tip. The change is the whole visible dart -- point, barrel, shaft, flight --
and the label wants the end that meets the board.

Geometry looks like the answer and is not. The tip lies on the board plane and
the flight stands off it toward the camera, so a homography -- which maps image
points as though they lay on the plane -- throws the flight outward and leaves
the tip where it is, making the tip the end nearer the bull. That was the first
rule here, and a fixture with the dart *drawn* rather than estimated rejected
it: at a typical mount, standing 90 mm off the plane displaces a flight about
5 mm, while the dart's own lean moves it nearly 40. The lean is seven times the
signal and points wherever the dart happens to point, so "nearer the bull"
tests nothing.

The pixels do answer it, through the difference the geometry overlooked. A
flight is some 35 mm across and a point is about one: the two ends of a dart
differ in *width* by more than an order of magnitude, whichever way it leans
and wherever it landed. So the tip is the thinner end, measured as how much of
the changed region lies near each end. A dart pointing almost straight at the
camera is a blob with no thin end, and that is reported rather than guessed.

Every suggestion is a suggestion. The output is a manifest to open in the
annotator and correct, not a label to trust: correcting a point is several
times quicker than placing one, and that is the whole saving being claimed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "Suggestion",
    "SuggestSettings",
    "changed_mask",
    "largest_component",
    "endpoints",
    "end_widths",
    "choose_tip",
    "suggest_pair",
]


@dataclass(frozen=True)
class SuggestSettings:
    """Thresholds, each with a reason.

    ``level`` is how far a pixel must move to count as changed, in grey levels.
    The same 25 the extractor uses, and for the same reason: sensor and
    compression noise rarely move a pixel that far, while a dart against a
    board is a large change over a small area.

    ``min_area`` and ``max_area`` are the share of the frame a single dart can
    plausibly cover. Below the floor is noise or a shadow; above the ceiling is
    not one dart -- it is the board being cleared, a light changing, or someone
    walking through, and a suggestion made from it would be confidently wrong.
    """

    level: float = 25.0
    min_area: float = 0.00005
    max_area: float = 0.01
    analysis_width: int = 640

    def __post_init__(self) -> None:
        if not 0 < self.level <= 255:
            raise ValueError("level must be in (0, 255]")
        if not 0 < self.min_area < self.max_area <= 1:
            raise ValueError("min_area must be below max_area, both in (0, 1]")
        if self.analysis_width < 64:
            raise ValueError("analysis_width must be at least 64")


@dataclass(frozen=True)
class Suggestion:
    """One proposed tip, with what it rests on.

    ``confident`` is the honest part. A dart whose ends are equally wide is one
    pointing at the camera, where the rule that picks the tip has nothing to
    work with -- and a reader correcting a hundred points should be told which
    few deserve a second look rather than left to find them.
    """

    point: tuple[float, float] | None
    area: int
    width_ratio: float
    reason: str = ""

    @property
    def confident(self) -> bool:
        return self.point is not None and self.width_ratio >= 1.6


def changed_mask(before: np.ndarray, after: np.ndarray, level: float) -> np.ndarray:
    """Pixels that moved by more than ``level`` grey levels."""
    difference = np.abs(
        np.asarray(after, dtype=np.int16) - np.asarray(before, dtype=np.int16)
    )
    return difference > level


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Coordinates of the biggest connected run of true pixels, ``(row, col)``.

    Flooded rather than labelled with a library, for the same reason the framing
    check floods: one call does not justify putting scipy in the install path of
    a capture tool.
    """
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []

    for start in np.argwhere(mask):
        y0, x0 = int(start[0]), int(start[1])
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        found: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            found.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width:
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(found) > len(best):
            best = found
    return np.array(best, dtype=np.int64) if best else np.zeros((0, 2), dtype=np.int64)


def endpoints(points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two ends of an elongated cloud, as ``(x, y)`` image coordinates.

    Taken along the cloud's own principal axis rather than along the image
    axes: a dart lies at whatever angle it lies at, and the extremes in x and y
    are corners of its bounding box rather than ends of the dart.
    """
    ys, xs = points[:, 0].astype(float), points[:, 1].astype(float)
    cloud = np.stack([xs, ys], axis=1)
    centred = cloud - cloud.mean(axis=0)
    # The principal axis is the first right singular vector; for a streak of
    # pixels that is the direction the streak runs in.
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = centred @ axis
    low, high = cloud[int(np.argmin(along))], cloud[int(np.argmax(along))]
    return (float(low[0]), float(low[1])), (float(high[0]), float(high[1]))


def end_widths(
    points: np.ndarray,
    ends: tuple[tuple[float, float], tuple[float, float]],
    share: float = 0.3,
) -> tuple[int, int]:
    """How many changed pixels sit near each end of the dart.

    A proxy for width, and a blunt one on purpose: fitting a thickness profile
    to forty pixels of shaft would be precision the input does not carry. What
    it has to separate is a flight from a point, and those differ by more than
    an order of magnitude.
    """
    cloud = np.stack([points[:, 1].astype(float), points[:, 0].astype(float)], axis=1)
    span = float(np.hypot(ends[0][0] - ends[1][0], ends[0][1] - ends[1][1]))
    reach = max(span * share, 3.0)
    return tuple(
        int((np.hypot(cloud[:, 0] - end[0], cloud[:, 1] - end[1]) <= reach).sum())
        for end in ends
    )


def choose_tip(
    points: np.ndarray,
    ends: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], float]:
    """Which end is the tip, and how lopsided the evidence was.

    The thinner end, by the count of changed pixels near each. The second
    return is the ratio of the two counts: 1.0 is a dart with no thin end --
    pointing at the camera, or too small to resolve -- where the answer is a
    coin toss and should be read as one.
    """
    fat, thin = end_widths(points, ends)
    ratio = max(fat, thin) / max(min(fat, thin), 1)
    return (ends[0] if fat < thin else ends[1]), float(ratio)


def suggest_pair(
    before: np.ndarray,
    after: np.ndarray,
    settings: SuggestSettings | None = None,
) -> Suggestion:
    """Propose the tip of the dart that arrived between two frames."""
    settings = settings or SuggestSettings()
    mask = changed_mask(before, after, settings.level)
    area = int(mask.sum())
    total = mask.size

    if area < settings.min_area * total:
        return Suggestion(None, area, 0.0, "nothing changed enough to be a dart")
    if area > settings.max_area * total:
        return Suggestion(
            None, area, 0.0,
            "too much changed for one dart — a cleared board, a light, or "
            "someone in the way",
        )

    points = largest_component(mask)
    if len(points) < 8:
        return Suggestion(None, area, 0.0, "the change is scattered, not a dart")

    ends = endpoints(points)
    tip, ratio = choose_tip(points, ends)
    return Suggestion(tip, len(points), ratio)
