"""Deterministic dartboard geometry and scoring.

Clean-room implementation derived from published BDO board dimensions, not from
any third-party source. Depends only on the standard library so it can be
imported by the training harness, the synthetic generator, the metrics code and
the eventual application without pulling in a numeric stack.

Coordinate system
-----------------
Board coordinates are millimetres in the plane of the board, origin at the
centre of the bull, ``+x`` to the right and ``+y`` up. Angles are measured
counter-clockwise from ``+x`` in degrees. The 20 bed is centred on ``+y``
(straight up), which is the physical orientation of a mounted board.

This module is intentionally free of any model or image concerns. Converting a
camera image to board coordinates is the homography's job (see ``homography``);
converting board coordinates to a score is this module's job. Keeping those
separate is what makes scoring exhaustively testable without a camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = [
    "BoardSpec",
    "BDO_BOARD",
    "SECTORS_CLOCKWISE_FROM_20",
    "Hit",
    "score_at",
    "margin_to_nearest_boundary",
]

# Sector numbers in clockwise order starting at 20 (which sits at the top).
SECTORS_CLOCKWISE_FROM_20: Final[tuple[int, ...]] = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
)

_SECTOR_ARC_DEG: Final[float] = 360.0 / 20.0  # 18 degrees per bed
# The 20 bed is centred at 90 degrees, so its counter-clockwise edge is at 99.
_FIRST_BOUNDARY_DEG: Final[float] = 90.0 + _SECTOR_ARC_DEG / 2.0


@dataclass(frozen=True)
class BoardSpec:
    """Radial dimensions of a dartboard, in millimetres from the bull centre.

    Defaults are the BDO standard measurements. ``r_double`` and ``r_treble``
    are the *outer* wire edges of their rings; ``ring_width`` is the wire
    apex-to-apex width shared by both rings.
    """

    r_board: float = 225.5       # full board radius, including the number ring
    r_double: float = 170.0      # outer edge of the double ring
    r_treble: float = 107.4      # outer edge of the treble ring
    r_outer_bull: float = 15.9   # outer edge of the 25 ring
    r_inner_bull: float = 6.35   # outer edge of the bullseye
    ring_width: float = 10.0     # radial width of the double and treble rings

    @property
    def r_double_inner(self) -> float:
        return self.r_double - self.ring_width

    @property
    def r_treble_inner(self) -> float:
        return self.r_treble - self.ring_width

    @property
    def radial_boundaries(self) -> tuple[float, ...]:
        """Every radius at which the score changes, ascending."""
        return (
            self.r_inner_bull,
            self.r_outer_bull,
            self.r_treble_inner,
            self.r_treble,
            self.r_double_inner,
            self.r_double,
        )

    def __post_init__(self) -> None:
        if not (
            0 < self.r_inner_bull < self.r_outer_bull
            < self.r_treble_inner < self.r_treble
            < self.r_double_inner < self.r_double <= self.r_board
        ):
            raise ValueError("board radii must be strictly increasing and within r_board")
        if self.ring_width <= 0:
            raise ValueError("ring_width must be positive")


BDO_BOARD: Final[BoardSpec] = BoardSpec()


@dataclass(frozen=True)
class Hit:
    """The scoring interpretation of one point on the board.

    ``segment`` is the bed number, or 25 for either bull. ``multiplier`` is 0
    for a miss, 1 for a single, 2 for a double and 3 for a treble; the double
    bull is reported as segment 25 with multiplier 2, giving 50.
    """

    segment: int
    multiplier: int
    score: int
    notation: str


def _sector_index(angle_deg: float) -> int:
    """Index into :data:`SECTORS_CLOCKWISE_FROM_20` for a board angle."""
    # Beds advance clockwise, i.e. as the angle decreases.
    return int(math.floor((_FIRST_BOUNDARY_DEG - angle_deg) / _SECTOR_ARC_DEG)) % 20


def score_at(x: float, y: float, board: BoardSpec = BDO_BOARD) -> Hit:
    """Score the board-coordinate point ``(x, y)``, in millimetres.

    Boundaries belong to the higher-scoring region: a point exactly on the outer
    treble wire is a treble. The convention only matters for measure-zero cases,
    but fixing it keeps the function total and the tests exact.
    """
    r = math.hypot(x, y)

    if r > board.r_double:
        return Hit(segment=0, multiplier=0, score=0, notation="MISS")
    if r <= board.r_inner_bull:
        return Hit(segment=25, multiplier=2, score=50, notation="DB")
    if r <= board.r_outer_bull:
        return Hit(segment=25, multiplier=1, score=25, notation="SB")

    segment = SECTORS_CLOCKWISE_FROM_20[_sector_index(math.degrees(math.atan2(y, x)))]

    if r > board.r_double_inner:
        multiplier, prefix = 2, "D"
    elif board.r_treble_inner < r <= board.r_treble:
        multiplier, prefix = 3, "T"
    else:
        multiplier, prefix = 1, "S"

    return Hit(
        segment=segment,
        multiplier=multiplier,
        score=segment * multiplier,
        notation=f"{prefix}{segment}",
    )


def margin_to_nearest_boundary(
    x: float, y: float, board: BoardSpec = BDO_BOARD
) -> float:
    """Millimetres from ``(x, y)`` to the nearest boundary that changes the score.

    This is the quantity that decides whether a localization error matters at
    all: an error smaller than the margin cannot change the score, while an
    error larger than it may. Reported alongside localization error, it
    separates "the model is imprecise" from "this dart was unscoreable at any
    achievable precision".

    Sector wires are ignored inside the outer bull, where the bed number does
    not affect the score.
    """
    r = math.hypot(x, y)
    radial = min(abs(r - b) for b in board.radial_boundaries)

    if r <= board.r_outer_bull:
        return radial

    # Angular distance to the nearer of the two wires bounding this bed, as an
    # arc length at this radius.
    angle = math.degrees(math.atan2(y, x))
    offset = (_FIRST_BOUNDARY_DEG - angle) % _SECTOR_ARC_DEG
    angular_deg = min(offset, _SECTOR_ARC_DEG - offset)
    return min(radial, r * math.radians(angular_deg))
