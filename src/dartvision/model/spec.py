"""What size image the model reads, and why that number and not a smaller one.

The input size is the dominant cost lever in the whole project -- compute scales
with its *area*, so halving it quarters the training bill -- which makes it the
first thing anyone reaches for when a run looks expensive. It also has a hard
floor, and the floor is not obvious from the arithmetic most people would do.

#15 set the precision budget: a 10 mm scoring ring must land on roughly 10-20
pixels for the tip head to have anything to localize with. The tempting way to
check that is `ring_width x fill x input / board_diameter`, which says 512 is
borderline-but-arguable. **That is the face-on number, and nobody mounts a phone
face-on** -- #2 established that face-on is where the darts fly. At the 45-65
degrees a real mount occupies, the board is tilted out of the image plane and
its rings compress with it, by roughly `sin(elevation)`.

That factor is the whole difference between "512 is arguable" and "512 is ruled
out", and it is why this module exists as code with a test rather than as a
number typed into three config files. Getting it wrong costs a training
campaign: the run completes, reports a number, and the number describes a model
the precision budget already excluded.

Stated as arithmetic so it can be re-derived rather than trusted, and depending
on nothing but the board's dimensions.
"""

from __future__ import annotations

import math

from dartvision.geometry.board import BDO_BOARD, BoardSpec

__all__ = [
    "DEFAULT_INPUT_PX",
    "MIN_RING_PX",
    "IDEAL_RING_PX",
    "MOUNT_BAND_DEG",
    "ring_width_px",
    "minimum_input_px",
]

#: #15's precision budget, in pixels across a 10 mm ring at the model's input.
MIN_RING_PX = 10.0
IDEAL_RING_PX = 12.0

#: The mounting band #2 settled on. The shallow end is what binds.
MOUNT_BAND_DEG = (45.0, 65.0)

#: Smallest input that clears the budget across that whole band. See
#: `test_model_spec.py`, which fails if this is lowered below what the
#: arithmetic supports.
DEFAULT_INPUT_PX = 768


def ring_width_px(
    input_px: int,
    board_fill: float = 0.85,
    elevation_deg: float = 55.0,
    board: BoardSpec = BDO_BOARD,
) -> float:
    """Pixels across a 10 mm scoring ring, as the model will see it.

    ``board_fill`` is the fraction of the frame's shorter side the board spans;
    0.85 is a well-framed mount. ``elevation_deg`` is the angle out of the board
    plane, where 90 is face-on -- and the ``sin`` of it is the part a face-on
    calculation silently omits.
    """
    if input_px < 1:
        raise ValueError("input_px must be positive")
    if not 0 < board_fill <= 1:
        raise ValueError("board_fill must be in (0, 1]")
    if not 0 < elevation_deg <= 90:
        raise ValueError("elevation_deg must be in (0, 90]")

    face_on = board.ring_width * board_fill * input_px / (2 * board.r_board)
    return face_on * math.sin(math.radians(elevation_deg))


def minimum_input_px(
    target_ring_px: float = MIN_RING_PX,
    board_fill: float = 0.85,
    elevation_deg: float | None = None,
    board: BoardSpec = BDO_BOARD,
    step: int = 64,
) -> int:
    """Smallest input, rounded up to ``step``, that reaches ``target_ring_px``.

    Defaults to the shallow end of the mount band, because that is the case
    that has to hold -- an input chosen for 65 degrees fails the player who
    mounts at 45.
    """
    elevation = MOUNT_BAND_DEG[0] if elevation_deg is None else elevation_deg
    exact = (
        target_ring_px * 2 * board.r_board
        / (board.ring_width * board_fill * math.sin(math.radians(elevation)))
    )
    return int(math.ceil(exact / step) * step)
