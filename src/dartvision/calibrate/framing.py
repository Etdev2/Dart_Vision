"""Telling a player whether their phone is mounted well enough to score.

The naive version of this screen asks the user for an angle: *"mount the phone
at 45 to 65 degrees."* Nobody knows what their phone is at, and a protractor is
not part of the product.

It turns out not to be necessary. Everything the guidance needs can be measured
from the eight landmarks the model already detects, in the frame the player is
already pointing at the board:

* **Elevation** falls out of the homography's local anisotropy at the bull. A
  plane viewed face-on scales equally in both directions; tilted, it compresses
  along the tilt. The ratio of the Jacobian's singular values is ``1/sin(el)``,
  so ``asin`` recovers the angle -- **with no knowledge of the phone's focal
  length, sensor size or distance from the board**, none of which a browser can
  be trusted to report. Measured exact at every azimuth, and stable to about a
  degree under 3 px of landmark noise.
* **Precision** is the local scale of the same homography on the double ring,
  converted to the pixels a 10 mm scoring ring will occupy once the frame is
  resized to the model's input. That is the quantity #15's budget is written in
  and the one #2 showed decides whether #21's gates are reachable at all.
* **Framing** is the board's extent in the frame, which is what the player
  actually adjusts.

So the screen never asks for an angle. It measures one, decides whether the
framing can support a score, and says the single most useful next thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dartvision.geometry.board import BDO_BOARD, BoardSpec
from dartvision.geometry.calibration import (
    assess_landmarks,
    estimate_homography,
    visible_landmarks,
)

__all__ = ["FramingLimits", "FramingReport", "assess_framing", "READY", "MARGINAL", "UNUSABLE"]

READY, MARGINAL, UNUSABLE = "ready", "marginal", "unusable"


@dataclass(frozen=True)
class FramingLimits:
    """Where the bands come from, so they can be argued with rather than guessed.

    ``min_elevation_deg`` is #2's floor: below roughly 30 degrees, one pixel of
    tip error buys more than a millimetre on the double ring, and #21's 4 mm
    gate starts demanding sub-pixel precision the representation cannot deliver.

    ``max_elevation_deg`` is not about accuracy at all -- face-on is the best
    view in every measurement #2 made. It is where the darts are. Ninety degrees
    is the board's normal axis, and a phone there eventually takes a dart.

    ``min_ring_px`` is #15's precision budget: a 10 mm ring on fewer than about
    10 pixels leaves the tip head nothing to localize with. It is measured at
    the model's input size, not the camera's, because resizing is where the
    resolution is actually lost -- and it is measured *through the perspective
    the player actually has*, which is the part that is easy to get wrong. A
    board tilted 55 degrees out of the image plane is compressed along the
    tilt, so its rings are narrower than the face-on arithmetic suggests by
    roughly ``sin(elevation)``. At 768 input and 85% board fill that is the
    difference between a comfortable 14.5 px and a marginal 11.8.

    ``model_input_px`` therefore defaults to 768 rather than the 640 a face-on
    calculation would justify.
    """

    min_elevation_deg: float = 30.0
    ideal_elevation_deg: tuple[float, float] = (45.0, 65.0)
    max_elevation_deg: float = 70.0
    min_ring_px: float = 10.0
    ideal_ring_px: float = 12.0
    model_input_px: int = 768

    def __post_init__(self) -> None:
        low, high = self.ideal_elevation_deg
        if not 0 < self.min_elevation_deg <= low < high <= self.max_elevation_deg <= 90:
            raise ValueError("elevation bands must be ordered and within (0, 90]")
        if not 0 < self.min_ring_px <= self.ideal_ring_px:
            raise ValueError("min_ring_px must be positive and no greater than ideal")
        if self.model_input_px < 1:
            raise ValueError("model_input_px must be positive")


@dataclass(frozen=True)
class FramingReport:
    """What the setup screen shows, and what it should say."""

    verdict: str
    guidance: str
    landmarks_found: int
    elevation_deg: float | None = None
    board_fill: float | None = None
    ring_px: float | None = None          # median around the double ring
    ring_px_worst: float | None = None    # the most compressed point on it
    mm_per_px_worst: float | None = None

    @property
    def ready(self) -> bool:
        return self.verdict == READY

    def to_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "guidance": self.guidance,
            "ready": self.ready,
            "landmarks_found": self.landmarks_found,
            "elevation_deg": _round(self.elevation_deg, 1),
            "board_fill": _round(self.board_fill, 3),
            "ring_px": _round(self.ring_px, 1),
            "ring_px_worst": _round(self.ring_px_worst, 1),
            "mm_per_px_worst": _round(self.mm_per_px_worst, 3),
        }


def _round(value: float | None, places: int) -> float | None:
    return None if value is None else round(value, places)


def _jacobian(matrix: np.ndarray, x: float, y: float, step: float = 0.5) -> np.ndarray:
    """Local board-to-image scale at a board point, in pixels per millimetre."""
    jacobian = np.zeros((2, 2))
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        ahead = matrix @ np.array([x + offset[0], y + offset[1], 1.0])
        behind = matrix @ np.array([x - offset[0], y - offset[1], 1.0])
        jacobian[:, axis] = (ahead[:2] / ahead[2] - behind[:2] / behind[2]) / (2 * step)
    return jacobian


def _elevation_deg(board_to_image: np.ndarray) -> float:
    """Angle above the board plane, from the anisotropy at the bull.

    A plane seen face-on scales the same in every direction; tilted by ``el``
    out of the image plane, it compresses by ``sin(el)`` along the tilt. So the
    singular values of the local Jacobian stand in ratio ``1/sin(el)``, and the
    angle follows without any camera intrinsics at all.
    """
    largest, smallest = np.linalg.svd(_jacobian(board_to_image, 0.0, 0.0), compute_uv=False)
    if largest <= 0:
        raise ValueError("degenerate homography")
    return math.degrees(math.asin(min(1.0, float(smallest / largest))))


def _ring_scale(
    board_to_image: np.ndarray, board: BoardSpec, samples: int = 180
) -> tuple[float, float, float]:
    """``(median px per mm, worst px per mm, board extent in px)``.

    Both statistics are taken in the *most compressed* direction at each point,
    since a single squashed axis is enough to lose a score however good the
    other one looks. The median across the ring is what the guidance judges --
    #21's gates are distributional over darts, and darts land all round the
    board -- while the worst point is reported for diagnosis.
    """
    scales = []
    rim = []
    for index in range(samples):
        angle = 2.0 * math.pi * index / samples
        x, y = board.r_double * math.cos(angle), board.r_double * math.sin(angle)
        singular = np.linalg.svd(_jacobian(board_to_image, x, y), compute_uv=False)
        scales.append(float(singular[1]))

        edge = board_to_image @ np.array([
            board.r_board * math.cos(angle), board.r_board * math.sin(angle), 1.0,
        ])
        rim.append(edge[:2] / edge[2])

    array = np.asarray(rim)
    extent = float(max(np.ptp(array[:, 0]), np.ptp(array[:, 1])))
    return float(np.median(scales)), float(np.min(scales)), extent


def assess_framing(
    landmarks: Sequence[Sequence[float] | None],
    image_size: tuple[int, int],
    limits: FramingLimits | None = None,
    board: BoardSpec = BDO_BOARD,
) -> FramingReport:
    """Judge one frame's framing and say what to do about it.

    ``landmarks`` are in normalized image coordinates, in the fixed order, with
    ``None`` for any the detector did not find. ``image_size`` is ``(height,
    width)`` in pixels.
    """
    limits = limits or FramingLimits()
    found = len(visible_landmarks(landmarks))

    if found < 4:
        return FramingReport(
            verdict=UNUSABLE, landmarks_found=found,
            guidance=(
                "Point the phone at the board — not enough of it is visible yet."
                if found == 0 else
                "Part of the board is out of frame or hidden. Show the whole board."
            ),
        )

    height, width = image_size
    pixels = [
        None if point is None else (point[0] * width, point[1] * height)
        for point in landmarks
    ]
    # A detector can report four points in a line -- three landmarks on one
    # wire plus a miss. Any four points admit an exact homography, so nothing
    # downstream would complain; the numbers would simply be nonsense.
    if not assess_landmarks(pixels, board=board).convex_and_ordered:
        return FramingReport(
            verdict=UNUSABLE, landmarks_found=found,
            guidance="The board's outline does not look right. Reposition and try again.",
        )

    try:
        board_to_image = np.linalg.inv(estimate_homography(pixels, board=board).matrix)
        elevation = _elevation_deg(board_to_image)
        median_scale, worst_scale, extent = _ring_scale(board_to_image, board)
    except (ValueError, np.linalg.LinAlgError):
        return FramingReport(
            verdict=UNUSABLE, landmarks_found=found,
            guidance="The board's outline does not look right. Reposition and try again.",
        )

    fill = extent / min(image_size)
    # What the model will actually see: the frame is resized to its input, and
    # that resize is where the resolution is lost.
    resize = limits.model_input_px / min(image_size)
    ring_px = board.ring_width * median_scale * resize
    ring_px_worst = board.ring_width * worst_scale * resize

    verdict, guidance = _judge(elevation, ring_px, limits)
    return FramingReport(
        verdict=verdict, guidance=guidance, landmarks_found=found,
        elevation_deg=elevation, board_fill=fill, ring_px=ring_px,
        ring_px_worst=ring_px_worst, mm_per_px_worst=1.0 / worst_scale,
    )


def _judge(elevation: float, ring_px: float, limits: FramingLimits) -> tuple[str, str]:
    """One verdict and the single most useful next action.

    Ordered by what the player should fix first. Being in the flight path
    outranks everything, because that one costs a phone rather than a score.
    """
    if elevation > limits.max_elevation_deg:
        return UNUSABLE, (
            "The phone is nearly in line with the board — that is where the darts "
            "fly. Move it further to one side, or higher up."
        )
    if elevation < limits.min_elevation_deg:
        return UNUSABLE, (
            "The view is too side-on to score accurately. Move the phone around "
            "toward the front of the board."
        )
    low, high = limits.ideal_elevation_deg

    if ring_px < limits.min_ring_px:
        # Two different causes look identical in this number. An oblique view
        # compresses the rings by roughly sin(elevation), so a player already
        # filling the frame at 35 degrees cannot fix it by moving closer --
        # and telling them to is the most annoying possible advice.
        if elevation < low:
            return UNUSABLE, (
                "Too side-on for this distance — the scoring rings are squashed. "
                "Move the phone around toward the front of the board."
            )
        return UNUSABLE, (
            "The board is too small in the frame. Move the phone closer until the "
            "board nearly fills it."
        )

    # Same reasoning as above, one tier softer: when the angle is shallow it is
    # the cause of the short rings *and* the cheaper thing to change, so it is
    # what to mention -- squaring up fixes both.
    if elevation < low:
        return MARGINAL, (
            "Good enough to score. A little more toward the front of the board "
            "would be better."
        )
    if ring_px < limits.ideal_ring_px:
        return MARGINAL, (
            "This will work, but scores near the wires will be less certain. "
            "Move a little closer if you can."
        )
    if elevation > high:
        return MARGINAL, (
            "Good enough to score, but close to the throwing line. Further to the "
            "side would be safer for the phone."
        )
    return READY, "Ready to throw."
