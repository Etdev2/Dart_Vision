"""Measuring how much of the frame the board fills, before anything is labelled.

The one thing in the capture chain that cannot be fixed later. A model can be
retrained, a threshold retuned, a session re-split -- but a double ring that
landed on four pixels was never in the image, and no amount of data recovers
it. #25 found that a board occupying too little of the frame carries no
recoverable precision at any model quality, and the protocol says "fill the
frame" without saying how anyone would check.

``calibrate.framing`` already answers this exactly, from the eight annotated
landmarks. That is the right answer at the wrong time: it arrives after a
session has been annotated, and the point of the question is to ask it before.

So this estimates the board's extent with no landmarks and no model, from the
one thing that is reliably true of a photograph of a dartboard -- it is much
the darkest large object in the frame. That is an approximation, and a
deliberately crude one. It is not used to score anything. It has to separate
"the ring will land on four pixels" from "the ring will land on twelve", and
for that a bounding box around the dark region is plenty.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dartvision.model.spec import DEFAULT_INPUT_PX, MIN_RING_PX, IDEAL_RING_PX

__all__ = [
    "BOARD_OUTER_MM",
    "RING_MM",
    "Framing",
    "otsu_threshold",
    "dark_threshold",
    "largest_dark_region",
    "measure_image",
]

# The board's full outer diameter, number ring included, because that is the
# edge a dark-region bounding box finds. The scoring diameter is 340 mm.
BOARD_OUTER_MM = 451.0

# The double ring, the narrowest thing the model has to resolve.
RING_MM = 10.0

# Frames are measured at this width. The board is hundreds of pixels across in
# any usable shot, so a bounding box is not improved by decoding more.
MEASURE_WIDTH = 256


@dataclass(frozen=True)
class Framing:
    """What the double ring will be worth, in pixels, at the model's input."""

    source: tuple[int, int]
    board: tuple[int, int]
    ring_across: float
    ring_down: float
    ring_squared: float

    @property
    def fills(self) -> float:
        """Share of the frame's width the board spans."""
        return self.board[0] / self.source[0]

    @property
    def verdict(self) -> str:
        worst = min(self.ring_across, self.ring_down)
        if worst >= IDEAL_RING_PX:
            return "good"
        if worst >= MIN_RING_PX:
            return "usable"
        return "too small"


def otsu_threshold(grey: np.ndarray) -> int:
    """The grey level that best separates the image into two groups.

    Otsu's method, which needs no parameter and no assumption about how dark a
    board is or how bright a wall is -- both of which vary with every session,
    which is the whole reason a fixed threshold was not used.
    """
    counts = np.bincount(np.asarray(grey, dtype=np.uint8).ravel(), minlength=256)
    weights = np.cumsum(counts)
    totals = np.cumsum(counts * np.arange(256))
    total, sum_all = weights[-1], totals[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        background = weights
        foreground = total - weights
        mean_b = np.where(background > 0, totals / np.maximum(background, 1), 0.0)
        mean_f = np.where(
            foreground > 0, (sum_all - totals) / np.maximum(foreground, 1), 0.0
        )
        between = background * foreground * (mean_b - mean_f) ** 2
    return int(np.argmax(between[:-1]))


def dark_threshold(grey: np.ndarray) -> int:
    """The grey level below which a pixel is board rather than room.

    Otsu twice, because these photographs have three tones and not two: a dark
    board, a bright wall, and a mid-grey ceiling or shadow between them. One
    split lands between the mid tone and the bright one, putting board and
    ceiling in the same class -- and a ceiling band is larger than a board, so
    the region search then measures the ceiling. Measured on a frame built to
    those proportions, that reported the board filling 100% of the width.

    The second pass, run on everything below the first, separates the board
    from the mid tone it was grouped with.
    """
    first = otsu_threshold(grey)
    below = np.asarray(grey)[np.asarray(grey) <= first]
    if below.size < 16 or below.min() == below.max():
        return first
    return min(first, otsu_threshold(below))


def _roundness(area: int, width: int, height: int) -> float:
    """How disc-like a region is, from its area against its bounding box.

    A circle fills pi/4 of the box that contains it; a wall, a ceiling band or
    a doorway fills close to all of it. The measure is the whole reason a
    ceiling cannot win: it is not a question of which dark thing is biggest but
    of which dark thing is *shaped like a dartboard*.
    """
    box = width * height
    if box <= 0:
        return 0.0
    circle = np.pi / 4
    return float(max(0.0, 1.0 - abs(area / box - circle) / circle))


def largest_dark_region(grey: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
    """Bounding box ``(left, top, right, bottom)`` of the most board-like blob.

    Labelled by flooding from each unvisited dark pixel, which is slower than a
    library routine and avoids a dependency for one call. At measurement width
    the whole image is some sixty thousand pixels, so the work is trivial and
    the alternative is putting scipy in the install path of a capture tool.

    Not simply the largest: a ceiling or a doorway can be both dark and bigger.
    Regions are scored by area weighted by how square their box is and how
    close their fill is to a circle's, so the winner is the biggest thing that
    is *shaped like a dartboard* rather than the biggest dark thing.
    """
    dark = np.asarray(grey) <= threshold
    height, width = dark.shape
    seen = np.zeros_like(dark, dtype=bool)
    best = (0.0, (0, 0, 0, 0))

    for start_y, start_x in np.argwhere(dark):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        size = 0
        top = bottom = int(start_y)
        left = right = int(start_x)
        while stack:
            y, x = stack.pop()
            size += 1
            top, bottom = min(top, y), max(bottom, y)
            left, right = min(left, x), max(right, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width:
                    if dark[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        box_w, box_h = right - left + 1, bottom - top + 1
        # Squareness allows for a board seen at an angle, which is an ellipse
        # and not a circle -- a shot oblique enough to halve one axis is still
        # a usable shot, and one flat enough to be a band is not a board.
        squareness = min(box_w, box_h) / max(box_w, box_h)
        score = size * squareness * _roundness(size, box_w, box_h)
        if score > best[0]:
            best = (score, (left, top, right, bottom))
    return best[1]


def _decode(image: Path, width: int, ffmpeg: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Read an image as grey at ``width``, and report its real size."""
    probe = subprocess.run(
        [ffmpeg, "-i", str(image), "-vf", f"scale={width}:-2,format=gray",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=False,
    )
    if probe.returncode != 0 or not probe.stdout:
        detail = probe.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            f"ffmpeg could not read {image.name}:\n"
            + (detail[-1] if detail else "no output")
        )
    height = len(probe.stdout) // width
    grey = np.frombuffer(probe.stdout, dtype=np.uint8).reshape(height, width)

    # The true size, read back off ffmpeg's own report of the input stream.
    text = probe.stderr.decode("utf-8", "replace")
    import re
    match = re.search(r",\s(\d{2,5})x(\d{2,5})[\s,]", text)
    source = (int(match.group(1)), int(match.group(2))) if match else (width, height)
    return grey, source


def measure_image(
    image: str | Path,
    input_px: int = DEFAULT_INPUT_PX,
    ffmpeg: str | None = None,
) -> Framing:
    """Estimate what one still leaves the model to work with."""
    from dartvision.capture.frames import find_ffmpeg

    image = Path(image)
    if not image.exists():
        raise FileNotFoundError(f"there is no file at {image}")
    grey, source = _decode(image, MEASURE_WIDTH, ffmpeg or find_ffmpeg())

    left, top, right, bottom = largest_dark_region(grey, dark_threshold(grey))
    scale = source[0] / grey.shape[1]
    board = (round((right - left + 1) * scale), round((bottom - top + 1) * scale))

    # The model resizes the whole frame onto a square, so each axis is scaled
    # by its own factor -- which is why a portrait recording loses far more
    # vertically than the framing alone would suggest.
    def ring(extent: int, source_extent: int) -> float:
        return extent * input_px / source_extent * RING_MM / BOARD_OUTER_MM

    return Framing(
        source=source, board=board,
        ring_across=ring(board[0], source[0]),
        ring_down=ring(board[1], source[1]),
        # What a square crop around the board would leave: the same scale on
        # both axes, which is the fix that does not need re-shooting.
        ring_squared=ring(board[1], min(source)),
    )
