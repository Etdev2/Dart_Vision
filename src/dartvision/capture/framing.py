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
    "draw_box",
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
        return self.diameter / self.source[0]

    @property
    def aspect(self) -> float:
        """Height over width of what was measured. A board is close to 1."""
        return self.board[1] / max(self.board[0], 1)

    @property
    def suspect(self) -> bool:
        """Whether what was found is the wrong shape to be a dartboard.

        A board photographed from any angle anyone throws from stays roughly as
        wide as it is tall. A measurement far outside that did not find a
        board: most often the board's dark pixels have merged with something
        dark touching it -- the box many boards are mounted against, a shadow,
        a doorway -- and the bounding box then spans both.

        Reported, and worked around rather than abandoned: see ``diameter``.
        """
        return not 0.6 <= self.aspect <= 1.6

    @property
    def diameter(self) -> int:
        """The board's size, taking the shorter side when the region is wrong.

        Merging can only ever make a region *larger* -- two dark things that
        touch produce a box spanning both, and nothing makes a box smaller than
        the board inside it. So whichever side is shorter is the side the merge
        did not inflate, and on a round object that side is the diameter. A box
        above a board inflates the height and leaves the width; a doorway
        beside it does the reverse; either way the smaller number is the honest
        one.

        It errs low for a board seen at a steep angle, whose shorter axis is
        genuinely foreshortened. That is the safe direction to err: it reports
        less precision available than there is, and the decision it feeds is
        whether to re-shoot.
        """
        return min(self.board) if self.suspect else self.board[0]

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


# How much of its bounding box a dartboard's dark pixels cover. Not a disc's
# pi/4: a board is black *segments*, and the cream ones between them are not
# dark at all. The dark part is the outer ring, the numbers ring and half the
# sectors -- roughly 0.4 to 0.6 of the box -- where a wall, a ceiling band or a
# doorway covers essentially all of theirs. The first version of this measure
# assumed a filled disc, was tested against a fixture that was one, and scored
# a real board as though it were barely round at all.
BOARD_FILL = (0.25, 0.9)


def _plausible(area: int, width: int, height: int) -> float:
    """How much a region looks like a board, rather than a wall or a band.

    Two things, and neither alone is enough. Squareness *squared*, because a
    board seen from an angle is an ellipse but never a 3:1 band, and a linear
    penalty left a band with four times the pixels still winning. And a fill
    inside the range a ring pattern produces: something covering all of its box
    is a flat surface, and something covering almost none of it is a wire.
    """
    box = width * height
    if box <= 0:
        return 0.0
    squareness = min(width, height) / max(width, height)
    fill = area / box
    inside = BOARD_FILL[0] <= fill <= BOARD_FILL[1]
    return squareness ** 2 * (1.0 if inside else 0.25)


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
        score = size * _plausible(size, box_w, box_h)
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


def draw_box(
    image: Path, box: tuple[int, int, int, int], destination: Path,
    ffmpeg: str | None = None,
) -> Path:
    """Write a copy of ``image`` with ``box`` drawn on it.

    So the measurement can be checked rather than believed. Everything else
    here is an estimate made from an assumption about what a photograph of a
    dartboard looks like, and the cheapest way to find out whether the
    assumption held on a particular photograph is to look.
    """
    from dartvision.capture.frames import find_ffmpeg

    left, top, right, bottom = box
    result = subprocess.run(
        [ffmpeg or find_ffmpeg(), "-y", "-i", str(image),
         "-vf", f"drawbox=x={left}:y={top}:w={right - left}:h={bottom - top}"
                ":color=red@0.9:t=6",
         "-frames:v", "1", str(destination)],
        capture_output=True, check=False,
    )
    if result.returncode != 0 or not destination.exists():
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            "ffmpeg could not draw the check image:\n"
            + (detail[-1] if detail else "no output")
        )
    return destination


def measure_image(
    image: str | Path,
    input_px: int = DEFAULT_INPUT_PX,
    ffmpeg: str | None = None,
    board_width: int | None = None,
    check_into: Path | None = None,
) -> Framing:
    """Estimate what one still leaves the model to work with.

    ``board_width`` overrides the measurement with one taken by hand, for the
    photographs where the detection is defeated -- a board mounted against
    something dark, most often. The height is taken to match, since a board is
    round.
    """
    from dartvision.capture.frames import find_ffmpeg

    image = Path(image)
    if not image.exists():
        raise FileNotFoundError(f"there is no file at {image}")
    ffmpeg = ffmpeg or find_ffmpeg()
    grey, source = _decode(image, MEASURE_WIDTH, ffmpeg)

    left, top, right, bottom = largest_dark_region(grey, dark_threshold(grey))
    scale = source[0] / grey.shape[1]
    board = (round((right - left + 1) * scale), round((bottom - top + 1) * scale))

    if check_into is not None:
        draw_box(
            image,
            (round(left * scale), round(top * scale),
             round((right + 1) * scale), round((bottom + 1) * scale)),
            check_into, ffmpeg,
        )
    if board_width is not None:
        board = (board_width, board_width)

    # The model resizes the whole frame onto a square, so each axis is scaled
    # by its own factor -- which is why a portrait recording loses far more
    # vertically than the framing alone would suggest.
    def ring(extent: int, source_extent: int) -> float:
        return extent * input_px / source_extent * RING_MM / BOARD_OUTER_MM

    # Sized from the shorter side whenever the region is the wrong shape to be
    # a board, since a merge inflates one axis and leaves the other.
    raw = Framing(source=source, board=board, ring_across=0, ring_down=0,
                  ring_squared=0)
    side = raw.diameter

    return Framing(
        source=source, board=board,
        ring_across=ring(side, source[0]),
        ring_down=ring(side, source[1]),
        # What a square crop around the board would leave: the same scale on
        # both axes, which is the fix that does not need re-shooting.
        ring_squared=ring(side, min(source)),
    )
