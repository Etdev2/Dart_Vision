"""Encoding annotations into model targets, and decoding predictions back.

Pure NumPy, so the encoding can be exercised without a training framework. This
is deliberate: target encoding is where keypoint pipelines quietly go wrong, and
a mistake here produces a model that trains happily on subtly wrong targets and
fails for reasons no later metric attributes to the labels.

Two heads, following #15:

* **Landmarks** -- one heatmap channel per landmark, decoded by soft-argmax.
  Exactly one instance per channel, so a heatmap fits, and soft-argmax recovers
  sub-cell position without a second head.
* **Dart tips** -- a single centre heatmap plus explicit sub-cell offsets. Tips
  are up to three instances of *one* class and may be clustered, which a
  per-channel heatmap cannot represent.

Sub-cell precision is not a refinement here. At an 800 px input with stride 4, a
cell is 4 px, while #15 puts the 10 mm double/treble ring at 10-20 px. A
whole-cell error is therefore a sizeable fraction of the ring width, and the
difference between a treble and a single.

Coordinate convention: points are normalized to ``[0, 1]``; on a grid of
``(H, W)``, cell ``i`` sits *at* integer coordinate ``i``, the continuous
position is ``cx = x * W``, the owning cell is ``ix = round(cx)`` and the offset
is ``cx - ix`` in ``[-0.5, 0.5]``, so ``(ix + offset) / W`` recovers ``x``
exactly.

Rounding rather than flooring is what keeps the encoder and decoder agreeing.
The Gaussian peaks at the *nearest* cell to ``cx``, so that is the cell the
decoder finds -- and therefore the cell the offset must be written to. Using
``floor`` for the offset while the peak lands on ``round`` puts them in
different cells whenever ``cx`` falls in the upper half of one, which silently
loses the sub-cell correction for about half of all points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "TipTargets",
    "recommended_window",
    "encode_landmark_heatmaps",
    "decode_landmark_heatmaps",
    "encode_tip_targets",
    "decode_tip_targets",
    "soft_argmax",
]

Point = tuple[float, float]


def _cell_and_offset(point: Point, grid: tuple[int, int]) -> tuple[int, int, float, float]:
    height, width = grid
    cx, cy = point[0] * width, point[1] * height
    ix = min(max(int(round(cx)), 0), width - 1)
    iy = min(max(int(round(cy)), 0), height - 1)
    return ix, iy, cx - ix, cy - iy


def _gaussian(grid: tuple[int, int], cx: float, cy: float, sigma: float) -> np.ndarray:
    height, width = grid
    ys = np.arange(height, dtype=float)[:, None]
    xs = np.arange(width, dtype=float)[None, :]
    return np.exp(-(((xs - cx) ** 2) + ((ys - cy) ** 2)) / (2.0 * sigma ** 2))


def encode_landmark_heatmaps(
    points, grid: tuple[int, int], sigma_cells: float = 2.0
) -> np.ndarray:
    """One channel per landmark. ``None`` yields an all-zero channel."""
    if sigma_cells <= 0:
        raise ValueError("sigma_cells must be positive")
    height, width = grid
    if height < 1 or width < 1:
        raise ValueError("grid must be at least 1x1")

    channels = []
    for point in points:
        if point is None:
            channels.append(np.zeros(grid, dtype=np.float32))
            continue
        cx, cy = point[0] * width, point[1] * height
        channels.append(_gaussian(grid, cx, cy, sigma_cells).astype(np.float32))
    return np.stack(channels) if channels else np.zeros((0, *grid), dtype=np.float32)


# Peak height is *not* a confidence signal on its own. A Gaussian centred
# between cells is never sampled at its maximum, so a perfect prediction peaks
# lower the further its true position sits from a cell centre -- roughly 0.78 to
# 1.00 at sigma=1, shrinking as sigma grows. Anything calibrating confidence
# from peak height (#21) must account for that, or read it from a head trained
# for the purpose.


def recommended_window(sigma_cells: float) -> int:
    """Smallest odd decode window that makes soft-argmax bias negligible.

    Soft-argmax over a truncated window is *biased*, not merely noisy: clipping
    the Gaussian asymmetrically pulls the centroid toward the peak cell.
    Measured mean error in cells, encoding and decoding the same point:

    ===========  ======  ======  ======  ======
    sigma        w=5     w=9     w=13    w=17
    ===========  ======  ======  ======  ======
    1.0          0.024   0.000   0.000   0.000
    2.0          0.106   0.044   0.005   0.000
    3.0          0.122   0.103   0.051   0.015
    ===========  ======  ======  ======  ======

    The bias disappears once the window covers roughly +/-3 sigma, which is what
    this returns. It matters because the error is systematic: at stride 4 on an
    800 px input, 0.1 cells is 0.4 px against a scoring ring 10-20 px wide
    (#15), and a bias applies to every landmark in the same direction rather
    than averaging out across a dataset.
    """
    if sigma_cells <= 0:
        raise ValueError("sigma_cells must be positive")
    return int(2 * math.ceil(3.0 * sigma_cells) + 1)


def soft_argmax(heat: np.ndarray, window: int = 5) -> tuple[Point, float]:
    """Sub-cell peak location, as normalized coordinates, plus peak value.

    Restricted to a window around the arg-max rather than run over the whole
    map: a global soft-argmax is pulled off-target by any other activity in the
    channel, which for a real prediction is common.
    """
    if heat.ndim != 2:
        raise ValueError("expected a 2-D heatmap")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number of cells")

    height, width = heat.shape
    peak = int(np.argmax(heat))
    py, px = divmod(peak, width)
    half = window // 2

    y0, y1 = max(0, py - half), min(height, py + half + 1)
    x0, x1 = max(0, px - half), min(width, px + half + 1)
    patch = heat[y0:y1, x0:x1].astype(float)

    weights = np.clip(patch - patch.min(), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return (px / width, py / height), float(heat[py, px])

    ys = np.arange(y0, y1, dtype=float)[:, None]
    xs = np.arange(x0, x1, dtype=float)[None, :]
    cx = float((weights * xs).sum() / total)
    cy = float((weights * ys).sum() / total)
    return (cx / width, cy / height), float(heat[py, px])


def decode_landmark_heatmaps(
    heatmaps: np.ndarray, window: int = 5, threshold: float = 0.0
) -> list[tuple[Point | None, float]]:
    """Decode each channel. Channels peaking at or below ``threshold`` give ``None``."""
    if heatmaps.ndim != 3:
        raise ValueError("expected (channels, height, width)")
    decoded: list[tuple[Point | None, float]] = []
    for channel in heatmaps:
        point, score = soft_argmax(channel, window)
        decoded.append((None, score) if score <= threshold else (point, score))
    return decoded


@dataclass(frozen=True)
class TipTargets:
    """Centre heatmap, sub-cell offsets, and the mask marking real tips."""

    heat: np.ndarray     # (H, W)
    offset: np.ndarray   # (2, H, W) -- x then y, in cells
    mask: np.ndarray     # (H, W) -- 1 where an offset is defined

    @property
    def count(self) -> int:
        return int(self.mask.sum())


def encode_tip_targets(
    points, grid: tuple[int, int], sigma_cells: float = 2.0
) -> TipTargets:
    """Centre heatmap with offsets. Overlapping tips take the elementwise max.

    Two darts landing in one cell is a genuine ambiguity this representation
    cannot express; the encoder keeps the stronger peak rather than pretending
    otherwise, and the cell count in ``count`` will be lower than the number of
    tips supplied. Callers checking cluster behaviour should compare the two.
    """
    if sigma_cells <= 0:
        raise ValueError("sigma_cells must be positive")
    height, width = grid
    heat = np.zeros(grid, dtype=np.float32)
    offset = np.zeros((2, *grid), dtype=np.float32)
    mask = np.zeros(grid, dtype=np.float32)

    for point in points:
        if point is None:
            continue
        ix, iy, ox, oy = _cell_and_offset(point, grid)
        cx, cy = point[0] * width, point[1] * height
        np.maximum(heat, _gaussian(grid, cx, cy, sigma_cells).astype(np.float32), out=heat)
        offset[0, iy, ix] = ox
        offset[1, iy, ix] = oy
        mask[iy, ix] = 1.0

    return TipTargets(heat=heat, offset=offset, mask=mask)


def _local_maxima(heat: np.ndarray) -> np.ndarray:
    """Boolean map of cells at least as large as all eight neighbours."""
    padded = np.pad(heat, 1, mode="constant", constant_values=-np.inf)
    keep = np.ones_like(heat, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[1 + dy: 1 + dy + heat.shape[0], 1 + dx: 1 + dx + heat.shape[1]]
            keep &= heat >= shifted
    return keep


def decode_tip_targets(
    heat: np.ndarray,
    offset: np.ndarray,
    threshold: float = 0.3,
    max_detections: int = 3,
) -> list[tuple[Point, float]]:
    """Peaks above ``threshold``, refined by their offsets, strongest first."""
    if heat.ndim != 2:
        raise ValueError("expected a 2-D heatmap")
    if offset.shape != (2, *heat.shape):
        raise ValueError(f"offset must be (2, {heat.shape[0]}, {heat.shape[1]})")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")

    height, width = heat.shape
    candidates = np.argwhere(_local_maxima(heat) & (heat > threshold))
    scored = sorted(
        ((float(heat[y, x]), int(x), int(y)) for y, x in candidates), reverse=True
    )

    results: list[tuple[Point, float]] = []
    for score, x, y in scored[:max_detections]:
        px = (x + float(offset[0, y, x])) / width
        py = (y + float(offset[1, y, x])) / height
        results.append(((px, py), score))
    return results
