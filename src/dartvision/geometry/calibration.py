"""Calibration landmarks and the image-to-board homography.

The board is planar and rigid, so a single homography maps image coordinates to
board millimetres. This module defines *which* points on the board serve as
calibration landmarks, estimates the homography from their detected image
positions, and reports how self-consistent that estimate is.

Landmark choice
---------------
Landmarks are intersections of the outer double wire with sector-boundary
wires. Four sector boundaries fall exactly 90 degrees apart -- at 9, 99, 189 and
279 degrees -- separating the 13|6, 20|5, 8|11 and 17|3 beds. These are visually
crisp features (a wire crossing a wire) rather than inferred points, and they
sit in two mutually perpendicular pairs.

The eight-point set adds the same four angles on the outer *treble* wire. Doing
so is not cosmetic: with more than four correspondences the homography becomes
over-determined, which yields a genuine least-squares residual usable as a
self-consistency signal, and averages down per-landmark noise. Since Dart Vision
generates and annotates its own data, the four-point contract is a choice rather
than a constraint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from dartvision.geometry.board import BDO_BOARD, BoardSpec

__all__ = [
    "CALIBRATION_ANGLES_DEG",
    "calibration_points_4",
    "calibration_points_8",
    "Homography",
    "estimate_homography",
    "HomographyQuality",
    "assess_landmarks",
]

# Sector-boundary wires that are mutually 90 degrees apart.
CALIBRATION_ANGLES_DEG: Final[tuple[float, ...]] = (9.0, 99.0, 189.0, 279.0)


def _ring_points(radius: float) -> list[tuple[float, float]]:
    return [
        (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)))
        for a in CALIBRATION_ANGLES_DEG
    ]


def calibration_points_4(board: BoardSpec = BDO_BOARD) -> list[tuple[float, float]]:
    """The four outer-double-wire landmarks, in board millimetres, fixed order."""
    return _ring_points(board.r_double)


def calibration_points_8(board: BoardSpec = BDO_BOARD) -> list[tuple[float, float]]:
    """Eight landmarks: the four double-wire points then the four treble-wire points."""
    return _ring_points(board.r_double) + _ring_points(board.r_treble)


def _default_board_points(count: int, board: BoardSpec) -> list[tuple[float, float]]:
    """Board coordinates for a standard landmark count.

    Only the 4- and 8-point sets have a canonical meaning; any other count must
    supply its own board coordinates rather than silently getting the wrong set.
    """
    if count == 4:
        return calibration_points_4(board)
    if count == 8:
        return calibration_points_8(board)
    raise ValueError(
        f"no canonical board points for {count} landmarks; "
        "pass board_points explicitly (4 and 8 are standard)"
    )


@dataclass(frozen=True)
class Homography:
    """A 3x3 projective map from image coordinates to board millimetres."""

    matrix: np.ndarray

    def to_board(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        return _apply(self.matrix, points)

    def to_image(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        return _apply(np.linalg.inv(self.matrix), points)


def _apply(m: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
    projected = homogeneous @ m.T
    w = projected[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("point maps to the line at infinity; homography is degenerate")
    return projected[:, :2] / w


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic normalization: centroid at origin, mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    mean_dist = float(np.sqrt((centred ** 2).sum(axis=1)).mean())
    if mean_dist < 1e-12:
        raise ValueError("degenerate point set: all points coincide")
    scale = math.sqrt(2.0) / mean_dist
    t = np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]]
    )
    return (centred * scale), t


def estimate_homography(
    image_points: Sequence[Sequence[float]],
    board_points: Sequence[Sequence[float]] | None = None,
    board: BoardSpec = BDO_BOARD,
) -> Homography:
    """Estimate the image-to-board homography by normalized DLT.

    ``image_points`` must be in the fixed landmark order. With four points the
    solution is exact; with more it is least-squares, which both averages down
    noise and makes the residual from :func:`assess_landmarks` meaningful.
    """
    src = np.asarray(image_points, dtype=float).reshape(-1, 2)
    if len(src) < 4:
        raise ValueError(
            f"a homography needs at least 4 correspondences, got {len(src)}"
        )

    if board_points is None:
        board_points = _default_board_points(len(src), board)
    dst = np.asarray(board_points, dtype=float).reshape(-1, 2)

    if len(src) != len(dst):
        raise ValueError(f"got {len(src)} image points but {len(dst)} board points")

    src_n, t_src = _normalize(src)
    dst_n, t_dst = _normalize(dst)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h_n = vt[-1].reshape(3, 3)

    h = np.linalg.inv(t_dst) @ h_n @ t_src
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate landmark configuration")
    return Homography(matrix=h / h[2, 2])


@dataclass(frozen=True)
class HomographyQuality:
    """Evidence about whether a landmark set can be trusted.

    ``residual_mm`` is only informative when there are more than four
    landmarks: four correspondences determine a homography exactly, so the
    residual is identically zero and carries no information. That is why the
    eight-point set exists.
    """

    convex_and_ordered: bool
    condition_number: float
    residual_mm: float | None

    @property
    def usable(self) -> bool:
        return self.convex_and_ordered and self.condition_number < 1e7


def _is_convex_in_order(pts: np.ndarray) -> bool:
    """True if the polygon through ``pts`` in the given order is convex."""
    n = len(pts)
    signs = []
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        (ax, ay), (bx, by) = b - a, c - b
        cross = ax * by - ay * bx
        if abs(cross) < 1e-12:
            return False  # collinear triple
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def assess_landmarks(
    image_points: Sequence[Sequence[float]],
    board: BoardSpec = BDO_BOARD,
) -> HomographyQuality:
    """Check a landmark set for the failure modes that four points can reveal.

    Note what is *not* checkable: any four points in general position admit an
    exact homography onto any other four, so there is no such thing as a
    "geometrically impossible" set of four landmarks to reject on residual
    grounds. What can be detected is a crossed or non-convex ordering, a
    near-degenerate configuration, and -- with more than four landmarks -- a
    genuine least-squares residual.
    """
    src = np.asarray(image_points, dtype=float).reshape(-1, 2)
    ordered = _is_convex_in_order(src[:4])

    try:
        h = estimate_homography(src, board=board)
        condition = float(np.linalg.cond(h.matrix))
    except (ValueError, np.linalg.LinAlgError):
        return HomographyQuality(
            convex_and_ordered=ordered, condition_number=math.inf, residual_mm=None
        )

    residual: float | None = None
    if len(src) > 4:
        expected = np.asarray(_default_board_points(len(src), board), dtype=float)
        errors = np.linalg.norm(h.to_board(src) - expected, axis=1)
        residual = float(np.sqrt((errors ** 2).mean()))

    return HomographyQuality(
        convex_and_ordered=ordered, condition_number=condition, residual_mm=residual
    )
