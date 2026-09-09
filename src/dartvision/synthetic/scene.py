"""Sampling synthetic scenes: camera poses, dart placements, and exact labels.

This is the half of the synthetic generator that produces *labels*. A renderer
turns a scene into pixels; everything that decides where the darts and
landmarks actually are lives here, which is why it can be built and tested
before any rendering exists.

The design goal from #24 and #14 is coverage the real corpus cannot supply on
demand: camera angles across the full viewing hemisphere, and -- critically --
dart tips at *controlled distances from scoring boundaries*. #14 established
that a benchmark with few near-wire darts cannot measure the thing that breaks
scoring, and thrown darts land near wires only by luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np

from dartvision.data.labels import Annotation, Origin
from dartvision.geometry.board import BDO_BOARD, BoardSpec, margin_to_nearest_boundary
from dartvision.geometry.calibration import calibration_points_4, calibration_points_8
from dartvision.geometry.camera import CameraPose, board_to_image_matrix, project

__all__ = [
    "PoseRanges",
    "sample_pose",
    "sample_tip_uniform",
    "sample_tip_near_boundary",
    "sample_cluster",
    "Scene",
    "build_scene",
]

Point = tuple[float, float]


@dataclass(frozen=True)
class PoseRanges:
    """Inclusive ranges the pose sampler draws from.

    Defaults span a plausible envelope for a phone watching a board: from a
    steep mount down to a low grazing view, and near enough that the board
    fills a useful share of the frame.
    """

    elevation_deg: tuple[float, float] = (8.0, 75.0)
    azimuth_deg: tuple[float, float] = (0.0, 360.0)
    distance_mm: tuple[float, float] = (1200.0, 3200.0)
    focal_px: tuple[float, float] = (1000.0, 2000.0)
    roll_deg: tuple[float, float] = (-12.0, 12.0)
    image_size: tuple[int, int] = (1200, 1600)


def _uniform(rng: np.random.Generator, span: tuple[float, float]) -> float:
    low, high = span
    return float(rng.uniform(low, high))


def sample_pose(rng: np.random.Generator, ranges: PoseRanges = PoseRanges()) -> CameraPose:
    return CameraPose(
        elevation_deg=_uniform(rng, ranges.elevation_deg),
        azimuth_deg=_uniform(rng, ranges.azimuth_deg),
        distance_mm=_uniform(rng, ranges.distance_mm),
        focal_px=_uniform(rng, ranges.focal_px),
        roll_deg=_uniform(rng, ranges.roll_deg),
        image_size=ranges.image_size,
    )


def sample_tip_uniform(
    rng: np.random.Generator, board: BoardSpec = BDO_BOARD
) -> Point:
    """A tip drawn uniformly *by area* over the scoring region."""
    radius = board.r_double * math.sqrt(float(rng.uniform(0.0, 1.0)))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    return (radius * math.cos(angle), radius * math.sin(angle))


def sample_tip_near_boundary(
    rng: np.random.Generator,
    target_margin_mm: float,
    board: BoardSpec = BDO_BOARD,
    tolerance_mm: float = 0.05,
    max_attempts: int = 64,
) -> Point:
    """A tip placed a chosen distance from the nearest scoring boundary.

    Constructed rather than rejection-sampled: pick a boundary, then step off
    it by the requested margin. The result is *verified* with the same
    ``margin_to_nearest_boundary`` used by the metrics, because stepping off one
    boundary can land nearer a different one -- so construction proposes and
    the tested measurement decides.
    """
    if target_margin_mm <= 0:
        raise ValueError("target_margin_mm must be positive")

    for _ in range(max_attempts):
        if rng.random() < 0.5:
            # Step radially off one of the ring boundaries.
            boundary = float(rng.choice(np.asarray(board.radial_boundaries)))
            radius = boundary + (target_margin_mm if rng.random() < 0.5 else -target_margin_mm)
            if radius <= 0 or radius > board.r_double:
                continue
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
        else:
            # Step angularly off a sector wire, outside the bull where wires matter.
            wire_index = int(rng.integers(0, 20))
            wire_deg = 99.0 - 18.0 * wire_index
            radius = float(rng.uniform(board.r_outer_bull + target_margin_mm + 1.0,
                                       board.r_double - 1.0))
            if radius <= 0:
                continue
            offset_rad = target_margin_mm / radius
            angle = math.radians(wire_deg) + (offset_rad if rng.random() < 0.5 else -offset_rad)

        candidate = (radius * math.cos(angle), radius * math.sin(angle))
        if abs(margin_to_nearest_boundary(*candidate, board) - target_margin_mm) <= tolerance_mm:
            return candidate

    raise RuntimeError(
        f"could not place a tip at margin {target_margin_mm} mm within {max_attempts} attempts"
    )


def sample_cluster(
    rng: np.random.Generator,
    count: int,
    spread_mm: float = 12.0,
    board: BoardSpec = BDO_BOARD,
    max_attempts: int = 64,
) -> tuple[Point, ...]:
    """``count`` tips within ``spread_mm`` of each other -- the hard case."""
    if not 1 <= count <= 3:
        raise ValueError("count must be between 1 and 3")
    if spread_mm <= 0:
        raise ValueError("spread_mm must be positive")

    for _ in range(max_attempts):
        anchor = sample_tip_uniform(rng, board)
        tips = [anchor]
        for _ in range(count - 1):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            offset = float(rng.uniform(0.0, spread_mm))
            tips.append((anchor[0] + offset * math.cos(angle), anchor[1] + offset * math.sin(angle)))
        if all(math.hypot(x, y) <= board.r_double for x, y in tips):
            return tuple(tips)

    raise RuntimeError("could not place a cluster inside the scoring area")


@dataclass(frozen=True)
class Scene:
    """A sampled scene and the label it produces."""

    pose: CameraPose
    tips_mm: tuple[Point, ...]
    annotation: Annotation
    landmarks_visible: bool

    @property
    def board_coverage(self) -> float:
        """Fraction of the shorter image dimension spanned by the board.

        A useful realism filter: a board occupying a handful of pixels carries
        no recoverable precision, whatever the labels say.
        """
        pts = np.asarray([p for p in self.annotation.landmarks if p is not None])
        width, height = self.annotation.image_size or self.pose.image_size
        extent = (pts.max(axis=0) - pts.min(axis=0)) * np.array([width, height])
        return float(extent.min() / min(width, height))


def build_scene(
    pose: CameraPose,
    tips_mm: Sequence[Point],
    *,
    image_id: str,
    setup_id: str = "synthetic",
    session_id: str | None = None,
    landmark_count: Literal[4, 8] = 8,
    board: BoardSpec = BDO_BOARD,
    meta: dict[str, object] | None = None,
) -> Scene:
    """Project a pose and a set of board-space tips into a labelled scene."""
    if len(tips_mm) > 3:
        raise ValueError("at most 3 dart tips")

    matrix = board_to_image_matrix(pose)
    width, height = pose.image_size
    board_landmarks = (
        calibration_points_4(board) if landmark_count == 4 else calibration_points_8(board)
    )

    def to_normalized(points) -> list[Point]:
        projected = project(matrix, points)
        return [(float(x) / width, float(y) / height) for x, y in projected]

    landmarks = to_normalized(board_landmarks)
    tips = to_normalized(tips_mm) if tips_mm else []
    visible = all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in landmarks)

    annotation = Annotation(
        image_id=image_id,
        setup_id=setup_id,
        session_id=session_id or f"{setup_id}/{pose.elevation_deg:.0f}-{pose.azimuth_deg:.0f}",
        origin=Origin.SYNTHETIC,
        landmarks=tuple(landmarks),
        tips=tuple(tips),
        image_size=(width, height),
        meta={
            "elevation_deg": round(pose.elevation_deg, 3),
            "azimuth_deg": round(pose.azimuth_deg, 3),
            "distance_mm": round(pose.distance_mm, 1),
            "focal_px": round(pose.focal_px, 1),
            "roll_deg": round(pose.roll_deg, 3),
            **(meta or {}),
        },
    )
    return Scene(pose=pose, tips_mm=tuple(tips_mm), annotation=annotation,
                 landmarks_visible=visible)
