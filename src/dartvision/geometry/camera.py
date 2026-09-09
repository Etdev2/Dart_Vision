"""A pinhole camera viewing the board plane.

The board is planar, so a camera pose plus intrinsics collapse to a single
homography from board millimetres to pixels. This module builds that mapping.
It is shared rather than test-only: the synthetic generator (#25) samples poses
from it, and the tests use it to exercise the projective geometry a real phone
produces instead of an arbitrary matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["CameraPose", "board_to_image_matrix", "project"]


@dataclass(frozen=True)
class CameraPose:
    """Where the camera is and what it sees.

    ``elevation_deg`` is the angle above the board plane: 90 is dead-on
    perpendicular to the board, small values are grazing. ``azimuth_deg``
    orbits the board. ``roll_deg`` rotates the image, as a handheld or
    imperfectly mounted phone does.
    """

    elevation_deg: float = 20.0
    azimuth_deg: float = 0.0
    distance_mm: float = 2400.0
    focal_px: float = 1400.0
    roll_deg: float = 0.0
    image_size: tuple[int, int] = (1200, 1600)

    def __post_init__(self) -> None:
        if not 0.0 < self.elevation_deg <= 90.0:
            raise ValueError("elevation_deg must be in (0, 90]")
        if self.distance_mm <= 0 or self.focal_px <= 0:
            raise ValueError("distance_mm and focal_px must be positive")
        if any(v <= 0 for v in self.image_size):
            raise ValueError("image_size must be positive")

    @property
    def principal_point(self) -> tuple[float, float]:
        width, height = self.image_size
        return (width / 2.0, height / 2.0)


def board_to_image_matrix(pose: CameraPose) -> np.ndarray:
    """Homography mapping board millimetres to pixels for ``pose``."""
    el, az = math.radians(pose.elevation_deg), math.radians(pose.azimuth_deg)
    centre = pose.distance_mm * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )

    forward = -centre / np.linalg.norm(centre)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    rotation = np.vstack([right, -up, forward])
    if pose.roll_deg:
        c, s = math.cos(math.radians(pose.roll_deg)), math.sin(math.radians(pose.roll_deg))
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ rotation

    translation = -rotation @ centre
    cx, cy = pose.principal_point
    intrinsics = np.array(
        [[pose.focal_px, 0.0, cx], [0.0, pose.focal_px, cy], [0.0, 0.0, 1.0]]
    )
    h = intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate camera pose")
    return h / h[2, 2]


def project(matrix: np.ndarray, points) -> np.ndarray:
    """Apply a homography to an ``(n, 2)`` array of points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ matrix.T
    w = homogeneous[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("point projects to the line at infinity")
    return homogeneous[:, :2] / w
