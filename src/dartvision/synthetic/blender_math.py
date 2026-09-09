"""Converting a :class:`CameraPose` into Blender's camera conventions.

Kept separate from the Blender script itself, and free of ``bpy``, so the risky
part -- reconciling two camera conventions -- can be tested without a renderer
present. A silent mismatch here would produce a dataset whose images and labels
disagree, which is the single worst failure this pipeline can have: everything
downstream would train happily on wrong data.

Conventions being reconciled:

* **Ours** (``geometry.camera``): board plane ``z = 0`` in millimetres, image
  ``x`` right and ``y`` down, focal length in pixels.
* **Blender**: metres, camera looks down its local ``-Z`` with ``+Y`` up, focal
  length in millimetres against a sensor of a given width.
"""

from __future__ import annotations

import math

import numpy as np

from dartvision.geometry.camera import CameraPose

__all__ = [
    "DEFAULT_SENSOR_WIDTH_MM",
    "MM_PER_METRE",
    "lens_mm_from_focal_px",
    "camera_world_matrix",
    "project_via_blender_convention",
]

DEFAULT_SENSOR_WIDTH_MM = 36.0  # Blender's default horizontal sensor size
MM_PER_METRE = 1000.0


def lens_mm_from_focal_px(
    focal_px: float, image_width_px: int, sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM
) -> float:
    """Focal length in millimetres for a horizontally-fit sensor."""
    if focal_px <= 0 or image_width_px <= 0 or sensor_width_mm <= 0:
        raise ValueError("focal_px, image_width_px and sensor_width_mm must be positive")
    return focal_px * sensor_width_mm / image_width_px


def camera_world_matrix(pose: CameraPose) -> np.ndarray:
    """Blender ``matrix_world`` for the camera, in metres (camera-to-world)."""
    el, az = math.radians(pose.elevation_deg), math.radians(pose.azimuth_deg)
    location = (pose.distance_mm / MM_PER_METRE) * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )

    forward = -location / np.linalg.norm(location)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    if pose.roll_deg:
        # Our image frame has +y down and Blender's camera has +y up, so the
        # same roll angle turns the opposite way in each. The sign here is what
        # reconciles them; without it, rolled scenes render mirrored against
        # their labels while unrolled ones look perfect.
        c, s = math.cos(math.radians(pose.roll_deg)), math.sin(math.radians(pose.roll_deg))
        right, up = c * right + s * up, -s * right + c * up

    # Blender's camera basis: +X right, +Y up, -Z forward.
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = location
    return matrix


def project_via_blender_convention(pose: CameraPose, points_mm) -> np.ndarray:
    """Project board points to pixels the way Blender's camera will see them.

    Used by the renderer's self-check: if this disagrees with the homography in
    ``geometry.camera``, the scene is set up wrong and the render must not
    proceed.
    """
    matrix = camera_world_matrix(pose)
    world_to_camera = np.linalg.inv(matrix)

    pts = np.asarray(points_mm, dtype=float).reshape(-1, 2) / MM_PER_METRE
    world = np.hstack([pts, np.zeros((len(pts), 1)), np.ones((len(pts), 1))])
    camera = world @ world_to_camera.T

    depth = -camera[:, 2]
    if np.any(depth <= 1e-9):
        raise ValueError("a point lies behind the camera")

    cx, cy = pose.principal_point
    u = cx + pose.focal_px * camera[:, 0] / depth
    v = cy - pose.focal_px * camera[:, 1] / depth
    return np.column_stack([u, v])
