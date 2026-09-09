"""Synthetic scene generation (#25): labels first, pixels later."""

from dartvision.synthetic.preview import scene_to_svg
from dartvision.synthetic.scene import (
    PoseRanges,
    focal_for_board_fill,
    nominal_ring_width_px,
    Scene,
    build_scene,
    sample_cluster,
    sample_pose,
    sample_tip_near_boundary,
    sample_tip_uniform,
)

__all__ = [
    "PoseRanges",
    "Scene",
    "build_scene",
    "focal_for_board_fill",
    "nominal_ring_width_px",
    "sample_cluster",
    "sample_pose",
    "sample_tip_near_boundary",
    "sample_tip_uniform",
    "scene_to_svg",
]
