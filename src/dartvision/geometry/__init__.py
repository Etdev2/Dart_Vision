"""Deterministic board geometry: no model, no images."""

from dartvision.geometry.board import (
    BDO_BOARD,
    SECTORS_CLOCKWISE_FROM_20,
    BoardSpec,
    Hit,
    margin_to_nearest_boundary,
    score_at,
)
from dartvision.geometry.camera import CameraPose, board_to_image_matrix, project
from dartvision.geometry.calibration import (
    CALIBRATION_ANGLES_DEG,
    Homography,
    HomographyQuality,
    assess_landmarks,
    calibration_points_4,
    calibration_points_8,
    estimate_homography,
)

__all__ = [
    "BDO_BOARD",
    "CALIBRATION_ANGLES_DEG",
    "SECTORS_CLOCKWISE_FROM_20",
    "BoardSpec",
    "CameraPose",
    "Hit",
    "Homography",
    "HomographyQuality",
    "assess_landmarks",
    "board_to_image_matrix",
    "calibration_points_4",
    "calibration_points_8",
    "estimate_homography",
    "margin_to_nearest_boundary",
    "project",
    "score_at",
]
