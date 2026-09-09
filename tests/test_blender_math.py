"""The two camera conventions must agree, or rendered images and their labels
will silently disagree. This is the check that makes the renderer safe to run
unattended."""

from __future__ import annotations

import numpy as np
import pytest

from dartvision.geometry.calibration import calibration_points_8
from dartvision.geometry.camera import CameraPose, board_to_image_matrix, project
from dartvision.synthetic.blender_math import (
    camera_world_matrix,
    lens_mm_from_focal_px,
    project_via_blender_convention,
)
from dartvision.synthetic.scene import PoseRanges, sample_pose, sample_tip_uniform


def test_lens_conversion_matches_the_standard_relation():
    assert lens_mm_from_focal_px(1200.0, 1200, 36.0) == pytest.approx(36.0)
    assert lens_mm_from_focal_px(2400.0, 1200, 36.0) == pytest.approx(72.0)


@pytest.mark.parametrize("bad", [(0.0, 100, 36.0), (100.0, 0, 36.0), (100.0, 100, 0.0)])
def test_lens_conversion_rejects_degenerate_inputs(bad):
    with pytest.raises(ValueError):
        lens_mm_from_focal_px(*bad)


def test_camera_matrix_is_a_rigid_transform():
    pose = CameraPose(elevation_deg=32.0, azimuth_deg=41.0, roll_deg=7.0)
    rotation = camera_world_matrix(pose)[:3, :3]
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_camera_looks_at_the_board_centre():
    pose = CameraPose(elevation_deg=25.0, azimuth_deg=110.0)
    matrix = camera_world_matrix(pose)
    location, forward = matrix[:3, 3], -matrix[:3, 2]
    assert forward @ (-location / np.linalg.norm(location)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("elevation", [10.0, 25.0, 50.0, 89.0])
@pytest.mark.parametrize("azimuth", [0.0, 73.0, 210.0, 355.0])
@pytest.mark.parametrize("roll", [0.0, 9.0, -14.0])
def test_blender_convention_reproduces_our_homography(elevation, azimuth, roll):
    """The load-bearing test: both paths must land on the same pixels."""
    pose = CameraPose(
        elevation_deg=elevation, azimuth_deg=azimuth, roll_deg=roll,
        distance_mm=1900.0, focal_px=1500.0,
    )
    points = calibration_points_8() + [(0.0, 0.0), (0.0, 160.0), (-95.0, 40.0)]

    ours = project(board_to_image_matrix(pose), points)
    blender = project_via_blender_convention(pose, points)
    assert blender == pytest.approx(ours, abs=1e-6)


def test_conventions_agree_across_randomly_sampled_scenes():
    rng = np.random.default_rng(11)
    for _ in range(300):
        pose = sample_pose(rng, PoseRanges())
        points = calibration_points_8() + [sample_tip_uniform(rng) for _ in range(3)]
        assert project_via_blender_convention(pose, points) == pytest.approx(
            project(board_to_image_matrix(pose), points), abs=1e-6
        )


def test_roll_is_where_the_conventions_diverge():
    """Regression guard for the sign that reconciles y-down and y-up frames.

    Unrolled scenes agreed before this was fixed, so a mistake here would have
    passed casual inspection and mislabelled only the rolled images.
    """
    points = [(0.0, 160.0), (160.0, 0.0)]
    unrolled = CameraPose(elevation_deg=25.0, distance_mm=1900.0, focal_px=1500.0)
    rolled = CameraPose(
        elevation_deg=25.0, distance_mm=1900.0, focal_px=1500.0, roll_deg=12.0
    )
    for pose in (unrolled, rolled):
        assert project_via_blender_convention(pose, points) == pytest.approx(
            project(board_to_image_matrix(pose), points), abs=1e-6
        )
    # And roll must actually change the image, or the test above proves nothing.
    assert project_via_blender_convention(rolled, points) != pytest.approx(
        project_via_blender_convention(unrolled, points), abs=1.0
    )
