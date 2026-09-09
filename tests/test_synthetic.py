"""Tests for synthetic scene sampling.

The important test here is the end-to-end round trip: a generated scene's
labels are fed back through homography estimation and scoring, and must
reproduce the score of the tips the generator placed. That closes the loop
between the generator, the calibration module and the scorer without any
pixels existing yet.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from dartvision.data.labels import Origin
from dartvision.geometry.board import BDO_BOARD, margin_to_nearest_boundary, score_at
from dartvision.geometry.calibration import estimate_homography
from dartvision.geometry.camera import CameraPose
from dartvision.synthetic import (
    PoseRanges,
    build_scene,
    sample_cluster,
    sample_pose,
    sample_tip_near_boundary,
    sample_tip_uniform,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260909)


# --------------------------------------------------------------------------
# Pose sampling
# --------------------------------------------------------------------------

def test_sampled_poses_stay_inside_their_ranges(rng):
    ranges = PoseRanges(elevation_deg=(15.0, 40.0), distance_mm=(1500.0, 1800.0))
    for _ in range(200):
        pose = sample_pose(rng, ranges)
        assert 15.0 <= pose.elevation_deg <= 40.0
        assert 1500.0 <= pose.distance_mm <= 1800.0
        assert -12.0 <= pose.roll_deg <= 12.0


def test_pose_rejects_impossible_geometry():
    for kwargs in ({"elevation_deg": 0.0}, {"elevation_deg": 95.0},
                   {"distance_mm": -1.0}, {"focal_px": 0.0}):
        with pytest.raises(ValueError):
            CameraPose(**kwargs)


# --------------------------------------------------------------------------
# Tip sampling
# --------------------------------------------------------------------------

def test_uniform_tips_stay_in_the_scoring_area(rng):
    for _ in range(500):
        x, y = sample_tip_uniform(rng)
        assert math.hypot(x, y) <= BDO_BOARD.r_double
        assert score_at(x, y).notation != "MISS"


def test_uniform_tips_are_area_uniform_not_radius_uniform(rng):
    """Half the scoring area lies outside r/sqrt(2), so half the tips should."""
    radii = [math.hypot(*sample_tip_uniform(rng)) for _ in range(4000)]
    outer = sum(1 for r in radii if r > BDO_BOARD.r_double / math.sqrt(2))
    assert 0.45 < outer / len(radii) < 0.55


@pytest.mark.parametrize("target", [0.25, 0.5, 1.0, 2.0, 5.0])
def test_tips_can_be_placed_at_a_chosen_margin(rng, target):
    """The coverage #14 requires and thrown darts supply only by luck."""
    for _ in range(60):
        tip = sample_tip_near_boundary(rng, target)
        assert margin_to_nearest_boundary(*tip) == pytest.approx(target, abs=0.05)
        assert math.hypot(*tip) <= BDO_BOARD.r_double


def test_near_boundary_tips_reach_both_radial_and_angular_boundaries(rng):
    """Both boundary kinds must be exercised, since they fail differently."""
    kinds = set()
    for _ in range(300):
        x, y = sample_tip_near_boundary(rng, 1.0)
        r = math.hypot(x, y)
        radial = min(abs(r - b) for b in BDO_BOARD.radial_boundaries)
        kinds.add("radial" if radial <= 1.0 + 1e-6 else "angular")
    assert kinds == {"radial", "angular"}


def test_rejects_a_non_positive_margin(rng):
    with pytest.raises(ValueError, match="must be positive"):
        sample_tip_near_boundary(rng, 0.0)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_clusters_are_tight_and_on_the_board(rng, count):
    for _ in range(100):
        tips = sample_cluster(rng, count, spread_mm=10.0)
        assert len(tips) == count
        for x, y in tips:
            assert math.hypot(x, y) <= BDO_BOARD.r_double
        for a in tips:
            for b in tips:
                assert math.dist(a, b) <= 20.0 + 1e-9


def test_cluster_count_is_bounded_by_the_darts_in_a_visit(rng):
    for bad in (0, 4):
        with pytest.raises(ValueError, match="between 1 and 3"):
            sample_cluster(rng, bad)


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------

def test_scene_labels_are_normalized_and_well_formed(rng):
    scene = build_scene(sample_pose(rng), [sample_tip_uniform(rng)], image_id="s1")
    annotation = scene.annotation
    assert annotation.origin is Origin.SYNTHETIC
    assert len(annotation.landmarks) == 8
    assert annotation.is_calibratable
    assert annotation.image_size == scene.pose.image_size
    assert {"elevation_deg", "azimuth_deg", "distance_mm", "focal_px", "roll_deg"} <= set(
        annotation.meta
    )


def test_scene_supports_the_four_landmark_contract(rng):
    scene = build_scene(sample_pose(rng), [], image_id="s1", landmark_count=4)
    assert len(scene.annotation.landmarks) == 4


def test_scene_rejects_more_than_three_tips(rng):
    with pytest.raises(ValueError, match="at most 3"):
        build_scene(sample_pose(rng), [(0.0, 0.0)] * 4, image_id="s1")


def test_a_dead_on_close_pose_keeps_the_board_in_frame_and_large():
    pose = CameraPose(elevation_deg=90.0, distance_mm=1400.0, focal_px=1400.0)
    scene = build_scene(pose, [], image_id="s1")
    assert scene.landmarks_visible
    assert scene.board_coverage > 0.2


def test_a_distant_pose_shrinks_the_board(rng):
    near = build_scene(CameraPose(elevation_deg=90.0, distance_mm=1400.0), [], image_id="a")
    far = build_scene(CameraPose(elevation_deg=90.0, distance_mm=3200.0), [], image_id="b")
    assert far.board_coverage < near.board_coverage


# --------------------------------------------------------------------------
# The round trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("landmark_count", [4, 8])
def test_generated_labels_score_back_to_the_placed_darts(rng, landmark_count):
    """Generator -> labels -> homography -> board coords -> score, over many scenes."""
    checked = 0
    for i in range(300):
        pose = sample_pose(rng)
        tips_mm = [sample_tip_uniform(rng) for _ in range(int(rng.integers(1, 4)))]
        scene = build_scene(
            pose, tips_mm, image_id=f"s{i}", landmark_count=landmark_count
        )
        if not scene.landmarks_visible or scene.board_coverage < 0.05:
            continue

        homography = estimate_homography(scene.annotation.landmarks)
        recovered = homography.to_board(scene.annotation.tips)

        for placed, (x, y) in zip(tips_mm, recovered):
            assert (x, y) == pytest.approx(placed, abs=1e-6)
            assert score_at(x, y) == score_at(*placed)
        checked += 1

    assert checked > 100, f"only {checked} scenes were usable; sampler may be broken"


def test_round_trip_holds_for_deliberately_near_wire_darts(rng):
    """The knife-edge cases must survive the projection, not just easy ones."""
    for i in range(200):
        pose = sample_pose(rng)
        tip = sample_tip_near_boundary(rng, float(rng.choice([0.25, 0.5, 1.0])))
        scene = build_scene(pose, [tip], image_id=f"n{i}")
        if not scene.landmarks_visible or scene.board_coverage < 0.05:
            continue
        recovered = estimate_homography(scene.annotation.landmarks).to_board(
            scene.annotation.tips
        )[0]
        assert score_at(*recovered) == score_at(*tip)


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------

def test_preview_svg_is_well_formed_and_marks_everything(rng):
    from xml.etree import ElementTree

    from dartvision.synthetic import scene_to_svg

    tips = [sample_tip_uniform(rng) for _ in range(3)]
    scene = build_scene(CameraPose(elevation_deg=40.0, distance_mm=1600.0), tips, image_id="p")
    svg = scene_to_svg(scene)

    root = ElementTree.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{ns}svg"
    # One circle per landmark, one crossed path per tip, seven scoring rings.
    assert len(root.findall(f"{ns}circle")) == len(scene.annotation.landmarks)
    paths = root.findall(f"{ns}path")
    assert sum(1 for p in paths if p.get("stroke") == "#dc2626") == len(tips)
    assert sum(1 for p in paths if p.get("stroke") == "#111827") == 7
    assert len(root.findall(f"{ns}line")) == 20  # sector wires


# --------------------------------------------------------------------------
# Structured dataset generation
# --------------------------------------------------------------------------

def test_dataset_has_the_requested_shape(rng):
    from dartvision.synthetic.dataset import DatasetSpec, generate_dataset

    spec = DatasetSpec(setups=2, sessions_per_setup=3, images_per_session=5)
    scenes = list(generate_dataset(rng, spec))

    assert len(scenes) == spec.total_images == 30
    assert len({s.annotation.setup_id for s in scenes}) == 2
    assert len({s.annotation.session_id for s in scenes}) == 6


def test_camera_pose_is_fixed_within_a_session(rng):
    """A mounted phone does not move mid-session, and #14 groups by session."""
    from dartvision.synthetic.dataset import DatasetSpec, generate_dataset

    scenes = list(generate_dataset(rng, DatasetSpec(setups=2, sessions_per_setup=2, images_per_session=6)))
    by_session: dict[str, set] = {}
    for scene in scenes:
        by_session.setdefault(scene.annotation.session_id, set()).add(scene.pose)
    assert all(len(poses) == 1 for poses in by_session.values())
    # ...but different sessions must genuinely differ.
    assert len({next(iter(p)) for p in by_session.values()}) == len(by_session)


def test_landmarks_are_identical_within_a_session(rng):
    """Which is exactly why #26 can annotate them once per session."""
    from dartvision.synthetic.dataset import DatasetSpec, generate_dataset

    scenes = list(generate_dataset(rng, DatasetSpec(setups=1, sessions_per_setup=1, images_per_session=8)))
    assert len({s.annotation.landmarks for s in scenes}) == 1


def test_every_scene_meets_the_coverage_floor(rng):
    from dartvision.synthetic.dataset import DatasetSpec, generate_dataset

    spec = DatasetSpec(setups=2, sessions_per_setup=2, images_per_session=4, min_board_coverage=0.15)
    for scene in generate_dataset(rng, spec):
        assert scene.landmarks_visible
        assert scene.board_coverage >= 0.15


def test_placement_mix_over_represents_near_boundary_darts(rng):
    """Deliberately unlike a real throw distribution -- see #14."""
    from dartvision.synthetic.dataset import DatasetSpec, generate_dataset, summarize

    spec = DatasetSpec(setups=1, sessions_per_setup=2, images_per_session=250)
    scenes = list(generate_dataset(rng, spec))
    placements = summarize(scenes)["placements"]

    near = sum(v for k, v in placements.items() if k.startswith("near-boundary"))
    assert 0.28 < near / len(scenes) < 0.42          # requested 0.35
    assert placements.get("cluster", 0) / len(scenes) > 0.13
    assert any(len(s.annotation.tips) == 0 for s in scenes)


def test_generated_dataset_scores_back_correctly(rng):
    """The round trip again, now over the structured generator."""
    from dartvision.synthetic.dataset import DatasetSpec, generate_dataset

    spec = DatasetSpec(setups=2, sessions_per_setup=2, images_per_session=25)
    for scene in generate_dataset(rng, spec):
        if not scene.annotation.tips:
            continue
        recovered = estimate_homography(scene.annotation.landmarks).to_board(
            scene.annotation.tips
        )
        for placed, (x, y) in zip(scene.tips_mm, recovered):
            assert score_at(x, y) == score_at(*placed)


def test_impossible_coverage_demand_fails_loudly(rng):
    from dartvision.synthetic.dataset import DatasetSpec, generate_dataset

    spec = DatasetSpec(setups=1, sessions_per_setup=1, images_per_session=1, min_board_coverage=0.99)
    with pytest.raises(RuntimeError, match="coverage"):
        list(generate_dataset(rng, spec))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"setups": 0},
        {"near_boundary_fraction": 0.9, "cluster_fraction": 0.9},
        {"landmark_count": 6},
        {"min_board_coverage": 1.0},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    from dartvision.synthetic.dataset import DatasetSpec

    with pytest.raises(ValueError):
        DatasetSpec(**kwargs)


def test_cli_writes_a_manifest_previews_and_a_summary(tmp_path):
    from dartvision.data.labels import read_jsonl
    from dartvision.synthetic.generate import main

    assert main(["--out", str(tmp_path), "--setups", "1", "--sessions", "2",
                 "--images", "5", "--previews", "3", "--seed", "3"]) == 0

    annotations = list(read_jsonl(tmp_path / "manifest.jsonl"))
    assert len(annotations) == 10
    assert len(list((tmp_path / "previews").glob("*.svg"))) == 3
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["images"] == 10 and summary["sessions"] == 2
