"""Generating a structured synthetic dataset.

Synthetic data is given the *same structure* as real captured data, not merely
the same file format: a small number of setups, each holding several sessions,
each holding many images. That matters because #14's benchmark tiers group by
session and hold out whole setups. If synthetic images were sampled
independently -- a fresh camera pose per image -- every image would be its own
session and the grouping machinery would silently do nothing.

The camera pose is therefore **fixed within a session** and only the darts
change, which is also how a real capture session behaves: the phone is mounted
once and the player throws. It is the same assumption that lets #26 annotate
landmarks once per session rather than once per image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from dartvision.geometry.board import BDO_BOARD, BoardSpec
from dartvision.synthetic.scene import (
    PoseRanges,
    Scene,
    build_scene,
    sample_cluster,
    sample_pose,
    sample_tip_near_boundary,
    sample_tip_uniform,
)

__all__ = ["DatasetSpec", "generate_dataset", "summarize"]

# Placeholders consumed by the renderer; kept here so a manifest fully
# determines its images and a scene can be re-rendered from its label alone.
BOARD_STYLES = ("classic-black", "classic-green", "worn-sisal", "high-contrast")
LIGHTING_STYLES = ("daylight-soft", "daylight-hard", "tungsten", "mixed-dim", "overhead-glare")


@dataclass(frozen=True)
class DatasetSpec:
    """How much data to make and in what proportions.

    The tip-placement mix is deliberately unlike a real throw distribution.
    Near-boundary darts are over-represented because #14 established that a
    benchmark with few near-wire darts cannot measure what breaks scoring, and
    synthetic data is the only place they can be produced on demand.
    """

    setups: int = 3
    sessions_per_setup: int = 4
    images_per_session: int = 50

    min_board_coverage: float = 0.10
    min_ring_px: float = 12.0
    near_boundary_fraction: float = 0.35
    cluster_fraction: float = 0.20
    empty_fraction: float = 0.05
    near_boundary_margins_mm: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    cluster_spread_mm: float = 12.0
    landmark_count: int = 8

    def __post_init__(self) -> None:
        if min(self.setups, self.sessions_per_setup, self.images_per_session) < 1:
            raise ValueError("setups, sessions and images must all be at least 1")
        total = self.near_boundary_fraction + self.cluster_fraction + self.empty_fraction
        if not 0.0 <= total <= 1.0:
            raise ValueError("placement fractions must sum to at most 1.0")
        if not 0.0 <= self.min_board_coverage < 1.0:
            raise ValueError("min_board_coverage must be in [0, 1)")
        if self.min_ring_px <= 0:
            raise ValueError("min_ring_px must be positive")
        if self.landmark_count not in (4, 8):
            raise ValueError("landmark_count must be 4 or 8")

    @property
    def total_images(self) -> int:
        return self.setups * self.sessions_per_setup * self.images_per_session


def _usable_pose(
    rng: np.random.Generator,
    spec: DatasetSpec,
    ranges: PoseRanges,
    board: BoardSpec,
    max_attempts: int = 200,
):
    """Draw a pose whose board is in frame and carries enough pixels to learn from.

    Gated on **ring width in pixels**, not just frame coverage. #15's budget is
    denominated in the pixel width of the 10 mm double/treble rings, since that
    is what decides whether a multiplier can be resolved at all; a scene below
    it is unlearnable however exact its labels.
    """
    for _ in range(max_attempts):
        pose = sample_pose(rng, ranges, board)
        probe = build_scene(
            pose, [], image_id="probe", landmark_count=spec.landmark_count, board=board
        )
        if (
            probe.landmarks_visible
            and probe.board_coverage >= spec.min_board_coverage
            and probe.nominal_ring_width_px >= spec.min_ring_px
        ):
            return pose
    raise RuntimeError(
        f"no pose met coverage >= {spec.min_board_coverage} and ring width >= "
        f"{spec.min_ring_px}px in {max_attempts} attempts; loosen PoseRanges or "
        "lower the thresholds"
    )


def _sample_tips(rng: np.random.Generator, spec: DatasetSpec, board: BoardSpec):
    draw = float(rng.random())
    if draw < spec.empty_fraction:
        return (), "empty"
    if draw < spec.empty_fraction + spec.cluster_fraction:
        count = int(rng.integers(2, 4))
        return sample_cluster(rng, count, spec.cluster_spread_mm, board), "cluster"
    if draw < spec.empty_fraction + spec.cluster_fraction + spec.near_boundary_fraction:
        margin = float(rng.choice(np.asarray(spec.near_boundary_margins_mm)))
        count = int(rng.integers(1, 4))
        return tuple(
            sample_tip_near_boundary(rng, margin, board) for _ in range(count)
        ), f"near-boundary-{margin}mm"
    count = int(rng.integers(1, 4))
    return tuple(sample_tip_uniform(rng, board) for _ in range(count)), "uniform"


def generate_dataset(
    rng: np.random.Generator,
    spec: DatasetSpec = DatasetSpec(),
    ranges: PoseRanges = PoseRanges(),
    board: BoardSpec = BDO_BOARD,
) -> Iterator[Scene]:
    """Yield scenes grouped into sessions and setups."""
    for setup_index in range(spec.setups):
        setup_id = f"synthetic-setup-{setup_index:02d}"
        setup_meta = {
            "board_style": BOARD_STYLES[setup_index % len(BOARD_STYLES)],
            "lighting": LIGHTING_STYLES[setup_index % len(LIGHTING_STYLES)],
        }

        for session_index in range(spec.sessions_per_setup):
            session_id = f"{setup_id}/session-{session_index:02d}"
            pose = _usable_pose(rng, spec, ranges, board)

            for image_index in range(spec.images_per_session):
                tips, placement = _sample_tips(rng, spec, board)
                yield build_scene(
                    pose,
                    tips,
                    image_id=f"{session_id}/img-{image_index:04d}",
                    setup_id=setup_id,
                    session_id=session_id,
                    landmark_count=spec.landmark_count,
                    board=board,
                    meta={**setup_meta, "placement": placement},
                )


def summarize(scenes: list[Scene]) -> dict[str, object]:
    """Counts a reviewer actually wants before spending money rendering."""
    placements: dict[str, int] = {}
    darts: dict[int, int] = {}
    for scene in scenes:
        key = str(scene.annotation.meta.get("placement", "unknown"))
        placements[key] = placements.get(key, 0) + 1
        n = len(scene.annotation.tips)
        darts[n] = darts.get(n, 0) + 1

    coverages = sorted(s.board_coverage for s in scenes)
    rings = sorted(s.nominal_ring_width_px for s in scenes)
    return {
        "images": len(scenes),
        "setups": len({s.annotation.setup_id for s in scenes}),
        "sessions": len({s.annotation.session_id for s in scenes}),
        "placements": dict(sorted(placements.items())),
        "darts_per_image": dict(sorted(darts.items())),
        "board_coverage_min": round(coverages[0], 3),
        "board_coverage_median": round(coverages[len(coverages) // 2], 3),
        "board_coverage_max": round(coverages[-1], 3),
        "ring_px_min": round(rings[0], 1),
        "ring_px_median": round(rings[len(rings) // 2], 1),
        "ring_px_max": round(rings[-1], 1),
    }
