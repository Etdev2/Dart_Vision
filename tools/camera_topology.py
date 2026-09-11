"""Measurements behind #2's camera-topology decision.

Two questions, both answered from the repository's own geometry rather than
from intuition:

1. How many board millimetres does one pixel of tip error buy, as a function of
   how face-on the camera is? That converts #21's tip-error gates into a pixel
   precision the model must actually reach.
2. How often does one dart hide another dart's tip? That is the failure mode a
   single viewpoint cannot fix, and the one that decides whether the MVP
   topology holds.

Run:  python tools/camera_topology.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dartvision.geometry.board import BDO_BOARD, BoardSpec
from dartvision.geometry.camera import CameraPose, board_to_image_matrix
from dartvision.synthetic.scene import focal_for_board_fill

# A steel-tip dart is about 150 mm long and penetrates 10-20 mm, so most of it
# stands proud of the board. It arrives descending, so it sits tilted rather
# than square to the face. Both make occlusion worse than a square 45 mm stub.
BARREL_RADIUS_MM = 3.5
PROTRUSION_MM = 130.0
TILT_DEG = 20.0

ELEVATIONS = (10, 15, 20, 30, 40, 45, 50, 60, 75, 90)


def pose_for(
    elevation_deg: float,
    board_fill: float = 0.80,
    distance_mm: float = 2000.0,
    image_size: tuple[int, int] = (1200, 1600),
) -> CameraPose:
    """A pose framed the way the pipeline actually frames a board."""
    return CameraPose(
        elevation_deg=elevation_deg, azimuth_deg=0.0, distance_mm=distance_mm,
        focal_px=focal_for_board_fill(distance_mm, board_fill, image_size),
        image_size=image_size,
    )


def full_projection(pose: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    """The 3x4 matrix and the camera centre.

    The plane homography is enough for points *on* the board; a dart standing
    out of it needs the full projection.
    """
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
    cx, cy = pose.principal_point
    intrinsics = np.array(
        [[pose.focal_px, 0.0, cx], [0.0, pose.focal_px, cy], [0.0, 0.0, 1.0]]
    )
    return intrinsics @ np.column_stack([rotation, -rotation @ centre]), centre


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points)
    hom = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return hom[:, :2] / hom[:, 2:3]


# --------------------------------------------------------------------------
# 1. Local scale
# --------------------------------------------------------------------------

def mm_per_pixel(pose: CameraPose, radius_mm: float, samples: int = 720) -> np.ndarray:
    """Board mm per pixel of image error, around a ring, worst direction.

    The largest singular value of the image-to-board Jacobian: how far a point
    can move on the board for one pixel of movement in the image.
    """
    h = board_to_image_matrix(pose)
    inverse = np.linalg.inv(h)

    scales = []
    for angle in np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False):
        board = np.array([radius_mm * math.cos(angle), radius_mm * math.sin(angle), 1.0])
        image = h @ board
        image = image[:2] / image[2]

        jacobian = np.zeros((2, 2))
        for axis in range(2):
            step = np.zeros(2)
            step[axis] = 0.5
            ahead = inverse @ np.array([*(image + step), 1.0])
            behind = inverse @ np.array([*(image - step), 1.0])
            jacobian[:, axis] = ahead[:2] / ahead[2] - behind[:2] / behind[2]
        scales.append(float(np.linalg.svd(jacobian, compute_uv=False)[0]))
    return np.array(scales)


# --------------------------------------------------------------------------
# 2. Occlusion
# --------------------------------------------------------------------------

def occluded_tips(
    pose: CameraPose,
    tips,
    protrusion_mm: float = PROTRUSION_MM,
    barrel_radius_mm: float = BARREL_RADIUS_MM,
    tilt_deg: float = TILT_DEG,
) -> set[int]:
    """Indices of tips whose board position is hidden behind another dart."""
    matrix, centre = full_projection(pose)
    tilt = math.radians(tilt_deg)
    axis = protrusion_mm * np.array([0.0, math.sin(tilt), math.cos(tilt)])

    hidden: set[int] = set()
    for i, tip in enumerate(tips):
        point = np.array([tip[0], tip[1], 0.0])
        image = _project(matrix, point)[0]
        depth = float(np.linalg.norm(point - centre))

        for j, other in enumerate(tips):
            if i == j:
                continue
            base = np.array([other[0], other[1], 0.0])
            other_depth = float(np.linalg.norm(base - centre))
            if other_depth > depth:
                continue                  # behind the tip; it cannot hide it
            near, far = _project(matrix, np.vstack([base, base + axis]))
            segment = far - near
            length = float(segment @ segment)
            along = 0.0 if length == 0 else float(
                np.clip((image - near) @ segment / length, 0.0, 1.0)
            )
            gap = float(np.linalg.norm(image - (near + along * segment)))
            if gap < pose.focal_px * barrel_radius_mm / other_depth:
                hidden.add(i)
                break
    return hidden


def occlusion_rate(
    pose: CameraPose,
    spread_mm: float,
    trials: int = 3000,
    seed: int = 7,
    board: BoardSpec = BDO_BOARD,
    barrel_radius_mm: float = BARREL_RADIUS_MM,
) -> float:
    """Share of darts hidden, over random three-dart groupings.

    Offsets are area-uniform and barrels never overlap: two darts closer than
    touching is not a grouping, it is a bounce-out.
    """
    rng = np.random.default_rng(seed)
    hidden = total = 0
    for _ in range(trials):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = math.sqrt(rng.uniform(0.0, 1.0)) * board.r_double * 0.9
        anchor = (radius * math.cos(angle), radius * math.sin(angle))

        tips = [anchor]
        while len(tips) < 3:
            direction = rng.uniform(0.0, 2.0 * math.pi)
            offset = math.sqrt(rng.uniform(0.0, 1.0)) * spread_mm
            candidate = (
                anchor[0] + offset * math.cos(direction),
                anchor[1] + offset * math.sin(direction),
            )
            if all(math.dist(candidate, t) >= 2 * barrel_radius_mm for t in tips):
                tips.append(candidate)

        hidden += len(occluded_tips(pose, tips, barrel_radius_mm=barrel_radius_mm))
        total += 3
    return hidden / total


def main() -> int:
    board = BDO_BOARD

    print("Board millimetres per pixel of tip error, on the double ring")
    print("and the pixel precision #21's gates therefore demand.")
    print()
    print(f"{'elev':>5} {'best':>7} {'median':>7} {'worst':>7} "
          f"{'px for 4mm':>11} {'px for 6mm':>11}")
    for elevation in ELEVATIONS:
        scale = mm_per_pixel(pose_for(elevation), board.r_double)
        worst = scale.max()
        print(f"{elevation:5d} {scale.min():7.3f} {np.median(scale):7.3f} {worst:7.3f} "
              f"{4.0 / worst:11.2f} {6.0 / worst:11.2f}")

    print()
    print("Share of darts whose board position is hidden by another dart.")
    print()
    print(f"{'elev':>5} {'12mm group':>11} {'20mm':>8} {'35mm':>8}")
    for elevation in ELEVATIONS:
        pose = pose_for(elevation)
        rates = [occlusion_rate(pose, spread) for spread in (12.0, 20.0, 35.0)]
        print(f"{elevation:5d} " + " ".join(f"{r * 100:9.1f}%" for r in rates))

    print()
    print("Occlusion tracks the barrel's width, not the camera's angle:")
    print(f"{'barrel':>8} {'elev 20':>9} {'elev 60':>9}")
    for radius in (2.0, 3.5, 5.0):
        low = occlusion_rate(pose_for(20), 12.0, trials=2500, barrel_radius_mm=radius)
        high = occlusion_rate(pose_for(60), 12.0, trials=2500, barrel_radius_mm=radius)
        print(f"{radius:6.1f}mm {low * 100:8.1f}% {high * 100:8.1f}%")

    print()
    print("Because the strip a dart hides is far longer than a grouping is wide:")
    for elevation in (10, 20, 30, 45, 60, 75):
        strip = PROTRUSION_MM / math.tan(math.radians(elevation))
        print(f"  {elevation:2d} deg -> {strip:7.0f} mm hidden behind each dart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
