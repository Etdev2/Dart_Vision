"""Wireframe SVG preview of a scene.

Not a renderer -- it draws the board's scoring geometry projected through the
scene's camera, so a human can confirm at a glance that the pose, the
landmarks and the dart placements are where they should be. Useful long before
a real renderer exists, and useful afterwards for debugging a scene whose
labels look wrong.
"""

from __future__ import annotations

import math
from typing import Sequence

from dartvision.geometry.board import BDO_BOARD, BoardSpec
from dartvision.geometry.camera import board_to_image_matrix, project
from dartvision.synthetic.scene import Scene

__all__ = ["scene_to_svg"]

_WIRE = "#6b7280"
_RING = "#111827"
_TIP = "#dc2626"
_LANDMARK = "#2563eb"


def _ring_path(matrix, radius: float, steps: int = 240) -> str:
    points = [
        (radius * math.cos(2 * math.pi * i / steps), radius * math.sin(2 * math.pi * i / steps))
        for i in range(steps + 1)
    ]
    projected = project(matrix, points)
    return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in projected)


def scene_to_svg(scene: Scene, board: BoardSpec = BDO_BOARD) -> str:
    matrix = board_to_image_matrix(scene.pose)
    width, height = scene.pose.image_size
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
    ]

    for radius, stroke in (
        (board.r_board, 1.0),
        (board.r_double, 2.0),
        (board.r_double_inner, 1.5),
        (board.r_treble, 1.5),
        (board.r_treble_inner, 1.5),
        (board.r_outer_bull, 1.5),
        (board.r_inner_bull, 1.5),
    ):
        parts.append(
            f'<path d="{_ring_path(matrix, radius)}" fill="none" '
            f'stroke="{_RING}" stroke-width="{stroke}" opacity="0.85"/>'
        )

    for i in range(20):
        angle = math.radians(99.0 - 18.0 * i)
        inner = (board.r_outer_bull * math.cos(angle), board.r_outer_bull * math.sin(angle))
        outer = (board.r_double * math.cos(angle), board.r_double * math.sin(angle))
        (x1, y1), (x2, y2) = project(matrix, [inner, outer])
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{_WIRE}" stroke-width="1"/>'
        )

    for point in scene.annotation.landmarks:
        if point is None:
            continue
        cx, cy = point[0] * width, point[1] * height
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="none" '
            f'stroke="{_LANDMARK}" stroke-width="2.5"/>'
        )

    for point in scene.annotation.tips:
        cx, cy = point[0] * width, point[1] * height
        parts.append(
            f'<path d="M {cx - 9:.2f},{cy - 9:.2f} L {cx + 9:.2f},{cy + 9:.2f} '
            f'M {cx - 9:.2f},{cy + 9:.2f} L {cx + 9:.2f},{cy - 9:.2f}" '
            f'stroke="{_TIP}" stroke-width="3"/>'
        )

    pose = scene.pose
    caption = (
        f"elev {pose.elevation_deg:.0f} deg | azim {pose.azimuth_deg:.0f} deg | "
        f"{pose.distance_mm:.0f} mm | f {pose.focal_px:.0f} px | roll {pose.roll_deg:+.1f} deg"
    )
    parts.append(
        f'<text x="16" y="{height - 18}" font-family="ui-monospace,monospace" '
        f'font-size="26" fill="#334155">{caption}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
