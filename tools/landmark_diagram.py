"""Draw the eight calibration landmarks on a dartboard, in click order.

"cal_1 (9 deg, 13|6 wire)" is precise and unreadable. The annotator asks for
eight specific points on a physical object, and the shortest correct
explanation of where they are is a picture of them. Generated rather than
drawn so it cannot drift from the geometry it documents: every radius and
angle here comes from ``geometry.board`` and ``geometry.calibration``.
"""

from __future__ import annotations

import math
from pathlib import Path

from dartvision.geometry.board import BDO_BOARD, SECTORS_CLOCKWISE_FROM_20
from dartvision.geometry.calibration import (
    CALIBRATION_ANGLES_DEG,
    calibration_points_8,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "landmarks.svg"

# The drawing is in board millimetres throughout, so the frame is stated the
# same way: the board is 451 mm across and everything else is margin. Wider
# than the board to fit the wire labels beside it, and taller to give the
# legend and the caption bands of their own rather than a corner of the board.
HALF_W, TOP, BOTTOM = 300.0, 352.0, 342.0


def _xy(x: float, y: float) -> tuple[float, float]:
    """Board millimetres to SVG units. SVG's y grows downward; a board's does not."""
    return x, -y


def build() -> str:
    board = BDO_BOARD
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-HALF_W} {-TOP} '
        f'{2 * HALF_W} {TOP + BOTTOM}" width="640" '
        f'height="{round(640 * (TOP + BOTTOM) / (2 * HALF_W))}" '
        'font-family="ui-sans-serif, system-ui, sans-serif">',
        f'<rect x="{-HALF_W}" y="{-TOP}" width="{2 * HALF_W}" '
        f'height="{TOP + BOTTOM}" fill="#faf9f6"/>',
        f'<circle cx="0" cy="0" r="{board.r_board}" fill="#1c1917"/>',
        f'<circle cx="0" cy="0" r="{board.r_double}" fill="#efe7d2"/>',
    ]

    # Alternating beds, so the board is recognisable rather than a set of rings.
    first_boundary = 90.0 + 9.0
    for index, number in enumerate(SECTORS_CLOCKWISE_FROM_20):
        start = first_boundary - index * 18.0
        end = start - 18.0
        if index % 2:
            a0, a1 = math.radians(start), math.radians(end)
            x0, y0 = _xy(board.r_double * math.cos(a0), board.r_double * math.sin(a0))
            x1, y1 = _xy(board.r_double * math.cos(a1), board.r_double * math.sin(a1))
            parts.append(
                f'<path d="M 0 0 L {x0:.2f} {y0:.2f} A {board.r_double} '
                f'{board.r_double} 0 0 1 {x1:.2f} {y1:.2f} Z" fill="#2b2622"/>'
            )
        mid = math.radians(start - 9.0)
        nx, ny = _xy((board.r_board - 28) * math.cos(mid),
                     (board.r_board - 28) * math.sin(mid))
        parts.append(
            f'<text x="{nx:.1f}" y="{ny + 7:.1f}" fill="#f5f5f4" font-size="17" '
            f'text-anchor="middle">{number}</text>'
        )

    # The wires the landmarks sit on.
    for radius, width in ((board.r_double, 2.4), (board.r_treble, 2.4),
                          (board.r_double_inner, 1.2), (board.r_treble_inner, 1.2),
                          (board.r_outer_bull, 1.2), (board.r_inner_bull, 1.2)):
        parts.append(
            f'<circle cx="0" cy="0" r="{radius}" fill="none" stroke="#a8a29e" '
            f'stroke-width="{width}"/>'
        )
    for index in range(20):
        angle = math.radians(first_boundary - index * 18.0)
        x0, y0 = _xy(board.r_outer_bull * math.cos(angle),
                     board.r_outer_bull * math.sin(angle))
        x1, y1 = _xy(board.r_double * math.cos(angle), board.r_double * math.sin(angle))
        parts.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            'stroke="#a8a29e" stroke-width="1.2"/>'
        )

    # The landmarks themselves, in the order the annotator asks for them.
    labels = {9.0: "13 | 6", 99.0: "20 | 5", 189.0: "8 | 11", 279.0: "17 | 3"}
    for index, (x, y) in enumerate(calibration_points_8(board), start=1):
        cx, cy = _xy(x, y)
        colour = "#dc2626" if index <= 4 else "#2563eb"
        parts.append(
            f'<circle class="landmark" cx="{cx:.2f}" cy="{cy:.2f}" r="7" '
            f'fill="{colour}" stroke="#fff" stroke-width="2.5"/>'
        )
        away = 1.16 if index <= 4 else 0.72
        parts.append(
            f'<text x="{cx * away:.1f}" y="{cy * away + 6:.1f}" fill="{colour}" '
            f'font-size="19" font-weight="700" text-anchor="middle">{index}</text>'
        )

    for angle, text in labels.items():
        a = math.radians(angle)
        tx, ty = _xy(243 * math.cos(a), 243 * math.sin(a))
        anchor = "start" if math.cos(a) > 0.5 else "end" if math.cos(a) < -0.5 else "middle"
        parts.append(
            f'<text x="{tx:.1f}" y="{ty + 5:.1f}" fill="#57534e" font-size="14" '
            f'text-anchor="{anchor}">{text}</text>'
        )

    parts += [
        f'<text x="0" y="{-TOP + 34}" fill="#1c1917" font-size="24" '
        'font-weight="700" text-anchor="middle">The eight landmarks, in click order</text>',
        f'<circle cx="-192" cy="{-TOP + 68}" r="7" fill="#dc2626" '
        'stroke="#fff" stroke-width="2.5"/>',
        f'<text x="-178" y="{-TOP + 74}" fill="#dc2626" font-size="16" '
        f'font-weight="600">1-4  outer edge of the DOUBLE ring ({board.r_double:.0f} mm out)</text>',
        f'<circle cx="-192" cy="{-TOP + 94}" r="7" fill="#2563eb" '
        'stroke="#fff" stroke-width="2.5"/>',
        f'<text x="-178" y="{-TOP + 100}" fill="#2563eb" font-size="16" '
        f'font-weight="600">5-8  outer edge of the TREBLE ring ({board.r_treble:.0f} mm out)</text>',
        f'<text x="0" y="{BOTTOM - 58}" fill="#1c1917" font-size="17" '
        'text-anchor="middle">Each point is where a radial wire crosses a ring wire.</text>',
        f'<text x="0" y="{BOTTOM - 32}" fill="#57534e" font-size="15" '
        'text-anchor="middle">The four wires are 90 degrees apart.</text>',
        "</svg>",
    ]
    return "\n".join(parts)


if __name__ == "__main__":  # pragma: no cover
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
