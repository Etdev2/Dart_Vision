"""Capture plans for hand-placed darts.

Thrown darts land near a scoring wire only by luck, and #14 established that a
benchmark with few near-wire darts cannot measure what actually breaks scoring.
Placing darts by hand fixes that: the intended segment is known before the
shutter opens, and the margin can be dialled in deliberately.

This module turns a set of target margins into instructions a person can follow
at the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dartvision.geometry.board import (
    BDO_BOARD,
    Boundary,
    BoardSpec,
    Hit,
    nearest_boundary,
    score_at,
)
from dartvision.synthetic.scene import sample_tip_near_boundary

__all__ = ["PlacedTarget", "plan_placed_capture", "render_plan"]


@dataclass(frozen=True)
class PlacedTarget:
    """One dart to place, and where."""

    board_xy: tuple[float, float]
    hit: Hit
    boundary: Boundary
    target_margin_mm: float

    @property
    def radius_mm(self) -> float:
        x, y = self.board_xy
        return float(np.hypot(x, y))

    @property
    def instruction(self) -> str:
        side = "inside" if self.boundary.inside else "outside"
        where = (
            f"{self.boundary.distance_mm:.2f} mm {side} the {self.boundary.name}"
            if self.boundary.kind == "radial"
            else f"{self.boundary.distance_mm:.2f} mm from the {self.boundary.name}"
        )
        return (
            f"{self.hit.notation:>5}  |  {where}, "
            f"about {self.radius_mm:.0f} mm out from the bull"
        )


def plan_placed_capture(
    rng: np.random.Generator,
    margins_mm: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    per_margin: int = 5,
    board: BoardSpec = BDO_BOARD,
) -> list[PlacedTarget]:
    """Targets covering each requested margin, spread across the board."""
    if per_margin < 1:
        raise ValueError("per_margin must be at least 1")
    if not margins_mm:
        raise ValueError("at least one margin is required")

    targets: list[PlacedTarget] = []
    for margin in margins_mm:
        for _ in range(per_margin):
            xy = sample_tip_near_boundary(rng, float(margin), board)
            targets.append(
                PlacedTarget(
                    board_xy=xy,
                    hit=score_at(*xy, board),
                    boundary=nearest_boundary(*xy, board),
                    target_margin_mm=float(margin),
                )
            )
    return targets


def render_plan(targets: Sequence[PlacedTarget]) -> str:
    """A checklist to read at the board."""
    if not targets:
        return "no targets"

    lines = [
        "Placed-dart capture plan",
        "=" * 68,
        "",
        "Place one dart per row, photograph, then move it. Record the row number",
        "with each photo so the label can be matched back to the intended target.",
        "",
    ]
    current: float | None = None
    for index, target in enumerate(targets, start=1):
        if target.target_margin_mm != current:
            current = target.target_margin_mm
            lines.append(f"\n-- margin {current} mm " + "-" * 46)
        lines.append(f"{index:>3}. {target.instruction}")

    kinds = {t.boundary.kind for t in targets}
    lines += [
        "",
        "=" * 68,
        f"{len(targets)} targets covering "
        f"{len(sorted({t.target_margin_mm for t in targets}))} margins "
        f"across {'radial and sector' if kinds == {'radial', 'sector'} else ' and '.join(sorted(kinds))} boundaries.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m dartvision.annotate.plan --margins 0.25,0.5,1,2 --per-margin 5``"""
    import argparse

    parser = argparse.ArgumentParser(description="Print a placed-dart capture plan")
    parser.add_argument("--margins", default="0.25,0.5,1,2",
                        help="comma-separated target margins in mm")
    parser.add_argument("--per-margin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write to a file instead of stdout")
    args = parser.parse_args(argv)

    margins = tuple(float(m) for m in args.margins.split(",") if m.strip())
    targets = plan_placed_capture(np.random.default_rng(args.seed), margins, args.per_margin)
    text = render_plan(targets)

    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {len(targets)} targets to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
