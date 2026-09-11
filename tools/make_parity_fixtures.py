"""Generate the fixtures `web/test/parity.test.mjs` checks the port against.

The Python is the reference implementation. A port that silently disagrees is
worse than no port, so rather than trusting a reading of the code, this dumps
what the Python actually computes on a spread of cases and the JS is asserted
to reproduce it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from dartvision.calibrate import assess_framing
from dartvision.geometry.board import BDO_BOARD, nearest_boundary, score_at
from dartvision.geometry.calibration import calibration_points_8, estimate_homography
from dartvision.game import Dart, X01Rules, play, replay
from dartvision.geometry.camera import CameraPose, board_to_image_matrix
from dartvision.synthetic.scene import focal_for_board_fill

IMAGE = (1600, 1200)   # (height, width)


def scoring_cases() -> list[dict]:
    """A spread over the whole board, plus every boundary radius deliberately."""
    rng = np.random.default_rng(11)
    points: list[tuple[float, float]] = []

    for _ in range(400):
        angle = float(rng.uniform(0, 2 * math.pi))
        radius = float(math.sqrt(rng.uniform(0, 1)) * BDO_BOARD.r_board)
        points.append((radius * math.cos(angle), radius * math.sin(angle)))

    # Straddle every radial boundary and every sector wire: the cases where a
    # convention disagreement would actually show up.
    for radius in BDO_BOARD.radial_boundaries:
        for delta in (-0.05, 0.0, 0.05):
            for degrees in (9.0, 45.0, 90.0, 99.0, 180.0, 279.0, 351.0):
                r = radius + delta
                points.append((r * math.cos(math.radians(degrees)),
                               r * math.sin(math.radians(degrees))))

    cases = []
    for x, y in points:
        hit = score_at(x, y)
        boundary = nearest_boundary(x, y)
        cases.append({
            "x": x, "y": y,
            "notation": hit.notation, "score": hit.score,
            "segment": hit.segment, "multiplier": hit.multiplier,
            "boundary": boundary.name, "margin_mm": boundary.distance_mm,
        })
    return cases


def framing_cases() -> list[dict]:
    """Real camera poses, with the landmarks a detector would report."""
    cases = []
    for elevation in (18.0, 25.0, 36.0, 45.0, 55.0, 65.0, 72.0, 84.0):
        for azimuth in (0.0, 47.0, 210.0):
            for fill in (0.45, 0.7, 0.85, 0.92):
                distance = 2000.0
                pose = CameraPose(
                    elevation_deg=elevation, azimuth_deg=azimuth, distance_mm=distance,
                    focal_px=focal_for_board_fill(distance, fill, IMAGE),
                    image_size=(IMAGE[1], IMAGE[0]),
                )
                matrix = board_to_image_matrix(pose)
                landmarks = []
                for bx, by in calibration_points_8():
                    projected = matrix @ np.array([bx, by, 1.0])
                    px, py = projected[:2] / projected[2]
                    landmarks.append([px / IMAGE[1], py / IMAGE[0]])

                report = assess_framing(landmarks, IMAGE)
                cases.append({
                    "landmarks": landmarks,
                    "true_elevation": elevation,
                    "expected": report.to_dict(),
                })
    return cases


def homography_cases() -> list[dict]:
    """Landmark sets, and where the Python maps a probe point through them."""
    probes = [(0.0, 0.0), (40.0, -90.0), (0.0, 160.0), (-120.0, 55.0)]
    cases = []
    for elevation, azimuth in ((25.0, 0.0), (50.0, 130.0), (70.0, 300.0)):
        pose = CameraPose(
            elevation_deg=elevation, azimuth_deg=azimuth, distance_mm=2200.0,
            focal_px=focal_for_board_fill(2200.0, 0.85, IMAGE),
            image_size=(IMAGE[1], IMAGE[0]),
        )
        matrix = board_to_image_matrix(pose)
        image_points = []
        for bx, by in calibration_points_8():
            projected = matrix @ np.array([bx, by, 1.0])
            image_points.append(list(projected[:2] / projected[2]))

        # Drop sets must leave the survivors in general position. Removing
        # 1, 5 and 7 would leave four of the five on the 9/189-degree
        # diameter -- collinear, so the homography is not unique and the two
        # implementations pick different, equally valid answers out of the
        # null space. A real degeneracy rather than a port bug, and exactly
        # what `convex_and_ordered` exists to reject.
        for drop in ([], [3], [1, 6], [4, 5, 6, 7]):
            partial = [None if i in drop else p for i, p in enumerate(image_points)]
            homography = estimate_homography(partial)
            projected_probes = [
                list(homography.to_image(np.array([probe]))[0]) for probe in probes
            ]
            cases.append({
                "image_points": partial,
                "probes_image": projected_probes,
                "expected_board": [
                    list(homography.to_board(np.array([p]))[0]) for p in projected_probes
                ],
            })
    return cases


def x01_cases() -> list[dict]:
    """Leg and match folds, weighted toward the busts.

    A bust is where a scorer goes wrong, and the three kinds do not look alike
    from the outside -- below zero, finishing on a single, and leaving one.
    """
    T20, T19, S20, S1 = Dart(20, 3), Dart(19, 3), Dart(20, 1), Dart(1, 1)
    nine = [T20] * 6 + [T20, T19, Dart(12, 2)]

    legs: list[tuple[list[Dart], X01Rules]] = [
        ([T20, T20, S20], X01Rules()),
        (nine, X01Rules()),
        (nine + [T20, T20], X01Rules()),                  # darts after the win
        ([T20], X01Rules(start=40)),                      # below zero
        ([Dart(13, 3)], X01Rules(start=40)),              # leaves 1
        ([S20, S20], X01Rules(start=40)),                 # finishes on a single
        ([Dart(20, 2)], X01Rules(start=40)),              # a clean checkout
        ([Dart(25, 2)], X01Rules(start=50)),              # the double bull
        ([Dart(25, 1)], X01Rules(start=50)),              # 25 is not a double
        ([S20, S20], X01Rules(start=40, double_out=False)),
        ([Dart(13, 3)], X01Rules(start=40, double_out=False)),
        ([T20, S20, T19, Dart(20, 2), T20], X01Rules(double_in=True)),
        ([T20, Dart(19, 1), T20], X01Rules(start=100)),   # bust mid-visit
        ([Dart(0, 0), T20, Dart(0, 0)], X01Rules()),      # misses
    ]

    cases = []
    for darts, rules in legs:
        state = replay(darts, rules)
        cases.append({
            "kind": "leg",
            "rules": {"start": rules.start, "doubleOut": rules.double_out,
                      "doubleIn": rules.double_in},
            "darts": [[d.segment, d.multiplier] for d in darts],
            "expected": {
                "remaining": state.remaining,
                "winner": state.winner,
                "darts_thrown": state.darts_thrown,
                "visit_score": state.visit_score,
                "three_dart_average": round(state.three_dart_average, 6),
                "outcomes": [r.outcome.value for r in state.results],
                "reasons": [r.reason for r in state.results],
                "notations": [r.dart.notation for r in state.results],
            },
        })

    # A whole match, including the throw alternating between legs.
    filler = [S1] * 3
    def leg_visits(a_first: bool) -> list[list[Dart]]:
        out: list[list[Dart]] = []
        for chunk in ([T20] * 3, [T20] * 3, [T20, T19, Dart(12, 2)]):
            if not a_first:
                out.append(filler)
            out.append(chunk)
            if a_first:
                out.append(filler)
        if a_first:
            out.pop()
        return out

    for legs_to_win, visits in (
        (1, leg_visits(True)),
        (2, leg_visits(True) + leg_visits(False)),
        (2, leg_visits(True) + leg_visits(False) + [[T20] * 3]),
    ):
        game = play(visits, players=("A", "B"), legs_to_win=legs_to_win)
        cases.append({
            "kind": "match",
            "legs_to_win": legs_to_win,
            "visits": [[[d.segment, d.multiplier] for d in v] for v in visits],
            "expected": {
                "remaining": game.remaining,
                "won_legs": game.won_legs,
                "winner": game.winner,
                "current_player": game.current_player,
                "leg_number": game.leg_number,
            },
        })
    return cases


def main(argv: list[str] | None = None) -> int:
    """Writes to `web/fixtures/parity.json`, or to a path given as argv[1].

    The override exists so `tests/test_web_parity.py` can regenerate into a
    temporary directory and compare, which is what keeps the fixtures honest.
    """
    argv = sys.argv[1:] if argv is None else argv
    out = (Path(argv[0]) if argv
           else Path(__file__).resolve().parent.parent / "web" / "fixtures" / "parity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "board": {
            "rBoard": BDO_BOARD.r_board, "rDouble": BDO_BOARD.r_double,
            "rTreble": BDO_BOARD.r_treble, "rOuterBull": BDO_BOARD.r_outer_bull,
            "rInnerBull": BDO_BOARD.r_inner_bull, "ringWidth": BDO_BOARD.ring_width,
        },
        "image_size": {"height": IMAGE[0], "width": IMAGE[1]},
        "scoring": scoring_cases(),
        "homography": homography_cases(),
        "framing": framing_cases(),
        "x01": x01_cases(),
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out} — {len(payload['scoring'])} scoring, "
          f"{len(payload['homography'])} homography, {len(payload['framing'])} framing, "
          f"{len(payload['x01'])} x01 cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
