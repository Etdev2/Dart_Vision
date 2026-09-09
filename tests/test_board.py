"""Tests for deterministic board geometry.

The positional tests are written against the physical layout of a real
dartboard (20 at the top, 3 at the bottom, 6 to the right, 11 to the left) so a
wrong sector order fails rather than merely differing from another
implementation.
"""

from __future__ import annotations

import math

import pytest

from dartvision.geometry import (
    BDO_BOARD,
    SECTORS_CLOCKWISE_FROM_20,
    BoardSpec,
    margin_to_nearest_boundary,
    score_at,
)


def polar(r_mm: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return r_mm * math.cos(a), r_mm * math.sin(a)


# --------------------------------------------------------------------------
# Sector layout
# --------------------------------------------------------------------------

def test_sectors_are_a_permutation_of_1_to_20():
    assert sorted(SECTORS_CLOCKWISE_FROM_20) == list(range(1, 21))


@pytest.mark.parametrize(
    ("angle_deg", "expected"),
    [
        (90.0, 20),    # straight up
        (270.0, 3),    # straight down
        (0.0, 6),      # straight right
        (180.0, 11),   # straight left
        (72.0, 1),     # first bed clockwise from 20
        (108.0, 5),    # first bed counter-clockwise from 20
    ],
)
def test_cardinal_beds_match_a_physical_board(angle_deg, expected):
    x, y = polar(140.0, angle_deg)
    assert score_at(x, y).segment == expected


def test_bed_centres_are_18_degrees_apart_and_clockwise():
    for i, expected in enumerate(SECTORS_CLOCKWISE_FROM_20):
        x, y = polar(140.0, 90.0 - 18.0 * i)
        assert score_at(x, y).segment == expected


def test_every_bed_is_reachable_and_beds_span_18_degrees():
    seen: dict[int, int] = {}
    for tenth_degree in range(3600):
        x, y = polar(140.0, tenth_degree / 10.0)
        seen[score_at(x, y).segment] = seen.get(score_at(x, y).segment, 0) + 1
    assert sorted(seen) == list(range(1, 21))
    assert set(seen.values()) == {180}  # 18 degrees at 0.1 degree steps


# --------------------------------------------------------------------------
# Radial bands
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("r_mm", "notation", "score"),
    [
        (0.0, "DB", 50),
        (6.0, "DB", 50),
        (10.0, "SB", 25),
        (15.0, "SB", 25),
        (50.0, "S20", 20),
        (100.0, "T20", 60),
        (130.0, "S20", 20),
        (165.0, "D20", 40),
        (175.0, "MISS", 0),
        (300.0, "MISS", 0),
    ],
)
def test_radial_bands_on_the_20_bed(r_mm, notation, score):
    hit = score_at(*polar(r_mm, 90.0))
    assert (hit.notation, hit.score) == (notation, score)


def test_treble_and_double_rings_are_ring_width_wide():
    board = BDO_BOARD
    eps = 1e-6
    for inner, outer, prefix in [
        (board.r_treble_inner, board.r_treble, "T"),
        (board.r_double_inner, board.r_double, "D"),
    ]:
        assert score_at(*polar(inner + eps, 90.0)).notation == f"{prefix}20"
        assert score_at(*polar(outer - eps, 90.0)).notation == f"{prefix}20"
        assert score_at(*polar(inner - eps, 90.0)).notation != f"{prefix}20"
        assert outer - inner == pytest.approx(board.ring_width)


def test_bull_ignores_the_sector_angle():
    scores = {score_at(*polar(3.0, a)).notation for a in range(0, 360, 7)}
    assert scores == {"DB"}


def test_maximum_score_is_the_treble_twenty():
    best = max(
        score_at(*polar(r / 2.0, a / 2.0)).score
        for r in range(0, 360)
        for a in range(0, 720)
    )
    assert best == 60


# --------------------------------------------------------------------------
# Margin to nearest boundary
# --------------------------------------------------------------------------

def test_margin_is_zero_on_a_radial_boundary():
    for b in BDO_BOARD.radial_boundaries:
        assert margin_to_nearest_boundary(*polar(b, 90.0)) == pytest.approx(0.0, abs=1e-9)


def test_margin_is_zero_on_a_sector_wire():
    # Wire between the 20 and 1 beds sits at 81 degrees.
    assert margin_to_nearest_boundary(*polar(140.0, 81.0)) == pytest.approx(0.0, abs=1e-9)


def test_margin_mid_bed_is_limited_by_the_sector_wire_not_the_rings():
    """Mid-bed, mid-radius, the angular margin binds before the radial one.

    Half-way between the treble and double rings the nearer ring is 26.3 mm
    away, but the sector wire is only 21.0 mm away along the arc. This is worth
    knowing: for singles, angular precision is the binding constraint, while
    radial precision dominates only near the rings.
    """
    r = (BDO_BOARD.r_treble + BDO_BOARD.r_double_inner) / 2.0
    radial = min(r - BDO_BOARD.r_treble, BDO_BOARD.r_double_inner - r)
    angular = r * math.radians(9.0)  # bed centre is 9 degrees from either wire
    assert angular < radial
    assert margin_to_nearest_boundary(*polar(r, 90.0)) == pytest.approx(angular)


def test_margin_inside_the_bull_ignores_sector_wires():
    # On a wire angle but well inside the bull: only radial boundaries count.
    r = 10.0
    expected = min(r - BDO_BOARD.r_inner_bull, BDO_BOARD.r_outer_bull - r)
    assert margin_to_nearest_boundary(*polar(r, 81.0)) == pytest.approx(expected)


# Points closer than this to a boundary are knife-edge cases: the margin is
# legitimately ~0, and floating-point error in the polar round-trip makes the
# baseline score itself ambiguous. The margin property is vacuous there.
BOUNDARY_EPS_MM = 1e-6


def test_a_displacement_smaller_than_the_margin_never_changes_the_score():
    """The defining property of the margin, checked over the scoring area."""
    checked = 0
    for r in range(2, 172, 3):
        for angle in range(0, 360, 7):
            x, y = polar(float(r), float(angle))
            baseline = score_at(x, y)
            margin = margin_to_nearest_boundary(x, y)
            if margin <= BOUNDARY_EPS_MM:
                continue
            step = margin * 0.98
            checked += 1
            for d in range(0, 360, 15):
                dx, dy = polar(step, float(d))
                moved = score_at(x + dx, y + dy)
                assert moved == baseline, (
                    f"r={r} angle={angle} margin={margin:.6f} "
                    f"moved {d} deg: {baseline.notation} -> {moved.notation}"
                )
    assert checked > 2000, "sampling grid degenerated"


def test_margin_is_tight_at_least_one_direction_crosses_just_beyond_it():
    for r in range(20, 170, 7):
        for angle in range(0, 360, 11):
            x, y = polar(float(r), float(angle))
            baseline = score_at(x, y)
            margin = margin_to_nearest_boundary(x, y)
            if margin <= BOUNDARY_EPS_MM:
                continue
            step = margin * 1.05
            changed = any(
                score_at(x + dx, y + dy) != baseline
                for dx, dy in (polar(step, float(d)) for d in range(0, 360, 3))
            )
            assert changed, f"margin not tight at r={r} angle={angle}"


# --------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------

def test_default_spec_matches_published_bdo_dimensions():
    b = BDO_BOARD
    assert (b.r_board, b.r_double, b.r_treble) == (225.5, 170.0, 107.4)
    assert (b.r_outer_bull, b.r_inner_bull, b.ring_width) == (15.9, 6.35, 10.0)
    assert b.r_treble_inner == pytest.approx(97.4)
    assert b.r_double_inner == pytest.approx(160.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r_double": 100.0},        # double ring inside the treble ring
        {"r_inner_bull": 20.0},     # bullseye outside the 25 ring
        {"ring_width": 0.0},        # degenerate rings
        {"r_board": 100.0},         # scoring area outside the board
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BoardSpec(**kwargs)
