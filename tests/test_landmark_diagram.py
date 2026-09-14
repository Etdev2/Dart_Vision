"""The landmark diagram.

It exists to be believed by someone about to click 184 images, so what is
tested is that it cannot quietly disagree with the geometry it documents: the
points it draws are the points the annotator asks for, and the wire labels name
the beds those wires actually separate.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from dartvision.geometry.board import BDO_BOARD, score_at
from dartvision.geometry.calibration import CALIBRATION_ANGLES_DEG, calibration_points_8

import landmark_diagram

SVG = Path(__file__).resolve().parent.parent / "docs" / "landmarks.svg"


def test_the_labels_name_the_beds_those_wires_separate():
    """A wrong label here puts every landmark in the wrong place, and the error
    would only surface as a model that never quite learns."""
    labels = {9.0: {"13", "6"}, 99.0: {"20", "5"}, 189.0: {"8", "11"},
              279.0: {"17", "3"}}
    radius = (BDO_BOARD.r_treble + BDO_BOARD.r_treble_inner) / 2

    for angle in CALIBRATION_ANGLES_DEG:
        either_side = set()
        for nudge in (+0.8, -0.8):
            a = math.radians(angle + nudge)
            either_side.add(str(score_at(radius * math.cos(a),
                                         radius * math.sin(a)).segment))
        assert either_side == labels[angle], f"at {angle} deg"


def test_the_drawn_points_are_the_ones_the_annotator_asks_for():
    svg = landmark_diagram.build()
    marks = re.findall(
        r'<circle class="landmark" cx="(-?\d+\.\d+)" cy="(-?\d+\.\d+)"', svg
    )
    drawn = [(float(x), -float(y)) for x, y in marks]      # SVG y grows downward
    expected = calibration_points_8(BDO_BOARD)

    assert len(drawn) == 8
    for (dx, dy), (ex, ey) in zip(drawn, expected):
        assert math.hypot(dx - ex, dy - ey) < 0.05


def test_the_first_four_are_the_double_ring_and_the_rest_the_treble():
    svg = landmark_diagram.build()
    order = re.findall(r'class="landmark"[^/]*fill="#(dc2626|2563eb)"', svg)

    assert order == ["dc2626"] * 4 + ["2563eb"] * 4, (
        "colour carries the meaning: red is the outer ring, blue the treble"
    )


def test_everything_drawn_is_inside_the_frame():
    """Clipped labels were the first two attempts at this drawing."""
    svg = landmark_diagram.build()
    half_w, top, bottom = (landmark_diagram.HALF_W, landmark_diagram.TOP,
                           landmark_diagram.BOTTOM)

    for x, y in re.findall(r'<text x="(-?\d+\.?\d*)" y="(-?\d+\.?\d*)"', svg):
        assert -half_w < float(x) < half_w, f"text at x={x} runs off the frame"
        assert -top < float(y) < bottom, f"text at y={y} runs off the frame"


def test_the_committed_file_matches_the_generator():
    """Generated, so it cannot drift from the geometry — which only holds if
    the copy in the repository is the one the generator produces."""
    assert SVG.exists(), "run tools/landmark_diagram.py"
    assert SVG.read_text(encoding="utf-8") == landmark_diagram.build()
