"""Tests for target encoding and decoding.

Target encoding is where keypoint pipelines quietly go wrong: a mistake trains
happily on subtly wrong targets and fails for reasons no later metric blames on
the labels. So the central tests here are round trips -- encode a known point,
decode it, and require the original back.
"""

from __future__ import annotations

import numpy as np
import pytest

from dartvision.model import (
    decode_landmark_heatmaps,
    decode_tip_targets,
    encode_landmark_heatmaps,
    encode_tip_targets,
    recommended_window,
    soft_argmax,
)

GRID = (128, 128)


def cell_error(a, b, grid=GRID) -> float:
    return float(np.hypot((a[0] - b[0]) * grid[1], (a[1] - b[1]) * grid[0]))


# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------

def test_encoding_shape_and_peak_location():
    heatmaps = encode_landmark_heatmaps([(0.25, 0.75), (0.5, 0.5)], GRID, sigma_cells=2.0)
    assert heatmaps.shape == (2, *GRID)
    assert heatmaps.dtype == np.float32
    peak = np.unravel_index(int(np.argmax(heatmaps[0])), GRID)
    assert peak == (int(0.75 * GRID[0]), int(0.25 * GRID[1]))


def test_a_missing_landmark_encodes_to_an_empty_channel_and_decodes_to_none():
    heatmaps = encode_landmark_heatmaps([(0.4, 0.4), None], GRID)
    assert heatmaps[1].max() == 0.0
    decoded = decode_landmark_heatmaps(heatmaps, window=recommended_window(2.0))
    assert decoded[0][0] is not None
    assert decoded[1][0] is None


@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
def test_round_trip_is_accurate_at_the_recommended_window(sigma):
    """The property that matters: decode(encode(p)) == p, to well under a cell."""
    rng = np.random.default_rng(2)
    window = recommended_window(sigma)
    worst = 0.0
    # A sub-cell peak is never sampled exactly, so the worst attainable peak
    # value is a point sitting at a cell corner: exp(-(0.5^2 + 0.5^2)/(2s^2)).
    floor_score = float(np.exp(-0.25 / sigma ** 2))
    for _ in range(200):
        point = (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))
        heatmaps = encode_landmark_heatmaps([point], GRID, sigma_cells=sigma)
        (decoded, score), = decode_landmark_heatmaps(heatmaps, window=window)
        worst = max(worst, cell_error(decoded, point))
        assert floor_score - 1e-6 <= score <= 1.0
    assert worst < 0.05, f"sigma={sigma} window={window} worst={worst:.4f} cells"


def test_peak_height_encodes_sub_cell_position_and_is_not_a_confidence():
    """A caveat for #21's calibration work.

    The sampled peak of a Gaussian centred between cells is lower than one
    centred on a cell, so peak height varies with sub-cell position even for a
    perfect prediction. At sigma=1 that spans roughly 0.78 to 1.00 -- a large
    apparent confidence swing carrying no information about correctness. The
    effect shrinks as sigma grows.
    """
    on_cell = encode_landmark_heatmaps([(32 / GRID[1], 32 / GRID[0])], GRID, sigma_cells=1.0)
    between = encode_landmark_heatmaps(
        [(32.5 / GRID[1], 32.5 / GRID[0])], GRID, sigma_cells=1.0
    )
    assert float(on_cell.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(between.max()) < 0.79

    wide_on = encode_landmark_heatmaps([(32 / GRID[1], 32 / GRID[0])], GRID, sigma_cells=3.0)
    wide_between = encode_landmark_heatmaps(
        [(32.5 / GRID[1], 32.5 / GRID[0])], GRID, sigma_cells=3.0
    )
    assert float(wide_on.max()) - float(wide_between.max()) < 0.03


def test_a_too_small_window_is_biased_not_merely_noisy():
    """Documents why recommended_window exists rather than a fixed default."""
    rng = np.random.default_rng(3)
    tight, generous = [], []
    for _ in range(200):
        point = (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))
        heatmaps = encode_landmark_heatmaps([point], GRID, sigma_cells=3.0)
        (a, _), = decode_landmark_heatmaps(heatmaps, window=5)
        (b, _), = decode_landmark_heatmaps(heatmaps, window=recommended_window(3.0))
        tight.append(cell_error(a, point))
        generous.append(cell_error(b, point))
    assert np.mean(tight) > 10 * np.mean(generous)


def test_recommended_window_is_odd_and_covers_three_sigma():
    for sigma in (0.5, 1.0, 2.0, 3.5):
        window = recommended_window(sigma)
        assert window % 2 == 1
        assert window // 2 >= 3 * sigma - 1e-9
    with pytest.raises(ValueError):
        recommended_window(0.0)


def test_soft_argmax_ignores_distant_activity():
    """A second blob elsewhere must not drag the estimate, which a global
    soft-argmax would allow."""
    point = (0.3, 0.3)
    heat = encode_landmark_heatmaps([point], GRID, sigma_cells=2.0)[0].copy()
    heat += 0.4 * encode_landmark_heatmaps([(0.8, 0.8)], GRID, sigma_cells=2.0)[0]
    decoded, _ = soft_argmax(heat, window=recommended_window(2.0))
    assert cell_error(decoded, point) < 0.05


def test_soft_argmax_rejects_bad_windows():
    heat = np.zeros(GRID, dtype=np.float32)
    for bad in (0, -1, 4):
        with pytest.raises(ValueError, match="odd"):
            soft_argmax(heat, window=bad)
    with pytest.raises(ValueError, match="2-D"):
        soft_argmax(np.zeros((1, *GRID)), window=5)


def test_flat_channel_degrades_gracefully():
    point, score = soft_argmax(np.zeros(GRID, dtype=np.float32), window=5)
    assert score == 0.0 and 0.0 <= point[0] <= 1.0


# --------------------------------------------------------------------------
# Tips
# --------------------------------------------------------------------------

def test_tip_round_trip_is_exact_because_offsets_are_explicit():
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(300):
        point = (float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95)))
        targets = encode_tip_targets([point], GRID)
        (decoded, _), = decode_tip_targets(targets.heat, targets.offset)
        worst = max(worst, cell_error(decoded, point))
    assert worst < 1e-5, f"worst {worst:.2e} cells"


def test_three_separated_tips_all_come_back():
    points = [(0.25, 0.3), (0.6, 0.4), (0.45, 0.75)]
    targets = encode_tip_targets(points, GRID, sigma_cells=1.5)
    assert targets.count == 3

    decoded = decode_tip_targets(targets.heat, targets.offset, max_detections=3)
    assert len(decoded) == 3
    for point in points:
        assert min(cell_error(d, point) for d, _ in decoded) < 1e-5


def test_an_empty_board_encodes_and_decodes_to_nothing():
    targets = encode_tip_targets([], GRID)
    assert targets.count == 0 and targets.heat.max() == 0.0
    assert decode_tip_targets(targets.heat, targets.offset) == []


def test_two_tips_in_one_cell_are_reported_as_a_lost_instance():
    """The representation cannot express it, so the count must say so rather
    than the encoder pretending otherwise."""
    close = [(0.5000, 0.5000), (0.50002, 0.50002)]
    targets = encode_tip_targets(close, GRID, sigma_cells=1.0)
    assert targets.count == 1 < len(close)


def test_clustered_but_resolvable_tips_survive():
    """Darts a few cells apart -- the case #14 cares about -- must stay separate."""
    points = [(0.50, 0.50), (0.50 + 4 / GRID[1], 0.50 + 4 / GRID[0])]
    targets = encode_tip_targets(points, GRID, sigma_cells=1.0)
    decoded = decode_tip_targets(targets.heat, targets.offset, threshold=0.2)
    assert targets.count == 2 and len(decoded) == 2


def test_detections_are_capped_and_ordered_by_score():
    points = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)]
    targets = encode_tip_targets(points, GRID, sigma_cells=1.5)
    decoded = decode_tip_targets(targets.heat, targets.offset, max_detections=2)
    assert len(decoded) == 2
    assert decoded[0][1] >= decoded[1][1]


def test_threshold_suppresses_weak_peaks():
    targets = encode_tip_targets([(0.4, 0.6)], GRID, sigma_cells=1.5)
    assert decode_tip_targets(targets.heat * 0.1, targets.offset, threshold=0.3) == []


@pytest.mark.parametrize("kwargs", [{"sigma_cells": 0.0}, {"sigma_cells": -1.0}])
def test_invalid_sigma_is_rejected(kwargs):
    with pytest.raises(ValueError, match="sigma_cells"):
        encode_tip_targets([(0.5, 0.5)], GRID, **kwargs)
    with pytest.raises(ValueError, match="sigma_cells"):
        encode_landmark_heatmaps([(0.5, 0.5)], GRID, **kwargs)


def test_decode_validates_its_inputs():
    targets = encode_tip_targets([(0.5, 0.5)], GRID)
    with pytest.raises(ValueError, match="offset must be"):
        decode_tip_targets(targets.heat, targets.offset[:, :4, :4])
    with pytest.raises(ValueError, match="at least 1"):
        decode_tip_targets(targets.heat, targets.offset, max_detections=0)


def test_points_at_the_frame_edge_stay_in_range():
    for point in [(0.0, 0.0), (0.999, 0.999)]:
        targets = encode_tip_targets([point], GRID)
        (decoded, _), = decode_tip_targets(targets.heat, targets.offset)
        assert 0.0 <= decoded[0] <= 1.0 and 0.0 <= decoded[1] <= 1.0
        assert cell_error(decoded, point) < 1e-5
