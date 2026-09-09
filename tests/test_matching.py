"""Tests for tip matching."""

from __future__ import annotations

import pytest

from dartvision.eval.matching import match_tips


def test_identical_sets_pair_exactly():
    tips = [(0.0, 100.0), (50.0, -30.0), (-80.0, 20.0)]
    matches = match_tips(tips, tips)
    assert len(matches) == 3
    assert all(m.is_paired and m.distance_mm == pytest.approx(0.0) for m in matches)


def test_nearest_pairing_wins_over_input_order():
    truth = [(0.0, 0.0), (100.0, 0.0)]
    predictions = [(99.0, 0.0), (1.0, 0.0)]   # reversed
    pairs = {(m.truth_index, m.prediction_index) for m in match_tips(truth, predictions)}
    assert pairs == {(0, 1), (1, 0)}


def test_a_lone_real_dart_is_a_miss_not_a_bad_match():
    matches = match_tips([(0.0, 0.0)], [])
    assert len(matches) == 1 and matches[0].is_missed
    assert not matches[0].is_phantom


def test_a_lone_prediction_is_a_phantom():
    matches = match_tips([], [(0.0, 0.0)])
    assert len(matches) == 1 and matches[0].is_phantom


def test_a_distant_prediction_is_a_phantom_plus_a_miss():
    """Without the threshold this would score as one huge localization error."""
    matches = match_tips([(0.0, 0.0)], [(200.0, 0.0)], max_distance_mm=25.0)
    assert len(matches) == 2
    assert sum(m.is_missed for m in matches) == 1
    assert sum(m.is_phantom for m in matches) == 1
    assert not any(m.is_paired for m in matches)


def test_a_close_prediction_inside_the_threshold_pairs():
    matches = match_tips([(0.0, 0.0)], [(20.0, 0.0)], max_distance_mm=25.0)
    assert len(matches) == 1 and matches[0].is_paired
    assert matches[0].distance_mm == pytest.approx(20.0)


def test_clustered_darts_are_not_double_matched():
    """Two predictions near one dart must not both claim it."""
    truth = [(0.0, 0.0), (10.0, 0.0)]
    predictions = [(1.0, 0.0), (2.0, 0.0)]
    matches = match_tips(truth, predictions)
    paired = [m for m in matches if m.is_paired]
    assert len(paired) == 2
    assert len({m.truth_index for m in paired}) == 2
    assert len({m.prediction_index for m in paired}) == 2


def test_every_truth_and_prediction_appears_exactly_once():
    truth = [(0.0, 0.0), (60.0, 0.0), (300.0, 300.0)]
    predictions = [(2.0, 0.0), (61.0, 0.0)]
    matches = match_tips(truth, predictions)

    seen_truth = [m.truth_index for m in matches if m.truth_index is not None]
    seen_prediction = [m.prediction_index for m in matches if m.prediction_index is not None]
    assert sorted(seen_truth) == [0, 1, 2]
    assert sorted(seen_prediction) == [0, 1]


def test_empty_on_both_sides_yields_nothing():
    assert match_tips([], []) == []


def test_threshold_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        match_tips([(0.0, 0.0)], [(0.0, 0.0)], max_distance_mm=0.0)
