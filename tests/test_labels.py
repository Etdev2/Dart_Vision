"""Tests for the annotation contract."""

from __future__ import annotations

import json

import pytest

from dartvision.data import Annotation, Origin, read_jsonl, sessions_by_setup, write_jsonl


def make(**overrides) -> Annotation:
    base = dict(
        image_id="img-1",
        setup_id="setup-a",
        session_id="setup-a/2026-09-09",
        origin=Origin.SYNTHETIC,
        landmarks=((0.2, 0.3), (0.8, 0.3), (0.8, 0.7), (0.2, 0.7)),
        tips=((0.5, 0.4),),
    )
    base.update(overrides)
    return Annotation(**base)


def test_round_trips_through_jsonl(tmp_path):
    annotations = [make(), make(image_id="img-2", tips=((0.4, 0.4), (0.6, 0.6)))]
    path = tmp_path / "manifest.jsonl"
    assert write_jsonl(annotations, path) == 2
    assert list(read_jsonl(path)) == annotations


def test_round_trips_with_missing_landmarks_and_metadata(tmp_path):
    annotation = make(
        landmarks=((0.2, 0.3), None, (0.8, 0.7), (0.2, 0.7)),
        image_size=(1200, 1600),
        meta={"lighting": "dim", "camera": "iphone"},
    )
    path = tmp_path / "m.jsonl"
    write_jsonl([annotation], path)
    assert list(read_jsonl(path))[0] == annotation


def test_missing_landmark_is_none_not_a_sentinel_coordinate(tmp_path):
    """A landmark just off the frame edge must stay distinguishable from absent."""
    off_frame = make(landmarks=((-0.02, 0.3), (0.8, 0.3), (0.8, 0.7), (0.2, 0.7)))
    absent = make(landmarks=(None, (0.8, 0.3), (0.8, 0.7), (0.2, 0.7)))
    assert off_frame.visible_landmarks == 4
    assert absent.visible_landmarks == 3
    assert off_frame.is_calibratable and not absent.is_calibratable


def test_eight_landmarks_are_accepted():
    eight = tuple((0.1 * i, 0.1 * i) for i in range(8))
    assert make(landmarks=eight).visible_landmarks == 8


@pytest.mark.parametrize("count", [0, 3, 5, 7, 9])
def test_rejects_non_standard_landmark_counts(count):
    with pytest.raises(ValueError, match="landmarks"):
        make(landmarks=tuple((0.1, 0.1) for _ in range(count)))


def test_rejects_more_than_three_tips():
    with pytest.raises(ValueError, match="at most 3"):
        make(tips=tuple((0.1 * i, 0.2) for i in range(4)))


@pytest.mark.parametrize("field", ["image_id", "setup_id", "session_id"])
def test_identity_fields_are_required(field):
    with pytest.raises(ValueError, match="required"):
        make(**{field: ""})


def test_rejects_non_finite_coordinates():
    with pytest.raises(ValueError, match="non-finite"):
        make(tips=((float("nan"), 0.5),))
    with pytest.raises(ValueError, match="non-finite"):
        make(tips=((float("inf"), 0.5),))


def test_read_error_names_the_offending_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(make().to_dict()) + "\n"
        + json.dumps({**make().to_dict(), "landmarks": [[0.1, 0.1]]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        list(read_jsonl(path))


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "gaps.jsonl"
    path.write_text("\n" + json.dumps(make().to_dict()) + "\n\n", encoding="utf-8")
    assert len(list(read_jsonl(path))) == 1


def test_sessions_group_under_their_setup():
    annotations = [
        make(image_id="a", setup_id="s1", session_id="s1/day1"),
        make(image_id="b", setup_id="s1", session_id="s1/day2"),
        make(image_id="c", setup_id="s2", session_id="s2/day1"),
    ]
    assert sessions_by_setup(annotations) == {
        "s1": {"s1/day1", "s1/day2"},
        "s2": {"s2/day1"},
    }


def test_origin_distinguishes_synthetic_from_real():
    assert not Origin.SYNTHETIC.is_real
    assert Origin.REAL_THROWN.is_real and Origin.REAL_PLACED.is_real
