"""Tests for the annotation tooling.

The point of #26's tooling is that labelling is affordable, so the tests check
the labour saving explicitly rather than only the correctness of the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from dartvision.annotate import (
    BurstError,
    CaptureSession,
    effort_saved,
    expand_burst,
    plan_placed_capture,
    render_plan,
)
from dartvision.data.labels import Origin
from dartvision.geometry.board import BDO_BOARD, margin_to_nearest_boundary

LANDMARKS_8 = tuple((0.1 + 0.1 * i, 0.2 + 0.05 * i) for i in range(8))
TIPS = ((0.50, 0.40), (0.52, 0.43), (0.48, 0.41))


def session(**overrides) -> CaptureSession:
    base = dict(
        setup_id="garage",
        session_id="garage/2026-09-09-a",
        landmarks=LANDMARKS_8,
        image_size=(3024, 4032),
    )
    base.update(overrides)
    return CaptureSession(**base)


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

def test_session_requires_enough_landmarks_to_score_anything():
    with pytest.raises(ValueError, match="at least 4 landmarks"):
        session(landmarks=(LANDMARKS_8[0], None, None, None))


def test_session_rejects_non_standard_landmark_counts():
    with pytest.raises(ValueError, match="4 or 8"):
        session(landmarks=LANDMARKS_8[:5])


@pytest.mark.parametrize("field", ["setup_id", "session_id"])
def test_session_identity_is_required(field):
    with pytest.raises(ValueError, match="required"):
        session(**{field: ""})


# --------------------------------------------------------------------------
# Bursts: the labour saving
# --------------------------------------------------------------------------

def test_a_burst_starting_empty_labels_zero_to_three_darts():
    frames = ["f0", "f1", "f2", "f3"]
    annotations = expand_burst(session(), frames, TIPS)

    assert [len(a.tips) for a in annotations] == [0, 1, 2, 3]
    assert annotations[-1].tips == TIPS
    assert annotations[2].tips == TIPS[:2]


def test_a_burst_starting_after_the_first_throw_is_inferred():
    annotations = expand_burst(session(), ["f1", "f2", "f3"], TIPS)
    assert [len(a.tips) for a in annotations] == [1, 2, 3]


def test_landmarks_are_shared_by_every_frame_in_the_session():
    annotations = expand_burst(session(), ["f0", "f1", "f2", "f3"], TIPS)
    assert {a.landmarks for a in annotations} == {LANDMARKS_8}
    assert {a.session_id for a in annotations} == {"garage/2026-09-09-a"}


def test_frames_record_their_position_and_dart_count():
    annotations = expand_burst(session(), ["f0", "f1"], TIPS[:1])
    assert [a.meta["burst_position"] for a in annotations] == [0, 1]
    assert [a.meta["darts_in_frame"] for a in annotations] == [0, 1]


def test_per_frame_metadata_is_merged():
    annotations = expand_burst(
        session(), ["f0", "f1"], TIPS[:1],
        frame_meta=[{"lighting": "dim"}, {"lighting": "dim", "glare": True}],
    )
    assert annotations[1].meta["glare"] is True


def test_origin_can_mark_a_placed_dart_capture():
    annotations = expand_burst(session(), ["f1"], TIPS[:1], origin=Origin.REAL_PLACED)
    assert annotations[0].origin is Origin.REAL_PLACED


def test_more_frames_than_darts_is_rejected_with_a_useful_message():
    with pytest.raises(BurstError, match="one dart per frame"):
        expand_burst(session(), ["f0", "f1", "f2", "f3", "f4"], TIPS)


def test_a_burst_needs_frames():
    with pytest.raises(BurstError, match="at least one frame"):
        expand_burst(session(), [], TIPS)


def test_a_visit_cannot_hold_more_than_three_darts():
    with pytest.raises(BurstError, match="at most 3 tips"):
        expand_burst(session(), ["a"], TIPS + ((0.3, 0.3),))


def test_frame_meta_length_is_checked():
    with pytest.raises(BurstError, match="one entry per frame"):
        expand_burst(session(), ["f0", "f1"], TIPS[:1], frame_meta=[{}])


def test_the_saving_is_real_and_measured():
    """8 landmark clicks plus 3 tip clicks should label a four-frame burst."""
    annotations = expand_burst(session(), ["f0", "f1", "f2", "f3"], TIPS)
    saved = effort_saved(annotations, landmark_clicks=8)

    assert saved["frames"] == 4
    assert saved["clicks_spent"] == 11
    assert saved["clicks_if_labelled_per_frame"] == 38  # 4*8 landmarks + 6 tips
    assert saved["clicks_saved"] == 27


def test_the_saving_grows_with_session_length():
    """Landmarks are paid for once, so longer sessions amortise them away."""
    short = effort_saved(expand_burst(session(), ["a", "b"], TIPS[:1]), 8)
    ratios = []
    for visits in (1, 10):
        frames, spent = 0, 8
        for visit in range(visits):
            annotations = expand_burst(
                session(), [f"v{visit}-{i}" for i in range(4)], TIPS
            )
            frames += len(annotations)
            spent += 3
        naive = frames * 8 + visits * 6
        ratios.append(naive / spent)
    assert ratios[1] > ratios[0] > 1.0
    assert short["clicks_saved"] > 0


# --------------------------------------------------------------------------
# Placed-dart plans
# --------------------------------------------------------------------------

def test_plan_hits_every_requested_margin():
    rng = np.random.default_rng(4)
    targets = plan_placed_capture(rng, margins_mm=(0.25, 1.0), per_margin=6)

    assert len(targets) == 12
    for target in targets:
        actual = margin_to_nearest_boundary(*target.board_xy)
        assert actual == pytest.approx(target.target_margin_mm, abs=0.05)
        assert target.boundary.distance_mm == pytest.approx(actual, abs=1e-9)


def test_plan_targets_are_on_the_board_and_score_something():
    rng = np.random.default_rng(5)
    for target in plan_placed_capture(rng, per_margin=4):
        assert np.hypot(*target.board_xy) <= BDO_BOARD.r_double
        assert target.hit.notation != "MISS"


def test_instructions_name_the_boundary_and_a_locating_radius():
    rng = np.random.default_rng(6)
    target = plan_placed_capture(rng, margins_mm=(0.5,), per_margin=1)[0]
    text = target.instruction
    assert target.boundary.name in text
    assert "mm out from the bull" in text
    assert target.hit.notation in text


def test_plan_covers_both_radial_and_sector_boundaries():
    """Both must be exercised: they are different failure modes."""
    rng = np.random.default_rng(7)
    kinds = {t.boundary.kind for t in plan_placed_capture(rng, per_margin=25)}
    assert kinds == {"radial", "sector"}


def test_rendered_plan_is_a_usable_checklist():
    rng = np.random.default_rng(8)
    targets = plan_placed_capture(rng, margins_mm=(0.25, 1.0), per_margin=3)
    text = render_plan(targets)

    assert "Placed-dart capture plan" in text
    assert "margin 0.25 mm" in text and "margin 1.0 mm" in text
    for i in range(1, 7):
        assert f"{i:>3}." in text
    assert "6 targets" in text


def test_empty_plan_renders_without_crashing():
    assert render_plan([]) == "no targets"


@pytest.mark.parametrize("kwargs", [{"per_margin": 0}, {"margins_mm": ()}])
def test_invalid_plans_are_rejected(kwargs):
    with pytest.raises(ValueError):
        plan_placed_capture(np.random.default_rng(0), **kwargs)


def test_plan_cli_prints_a_checklist(capsys):
    from dartvision.annotate.plan import main

    assert main(["--margins", "0.5,2", "--per-margin", "2", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Placed-dart capture plan" in out and "4 targets" in out


def test_plan_cli_writes_a_file(tmp_path):
    from dartvision.annotate.plan import main

    path = tmp_path / "plan.txt"
    assert main(["--per-margin", "1", "--out", str(path)]) == 0
    assert "margin 0.25 mm" in path.read_text()
