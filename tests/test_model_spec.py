"""The input size has a floor, and this is what holds it there.

Compute scales with the input's *area*, so this is the first number anyone
lowers when a training run looks expensive. Lowering it past the floor does not
fail loudly -- the run completes and reports an accuracy, and that accuracy
describes a model #15's precision budget already excluded. A wasted campaign
looks exactly like a successful one until someone re-derives the arithmetic.

So the arithmetic lives in code and is asserted here. Change the default below
what the budget supports and this fails, with the reason attached.
"""

from __future__ import annotations

import math

import pytest

from dartvision.geometry.board import BDO_BOARD
from dartvision.model.spec import (
    DEFAULT_INPUT_PX,
    IDEAL_RING_PX,
    MIN_RING_PX,
    MOUNT_BAND_DEG,
    minimum_input_px,
    ring_width_px,
)


def test_the_default_clears_the_precision_budget_across_the_whole_mount_band():
    """#2's band is 45-65 degrees, and the *shallow* end is what binds. An
    input chosen for 65 fails the player who mounts at 45."""
    for elevation in (MOUNT_BAND_DEG[0], 55.0, MOUNT_BAND_DEG[1]):
        ring = ring_width_px(DEFAULT_INPUT_PX, elevation_deg=elevation)
        assert ring >= MIN_RING_PX, (
            f"at {elevation:g} degrees a 10 mm ring lands on {ring:.1f} px, under "
            f"#15's {MIN_RING_PX:g} px floor. Raise DEFAULT_INPUT_PX — training at "
            "this size measures a model the precision budget already rules out."
        )


def test_the_default_is_not_larger_than_it_needs_to_be():
    """The floor is a floor, not an excuse. Compute scales with area, so an
    input well past what the budget needs is money spent for nothing."""
    assert DEFAULT_INPUT_PX == minimum_input_px()


def test_the_sizes_that_were_ruled_out_stay_ruled_out():
    """512 and 640 were both proposed and both fail, for the same reason:
    a face-on calculation omits the tilt."""
    for rejected in (512, 640):
        assert ring_width_px(rejected, elevation_deg=MOUNT_BAND_DEG[0]) < MIN_RING_PX


def test_the_tilt_is_what_the_easy_calculation_misses():
    """The face-on arithmetic says 512 is borderline-but-arguable. Through a
    real mount it is not close. That gap is the whole point of this module."""
    face_on = BDO_BOARD.ring_width * 0.85 * 512 / (2 * BDO_BOARD.r_board)
    tilted = ring_width_px(512, elevation_deg=55.0)

    assert face_on == pytest.approx(9.7, abs=0.2)
    assert tilted == pytest.approx(face_on * math.sin(math.radians(55.0)), rel=1e-6)
    assert tilted < face_on * 0.85


def test_more_pixels_and_a_squarer_view_both_help():
    assert ring_width_px(896) > ring_width_px(768) > ring_width_px(512)
    assert ring_width_px(768, elevation_deg=65.0) > ring_width_px(768, elevation_deg=45.0)
    assert ring_width_px(768, board_fill=0.9) > ring_width_px(768, board_fill=0.6)


def test_a_face_on_view_loses_nothing():
    face_on = BDO_BOARD.ring_width * 0.85 * 768 / (2 * BDO_BOARD.r_board)
    assert ring_width_px(768, elevation_deg=90.0) == pytest.approx(face_on)


def test_margin_would_cost_a_bigger_input():
    """Recorded so the trade is visible: clearing the *comfortable* bar rather
    than the floor is another third of the compute again."""
    assert minimum_input_px(IDEAL_RING_PX) > DEFAULT_INPUT_PX


def test_the_default_divides_by_every_stride_the_model_offers():
    """The heatmap grid is the input divided by the stride; a remainder would
    misalign every target by a fraction of a cell."""
    for stride in (2, 4, 8):
        assert DEFAULT_INPUT_PX % stride == 0


@pytest.mark.parametrize("kwargs", [
    {"input_px": 0}, {"board_fill": 0.0}, {"board_fill": 1.5},
    {"elevation_deg": 0.0}, {"elevation_deg": 91.0},
])
def test_impossible_framings_are_refused(kwargs):
    with pytest.raises(ValueError):
        ring_width_px(**{"input_px": 768, **kwargs})


def test_the_shipped_defaults_all_agree():
    """Three modules used to say 512 independently, which is how they drift."""
    from dartvision.data.torch_dataset import SceneDataset
    from dartvision.train import TrainConfig
    import inspect

    assert TrainConfig(manifest="m", image_root="i").input_size == (
        DEFAULT_INPUT_PX, DEFAULT_INPUT_PX)
    assert inspect.signature(SceneDataset.__init__).parameters["input_size"].default == (
        DEFAULT_INPUT_PX, DEFAULT_INPUT_PX)
