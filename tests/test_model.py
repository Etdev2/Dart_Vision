"""Tests for the network, heads and losses.

The load-bearing test is the overfit: encode targets from a known scene, train
on that single batch, and require the decoded predictions to come back as the
points that went in. That exercises the whole chain -- target encoding, the
backbone, both heads, both losses, the optimiser and the decoders -- and is the
only way to catch a wiring error that otherwise shows up as a model that trains
smoothly to nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from dartvision.model import (  # noqa: E402
    decode_landmark_heatmaps,
    decode_tip_targets,
    encode_landmark_heatmaps,
    encode_tip_targets,
    recommended_window,
)
from dartvision.model.losses import (  # noqa: E402
    channel_peak_mask,
    heatmap_focal_loss,
    offset_l1_loss,
)
from dartvision.model.net import BrainOutputs, DartVisionBrain  # noqa: E402

IMAGE = (128, 128)
GRID = (32, 32)  # stride 4


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DartVisionBrain(backbone="resnet18", landmarks=8, width=32, stride=4)


# --------------------------------------------------------------------------
# Shapes and wiring
# --------------------------------------------------------------------------

def test_output_grid_matches_the_declared_stride(model):
    out = model(torch.zeros(2, 3, *IMAGE))
    assert isinstance(out, BrainOutputs)
    assert out.landmark_heat.shape == (2, 8, *GRID)
    assert out.tip_heat.shape == (2, 1, *GRID)
    assert out.tip_offset.shape == (2, 2, *GRID)
    assert out.grid == GRID == model.grid_for(IMAGE)


def test_offsets_are_bounded_to_the_encoders_range(model):
    """The encoder emits cx - round(cx), so anything outside +/-0.5 is wrong."""
    out = model(torch.randn(2, 3, *IMAGE) * 4.0)
    assert out.tip_offset.abs().max().item() <= 0.5 + 1e-6


def test_four_landmark_variant_changes_only_that_head():
    small = DartVisionBrain(backbone="resnet18", landmarks=4, width=16, stride=4)
    out = small(torch.zeros(1, 3, *IMAGE))
    assert out.landmark_heat.shape[1] == 4
    assert out.tip_heat.shape[1] == 1


def test_rejects_a_stride_the_backbone_cannot_provide():
    with pytest.raises(ValueError, match="stride"):
        DartVisionBrain(backbone="resnet18", stride=3)


def test_rejects_a_non_standard_landmark_count():
    with pytest.raises(ValueError, match="4 or 8"):
        DartVisionBrain(backbone="resnet18", landmarks=6)


def test_model_satisfies_the_brain_protocol(model):
    from dartvision.model.net import Brain

    assert isinstance(model, Brain)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def test_positives_are_found_even_though_targets_never_reach_one():
    """A sub-cell peak tops out near 0.78 at sigma=1, so `target == 1` finds none."""
    heat = encode_landmark_heatmaps([(0.5031, 0.4967)], GRID, sigma_cells=1.0)
    target = torch.from_numpy(heat).unsqueeze(0)
    assert target.max().item() < 1.0
    assert channel_peak_mask(target).sum().item() == 1


def test_an_absent_landmark_channel_has_no_positives():
    heat = encode_landmark_heatmaps([(0.4, 0.4), None], GRID)
    target = torch.from_numpy(heat).unsqueeze(0)
    mask = channel_peak_mask(target)
    assert mask[0, 0].sum().item() == 1
    assert mask[0, 1].sum().item() == 0


def test_channel_peak_mask_finds_only_one_of_several_tips():
    """Why `positives` is a required argument rather than inferred.

    Both tips live in one channel, so the per-channel maximum marks only the
    stronger. Supervising the other as background actively suppresses it: two
    tips whose targets differed by 0.001 trained to 0.943 and 0.001.
    """
    tip = encode_tip_targets([(0.30, 0.30), (0.62, 0.66)], GRID, sigma_cells=1.5)
    target = torch.from_numpy(tip.heat).unsqueeze(0).unsqueeze(0)
    assert tip.count == 2
    assert channel_peak_mask(target).sum().item() == 1  # the trap
    assert torch.from_numpy(tip.mask).sum().item() == 2  # what to pass instead


def test_focal_loss_is_lower_when_the_prediction_is_right():
    heat = torch.from_numpy(encode_landmark_heatmaps([(0.5, 0.5)], GRID)).unsqueeze(0)
    positives = channel_peak_mask(heat)
    confident = torch.logit(heat.clamp(1e-4, 1 - 1e-4))
    wrong = torch.full_like(heat, -4.0)
    assert heatmap_focal_loss(confident, heat, positives) < heatmap_focal_loss(
        wrong, heat, positives
    )


def test_focal_loss_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        heatmap_focal_loss(
            torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 4, 4)
        )


def test_focal_loss_rejects_a_mismatched_positive_grid():
    with pytest.raises(ValueError, match="positives and target grids differ"):
        heatmap_focal_loss(
            torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8), torch.ones(1, 4, 4)
        )


def test_offset_loss_ignores_background_cells():
    prediction = torch.full((1, 2, *GRID), 0.4)
    target = torch.zeros(1, 2, *GRID)
    mask = torch.zeros(1, *GRID)
    mask[0, 10, 10] = 1.0
    target[0, :, 10, 10] = 0.4

    assert offset_l1_loss(prediction, target, mask).item() == pytest.approx(0.0)
    # Nudge only the supervised cell and the loss must react.
    target[0, 0, 10, 10] = -0.1
    assert offset_l1_loss(prediction, target, mask).item() > 0.2


def test_offset_loss_validates_grids():
    with pytest.raises(ValueError, match="grids differ"):
        offset_l1_loss(
            torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8), torch.zeros(1, 4, 4)
        )


# --------------------------------------------------------------------------
# The overfit
# --------------------------------------------------------------------------

def test_overfits_a_single_batch_and_decodes_the_points_back():
    """End to end: targets -> model -> loss -> optimiser -> decode.

    Deliberately tiny and deliberately overfitting. If the chain is wired
    correctly this converges; if anything is transposed, mis-strided or
    mis-masked, it will not -- and it would otherwise look like a model that
    trains smoothly and predicts nothing.
    """
    torch.manual_seed(7)
    rng = np.random.default_rng(3)

    landmarks = [(0.30, 0.28), (0.72, 0.31), (0.70, 0.74), (0.27, 0.71),
                 (0.42, 0.40), (0.60, 0.42), (0.59, 0.61), (0.40, 0.59)]
    tips = [(0.48, 0.36), (0.55, 0.58)]

    landmark_target = torch.from_numpy(
        encode_landmark_heatmaps(landmarks, GRID, sigma_cells=1.5)
    ).unsqueeze(0)
    tip = encode_tip_targets(tips, GRID, sigma_cells=1.5)
    tip_heat = torch.from_numpy(tip.heat).unsqueeze(0).unsqueeze(0)
    tip_offset = torch.from_numpy(tip.offset).unsqueeze(0)
    tip_mask = torch.from_numpy(tip.mask).unsqueeze(0)

    image = torch.from_numpy(rng.random((1, 3, *IMAGE)).astype("float32"))

    model = DartVisionBrain(backbone="resnet18", landmarks=8, width=32, stride=4)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = last = None
    for step in range(240):
        optimiser.zero_grad()
        out = model(image)
        loss = (
            heatmap_focal_loss(
                out.landmark_heat, landmark_target, channel_peak_mask(landmark_target)
            )
            # Tips share one channel, so positives come from the encoder's mask.
            + heatmap_focal_loss(out.tip_heat, tip_heat, tip_mask)
            + 5.0 * offset_l1_loss(out.tip_offset, tip_offset, tip_mask)
        )
        loss.backward()
        optimiser.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first * 0.25, f"loss {first:.3f} -> {last:.3f}: not converging"

    model.eval()
    with torch.no_grad():
        out = model(image)
        landmark_probs = torch.sigmoid(out.landmark_heat)[0].numpy()
        decoded = decode_landmark_heatmaps(
            landmark_probs, window=recommended_window(1.5), threshold=0.15
        )
        tip_decoded = decode_tip_targets(
            torch.sigmoid(out.tip_heat)[0, 0].numpy(),
            out.tip_offset[0].numpy(),
            threshold=0.2,
            max_detections=3,
        )

    for (point, _), truth in zip(decoded, landmarks):
        assert point is not None, "a landmark channel failed to fire"
        error = np.hypot((point[0] - truth[0]) * GRID[1], (point[1] - truth[1]) * GRID[0])
        assert error < 1.0, f"landmark off by {error:.2f} cells"

    assert len(tip_decoded) == len(tips)
    for truth in tips:
        best = min(
            np.hypot((p[0] - truth[0]) * GRID[1], (p[1] - truth[1]) * GRID[0])
            for p, _ in tip_decoded
        )
        assert best < 1.0, f"tip off by {best:.2f} cells"
