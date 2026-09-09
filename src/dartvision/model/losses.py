"""Losses for the two heads.

The heatmap loss is the penalty-reduced focal loss from the CenterNet line of
work, with two necessary adaptations. Both were found by an overfit test after
a dozen unit tests had passed, and either one silently breaks training.

**Targets never reach 1.** A Gaussian centred between cells is never sampled at
its maximum, so a sub-cell-accurate peak lands anywhere from about 0.78 (at
sigma=1) to 1.0. The usual ``target == 1`` test for positives would find none at
all and train the model to predict background everywhere.

**Positives cannot be inferred from the target alone.** The tip head carries
*several instances in one channel*, so taking the per-channel maximum finds only
the strongest and supervises every other tip as background -- actively
suppressing it. Observed directly: two tips whose target peaks differed by
0.001 trained to 0.943 and 0.001 respectively, with the loss sitting happily at
0.019.

So ``positives`` is a **required argument**. Use :func:`channel_peak_mask` for
the landmark head, where exactly one instance per channel is guaranteed by
construction, and the encoder's own ``TipTargets.mask`` for tips. Making the
caller say which cells are positive is the only way to stop this being wrong
for one of the two heads.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["heatmap_focal_loss", "offset_l1_loss", "channel_peak_mask"]


def channel_peak_mask(target: Tensor, tolerance: float = 1e-4) -> Tensor:
    """Cells at the per-channel maximum of a non-empty channel.

    Correct **only for one instance per channel**, which the landmark head
    guarantees by construction. Using it on a multi-instance channel supervises
    every instance but the strongest as background. An all-zero channel -- an
    absent landmark -- yields no positives, so it is supervised entirely as
    background, which is what we want.
    """
    if target.ndim != 4:
        raise ValueError("expected (batch, channels, height, width)")
    peak = target.amax(dim=(-2, -1), keepdim=True)
    return (target >= peak - tolerance) & (peak > tolerance)


def heatmap_focal_loss(
    logits: Tensor,
    target: Tensor,
    positives: Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> Tensor:
    """Penalty-reduced focal loss, normalized by the number of positives.

    ``positives`` marks the cells owning an instance and is required -- see the
    module docstring for why inferring it is unsafe. ``beta`` down-weights
    negatives near a peak, so cells one away from the true position are not
    punished as hard as distant background, which is what makes a Gaussian
    target trainable at all.
    """
    if logits.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    if positives.ndim == 3:
        positives = positives.unsqueeze(1)
    if positives.shape[-2:] != target.shape[-2:]:
        raise ValueError("positives and target grids differ")
    positives = positives.to(dtype=torch.bool).expand_as(target)

    probability = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    positive_loss = -((1.0 - probability) ** alpha) * torch.log(probability)
    negative_loss = (
        -((1.0 - target) ** beta) * (probability ** alpha) * torch.log(1.0 - probability)
    )

    total = torch.where(positives, positive_loss, negative_loss).sum()
    count = positives.sum().clamp(min=1)
    return total / count


def offset_l1_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """L1 on sub-cell offsets, only where a tip actually is.

    Offsets at background cells are meaningless -- there is no tip to be offset
    from -- so supervising them would teach the head to regress toward zero
    everywhere and blunt the predictions that matter.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("mask and offset grids differ")

    weights = mask.expand_as(prediction)
    count = weights.sum().clamp(min=1)
    return ((prediction - target).abs() * weights).sum() / count
