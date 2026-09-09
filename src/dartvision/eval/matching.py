"""Matching predicted dart tips to the darts that were actually there.

Evaluation cannot begin until predictions and truth are paired, and that pairing
is not incidental: get it wrong and every metric downstream is wrong in a way
that looks like model error.

Two failure modes must stay distinguishable, because #14's taxonomy treats them
separately and they have different causes:

* a **missed dart** -- a real dart with no prediction near it;
* a **phantom dart** -- a prediction with no real dart near it.

A greedy nearest-first assignment produces those naturally, provided pairs
beyond a distance threshold are refused. Without that threshold a lone
prediction on the far side of the board would be "matched" to a real dart and
counted as a large localization error rather than as one phantom plus one miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["Match", "match_tips"]

Point = tuple[float, float]


@dataclass(frozen=True)
class Match:
    """One pairing. Either side may be absent, but never both."""

    truth_index: int | None
    prediction_index: int | None
    distance_mm: float | None

    @property
    def is_missed(self) -> bool:
        return self.prediction_index is None

    @property
    def is_phantom(self) -> bool:
        return self.truth_index is None

    @property
    def is_paired(self) -> bool:
        return self.truth_index is not None and self.prediction_index is not None


def match_tips(
    truth: Sequence[Point],
    predictions: Sequence[Point],
    max_distance_mm: float = 25.0,
) -> list[Match]:
    """Pair tips nearest-first, refusing pairs further apart than the threshold.

    ``max_distance_mm`` defaults to 25 mm -- wider than any plausible
    localization error, narrower than the gap between distinct darts on a board
    of 225 mm radius. Beyond it, a prediction is a phantom and the dart it might
    have been matched to is a miss.
    """
    if max_distance_mm <= 0:
        raise ValueError("max_distance_mm must be positive")

    candidates = sorted(
        (
            ((tx - px) ** 2 + (ty - py) ** 2) ** 0.5,
            t_index,
            p_index,
        )
        for t_index, (tx, ty) in enumerate(truth)
        for p_index, (px, py) in enumerate(predictions)
    )

    matches: list[Match] = []
    used_truth: set[int] = set()
    used_prediction: set[int] = set()
    for distance, t_index, p_index in candidates:
        if distance > max_distance_mm:
            break
        if t_index in used_truth or p_index in used_prediction:
            continue
        used_truth.add(t_index)
        used_prediction.add(p_index)
        matches.append(
            Match(truth_index=t_index, prediction_index=p_index, distance_mm=distance)
        )

    matches += [
        Match(truth_index=i, prediction_index=None, distance_mm=None)
        for i in range(len(truth))
        if i not in used_truth
    ]
    matches += [
        Match(truth_index=None, prediction_index=i, distance_mm=None)
        for i in range(len(predictions))
        if i not in used_prediction
    ]
    return matches
