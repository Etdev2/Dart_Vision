"""Deterministic board geometry: no model, no images, no dependencies."""

from dartvision.geometry.board import (
    BDO_BOARD,
    SECTORS_CLOCKWISE_FROM_20,
    BoardSpec,
    Hit,
    margin_to_nearest_boundary,
    score_at,
)

__all__ = [
    "BDO_BOARD",
    "SECTORS_CLOCKWISE_FROM_20",
    "BoardSpec",
    "Hit",
    "margin_to_nearest_boundary",
    "score_at",
]
