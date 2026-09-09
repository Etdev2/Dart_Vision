"""Model-facing encoding, decoding, and the network itself.

``targets`` is pure NumPy and importable without a training framework; the
network, heads and losses need PyTorch and are imported lazily so the rest of
the package stays usable without it.
"""

from dartvision.model.targets import (
    TipTargets,
    decode_landmark_heatmaps,
    decode_tip_targets,
    encode_landmark_heatmaps,
    encode_tip_targets,
    recommended_window,
    soft_argmax,
)

__all__ = [
    "TipTargets",
    "decode_landmark_heatmaps",
    "decode_tip_targets",
    "encode_landmark_heatmaps",
    "encode_tip_targets",
    "recommended_window",
    "soft_argmax",
]
