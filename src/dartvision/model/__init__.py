"""Model-facing encoding, decoding, and the network itself.

``targets`` is pure NumPy and importable without a training framework; the
network, heads and losses need PyTorch and are imported lazily so the rest of
the package stays usable without it.
"""

from dartvision.model.spec import (
    DEFAULT_INPUT_PX,
    IDEAL_RING_PX,
    MIN_RING_PX,
    MOUNT_BAND_DEG,
    minimum_input_px,
    ring_width_px,
)
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
    "DEFAULT_INPUT_PX",
    "IDEAL_RING_PX",
    "MIN_RING_PX",
    "MOUNT_BAND_DEG",
    "TipTargets",
    "minimum_input_px",
    "ring_width_px",
    "decode_landmark_heatmaps",
    "decode_tip_targets",
    "encode_landmark_heatmaps",
    "encode_tip_targets",
    "recommended_window",
    "soft_argmax",
]
