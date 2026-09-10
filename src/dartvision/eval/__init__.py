"""Evaluation: matching, scoring and #21's gate report."""

from dartvision.eval.evaluate import FrameResult, evaluate_all, evaluate_frame
from dartvision.eval.matching import Match, match_tips

__all__ = [
    "FrameResult",
    "Match",
    "evaluate_all",
    "evaluate_frame",
    "match_tips",
]


def __getattr__(name: str):
    # The runner needs torch; keep the light evaluation helpers importable
    # without it so metrics can be computed anywhere.
    if name in ("evaluate_checkpoint", "load_model", "predict_frames", "checkpoint_digest"):
        from dartvision.eval import runner

        return getattr(runner, name)
    raise AttributeError(name)
