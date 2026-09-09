"""Annotating a capture session with as few clicks as the geometry allows.

Labelling every image independently -- eight landmarks plus up to three tips per
frame -- would make #26 a month of clicking. Two properties of a real capture
session remove most of that work:

1. **The board and camera do not move within a session**, so the calibration
   landmarks are annotated *once* and propagated to every frame. Re-annotating
   is what defines a new session, which is also exactly the grouping #14's
   benchmark tiers need.
2. **Darts arrive one at a time.** Capturing after each throw means frame *k*
   holds the first *k* tips. Clicking the three tips once, in throw order, on
   the final frame therefore labels the whole burst.

Together those turn "11 clicks per image" into "8 per session plus 3 per visit".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from dartvision.data.labels import Annotation, Origin

__all__ = ["CaptureSession", "BurstError", "expand_burst", "effort_saved"]

Point = tuple[float, float]


class BurstError(ValueError):
    """The frames and tips given cannot describe an incremental capture."""


@dataclass(frozen=True)
class CaptureSession:
    """A fixed board and camera. Landmarks are annotated once, here."""

    setup_id: str
    session_id: str
    landmarks: tuple[Point | None, ...]
    image_size: tuple[int, int] | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.setup_id or not self.session_id:
            raise ValueError("setup_id and session_id are required")
        if len(self.landmarks) not in (4, 8):
            raise ValueError(f"expected 4 or 8 landmarks, got {len(self.landmarks)}")
        if sum(1 for p in self.landmarks if p is not None) < 4:
            raise ValueError(
                "at least 4 landmarks must be annotated, or no frame in this "
                "session can be scored at all"
            )


def expand_burst(
    session: CaptureSession,
    frame_ids: Sequence[str],
    tips_in_throw_order: Sequence[Point],
    origin: Origin = Origin.REAL_THROWN,
    frame_meta: Sequence[dict[str, object]] | None = None,
) -> list[Annotation]:
    """Label a whole visit from one set of clicks.

    ``frame_ids`` are in capture order. The number of frames decides whether the
    empty board was captured: with three tips, four frames means the burst
    started empty and three frames means it started after the first throw.
    """
    tips = tuple(tips_in_throw_order)
    frames = list(frame_ids)
    if not frames:
        raise BurstError("a burst needs at least one frame")
    if len(tips) > 3:
        raise BurstError(f"at most 3 tips in a visit, got {len(tips)}")

    first_count = len(tips) - len(frames) + 1
    if first_count < 0:
        raise BurstError(
            f"{len(frames)} frames but only {len(tips)} tips: a burst gains one "
            "dart per frame, so there cannot be more frames than tips plus one"
        )

    if frame_meta is not None and len(frame_meta) != len(frames):
        raise BurstError("frame_meta must have one entry per frame")

    annotations = []
    for index, frame_id in enumerate(frames):
        count = first_count + index
        annotations.append(
            Annotation(
                image_id=frame_id,
                setup_id=session.setup_id,
                session_id=session.session_id,
                origin=origin,
                landmarks=session.landmarks,
                tips=tips[:count],
                image_size=session.image_size,
                meta={
                    **session.meta,
                    "burst_position": index,
                    "darts_in_frame": count,
                    **((frame_meta[index] if frame_meta else {}) or {}),
                },
            )
        )
    return annotations


def effort_saved(annotations: Sequence[Annotation], landmark_clicks: int) -> dict[str, int]:
    """Clicks actually spent versus labelling every frame independently.

    Reported by the tooling because the saving is the entire reason #26 is
    affordable, and it is worth seeing rather than assuming.
    """
    if not annotations:
        raise ValueError("no annotations")
    frames = len(annotations)
    tips_clicked = max(len(a.tips) for a in annotations)
    spent = landmark_clicks + tips_clicked
    naive = sum(len(a.landmarks) + len(a.tips) for a in annotations)
    return {
        "frames": frames,
        "clicks_spent": spent,
        "clicks_if_labelled_per_frame": naive,
        "clicks_saved": naive - spent,
    }
