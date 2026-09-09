"""The annotation contract: what one labelled image carries.

Coordinates are normalized to the image, ``x/width`` and ``y/height``, matching
the contract recovered in #13 so that synthetic output (#25) and captured data
(#26) are interchangeable inputs to the same pipeline.

Two deliberate departures from the reference contract:

* **Missing landmarks are ``None``, not a sentinel coordinate.** The reference
  encoded "not annotated" as a non-positive coordinate, which is silently
  indistinguishable from a landmark that genuinely projects just off the frame
  edge. An explicit ``None`` cannot be misread.
* **Eight landmarks are supported, not just four.** Four correspondences
  determine a homography exactly and so carry no residual; eight over-determine
  it, exposing noise and averaging it down (see ``geometry.calibration``).

``setup_id`` and ``session_id`` are required rather than optional because
#14's benchmark tiers are defined in terms of them: splits group by session and
never span one, and the cross-setup tier holds out whole setups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = ["Origin", "Annotation", "image_filename", "read_jsonl", "write_jsonl"]

Point = tuple[float, float]

MAX_DARTS = 3
VALID_LANDMARK_COUNTS = (4, 8)


def image_filename(image_id: str, suffix: str = ".jpg") -> str:
    """File name for an image id.

    Ids are hierarchical (``setup/session/img-0001``) but file systems are not,
    so separators become underscores. Shared by the renderer and the dataset:
    when the two derive this independently they eventually disagree, and the
    failure looks like missing data rather than a naming mismatch.
    """
    return image_id.replace("/", "_") + suffix


class Origin(str, Enum):
    """Where an image came from. Kept in the label so metrics can slice by it."""

    SYNTHETIC = "synthetic"
    REAL_THROWN = "real-thrown"
    REAL_PLACED = "real-placed"

    @property
    def is_real(self) -> bool:
        return self is not Origin.SYNTHETIC


@dataclass(frozen=True)
class Annotation:
    """One labelled image."""

    image_id: str
    setup_id: str
    session_id: str
    origin: Origin
    landmarks: tuple[Point | None, ...]
    tips: tuple[Point, ...] = ()
    image_size: tuple[int, int] | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_id or not self.setup_id or not self.session_id:
            raise ValueError("image_id, setup_id and session_id are all required")
        if len(self.landmarks) not in VALID_LANDMARK_COUNTS:
            raise ValueError(
                f"expected {' or '.join(map(str, VALID_LANDMARK_COUNTS))} landmarks, "
                f"got {len(self.landmarks)}"
            )
        if len(self.tips) > MAX_DARTS:
            raise ValueError(f"at most {MAX_DARTS} dart tips, got {len(self.tips)}")
        for name, points in (("landmark", self.landmarks), ("tip", self.tips)):
            for i, p in enumerate(points):
                if p is None:
                    continue
                if len(p) != 2 or not all(isinstance(v, (int, float)) for v in p):
                    raise ValueError(f"{name} {i} must be an (x, y) pair, got {p!r}")
                if not all(v == v and abs(v) != float("inf") for v in p):
                    raise ValueError(f"{name} {i} has a non-finite coordinate: {p!r}")

    @property
    def visible_landmarks(self) -> int:
        return sum(1 for p in self.landmarks if p is not None)

    @property
    def is_calibratable(self) -> bool:
        """Whether enough landmarks are present to estimate a homography.

        Below this, no dart in the image can be scored at all -- a cliff rather
        than a slope, which is why #21 gates on the detection rate directly.
        """
        return self.visible_landmarks >= 4

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "image_id": self.image_id,
            "setup_id": self.setup_id,
            "session_id": self.session_id,
            "origin": self.origin.value,
            "landmarks": [list(p) if p is not None else None for p in self.landmarks],
            "tips": [list(p) for p in self.tips],
        }
        if self.image_size is not None:
            payload["image_size"] = list(self.image_size)
        if self.meta:
            payload["meta"] = self.meta
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Annotation":
        def as_point(value: object) -> Point | None:
            if value is None:
                return None
            x, y = value  # type: ignore[misc]
            return (float(x), float(y))

        size = payload.get("image_size")
        return cls(
            image_id=str(payload["image_id"]),
            setup_id=str(payload["setup_id"]),
            session_id=str(payload["session_id"]),
            origin=Origin(str(payload["origin"])),
            landmarks=tuple(as_point(p) for p in payload["landmarks"]),  # type: ignore[union-attr]
            tips=tuple(as_point(p) for p in payload.get("tips", [])),  # type: ignore[misc]
            image_size=(int(size[0]), int(size[1])) if size else None,  # type: ignore[index]
            meta=dict(payload.get("meta", {})),  # type: ignore[arg-type]
        )


def write_jsonl(annotations: Iterable[Annotation], path: str | Path) -> int:
    """Write annotations one per line. Returns the count written."""
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            handle.write(json.dumps(annotation.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[Annotation]:
    """Stream annotations from a JSONL manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Annotation.from_dict(json.loads(line))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc


def sessions_by_setup(annotations: Sequence[Annotation]) -> dict[str, set[str]]:
    """Sessions grouped by setup -- the structure #14's tiers are built on."""
    grouped: dict[str, set[str]] = {}
    for a in annotations:
        grouped.setdefault(a.setup_id, set()).add(a.session_id)
    return grouped
