"""Pulling one usable still per dart out of a video.

Recording a session as video is easier than photographing it, and it is also
closer to what the product does: the app reads video frames, so training on
video frames removes a domain gap rather than adding one. What it creates is a
sorting problem. Thirty frames a second of a still board is thirty nearly
identical images, and a naive extraction produces a corpus that is large and
nearly empty -- the exact thing #14's effective-size check exists to catch.

So this keeps one frame per *board state*, using the same idea as #3's stream
gate, one level lower down. The gate there asks "is the scene still?" of
detections; here the same question is asked of raw pixels, which needs no model
and can run before any of the Brain exists.

Three steps, and each drops something for a different reason:

1. **Still runs.** Consecutive frames that barely differ. A dart in flight, a
   hand over the board or the player walking up all move, and everything that
   moves is dropped.
2. **The sharpest frame of each run.** A still run still contains focus breathing and
   compression noise; the frame with the most high-frequency detail is the one
   whose wires are crispest, and wire sharpness is what the landmark head needs.
3. **A localized change against the frame already kept.** Two still runs either
   side of a pause with nothing thrown are the same board twice.

What survives is the empty board, then each dart as it lands.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "ExtractionSettings",
    "Extraction",
    "frame_differences",
    "still_runs",
    "sharpness",
    "changed_pixel_count",
    "choose_frames",
    "find_ffmpeg",
    "extract",
]

# Frames are analysed at this width. Small enough that a long session decodes in
# seconds, large enough that a dart landing is a clear change.
ANALYSIS_WIDTH = 192


@dataclass(frozen=True)
class ExtractionSettings:
    """Thresholds, each with a reason rather than a round number.

    ``still`` is the mean absolute difference, in 0-255 grey levels, below which
    consecutive frames count as the same scene. Sensor noise on a static shot
    sits near 1; a dart entering the frame moves it far higher.

    ``min_still_frames`` is how long a scene must hold. A dart takes a moment to
    stop wobbling, and a hand pausing mid-reach should not read as a state.

    ``change_level`` and ``changed_pixels`` decide whether anything was actually
    thrown between two still runs. A perceptual hash is the obvious tool and the
    wrong one: at 192 px of analysis width a dart spans about ten pixels, and a
    difference hash downsamples to 8x8 -- so an entire dart lands inside a single
    hash cell and disappears. Measured on synthetic sessions, that silently
    dropped real dart-landing frames as duplicates.

    A dart is instead exactly what it looks like: a *small* number of pixels
    changing by a *lot*. Sensor noise is the opposite -- many pixels changing by
    one or two grey levels -- so counting pixels past a level separates them
    cleanly where an average of either cannot.
    """

    still: float = 2.0
    min_still_frames: int = 5
    change_level: float = 25.0
    changed_pixels: int = 10
    analysis_width: int = ANALYSIS_WIDTH

    def __post_init__(self) -> None:
        if self.still <= 0:
            raise ValueError("still must be positive")
        if self.min_still_frames < 1:
            raise ValueError("min_still_frames must be at least 1")
        if not 0 < self.change_level <= 255:
            raise ValueError("change_level must be in (0, 255]")
        if self.changed_pixels < 1:
            raise ValueError("changed_pixels must be at least 1")
        if self.analysis_width < 32:
            raise ValueError("analysis_width must be at least 32")


@dataclass(frozen=True)
class Extraction:
    """What came out, and enough to tell whether it went well."""

    frames_read: int
    still_runs: int
    kept: tuple[int, ...]
    dropped_as_duplicate: int
    files: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "frames_read": self.frames_read,
            "still_runs": self.still_runs,
            "kept": list(self.kept),
            "dropped_as_duplicate": self.dropped_as_duplicate,
            "files": [str(f) for f in self.files],
        }


# --------------------------------------------------------------------------
# The analysis, as pure functions over arrays
# --------------------------------------------------------------------------

def frame_differences(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Mean absolute difference between each frame and the one before it."""
    if len(frames) < 2:
        return np.zeros(max(len(frames), 0))
    stack = np.asarray(frames, dtype=np.float32)
    return np.concatenate([[0.0], np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))])


def still_runs(
    differences: Sequence[float], threshold: float, min_length: int
) -> list[tuple[int, int]]:
    """Index ranges (inclusive) over which the scene barely changed."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(differences):
        if value <= threshold:
            start = index if start is None else start
            continue
        if start is not None and index - start >= min_length:
            runs.append((start, index - 1))
        start = None
    if start is not None and len(differences) - start >= min_length:
        runs.append((start, len(differences) - 1))
    return runs


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian -- more high-frequency detail, crisper wires.

    The standard focus measure, and the right one here: a soft frame loses the
    thin high-contrast lines the landmark head keys on long before it looks
    obviously blurry to a person.
    """
    image = np.asarray(frame, dtype=np.float32)
    laplacian = (
        4 * image[1:-1, 1:-1]
        - image[:-2, 1:-1] - image[2:, 1:-1]
        - image[1:-1, :-2] - image[1:-1, 2:]
    )
    return float(laplacian.var())


def changed_pixel_count(before: np.ndarray, after: np.ndarray, level: float) -> int:
    """Pixels that changed by more than ``level`` grey levels.

    The test for "did something land here". A dart is a small region changing
    a lot; noise is a large region changing a little. Counting past a level
    tells them apart, where comparing either average would not.
    """
    difference = np.abs(np.asarray(after, dtype=np.int16) - np.asarray(before, dtype=np.int16))
    return int((difference > level).sum())


def choose_frames(
    frames: Sequence[np.ndarray], settings: ExtractionSettings | None = None
) -> tuple[list[int], int, int]:
    """``(frame indices to keep, still runs found, duplicates dropped)``."""
    settings = settings or ExtractionSettings()
    runs = still_runs(
        frame_differences(frames), settings.still, settings.min_still_frames
    )

    kept: list[int] = []
    duplicates = 0
    previous: np.ndarray | None = None
    for start, end in runs:
        best = max(range(start, end + 1), key=lambda i: sharpness(frames[i]))
        if previous is not None:
            changed = changed_pixel_count(previous, frames[best], settings.change_level)
            if changed < settings.changed_pixels:
                # The same board twice, either side of a pause. Nothing landed.
                duplicates += 1
                continue
        kept.append(best)
        previous = frames[best]
    return kept, len(runs), duplicates


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------

def find_ffmpeg() -> str:
    """Locate ffmpeg, preferring one on PATH.

    Not a dependency of this package -- it is invoked, never shipped -- so a
    missing one is a setup problem to report clearly rather than a crash.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    bundled = Path("/opt/pw-browsers/ffmpeg-1011/ffmpeg-linux")
    if bundled.exists():
        return str(bundled)
    raise RuntimeError(
        "ffmpeg not found. Install it (macOS: `brew install ffmpeg`) — it reads "
        "the video; nothing in this package ships it."
    )


def _analysis_frames(video: Path, width: int, ffmpeg: str) -> Iterator[np.ndarray]:
    """Decode the whole video small and grey, one frame at a time."""
    probe = subprocess.run(
        [ffmpeg, "-i", str(video), "-vf", f"scale={width}:-2,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not read {video.name}:\n"
            + probe.stderr.decode("utf-8", "replace").strip().splitlines()[-1]
        )

    # The scale filter keeps the aspect ratio, so the height follows from the
    # payload rather than being assumed.
    height = _height_from(probe.stderr.decode("utf-8", "replace"), width)
    data = np.frombuffer(probe.stdout, dtype=np.uint8)
    count = len(data) // (width * height)
    for index in range(count):
        offset = index * width * height
        yield data[offset:offset + width * height].reshape(height, width)


def _height_from(stderr: str, width: int) -> int:
    """Read the scaled height out of ffmpeg's own report of the output stream."""
    import re

    matches = re.findall(r"(\d{2,5})x(\d{2,5})", stderr)
    for found_width, found_height in reversed(matches):
        if int(found_width) == width:
            return int(found_height)
    raise RuntimeError("could not determine the decoded frame height from ffmpeg")


def extract(
    video: str | Path,
    out_dir: str | Path,
    settings: ExtractionSettings | None = None,
    prefix: str = "frame",
) -> Extraction:
    """Pull one still per board state out of ``video``.

    Filenames are zero-padded and sequential because the annotator sorts by
    filename and burst expansion depends on that order being right (#26).
    """
    settings = settings or ExtractionSettings()
    video, out_dir = Path(video), Path(out_dir)
    if not video.exists():
        raise FileNotFoundError(video)
    ffmpeg = find_ffmpeg()

    frames = list(_analysis_frames(video, settings.analysis_width, ffmpeg))
    if not frames:
        raise RuntimeError(f"{video.name} decoded to no frames")

    kept, runs, duplicates = choose_frames(frames, settings)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    if kept:
        # One pass over the video, pulling the chosen frames at full size.
        selection = "+".join(f"eq(n\\,{index})" for index in kept)
        pattern = str(out_dir / f"{prefix}-%04d.jpg")
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(video), "-vf", f"select='{selection}'",
             "-vsync", "0", "-q:v", "2", pattern],
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg could not write the frames:\n"
                + result.stderr.decode("utf-8", "replace").strip().splitlines()[-1]
            )
        files = sorted(out_dir.glob(f"{prefix}-*.jpg"))

    extraction = Extraction(
        frames_read=len(frames), still_runs=runs, kept=tuple(kept),
        dropped_as_duplicate=duplicates, files=tuple(files),
    )
    (out_dir / "extraction.json").write_text(
        json.dumps(extraction.to_dict(), indent=2), encoding="utf-8"
    )
    return extraction
