"""Explaining what an extraction did, before it writes anything.

``extract`` reports three numbers -- frames read, runs found, duplicates dropped
-- and when the answer is wrong those numbers do not say why. Session-01 read
1059 frames and kept 9, and the only way to find out that a whole visit was
collapsing into one still run was to work backwards from the timestamps by hand.

So this decodes the same video and reports the evidence instead of the verdict:
what the two stillness measures actually did over the recording, how often the
camera itself moved, and what a different setting would have produced. Nothing
is written to disk. Thresholds in this package are all defended by a paragraph
of reasoning about sensor noise and dart geometry, and reasoning is a good way
to pick a starting value and a poor way to confirm one -- the footage in hand
settles it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from dartvision.capture.frames import (
    ExtractionSettings,
    _analysis_frames,
    choose_frames,
    find_ffmpeg,
    frame_statistics,
)

__all__ = ["Scan", "scan", "scan_video", "sweep", "render"]

# A change touching this share of the frame is not something *in* the scene --
# nothing on a dartboard is a sixth of the picture. It is the camera moving.
CAMERA_MOVE_FRACTION = 0.15


@dataclass(frozen=True)
class Scan:
    """Both stillness measures across a whole recording."""

    fps: float
    pixels: int
    means: np.ndarray
    changed: np.ndarray

    @property
    def frames_read(self) -> int:
        return len(self.means)

    @property
    def seconds(self) -> float:
        return self.frames_read / self.fps

    def settled(self, settings: ExtractionSettings) -> np.ndarray:
        """Which frames a given setting would call still."""
        return (self.means <= settings.still) & (self.changed <= settings.still_pixels)

    def camera_moves(self) -> list[tuple[float, float]]:
        """Time spans over which the camera itself was moving.

        Consecutive offending frames are one move, not many: a phone being
        re-propped is a single event that happens to span a second of video.
        """
        moving = self.changed > self.pixels * CAMERA_MOVE_FRACTION
        spans: list[tuple[float, float]] = []
        start: int | None = None
        for index, value in enumerate(moving):
            if value and start is None:
                start = index
            elif not value and start is not None:
                spans.append((start / self.fps, (index - 1) / self.fps))
                start = None
        if start is not None:
            spans.append((start / self.fps, (len(moving) - 1) / self.fps))
        return spans


def scan(frames: Sequence[np.ndarray], settings: ExtractionSettings) -> Scan:
    means, changed = frame_statistics(frames, settings.change_level)
    height, width = np.asarray(frames[0]).shape
    return Scan(fps=settings.analysis_fps, pixels=height * width,
                means=means, changed=changed)


def scan_video(
    video: str | Path, settings: ExtractionSettings | None = None
) -> tuple[Scan, list[np.ndarray]]:
    """Decode a video for analysis and measure it. Writes nothing."""
    settings = settings or ExtractionSettings()
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)
    frames = list(_analysis_frames(
        video, settings.analysis_width, settings.analysis_fps, find_ffmpeg()
    ))
    if not frames:
        raise RuntimeError(f"{video.name} decoded to no frames")
    return scan(frames, settings), frames


def sweep(
    frames: Sequence[np.ndarray],
    settings: ExtractionSettings,
    still_pixels: Sequence[int] = (5, 10, 25, 60, 150),
    min_still_frames: Sequence[int] = (3, 5),
) -> list[dict[str, object]]:
    """What each candidate setting would have kept, on this footage.

    Re-runs the real chooser rather than reimplementing it, so the table cannot
    drift away from what ``extract`` would actually do.
    """
    rows: list[dict[str, object]] = []
    for minimum in min_still_frames:
        for budget in still_pixels:
            candidate = ExtractionSettings(
                still=settings.still, still_pixels=budget,
                min_still_frames=minimum, change_level=settings.change_level,
                changed_pixels=settings.changed_pixels,
                analysis_width=settings.analysis_width,
                analysis_fps=settings.analysis_fps,
            )
            kept, runs, duplicates, blocked = choose_frames(frames, candidate)
            rows.append({
                "still_pixels": budget, "min_still_frames": minimum,
                "runs": runs, "kept": len(kept), "duplicates": duplicates,
                "obstructed": blocked,
                "current": (budget == settings.still_pixels
                            and minimum == settings.min_still_frames),
            })
    return rows


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


def render(scan: Scan, rows: Sequence[dict[str, object]], settings: ExtractionSettings) -> str:
    """The report, as text a person reads and then changes one number."""
    lines = [
        f"scanned {scan.frames_read} frames at {scan.fps:g} fps "
        f"(~{scan.seconds / 60:.1f} minutes), {scan.pixels:,} pixels each",
        "",
        "how much changed between frames",
    ]

    steps = scan.means[1:]
    counts = scan.changed[1:]
    if len(steps):
        for label, series, unit in (
            ("mean brightness", steps, "grey levels"),
            ("pixels changed", counts, "pixels"),
        ):
            percentiles = np.percentile(series, [50, 90, 99])
            lines.append(
                f"  {label:<16} median {percentiles[0]:8.1f}   "
                f"90th {percentiles[1]:8.1f}   99th {percentiles[2]:8.1f}  {unit}"
            )

    settled = scan.settled(settings)
    lines += [
        "",
        f"at the current setting (still {settings.still:g}, "
        f"still_pixels {settings.still_pixels}) "
        f"{100 * settled.mean():.0f}% of frames count as unchanged",
    ]

    moves = scan.camera_moves()
    if moves:
        shown = ", ".join(f"{_clock(a)}-{_clock(b)}" for a, b in moves[:8])
        more = f" and {len(moves) - 8} more" if len(moves) > 8 else ""
        lines += [
            "",
            f"the camera itself moved {len(moves)} time(s): {shown}{more}",
            "  Each move changes where the board sits in frame. Frames either "
            "side of one are not the same viewpoint,",
            "  and a still caught during one is unusable. A fixed mount removes "
            "all of them.",
        ]
    else:
        lines += ["", "the camera never moved — good, that is the whole game"]

    lines += ["", "what other settings would have given", "",
              "  still_pixels  hold for  runs  kept  blocked"]
    for row in rows:
        mark = "  <- current" if row["current"] else ""
        held = int(row["min_still_frames"]) / scan.fps
        lines.append(
            f"  {row['still_pixels']:>12}  {held:>6.1f}s  "
            f"{row['runs']:>4}  {row['kept']:>4}  {row['obstructed']:>7}{mark}"
        )
    lines += [
        "",
        "A lower still_pixels splits more finely; too low and noise reads as a "
        "dart and nothing holds still.",
        "Aim for roughly one kept frame per dart thrown, plus one empty board "
        "per visit.",
    ]
    return "\n".join(lines)
