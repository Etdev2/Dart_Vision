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

1. **Still runs.** Consecutive frames in which nothing happened. A dart in
   flight, a hand over the board or the player walking up all move, and
   everything that moves is dropped.
2. **The sharpest frame of each run.** A still run still contains focus breathing and
   compression noise; the frame with the most high-frequency detail is the one
   whose wires are crispest, and wire sharpness is what the landmark head needs.
3. **Runs that are something standing in front of the board.** Pulling darts
   means standing at the board, and a person standing still is as still as a
   board is. Such a run is recognised by its neighbours rather than its
   contents: it differs wildly from the run before it *and* from the run after
   it, while those two resemble each other. A camera being moved fails that
   last test -- the view after a move does not match the view before it -- which
   is what separates an obstruction, which passes, from a new viewpoint, which
   stays.
4. **A localized change against the frame already kept.** Two still runs either
   side of a pause with nothing thrown are the same board twice.

What survives is the empty board, then each dart as it lands.

Step 1 asks "did anything happen?" twice over, and both questions are needed
because a dart is *small*. Measured on a real session -- a board filling 43% of
a portrait frame's width -- a landed dart covers about 90 of 65,000 analysis
pixels, so it moves the frame's mean brightness by 0.2 grey levels against a
threshold of 2.0. A mean cannot see a dart at all. Segmenting on the mean alone
therefore never split a visit: a run began when the player stepped out of shot
and ended when they stepped back in, spanning all three darts, and the session
yielded one still per visit instead of one per dart.

So stillness is two tests, and a frame must pass both. The mean catches motion
that is *large* -- the camera moving, a body crossing the frame. A count of
pixels changing past a level catches motion that is *small and bright* -- a dart
arriving. That is the same measure, and the same reasoning, step 3 already uses
to tell a dart from sensor noise; it simply belongs in both places.
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
    "frame_statistics",
    "runs_of_true",
    "still_runs",
    "sharpness",
    "changed_pixel_count",
    "obstructed",
    "viewpoint_groups",
    "viewpoint_splits",
    "choose_frames",
    "find_ffmpeg",
    "extract",
    "extract_sessions",
    "extract_folder",
    "find_videos",
    "VIDEO_SUFFIXES",
]

# Frames are analysed at this width. Small enough that a long session decodes in
# seconds, large enough that a dart landing is a clear change.
ANALYSIS_WIDTH = 192

# And at this rate, rather than the video's own. A session is minutes long and
# what it is searched for is a pause of a second or more, so decoding every
# frame of 30 fps footage buys nothing and costs a great deal: a 20-minute 4K
# recording is 36,000 frames, and holding them even at analysis size is 746 MB.
# At 5 fps the same recording is 124 MB and each pause still spans several
# frames.
ANALYSIS_FPS = 5.0


@dataclass(frozen=True)
class ExtractionSettings:
    """Thresholds, each with a reason rather than a round number.

    ``still`` is the mean absolute difference, in 0-255 grey levels, below which
    consecutive frames count as the same scene. Sensor noise on a static shot
    sits near 1; a dart entering the frame moves it far higher.

    ``min_still_frames`` is how long a scene must hold, counted in *analysis*
    frames -- so at the default 5 fps, five of them is one second. A dart takes
    a moment to stop wobbling, and a hand pausing mid-reach should not read as
    a state.

    ``still_pixels`` is the companion test: how many pixels may change by more
    than ``change_level`` between consecutive frames while the scene still counts
    as unchanged. Sensor and compression noise rarely move a pixel by 25 grey
    levels at all, so the floor sits near zero; an arriving dart is roughly 90
    pixels. The default leaves a wide margin either side, and ``--diagnose``
    prints the distribution from your own footage so it can be set on evidence
    rather than on this paragraph.

    ``obstruction_fraction`` is the share of the frame that has to change before
    a difference stops being a dart and starts being a body. Three darts are
    a few hundred pixels; someone standing at the board is half the picture.
    Anything in between is not a thing that happens.

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
    still_pixels: int = 25
    min_still_frames: int = 5
    change_level: float = 25.0
    changed_pixels: int = 10
    obstruction_fraction: float = 0.05
    analysis_width: int = ANALYSIS_WIDTH
    analysis_fps: float = ANALYSIS_FPS

    def __post_init__(self) -> None:
        if self.still <= 0:
            raise ValueError("still must be positive")
        if self.still_pixels < 0:
            raise ValueError("still_pixels cannot be negative")
        if self.min_still_frames < 1:
            raise ValueError("min_still_frames must be at least 1")
        if not 0 < self.change_level <= 255:
            raise ValueError("change_level must be in (0, 255]")
        if self.changed_pixels < 1:
            raise ValueError("changed_pixels must be at least 1")
        if not 0 < self.obstruction_fraction <= 1:
            raise ValueError("obstruction_fraction must be in (0, 1]")
        if self.analysis_width < 32:
            raise ValueError("analysis_width must be at least 32")
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be positive")


@dataclass(frozen=True)
class Extraction:
    """What came out, and enough to tell whether it went well."""

    frames_read: int
    still_runs: int
    kept: tuple[int, ...]
    dropped_as_duplicate: int
    dropped_as_obstructed: int = 0
    files: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "frames_read": self.frames_read,
            "still_runs": self.still_runs,
            "kept": list(self.kept),
            "dropped_as_duplicate": self.dropped_as_duplicate,
            "dropped_as_obstructed": self.dropped_as_obstructed,
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


def frame_statistics(
    frames: Sequence[np.ndarray], level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Both stillness measures, against the previous frame, in one pass.

    ``(mean absolute difference, pixels changed by more than ``level``)``, with
    a leading zero apiece so index *i* describes the step into frame *i* and the
    first frame is still by definition.

    Computed pair by pair rather than by stacking the whole session: a
    twenty-minute recording is six thousand analysis frames, and one float32
    stack of those is half a gigabyte for a statistic that only ever looks at
    two frames at a time.
    """
    count = len(frames)
    means = np.zeros(count, dtype=np.float64)
    changed = np.zeros(count, dtype=np.int64)
    for index in range(1, count):
        difference = np.abs(
            np.asarray(frames[index], dtype=np.int16)
            - np.asarray(frames[index - 1], dtype=np.int16)
        )
        means[index] = difference.mean()
        changed[index] = int((difference > level).sum())
    return means, changed


def runs_of_true(mask: Sequence[bool], min_length: int) -> list[tuple[int, int]]:
    """Index ranges (inclusive) over which ``mask`` held, lasting long enough."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value:
            start = index if start is None else start
            continue
        if start is not None and index - start >= min_length:
            runs.append((start, index - 1))
        start = None
    if start is not None and len(mask) - start >= min_length:
        runs.append((start, len(mask) - 1))
    return runs


def still_runs(
    differences: Sequence[float], threshold: float, min_length: int
) -> list[tuple[int, int]]:
    """Index ranges (inclusive) over which the scene barely changed.

    The mean test on its own. Kept because it is the readable half of the rule
    and is what a timeline plot shows; ``choose_frames`` applies it together
    with the pixel-count test, which is the half that can see a dart.
    """
    return runs_of_true([value <= threshold for value in differences], min_length)


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


def obstructed(
    frames: Sequence[np.ndarray], candidates: Sequence[int], settings: ExtractionSettings
) -> set[int]:
    """Which candidates are something standing in front of the board.

    Judged by a candidate's neighbours, not by itself, because nothing in a
    single frame says whether the dark shape in the middle is a person: the
    board is never located, and locating it is the model's job, not this one's.
    What is decidable is whether the view *came back*. An obstruction is a
    parenthesis -- the frames either side of it match each other and neither
    matches the middle. A camera being moved looks identical from the inside
    and different from the outside: the views either side do not match, so the
    test declines to call it an obstruction and the new viewpoint is kept.
    """
    budget = settings.obstruction_fraction * np.asarray(frames[0]).size
    level = settings.change_level

    def far(left: int, right: int) -> bool:
        return changed_pixel_count(frames[left], frames[right], level) > budget

    blocked = {
        middle
        for before, middle, after in zip(candidates, candidates[1:], candidates[2:])
        if far(before, middle) and far(middle, after) and not far(before, after)
    }

    # The first and last candidates have no pair of neighbours to sit between,
    # so the parenthesis test cannot reach them -- and a recording that opens or
    # closes with someone at the board is the ordinary case, not a rare one.
    # What decides them instead is that a board state cannot differ from the
    # state beside it by this much: three darts are a few hundred pixels and the
    # budget is thousands. An end candidate that far from its neighbour is an
    # obstruction or a camera move, and as a lone frame at one end of a
    # recording it is worth little either way.
    #
    # Measured against the nearest *surviving* candidate rather than the
    # adjacent one, because the adjacent one may be the obstruction just found
    # -- and comparing a good opening frame against the body that walked in
    # front of it would discard the frame for the body's sin.
    survivors = [index for index in candidates if index not in blocked]
    if len(survivors) >= 2:
        if far(survivors[0], survivors[1]):
            blocked.add(survivors[0])
        if far(survivors[-2], survivors[-1]):
            blocked.add(survivors[-1])
    return blocked


def viewpoint_groups(
    frames: Sequence[np.ndarray],
    kept: Sequence[int],
    settings: ExtractionSettings | None = None,
) -> list[list[int]]:
    """Split chosen stills wherever the camera ended up pointing somewhere new.

    A *session* is one fixed camera position, because landmarks are annotated
    once per session and reused across every frame in it. So a recording made
    of three camera positions is three sessions, and the split has to happen
    before annotation rather than be discovered during it.

    The judgement is the same one ``obstructed`` makes and rests on the same
    fact: consecutive board states differ by a few hundred pixels at most,
    because that is what a dart is. A jump of thousands is the camera. Asked of
    settled states rather than the raw timeline, because a body crossing the
    shot moves as much of the picture as a camera move does and is
    indistinguishable from one until it leaves again.
    """
    settings = settings or ExtractionSettings()
    if not kept:
        return []

    return [group for group, _ in viewpoint_splits(frames, kept, settings)]


def viewpoint_splits(
    frames: Sequence[np.ndarray],
    kept: Sequence[int],
    settings: ExtractionSettings | None = None,
) -> list[tuple[list[int], float]]:
    """``viewpoint_groups``, each group paired with how hard its split was.

    The size of the jump that opened a group, as a multiple of the threshold --
    ``0.0`` for the first group, which nothing opened. A recording that splits
    into twenty is either a phone picked up twenty times or a threshold sitting
    too close to the ordinary business of darts being pulled, and the two look
    identical in a count. They do not look identical in the margins: a real
    reposition clears the line many times over, and a borderline one sits just
    past it.
    """
    settings = settings or ExtractionSettings()
    if not kept:
        return []

    budget = settings.obstruction_fraction * np.asarray(frames[0]).size
    found: list[tuple[list[int], float]] = [([kept[0]], 0.0)]
    for previous, index in zip(kept, kept[1:]):
        moved = changed_pixel_count(
            frames[previous], frames[index], settings.change_level
        )
        if moved > budget:
            found.append(([index], moved / budget))
        else:
            found[-1][0].append(index)
    return found


def choose_frames(
    frames: Sequence[np.ndarray], settings: ExtractionSettings | None = None
) -> tuple[list[int], int, int, int]:
    """``(indices to keep, still runs, duplicates dropped, obstructions dropped)``."""
    settings = settings or ExtractionSettings()
    means, changed = frame_statistics(frames, settings.change_level)
    settled = [
        mean <= settings.still and count <= settings.still_pixels
        for mean, count in zip(means, changed)
    ]
    runs = runs_of_true(settled, settings.min_still_frames)

    candidates = [
        max(range(start, end + 1), key=lambda i: sharpness(frames[i]))
        for start, end in runs
    ]
    blocked = obstructed(frames, candidates, settings)

    kept: list[int] = []
    duplicates = 0
    previous: np.ndarray | None = None
    for index in candidates:
        if index in blocked:
            continue
        if previous is not None:
            moved = changed_pixel_count(previous, frames[index], settings.change_level)
            if moved < settings.changed_pixels:
                # The same board twice, either side of a pause. Nothing landed.
                duplicates += 1
                continue
        kept.append(index)
        previous = frames[index]
    return kept, len(runs), duplicates, len(blocked)


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


def _frame_size(video: Path, width: int, ffmpeg: str) -> tuple[int, int]:
    """``(width, height)`` of the scaled analysis frame.

    Learned by decoding a single frame and measuring it, rather than by parsing
    ffmpeg's report of the stream. One frame is a few kilobytes, and the length
    of it divided by the width *is* the height -- no text to misread, and no
    dependency on ffprobe being installed alongside.
    """
    probe = subprocess.run(
        [ffmpeg, "-i", str(video), "-vf", f"scale={width}:-2,format=gray",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=False,
    )
    if probe.returncode != 0 or not probe.stdout:
        detail = probe.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            f"ffmpeg could not read {video.name}:\n" + (detail[-1] if detail else "no output")
        )
    return width, len(probe.stdout) // width


def _analysis_frames(
    video: Path, width: int, fps: float, ffmpeg: str
) -> Iterator[np.ndarray]:
    """Decode the video small, grey and slow, streaming one frame at a time.

    Streamed rather than collected because the whole point of this module is
    long recordings: buffering a session's worth of decoded video before
    looking at any of it is how a twenty-minute clip turns into a gigabyte.
    """
    width, height = _frame_size(video, width, ffmpeg)
    stride = width * height

    process = subprocess.Popen(
        [ffmpeg, "-i", str(video), "-vf", f"fps={fps},scale={width}:-2,format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdout is not None
        while True:
            payload = process.stdout.read(stride)
            if len(payload) < stride:
                break
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width)
    finally:
        if process.stdout:
            process.stdout.close()
        process.wait()


VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg")


def _ensure_writable(out_dir: Path) -> None:
    """Prove the destination is writable *before* decoding, not after.

    Scanning a session takes minutes, and macOS refuses Terminal write access to
    the Desktop and Documents by default -- so the obvious place to send output
    is exactly the one that fails, and failing at the end throws away all of
    that work for a reason that was knowable at the start.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        raise PermissionError(
            f"cannot write to {out_dir}. On macOS, Terminal needs explicit "
            "permission for the Desktop and Documents folders (System Settings "
            "> Privacy & Security > Files and Folders). Writing somewhere inside "
            "the project instead — data/captures/… — avoids that entirely, and "
            "is where captures belong."
        ) from error


def _decode(video: Path, settings: ExtractionSettings, ffmpeg: str) -> list[np.ndarray]:
    frames = list(_analysis_frames(
        video, settings.analysis_width, settings.analysis_fps, ffmpeg
    ))
    if not frames:
        raise RuntimeError(f"{video.name} decoded to no frames")
    return frames


def _write_stills(
    video: Path, indices: Sequence[int], out_dir: Path,
    settings: ExtractionSettings, prefix: str, ffmpeg: str,
) -> list[Path]:
    """Pull each chosen still out of the source at full resolution.

    Filenames are zero-padded and sequential because the annotator sorts by
    filename and burst expansion depends on that order being right (#26).
    """
    files: list[Path] = []
    for position, index in enumerate(indices, start=1):
        # An analysis frame is a *time*, not a frame number in the source, so
        # the still is pulled by seeking. Seeking before the input is the fast
        # form, and the scene is by definition not moving, so landing a few
        # milliseconds either side costs nothing.
        at = index / settings.analysis_fps
        destination = out_dir / f"{prefix}-{position:04d}.jpg"
        result = subprocess.run(
            [ffmpeg, "-y", "-ss", f"{at:.3f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "2", str(destination)],
            capture_output=True, check=False,
        )
        if result.returncode != 0 or not destination.exists():
            detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(
                f"ffmpeg could not write a still at {at:.1f}s:\n"
                + (detail[-1] if detail else "no output")
            )
        files.append(destination)
    return files


def _record(
    out_dir: Path, frames_read: int, runs: int, indices: Sequence[int],
    duplicates: int, blocked: int, files: Sequence[Path],
) -> Extraction:
    extraction = Extraction(
        frames_read=frames_read, still_runs=runs, kept=tuple(indices),
        dropped_as_duplicate=duplicates, dropped_as_obstructed=blocked,
        files=tuple(files),
    )
    (out_dir / "extraction.json").write_text(
        json.dumps(extraction.to_dict(), indent=2), encoding="utf-8"
    )
    return extraction


def extract(
    video: str | Path,
    out_dir: str | Path,
    settings: ExtractionSettings | None = None,
    prefix: str = "frame",
) -> Extraction:
    """Pull one still per board state out of ``video``, into one folder.

    Correct only when the recording holds a single camera position. Use
    ``extract_sessions`` when it might not, which is most of the time.
    """
    settings = settings or ExtractionSettings()
    video, out_dir = Path(video), Path(out_dir)
    if not video.exists():
        raise FileNotFoundError(video)
    ffmpeg = find_ffmpeg()
    _ensure_writable(out_dir)

    frames = _decode(video, settings, ffmpeg)
    kept, runs, duplicates, blocked = choose_frames(frames, settings)
    files = _write_stills(video, kept, out_dir, settings, prefix, ffmpeg)
    return _record(out_dir, len(frames), runs, kept, duplicates, blocked, files)


def extract_sessions(
    video: str | Path,
    out_dir: str | Path,
    settings: ExtractionSettings | None = None,
    prefix: str = "frame",
    min_states: int = 2,
) -> list[Extraction]:
    """Extract ``video`` into one folder per camera position.

    The form to reach for when a recording was not made to a protocol -- the
    phone picked up between games, a folder of clips shot across an afternoon.
    Splitting is not a tidiness preference: landmarks are annotated once per
    folder and applied to everything in it, so stills from two camera positions
    sharing a folder means one of the two gets labels belonging to the other,
    silently and everywhere.

    ``min_states`` drops a viewpoint too small to be worth annotating. Eight
    landmark clicks to gain one image is not a trade worth making, and a
    one-state viewpoint is usually the moment during a move rather than a
    position anybody threw from.
    """
    settings = settings or ExtractionSettings()
    video, out_dir = Path(video), Path(out_dir)
    if not video.exists():
        raise FileNotFoundError(video)
    ffmpeg = find_ffmpeg()
    _ensure_writable(out_dir)

    frames = _decode(video, settings, ffmpeg)
    kept, runs, duplicates, blocked = choose_frames(frames, settings)
    groups = [
        group for group in viewpoint_groups(frames, kept, settings)
        if len(group) >= min_states
    ]

    written: list[Extraction] = []
    for number, group in enumerate(groups, start=1):
        session = out_dir / f"{video.stem.lower()}-{number:02d}"
        _ensure_writable(session)
        files = _write_stills(video, group, session, settings, prefix, ffmpeg)
        written.append(_record(
            session, len(frames), runs, group, duplicates, blocked, files
        ))
    return written


def find_videos(directory: str | Path) -> list[Path]:
    """Every video in ``directory``, in a stable order.

    Sorted by name, which for phone footage is chronological -- ``IMG_0624``
    precedes ``IMG_0625`` -- so session numbering follows the order they were
    shot in rather than whatever order the filesystem happens to return.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        and not path.name.startswith(".")
    )


def extract_folder(
    directory: str | Path,
    out_dir: str | Path,
    settings: ExtractionSettings | None = None,
    prefix: str = "frame",
    min_states: int = 2,
) -> dict[Path, list[Extraction]]:
    """Run ``extract_sessions`` over every video in a folder.

    One video failing does not stop the rest: a folder of phone footage
    reliably contains something that is not a video, or is a video ffmpeg
    dislikes, and losing an hour of decoding to the last file in the list would
    be a poor trade. Failures are raised together at the end, once everything
    that could be salvaged has been.
    """
    videos = find_videos(directory)
    results: dict[Path, list[Extraction]] = {}
    failures: list[str] = []
    for video in videos:
        try:
            results[video] = extract_sessions(
                video, out_dir, settings, prefix, min_states
            )
        except (RuntimeError, OSError) as error:
            failures.append(f"{video.name}: {error}")
    if failures and not results:
        raise RuntimeError("no video could be read:\n  " + "\n  ".join(failures))
    if failures:
        print("could not read:\n  " + "\n  ".join(failures))
    return results
