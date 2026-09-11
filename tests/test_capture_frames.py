"""Tests for pulling stills out of a recorded session.

The analysis is pure functions over arrays, so most of this needs no video at
all -- which is the point of keeping the decoding at the edge. One test does
build a real video and run the whole thing, because the ffmpeg wiring is
exactly the part that unit tests cannot vouch for.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from dartvision.capture import (
    ExtractionSettings,
    choose_frames,
    extract,
    find_ffmpeg,
    frame_differences,
    sharpness,
    still_runs,
)

HEIGHT, WIDTH = 96, 192


def board(darts: int = 0, seed: int = 0, noise: float = 1.0) -> np.ndarray:
    """A frame: a fixed background, plus a bright blob per dart."""
    rng = np.random.default_rng(seed)
    frame = np.full((HEIGHT, WIDTH), 40.0)
    # Something static with detail in it, so sharpness means something.
    frame[::4, :] = 200.0
    frame[:, ::6] = 180.0
    for index in range(darts):
        y, x = 30 + index * 12, 60 + index * 15
        frame[y:y + 7, x:x + 7] = 255.0
    return np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)


def moving(seed: int = 0) -> np.ndarray:
    """A frame mid-throw: the whole scene shifted, as a hand or a dart does."""
    rng = np.random.default_rng(seed)
    return np.clip(board(seed=seed) + rng.normal(60, 40, (HEIGHT, WIDTH)), 0, 255).astype(np.uint8)


def session_frames() -> list[np.ndarray]:
    """Empty board, then three darts, with movement between each."""
    frames: list[np.ndarray] = []
    for darts in range(4):
        frames += [board(darts, seed=darts * 10 + i) for i in range(8)]   # settled
        if darts < 3:
            frames += [moving(seed=100 + darts * 5 + i) for i in range(4)]
    return frames


# --------------------------------------------------------------------------
# The analysis
# --------------------------------------------------------------------------

def test_movement_registers_and_stillness_does_not():
    still = [board(seed=i) for i in range(4)]
    assert frame_differences(still)[1:].max() < 4.0

    mixed = [board(seed=0), moving(seed=1)]
    assert frame_differences(mixed)[1] > 20.0


def test_a_single_frame_has_no_differences():
    assert len(frame_differences([board()])) == 1
    assert len(frame_differences([])) == 0


def test_still_runs_need_to_hold_for_long_enough():
    """A hand pausing mid-reach is not a board state."""
    differences = [0, 0, 0, 0, 0, 9, 9, 0, 0, 9, 0, 0, 0, 0]
    assert still_runs(differences, threshold=1.0, min_length=4) == [(0, 4), (10, 13)]
    assert still_runs(differences, threshold=1.0, min_length=2) == [(0, 4), (7, 8), (10, 13)]


def test_a_run_reaching_the_end_of_the_video_still_counts():
    assert still_runs([9, 0, 0, 0], threshold=1.0, min_length=3) == [(1, 3)]


def test_nothing_still_yields_no_runs():
    assert still_runs([9, 9, 9], threshold=1.0, min_length=1) == []


def test_sharpness_prefers_the_crisper_frame():
    """The focus measure that decides which frame of a still run to keep."""
    from PIL import Image, ImageFilter

    crisp = board(darts=2)
    soft = np.asarray(
        Image.fromarray(crisp).filter(ImageFilter.GaussianBlur(1.6)), dtype=np.uint8
    )
    assert sharpness(crisp) > sharpness(soft) * 2


# --------------------------------------------------------------------------
# Choosing
# --------------------------------------------------------------------------

def test_one_frame_is_kept_per_board_state():
    kept, runs, duplicates = choose_frames(session_frames())
    assert runs == 4
    assert len(kept) == 4, "expected the empty board and each dart as it landed"
    assert duplicates == 0


def test_the_same_board_twice_is_not_two_states():
    """Two pauses with nothing thrown between them. A player stepping back and
    forward should not produce a second copy of the same board."""
    frames = (
        [board(1, seed=i) for i in range(8)]
        + [moving(seed=50 + i) for i in range(4)]
        + [board(1, seed=200 + i) for i in range(8)]     # nothing was thrown
    )
    kept, runs, duplicates = choose_frames(frames)

    assert runs == 2
    assert len(kept) == 1
    assert duplicates == 1


def test_a_single_dart_is_not_mistaken_for_the_same_board():
    """A perceptual hash cannot see this, which is why it is not used. At
    analysis width a dart spans about ten pixels and a difference hash
    downsamples to 8x8, so an entire dart fits inside one hash cell."""
    from dartvision.capture import changed_pixel_count

    before, after = board(darts=1, noise=0.0), board(darts=2, noise=0.0)
    assert changed_pixel_count(before, after, level=25.0) > 10

    # And noise alone must not look like a dart.
    noisy = board(darts=1, seed=99, noise=2.0)
    assert changed_pixel_count(board(darts=1, seed=1, noise=2.0), noisy, level=25.0) < 10


def test_a_video_of_nothing_but_movement_keeps_nothing():
    kept, runs, _ = choose_frames([moving(seed=i) for i in range(30)])
    assert runs == 0 and kept == []


def test_the_kept_frame_is_the_sharpest_of_its_run():
    """Focus breathing within a held shot: the crispest frame is the one worth
    keeping, because wire sharpness is what the landmark head reads.

    The still threshold is opened right up so this tests the *selection* alone.
    A blur large enough to measure is also large enough to break a still run,
    and the two concerns should not be tangled in one assertion.
    """
    from PIL import Image, ImageFilter

    crisp = board(darts=1, noise=0.0)
    blurred = np.asarray(
        Image.fromarray(crisp).filter(ImageFilter.GaussianBlur(1.4)), dtype=np.uint8
    )
    frames = [blurred, blurred, crisp, blurred, blurred]
    kept, runs, _ = choose_frames(
        frames, ExtractionSettings(min_still_frames=3, still=255.0)
    )

    assert runs == 1
    assert kept == [2]


@pytest.mark.parametrize("kwargs", [
    {"still": 0.0}, {"min_still_frames": 0}, {"change_level": 0.0},
    {"changed_pixels": 0}, {"analysis_width": 8},
])
def test_impossible_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        ExtractionSettings(**kwargs)


# --------------------------------------------------------------------------
# End to end, through ffmpeg
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ffmpeg():
    """An ffmpeg that can also *build* a test video.

    Playwright ships a minimal screen-recording build with no image demuxer and
    no pipe protocol, which can decode but cannot encode a fixture. Rather than
    pretend the end-to-end path is covered on such a machine, this skips and
    says so: the test runs wherever a normal ffmpeg is installed, which is
    where the capture work actually happens.
    """
    try:
        binary = find_ffmpeg()
    except RuntimeError as error:
        pytest.skip(str(error))

    return binary


def encode(ffmpeg: str, frames: list[np.ndarray], video) -> None:
    """Write a test video, or skip if this ffmpeg cannot.

    Asked by trying rather than by inspecting the feature list -- the demuxer
    listing names formats in a description column too, so a substring check
    reported a capability this build does not have.
    """
    from PIL import Image

    source = video.parent / "frames"
    source.mkdir(exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).convert("RGB").resize(
            (frame.shape[1] * 4, frame.shape[0] * 4)
        ).save(source / f"{index:05d}.png")

    result = subprocess.run(
        [ffmpeg, "-y", "-framerate", "30", "-i", str(source / "%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(video)],
        capture_output=True, check=False,
    )
    if result.returncode != 0 or not video.exists():
        pytest.skip(
            f"{ffmpeg} cannot encode a test video (it may be a minimal build). "
            "Install a normal ffmpeg to exercise the end-to-end path."
        )


def test_a_recorded_session_yields_one_still_per_dart(ffmpeg, tmp_path):
    """The whole path: write a video, read it back, keep four frames.

    The decoding and the frame-selection filter are the parts unit tests cannot
    vouch for -- in particular that the frames written back out are the ones
    that were chosen.
    """
    from PIL import Image

    video = tmp_path / "session.mp4"
    encode(ffmpeg, session_frames(), video)

    result = extract(video, tmp_path / "out")

    assert result.frames_read > 30
    assert len(result.files) == 4, [f.name for f in result.files]
    assert [f.name for f in result.files] == [f"frame-{i:04d}.jpg" for i in range(1, 5)]
    assert (tmp_path / "out" / "extraction.json").exists()

    # The stills must get brighter as darts accumulate -- i.e. the frames written
    # really are the ones chosen, in order.
    means = [np.asarray(Image.open(f).convert("L"), dtype=float).mean() for f in result.files]
    assert means == sorted(means), means


def test_a_missing_video_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract(tmp_path / "nope.mp4", tmp_path / "out")
