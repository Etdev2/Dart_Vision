"""Tests for pulling stills out of a recorded session.

The analysis is pure functions over arrays, so most of this needs no video at
all -- which is the point of keeping the decoding at the edge. One test does
build a real video and run the whole thing, because the ffmpeg wiring is
exactly the part that unit tests cannot vouch for.
"""

from __future__ import annotations

import pathlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dartvision.capture import (
    ExtractionSettings,
    changed_pixel_count,
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
    kept, runs, duplicates, _ = choose_frames(session_frames())
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
    kept, runs, duplicates, _ = choose_frames(frames)

    assert runs == 2
    assert len(kept) == 1
    assert duplicates == 1


def test_a_single_dart_is_not_mistaken_for_the_same_board():
    """A perceptual hash cannot see this, which is why it is not used. At
    analysis width a dart spans about ten pixels and a difference hash
    downsamples to 8x8, so an entire dart fits inside one hash cell."""
    before, after = board(darts=1, noise=0.0), board(darts=2, noise=0.0)
    assert changed_pixel_count(before, after, level=25.0) > 10

    # And noise alone must not look like a dart.
    noisy = board(darts=1, seed=99, noise=2.0)
    assert changed_pixel_count(board(darts=1, seed=1, noise=2.0), noisy, level=25.0) < 10


def test_a_video_of_nothing_but_movement_keeps_nothing():
    kept, runs, _, _ = choose_frames([moving(seed=i) for i in range(30)])
    assert runs == 0 and kept == []


def test_the_kept_frame_is_the_sharpest_of_its_run():
    """Focus breathing within a held shot: the crispest frame is the one worth
    keeping, because wire sharpness is what the landmark head reads.

    Both still thresholds are opened right up so this tests the *selection*
    alone. A blur large enough to measure is also large enough to break a still
    run -- on the mean and on the pixel count alike -- and the two concerns
    should not be tangled in one assertion.
    """
    from PIL import Image, ImageFilter

    crisp = board(darts=1, noise=0.0)
    blurred = np.asarray(
        Image.fromarray(crisp).filter(ImageFilter.GaussianBlur(1.4)), dtype=np.uint8
    )
    frames = [blurred, blurred, crisp, blurred, blurred]
    kept, runs, _, _ = choose_frames(
        frames,
        ExtractionSettings(min_still_frames=3, still=255.0, still_pixels=10_000_000),
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


def encode_still(ffmpeg: str, frame: np.ndarray, path) -> None:
    """Write one frame as a photograph, or skip if this ffmpeg cannot."""
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-i", "-",
         "-frames:v", "1", str(path)],
        input=frame.tobytes(), capture_output=True,
    )
    if result.returncode != 0 or not path.exists():
        pytest.skip(f"{ffmpeg} cannot write a test image")


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

    # Encoded slowly on purpose. Analysis runs at 5 fps and a board state must
    # hold for a second, so what the fixture needs is realistic *duration* --
    # eight frames at 4 fps is the two-second pause the protocol asks for,
    # where eight frames at 30 fps is a quarter of a second and finds nothing.
    result = subprocess.run(
        [ffmpeg, "-y", "-framerate", "4", "-i", str(source / "%05d.png"),
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

    # 44 source frames at 4 fps is 11 seconds, which is 55 frames at the 5 fps
    # analysis rate.
    assert result.frames_read > 40
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


def test_an_unwritable_destination_fails_before_the_video_is_decoded(tmp_path, monkeypatch):
    """Scanning a session takes minutes, and macOS refuses Terminal write access
    to the Desktop by default -- so the obvious place to send output is exactly
    the one that fails. Failing at the end throws away all that work for a
    reason that was knowable at the start.

    The claim tested is the *ordering*, so the decoder is replaced with
    something that fails loudly if it is ever reached. Real permissions cannot
    be used: the suite may run as root, which bypasses them.
    """
    from dartvision.capture import frames as module

    video = tmp_path / "session.mp4"
    video.write_bytes(b"not really a video, and never read")

    monkeypatch.setattr(module, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(module, "_analysis_frames", lambda *a, **k: pytest.fail(
        "the video was decoded before the destination was checked"))

    real_mkdir = pathlib.Path.mkdir

    def refuse(self, *args, **kwargs):
        if "blocked" in str(self):
            raise PermissionError(1, "Operation not permitted")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", refuse)

    with pytest.raises(PermissionError, match="cannot write to"):
        extract(video, tmp_path / "blocked" / "frames")


def test_a_missing_video_is_caught_before_anything_else(tmp_path):
    """The order matters: a bad path should not first spend minutes decoding."""
    with pytest.raises(FileNotFoundError):
        extract(tmp_path / "nope.mp4", tmp_path / "out")


# --------------------------------------------------------------------------
# The regression this module was rewritten for
# --------------------------------------------------------------------------

def unattended_board(darts: int = 0, seed: int = 0) -> np.ndarray:
    """A frame shot the way a session actually gets recorded.

    The phone is propped and pointed up at the board, so the player is never in
    shot: between one dart and the next, *nothing* in the frame moves except the
    dart itself. The board fills a fraction of a portrait frame and the rest is
    a flat wall -- the proportions measured off a real session, where the board
    spanned 43% of the frame's width.
    """
    height, width = 341, 192                       # 1080x1920 at analysis width
    rng = np.random.default_rng(seed)
    frame = np.full((height, width), 230.0)        # a pale wall
    centre_y, centre_x, radius = 110, 96, 41       # the board, 43% of the width
    ys, xs = np.ogrid[:height, :width]
    inside = (ys - centre_y) ** 2 + (xs - centre_x) ** 2 <= radius ** 2
    frame[inside] = 60.0
    for index in range(darts):
        y, x = centre_y - 20 + index * 14, centre_x - 12 + index * 12
        frame[y:y + 30, x:x + 3] = 240.0           # ~90 px of protruding dart
    return np.clip(frame + rng.normal(0, 1.2, frame.shape), 0, 255).astype(np.uint8)


def unattended_visit() -> list[np.ndarray]:
    """One visit: an empty board, then three darts, nobody else in frame."""
    frames: list[np.ndarray] = []
    for darts in range(4):
        if darts:
            frames.append(unattended_board(darts - 1, seed=900 + darts))  # in flight
        frames += [unattended_board(darts, seed=darts * 20 + i) for i in range(10)]
    return frames


def test_a_landing_dart_is_too_small_for_the_mean_to_see():
    """The measurement behind the rewrite, asserted so it cannot quietly change."""
    before, after = unattended_board(1, seed=0), unattended_board(2, seed=0)
    mean = float(np.abs(after.astype(float) - before.astype(float)).mean())

    assert mean < ExtractionSettings.still, (
        "a dart must move the frame mean by less than the still threshold -- "
        "that is precisely why segmenting on the mean alone loses darts"
    )
    assert changed_pixel_count(before, after, level=25.0) > 50, (
        "and must be plainly visible to the pixel-count test that replaced it"
    )


def test_every_dart_of_an_unattended_visit_is_kept():
    """The bug from session-01: three darts thrown with nobody in shot came back
    as a single still, because only a person walking into frame ever ended a
    run. Each board state must now get its own."""
    kept, runs, duplicates, _ = choose_frames(unattended_visit())

    assert runs == 4, "the empty board and each dart in turn"
    assert len(kept) == 4
    assert duplicates == 0


def test_the_old_mean_only_rule_would_have_lost_them():
    """Guards the fix itself: relax the pixel test and the old failure returns."""
    blind = ExtractionSettings(still_pixels=10_000_000)
    kept, runs, _, _ = choose_frames(unattended_visit(), blind)

    assert runs == 1 and len(kept) == 1


def test_a_frame_mid_flight_is_never_the_one_kept():
    """session-01's first still had a dart in mid-air. A moving dart changes
    pixels, so the frame holding it cannot be inside a still run."""
    frames = unattended_visit()
    kept, _, _, _ = choose_frames(frames)

    in_flight = {1, 12, 23}          # where unattended_visit puts the throws
    assert not (set(kept) & in_flight)


def test_a_negative_pixel_budget_is_refused():
    with pytest.raises(ValueError):
        ExtractionSettings(still_pixels=-1)


def obstruction(seed: int = 0) -> np.ndarray:
    """Someone standing at the board, pulling their darts out."""
    rng = np.random.default_rng(seed)
    frame = unattended_board(0, seed=seed).astype(float)
    frame[:, 40:150] = 90.0
    return np.clip(frame + rng.normal(0, 1.2, frame.shape), 0, 255).astype(np.uint8)


def test_someone_standing_at_the_board_is_not_a_board_state():
    """A person pulling darts stands still, and a still person reads as a still
    scene. The run is recognised by its neighbours: the board before and the
    board after match each other, and neither matches the middle."""
    frames = (
        [unattended_board(3, seed=i) for i in range(10)]
        + [obstruction(seed=300 + i) for i in range(10)]
        + [unattended_board(0, seed=400 + i) for i in range(10)]
    )
    kept, runs, _, blocked = choose_frames(frames)

    assert runs == 3, "all three stretches are genuinely still"
    assert blocked == 1
    assert len(kept) == 2, "the full board and the empty board, not the body"


def test_a_camera_move_is_not_mistaken_for_an_obstruction():
    """The view after a move does not match the view before it, so the middle
    run is a new viewpoint to keep rather than a body to discard."""
    def shifted(darts, seed):
        return np.roll(unattended_board(darts, seed=seed), 40, axis=1)

    frames = (
        [unattended_board(1, seed=i) for i in range(10)]
        + [unattended_board(2, seed=50 + i) for i in range(10)]
        + [shifted(2, seed=500 + i) for i in range(10)]
        + [shifted(3, seed=600 + i) for i in range(10)]
    )
    kept, runs, _, blocked = choose_frames(frames)

    assert runs == 4
    assert blocked == 0, "a new viewpoint is not an obstruction"
    assert len(kept) == 4


@pytest.mark.parametrize("value", [0.0, 1.5, -0.2])
def test_an_impossible_obstruction_fraction_is_refused(value):
    with pytest.raises(ValueError):
        ExtractionSettings(obstruction_fraction=value)


# --------------------------------------------------------------------------
# Footage that was not shot to a protocol
# --------------------------------------------------------------------------

def moved_visit() -> list[np.ndarray]:
    """Two camera positions in one recording: the phone was picked up."""
    def shifted(darts, seed):
        return np.roll(unattended_board(darts, seed=seed), 70, axis=1)

    return (
        [unattended_board(1, seed=i) for i in range(10)]
        + [unattended_board(2, seed=40 + i) for i in range(10)]
        + [shifted(0, seed=500 + i) for i in range(10)]
        + [shifted(1, seed=600 + i) for i in range(10)]
        + [shifted(2, seed=700 + i) for i in range(10)]
    )


def test_viewpoint_groups_split_where_the_camera_moved():
    from dartvision.capture.frames import viewpoint_groups

    frames = moved_visit()
    kept, _, _, _ = choose_frames(frames)
    groups = viewpoint_groups(frames, kept)

    assert [len(group) for group in groups] == [2, 3]
    assert groups[0][-1] < groups[1][0]


def test_a_steady_recording_is_one_group():
    from dartvision.capture.frames import viewpoint_groups

    frames = unattended_visit()
    kept, _, _, _ = choose_frames(frames)

    assert len(viewpoint_groups(frames, kept)) == 1


def test_grouping_an_empty_extraction_is_not_an_error():
    from dartvision.capture.frames import viewpoint_groups

    assert viewpoint_groups(unattended_visit(), []) == []


def test_find_videos_ignores_everything_that_is_not_one(tmp_path):
    from dartvision.capture.frames import find_videos

    for name in ("IMG_0625.MOV", "IMG_0624.mp4", "notes.txt", ".DS_Store",
                 "._IMG_0624.mp4"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "subfolder").mkdir()

    found = find_videos(tmp_path)

    assert [path.name for path in found] == ["IMG_0624.mp4", "IMG_0625.MOV"], (
        "filename order, which for phone footage is the order they were shot in"
    )


def test_find_videos_needs_a_directory(tmp_path):
    from dartvision.capture.frames import find_videos

    video = tmp_path / "one.mp4"
    video.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        find_videos(video)


def test_a_moved_camera_becomes_two_session_folders(ffmpeg, tmp_path):
    """The question this answers: footage shot without a protocol, where the
    phone was picked up partway. Splitting is not tidiness — landmarks are
    annotated once per folder, so a shared folder mislabels one position."""
    from dartvision.capture.frames import extract_sessions

    video = tmp_path / "IMG_0624.mp4"
    encode(ffmpeg, moved_visit(), video)
    sessions = extract_sessions(video, tmp_path / "out")

    assert len(sessions) == 2
    folders = [session.files[0].parent.name for session in sessions]
    assert folders == ["img_0624-01", "img_0624-02"]
    for session in sessions:
        assert (session.files[0].parent / "extraction.json").exists()
        assert len(session.files) >= 2


def test_a_viewpoint_too_small_to_annotate_is_dropped(ffmpeg, tmp_path):
    """Eight landmark clicks to gain one image is not a trade worth making."""
    from dartvision.capture.frames import extract_sessions

    video = tmp_path / "IMG_0624.mp4"
    encode(ffmpeg, moved_visit(), video)

    assert len(extract_sessions(video, tmp_path / "a", min_states=3)) == 1
    assert len(extract_sessions(video, tmp_path / "b", min_states=1)) == 2


def test_a_folder_of_recordings_is_read_in_order(ffmpeg, tmp_path):
    from dartvision.capture.frames import extract_folder

    source = tmp_path / "Darts"
    source.mkdir()
    for name in ("IMG_0624.mp4", "IMG_0625.mp4"):
        encode(ffmpeg, unattended_visit(), source / name)
    (source / "notes.txt").write_text("not a video")

    results = extract_folder(source, tmp_path / "out")

    assert [path.name for path in results] == ["IMG_0624.mp4", "IMG_0625.mp4"]
    assert all(len(sessions) == 1 for sessions in results.values())


def test_one_unreadable_video_does_not_lose_the_rest(ffmpeg, tmp_path, capsys):
    """A folder of phone footage reliably contains something ffmpeg dislikes,
    and losing an hour of decoding to the last file would be a poor trade."""
    from dartvision.capture.frames import extract_folder

    source = tmp_path / "Darts"
    source.mkdir()
    encode(ffmpeg, unattended_visit(), source / "IMG_0624.mp4")
    (source / "IMG_0625.mp4").write_bytes(b"not actually a video")

    results = extract_folder(source, tmp_path / "out")

    assert [path.name for path in results] == ["IMG_0624.mp4"]
    assert "IMG_0625" in capsys.readouterr().out


def test_a_folder_with_nothing_readable_raises(ffmpeg, tmp_path):
    from dartvision.capture.frames import extract_folder

    source = tmp_path / "Darts"
    source.mkdir()
    (source / "IMG_0625.mp4").write_bytes(b"not actually a video")

    with pytest.raises(RuntimeError, match="no video could be read"):
        extract_folder(source, tmp_path / "out")


# --------------------------------------------------------------------------
# The board framed properly, which is where the viewpoint rule got it wrong
# --------------------------------------------------------------------------

def closely_framed(darts: int = 0, seed: int = 0) -> np.ndarray:
    """A board filling enough of the frame that a visit's darts are big.

    The framing the precision budget asks for, and the framing that broke the
    viewpoint rule: three darts here cover more of the frame than the line that
    splits a session, so clearing them looked exactly like the camera moving.
    Measured on a real 12-minute recording that split into twenty viewpoints,
    every one of them between 1.0x and 1.3x the line.
    """
    height, width = 341, 192
    rng = np.random.default_rng(seed)
    frame = np.full((height, width), 225.0)
    centre_y, centre_x, radius = 150, 96, 88
    ys, xs = np.ogrid[:height, :width]
    frame[(ys - centre_y) ** 2 + (xs - centre_x) ** 2 <= radius ** 2] = 55.0
    for index in range(darts):
        y, x = centre_y - 40 + index * 26, centre_x - 30 + index * 24
        frame[y:y + 64, x:x + 20] = 245.0        # ~1280 px apiece
    return np.clip(frame + rng.normal(0, 1.2, frame.shape), 0, 255).astype(np.uint8)


def closely_framed_visits(visits: int = 3) -> list[np.ndarray]:
    """Several visits from one fixed camera: fill the board, clear it, repeat."""
    frames: list[np.ndarray] = []
    seed = 0
    for visit in range(visits):
        for darts in range(4):
            seed += 1
            frames.append(closely_framed(max(darts - 1, 0), seed=900 + seed))
            frames += [closely_framed(darts, seed=seed * 20 + i) for i in range(10)]
        seed += 1
        frames += [closely_framed(0, seed=5000 + seed * 10 + i) for i in range(10)]
    return frames


def test_clearing_the_board_exceeds_the_line_when_framed_closely():
    """The measurement behind the rule change, asserted so it cannot drift."""
    from dartvision.capture.frames import changed_pixel_count

    settings = ExtractionSettings()
    full, empty = closely_framed(3, seed=1), closely_framed(0, seed=2)
    budget = settings.obstruction_fraction * full.size
    jump = changed_pixel_count(full, empty, settings.change_level)

    assert jump > budget, "pulling three darts must clear the line"
    assert jump < 2 * budget, (
        "and only barely — which is exactly what made it indistinguishable "
        "from a camera move by size alone"
    )


def test_pulling_darts_is_not_a_camera_move_however_big_they_are():
    """IMG_0632: twelve minutes from one fixed phone that came back as twenty
    sessions. The board returning to how it looked at the start of the visit is
    a board being cleared, not a camera pointing somewhere new."""
    from dartvision.capture.frames import viewpoint_groups

    frames = closely_framed_visits()
    kept, _, _, _ = choose_frames(frames)
    groups = viewpoint_groups(frames, kept)

    assert len(groups) == 1, (
        f"one fixed camera throughout, got {len(groups)} viewpoints"
    )


def test_the_old_size_only_rule_would_have_split_it(monkeypatch):
    """Guards the fix: stop looking for the view coming back and the twenty
    spurious sessions return."""
    from dartvision.capture import frames as module

    monkeypatch.setattr(module, "RETURN_LOOKBACK", 0)
    frames = closely_framed_visits()
    kept, _, _, _ = choose_frames(frames)

    assert len(module.viewpoint_groups(frames, kept)) > 1


def test_a_real_move_still_splits_a_closely_framed_recording():
    """The rule must not have been softened into never splitting anything."""
    from dartvision.capture.frames import viewpoint_groups

    before = closely_framed_visits(visits=2)
    after = [np.roll(frame, 60, axis=1) for frame in closely_framed_visits(visits=2)]
    frames = before + after

    kept, _, _, _ = choose_frames(frames)
    groups = viewpoint_groups(frames, kept)

    assert len(groups) == 2, f"a phone moved somewhere new, got {len(groups)}"


def test_a_zero_lookback_means_none_rather_than_all():
    """`x[-0:]` is the whole list, not an empty one, so the bound has to be
    spelled with a guard or the number means the opposite of what it says."""
    from dartvision.capture import frames as module

    assert "if RETURN_LOOKBACK else []" in Path(module.__file__).read_text()


# --------------------------------------------------------------------------
# Re-extracting a recording that was already extracted
# --------------------------------------------------------------------------

def test_stale_session_folders_are_found_by_the_videos_own_name(tmp_path):
    from dartvision.capture.frames import existing_sessions

    out = tmp_path / "garage"
    for name in ("img_0632-01", "img_0632-02", "img_0635-01", "img_0632x-01",
                 "img_0632-01-notes"):
        (out / name).mkdir(parents=True)
    (out / "img_0632-03.txt").write_text("not a folder")

    found = existing_sessions(tmp_path / "IMG_0632.MOV", out)

    assert [path.name for path in found] == ["img_0632-01", "img_0632-02"], (
        "another recording's sessions are not this one's to touch"
    )


def test_clearing_removes_only_that_recordings_sessions(tmp_path):
    from dartvision.capture.frames import clear_sessions

    out = tmp_path / "garage"
    for name in ("img_0632-01", "img_0632-02", "img_0635-01"):
        (out / name).mkdir(parents=True)
        (out / name / "frame-0001.jpg").write_bytes(b"")

    clear_sessions(tmp_path / "IMG_0632.MOV", out)

    assert sorted(path.name for path in out.iterdir()) == ["img_0635-01"]


def test_existing_sessions_of_a_fresh_destination_is_empty(tmp_path):
    from dartvision.capture.frames import existing_sessions

    assert existing_sessions(tmp_path / "IMG_0632.MOV", tmp_path / "nothing") == []


def test_re_extracting_refuses_to_write_beside_the_old_folders(ffmpeg, tmp_path):
    """A second run can split the same recording differently, and the leftovers
    are indistinguishable from the new ones by name. Annotating that mixture
    means labelling the same frames twice under two landmark sets."""
    from dartvision.capture.frames import extract_sessions

    video = tmp_path / "IMG_0632.mp4"
    encode(ffmpeg, unattended_visit(), video)
    out = tmp_path / "garage"
    extract_sessions(video, out)

    with pytest.raises(FileExistsError, match="earlier extraction"):
        extract_sessions(video, out)


def test_the_refusal_comes_before_the_decoding(ffmpeg, tmp_path, monkeypatch):
    """Scanning a session takes minutes; discovering the conflict afterwards
    throws all of it away for a reason knowable at the start."""
    from dartvision.capture import frames as module

    video = tmp_path / "IMG_0632.mp4"
    encode(ffmpeg, unattended_visit(), video)
    out = tmp_path / "garage"
    module.extract_sessions(video, out)

    def refuse(*args, **kwargs):
        raise AssertionError("decoded before checking the destination")

    monkeypatch.setattr(module, "_decode", refuse)
    with pytest.raises(FileExistsError):
        module.extract_sessions(video, out)


def test_overwrite_replaces_the_earlier_extraction(ffmpeg, tmp_path):
    from dartvision.capture.frames import extract_sessions

    video = tmp_path / "IMG_0632.mp4"
    encode(ffmpeg, unattended_visit(), video)
    out = tmp_path / "garage"

    first = extract_sessions(video, out)
    (first[0].files[0].parent / "left-over.jpg").write_bytes(b"")

    second = extract_sessions(video, out, overwrite=True)

    assert len(second) == len(first)
    assert not (second[0].files[0].parent / "left-over.jpg").exists()


def test_a_folder_run_names_every_conflict_before_writing_anything(ffmpeg, tmp_path):
    """A folder of recordings is an hour of decoding. Refusing the fourth video
    after writing the first three leaves a half-extraction to unpick by hand."""
    from dartvision.capture.frames import extract_folder

    source = tmp_path / "Darts"
    source.mkdir()
    for name in ("IMG_0624.mp4", "IMG_0625.mp4", "IMG_0626.mp4"):
        encode(ffmpeg, unattended_visit(), source / name)

    out = tmp_path / "garage"
    (out / "img_0624-01").mkdir(parents=True)
    (out / "img_0626-01").mkdir(parents=True)

    with pytest.raises(FileExistsError) as raised:
        extract_folder(source, out)

    message = str(raised.value)
    assert "IMG_0624.mp4" in message and "IMG_0626.mp4" in message
    assert "IMG_0625.mp4" not in message, "only the ones that actually conflict"
    assert not (out / "img_0625-01").exists(), "nothing written before refusing"


def test_a_folder_run_proceeds_once_told_to_overwrite(ffmpeg, tmp_path):
    from dartvision.capture.frames import extract_folder

    source = tmp_path / "Darts"
    source.mkdir()
    encode(ffmpeg, unattended_visit(), source / "IMG_0624.mp4")

    out = tmp_path / "garage"
    extract_folder(source, out)
    results = extract_folder(source, out, overwrite=True)

    assert [path.name for path in results] == ["IMG_0624.mp4"]


# --------------------------------------------------------------------------
# Photographs, which arrive as files rather than frames
# --------------------------------------------------------------------------

def test_find_stills_separates_what_ffmpeg_cannot_open(tmp_path):
    """iPhones shoot HEIC by default and ffmpeg does not read it. Reporting a
    file that plainly is a photograph as 'not a photograph' helps nobody."""
    from dartvision.capture.frames import find_stills

    for name in ("b.JPG", "a.jpg", "c.png", "d.HEIC", "notes.txt", ".hidden.jpg"):
        (tmp_path / name).write_bytes(b"")

    readable, unreadable = find_stills(tmp_path)

    assert [p.name for p in readable] == ["a.jpg", "b.JPG", "c.png"]
    assert [p.name for p in unreadable] == ["d.HEIC"]


def test_photographs_from_several_angles_become_several_sessions(ffmpeg, tmp_path):
    """Landmarks are annotated once per folder, so an angle is a folder —
    nothing about the input being files rather than frames changes that."""
    from dartvision.capture.frames import collect_stills

    source = tmp_path / "T2"
    source.mkdir()
    for angle, shift in enumerate([0, 70, -65], start=1):
        for shot in range(2):
            frame = np.roll(unattended_board(shot, seed=angle * 10 + shot), shift, axis=1)
            encode_still(ffmpeg, frame, source / f"IMG_{7000 + angle * 10 + shot}.jpg")

    sessions = collect_stills(source, tmp_path / "out")

    assert len(sessions) == 3
    assert [len(s.files) for s in sessions] == [2, 2, 2]
    assert [s.files[0].parent.name for s in sessions] == ["t2-01", "t2-02", "t2-03"]


def test_sorting_photographs_copies_rather_than_moves(ffmpeg, tmp_path):
    """The originals are the only copy of a session that cannot be shot again."""
    from dartvision.capture.frames import collect_stills

    source = tmp_path / "T2"
    source.mkdir()
    for shot in range(2):
        encode_still(ffmpeg, unattended_board(shot, seed=shot), source / f"a{shot}.jpg")

    collect_stills(source, tmp_path / "out")

    assert sorted(p.name for p in source.iterdir()) == ["a0.jpg", "a1.jpg"]


def test_a_folder_of_heic_says_what_to_do(tmp_path):
    from dartvision.capture.frames import collect_stills

    source = tmp_path / "T2"
    source.mkdir()
    (source / "IMG_0001.HEIC").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="Most Compatible"):
        collect_stills(source, tmp_path / "out")


def test_a_change_of_orientation_splits_without_measuring(ffmpeg, tmp_path):
    """A portrait photograph and a landscape one were not taken from the same
    position, and their pixels cannot be compared anyway."""
    from dartvision.capture.frames import collect_stills

    source = tmp_path / "T2"
    source.mkdir()
    for shot in range(2):
        encode_still(ffmpeg, unattended_board(shot, seed=shot), source / f"a{shot}.jpg")
    for shot in range(2):
        turned = np.rot90(unattended_board(shot, seed=50 + shot))
        encode_still(ffmpeg, np.ascontiguousarray(turned), source / f"b{shot}.jpg")

    sessions = collect_stills(source, tmp_path / "out")

    assert len(sessions) == 2


def test_re_sorting_photographs_is_refused_then_allowed(ffmpeg, tmp_path):
    from dartvision.capture.frames import collect_stills

    source = tmp_path / "T2"
    source.mkdir()
    for shot in range(2):
        encode_still(ffmpeg, unattended_board(shot, seed=shot), source / f"a{shot}.jpg")
    out = tmp_path / "out"
    collect_stills(source, out)

    with pytest.raises(FileExistsError, match="earlier sort"):
        collect_stills(source, out)
    assert len(collect_stills(source, out, overwrite=True)) == 1
