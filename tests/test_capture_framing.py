"""Measuring how much of the frame the board fills.

The one thing in the capture chain that cannot be fixed afterwards, so what is
tested is that the number is *right* on frames whose answer is known by
construction -- and that the two things which defeated the first attempt, a
ceiling and a wall, still do not.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from dartvision.capture.framing import (
    BOARD_OUTER_MM,
    RING_MM,
    dark_threshold,
    largest_dark_region,
    measure_image,
    otsu_threshold,
)
from dartvision.model.spec import DEFAULT_INPUT_PX

WIDTH, HEIGHT = 1080, 1920


def room(diameter: int, centre=(900, 520), ceiling: int = 150) -> np.ndarray:
    """A photograph of a dartboard, in the three tones one actually has.

    A dark board on a bright wall, under a mid-grey ceiling that spans the
    whole frame -- the arrangement that made the first implementation report
    the board filling 100% of the width, because the ceiling is both darker
    than the wall and larger than the board.
    """
    frame = np.full((HEIGHT, WIDTH), 225, dtype=np.uint8)
    frame[:400, :] = ceiling
    ys, xs = np.ogrid[:HEIGHT, :WIDTH]
    frame[(ys - centre[0]) ** 2 + (xs - centre[1]) ** 2 <= (diameter // 2) ** 2] = 45
    return frame


def dartboard_room(diameter: int, centre=(900, 540), doorway=True,
                   box_above=False) -> np.ndarray:
    """The scene as photographed, not as imagined.

    A dartboard is *segments*: a black outer ring and numbers ring, with
    alternating black and cream sectors inside. Its dark pixels cover about
    half its bounding box, not the pi/4 a filled disc covers -- and the first
    version of this measure assumed a disc, was tested against a fixture that
    was one, and so never noticed.

    The distractors are the ones a garage actually contains: a ceiling band
    darker than the wall and larger than the board, and a tall dark doorway
    beside it. On real footage those between them produced a measurement
    exactly twice as tall as it was wide, which no dartboard is.
    """
    frame = np.full((HEIGHT, WIDTH), 225, dtype=np.uint8)
    frame[:400, :] = 150                       # ceiling: darker, and bigger
    if doorway:
        frame[1500:1900, 30:200] = 70          # doorway: tall, dark, well clear
    if box_above:
        # The case that defeats it, and the one most boards present: a dark box
        # mounted directly behind and above the board, touching it.
        top = int(centre[0] - diameter / 2)
        frame[top - 320:top + 20, centre[1] - 190:centre[1] + 190] = 50

    ys, xs = np.ogrid[:HEIGHT, :WIDTH]
    radius = diameter / 2
    distance = np.sqrt((ys - centre[0]) ** 2 + (xs - centre[1]) ** 2)
    angle = np.arctan2(ys - centre[0], xs - centre[1])

    board = distance <= radius
    frame[board] = 215                         # cream sectors
    sector = (((angle + np.pi) / (2 * np.pi) * 20).astype(int) % 2 == 0)
    frame[board & sector] = 40                 # black sectors
    frame[board & (distance > radius * 0.88)] = 35   # outer and numbers ring
    return frame


def write(frame: np.ndarray, path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-i", "-",
         "-frames:v", "1", str(path)],
        input=frame.tobytes(), capture_output=True,
    )
    if result.returncode != 0 or not path.exists():
        pytest.skip("this ffmpeg cannot write a test image")


# --------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------

def test_otsu_splits_two_tones_where_they_separate():
    """Asserted by what the threshold selects, not by where it lands. Otsu
    returns the level at which `<=` is the dark class, so for two tones it
    returns the darker tone itself rather than a value between them."""
    grey = np.concatenate([np.full(500, 40), np.full(500, 200)]).reshape(50, 20)
    grey = grey.astype(np.uint8)

    dark = grey <= otsu_threshold(grey)

    assert dark.sum() == 500
    assert set(np.unique(grey[dark])) == {40}


def test_a_third_tone_needs_the_second_pass():
    """One split lands between the mid tone and the bright one, grouping board
    with ceiling. The second separates them."""
    grey = room(400)

    one_pass = set(np.unique(grey[grey <= otsu_threshold(grey)]))
    two_pass = set(np.unique(grey[grey <= dark_threshold(grey)]))

    assert one_pass == {45, 150}, "one split leaves the ceiling on the dark side"
    assert two_pass == {45}, "two splits put the board alone on the dark side"


def test_a_flat_image_does_not_break_the_second_pass():
    assert dark_threshold(np.full((20, 20), 128, dtype=np.uint8)) >= 0


def test_the_board_wins_over_a_larger_darker_band():
    """A ceiling band is bigger than a board and darker than a wall. It is not
    shaped like a dartboard, which is what decides it."""
    grey = room(400)
    left, top, right, bottom = largest_dark_region(grey, dark_threshold(grey))

    assert abs((right - left + 1) - 400) < 30
    assert abs((bottom - top + 1) - 400) < 30


def test_no_dark_pixels_at_all_is_not_a_crash():
    grey = np.full((40, 40), 200, dtype=np.uint8)
    assert largest_dark_region(grey, 10) == (0, 0, 0, 0)


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------

@pytest.mark.parametrize("diameter", [400, 490, 700])
def test_the_measured_board_matches_the_one_that_was_drawn(diameter, tmp_path):
    image = tmp_path / "still.png"
    write(room(diameter), image)

    measured = measure_image(image)

    assert measured.source == (WIDTH, HEIGHT)
    assert abs(measured.board[0] - diameter) / diameter < 0.08


def test_the_ring_estimate_follows_from_the_board_size(tmp_path):
    """Not an independent number: the arithmetic the whole report rests on."""
    image = tmp_path / "still.png"
    write(room(490), image)

    measured = measure_image(image)
    expected = (measured.board[0] * DEFAULT_INPUT_PX / WIDTH
                * RING_MM / BOARD_OUTER_MM)

    assert measured.ring_across == pytest.approx(expected, rel=1e-6)


def test_a_portrait_frame_loses_far_more_down_than_across(tmp_path):
    """The finding that squaring up would recover: the model scales each axis
    by its own factor, so 1920 of height is squeezed harder than 1080 of
    width, and the ring pays for it in one direction only."""
    image = tmp_path / "still.png"
    write(room(490), image)

    measured = measure_image(image)

    assert measured.ring_down < measured.ring_across / 1.5
    assert measured.ring_squared > measured.ring_down


def test_a_board_filling_the_frame_clears_the_floor(tmp_path):
    from dartvision.model.spec import MIN_RING_PX

    image = tmp_path / "still.png"
    write(room(1000, centre=(960, 540)), image)

    measured = measure_image(image)

    assert measured.fills > 0.85
    assert measured.ring_squared > MIN_RING_PX, (
        "a board filling a squared-up frame must carry the precision the "
        "model needs, or the target in the protocol is wrong"
    )


def test_a_missing_still_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no file at"):
        measure_image(tmp_path / "nothing.jpg")


def test_the_command_line_measures_a_folder_of_stills(tmp_path, capsys):
    from dartvision.capture import __main__ as cli

    for index in range(3):
        write(room(490), tmp_path / f"frame-{index:04d}.jpg")

    assert cli.main(["--framing", str(tmp_path)]) == 0
    printed = capsys.readouterr().out

    assert "measured 3 still(s)" in printed
    assert "45% of the frame's width" in printed
    assert "Too small" in printed


def test_a_folder_with_no_stills_is_a_sentence(tmp_path, capsys):
    from dartvision.capture import __main__ as cli

    assert cli.main(["--framing", str(tmp_path)]) == 1
    assert "no .jpg stills" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The board as it actually photographs: segments, not a disc
# --------------------------------------------------------------------------

def test_a_segmented_board_covers_far_less_of_its_box_than_a_disc():
    """The measurement behind the fix, so the assumption cannot come back."""
    frame = dartboard_room(490)
    dark = frame <= dark_threshold(frame)

    left, top, right, bottom = largest_dark_region(frame, dark_threshold(frame))
    box = (right - left + 1) * (bottom - top + 1)
    inside = dark[top:bottom + 1, left:right + 1].sum()

    assert inside / box < np.pi / 4 * 0.95, (
        "a real board's dark pixels do not fill a disc's share of its box"
    )


def test_the_board_is_found_beside_a_ceiling_and_a_doorway(tmp_path):
    """The two distractors that beat the first implementation, together."""
    image = tmp_path / "still.png"
    write(dartboard_room(490), image)

    measured = measure_image(image)

    assert abs(measured.board[0] - 490) / 490 < 0.10
    assert not measured.suspect
    assert 0.75 < measured.aspect < 1.35


def test_a_board_mounted_against_a_dark_box_is_reported_not_guessed(tmp_path):
    """The real failure: most boards hang against something, and dark pixels
    that touch are one region. It cannot be separated without knowing where the
    board is, which is the question — so it is declared rather than fudged."""
    image = tmp_path / "still.png"
    write(dartboard_room(490, box_above=True), image)

    measured = measure_image(image)

    assert measured.suspect, "a region far taller than wide is not a dartboard"
    assert measured.aspect > 1.6
    assert measured.diameter == min(measured.board), (
        "the merge inflated the height, so the width is the side to trust"
    )
    assert measured.ring_across == pytest.approx(
        measured.diameter * DEFAULT_INPUT_PX / WIDTH * RING_MM / BOARD_OUTER_MM
    )


def test_the_command_line_says_when_the_region_is_the_wrong_shape(tmp_path, capsys):
    """Still answers, rather than sending the reader away to measure by hand:
    the shorter side is the one the merge did not inflate."""
    from dartvision.capture import __main__ as cli

    write(dartboard_room(490, box_above=True), tmp_path / "frame-0001.jpg")

    assert cli.main(["--framing", str(tmp_path)]) == 0
    printed = capsys.readouterr().out

    assert "a dartboard is not" in printed
    assert "shorter side" in printed and "--board-width" in printed
    assert (tmp_path / "framing-check.jpg").exists()


def test_a_merged_region_is_measured_as_though_it_were_round(tmp_path):
    """The board is the same size whether or not a box is bolted above it."""
    alone = tmp_path / "alone.png"
    against = tmp_path / "against.png"
    write(dartboard_room(490, box_above=False), alone)
    write(dartboard_room(490, box_above=True), against)

    clean, merged = measure_image(alone), measure_image(against)

    assert not clean.suspect and merged.suspect
    assert abs(merged.ring_across - clean.ring_across) < 0.6, (
        "the same board must measure the same either way"
    )


def test_a_hand_measurement_overrides_the_detection(tmp_path, capsys):
    from dartvision.capture import __main__ as cli

    write(dartboard_room(490, box_above=True), tmp_path / "frame-0001.jpg")

    assert cli.main(["--framing", str(tmp_path), "--board-width", "700"]) == 0
    assert "65% of the frame's width" in capsys.readouterr().out


def test_every_measurement_leaves_a_picture_of_what_it_found(tmp_path):
    """Everything here rests on an assumption about what a photograph of a
    dartboard looks like. The cheapest way to check it held is to look."""
    from dartvision.capture.framing import draw_box

    image = tmp_path / "still.png"
    write(dartboard_room(490), image)
    check = tmp_path / "framing-check.jpg"

    measure_image(image, check_into=check)

    assert check.exists() and check.stat().st_size > 0
    assert draw_box(image, (10, 10, 200, 200), tmp_path / "other.jpg").exists()


@pytest.mark.parametrize("diameter", [360, 490, 680])
def test_a_segmented_board_measures_true_at_several_sizes(diameter, tmp_path):
    image = tmp_path / "still.png"
    write(dartboard_room(diameter, centre=(900, 540)), image)

    measured = measure_image(image)

    assert abs(measured.board[0] - diameter) / diameter < 0.10
