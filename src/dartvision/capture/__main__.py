"""Turning recorded video into labelled-ready stills.

    python -m dartvision.capture --video session.mov --out data/captures/garage
    python -m dartvision.capture --video-dir ~/Desktop/Darts --out data/captures/garage
    python -m dartvision.capture --video session.mov --diagnose

Output is always one folder per camera position, never one folder per video.
Landmarks are annotated once per folder and applied to everything in it, so two
camera positions sharing a folder means one of them is labelled against a
viewpoint it was never shot from -- silently, and in every frame. Recordings
made without a protocol routinely hold several positions, so the safe
arrangement is the only one offered rather than a flag to remember.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dartvision.capture.diagnose import render, scan_video, sweep, viewpoints
from dartvision.capture.frames import choose_frames, state_jumps
from dartvision.capture.frames import (
    Extraction,
    ExtractionSettings,
    extract_folder,
    extract_sessions,
    find_videos,
)


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{seconds % 60:04.1f}"


def _describe(sessions: list[Extraction], fps: float) -> list[str]:
    lines = []
    for session in sessions:
        folder = session.files[0].parent.name if session.files else "?"
        span = (f"{_clock(session.kept[0] / fps)}-{_clock(session.kept[-1] / fps)}"
                if session.kept else "-")
        lines.append(f"    {folder:<20} {span:>15}   {len(session.files):>3} stills")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dartvision.capture",
        description="Pull one still per board state out of recorded sessions",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="one recording")
    source.add_argument("--video-dir", help="a folder of recordings")
    source.add_argument(
        "--framing",
        help="a still, or a folder of them, to measure the board's size in — "
             "the one thing in the whole chain that cannot be fixed later",
    )

    parser.add_argument("--out", help="parent directory; not needed with --diagnose")
    parser.add_argument(
        "--diagnose", action="store_true",
        help="report what the video looks like and what each setting would keep, "
             "without writing any stills",
    )
    parser.add_argument("--prefix", default="frame")
    parser.add_argument(
        "--still", type=float, default=ExtractionSettings.still,
        help="mean grey-level change below which the scene counts as still",
    )
    parser.add_argument(
        "--still-pixels", type=int, default=ExtractionSettings.still_pixels,
        help="pixels that may change a lot between frames while the scene still "
             "counts as unchanged; this is the test that can see a dart",
    )
    parser.add_argument(
        "--min-still-frames", type=int, default=ExtractionSettings.min_still_frames,
        help="how long a scene must hold to count as a board state",
    )
    parser.add_argument(
        "--analysis-fps", type=float, default=ExtractionSettings.analysis_fps,
        help="rate the video is scanned at; lower is faster and uses less memory",
    )
    parser.add_argument(
        "--obstruction-fraction", type=float,
        default=ExtractionSettings.obstruction_fraction,
        help="share of the frame that must change before a difference is a body "
             "in the way, or the camera moving, rather than a dart",
    )
    parser.add_argument(
        "--changed-pixels", type=int, default=ExtractionSettings.changed_pixels,
        help="pixels that must change before two stills count as different states",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace session folders left by an earlier extraction of the same "
             "recording, instead of refusing to write beside them",
    )
    parser.add_argument(
        "--min-states", type=int, default=2,
        help="viewpoints yielding fewer stills than this are dropped; eight "
             "landmark clicks to gain one image is not a trade worth making",
    )
    return parser


# Failures a person caused and can fix: a destination that is not writable, a
# folder already holding an earlier extraction, a path that is not there. They
# are reported as sentences, because a traceback in front of a sentence written
# for the reader buries it under thirty lines saying nothing they can act on.
EXPECTED = (FileExistsError, FileNotFoundError, NotADirectoryError,
            PermissionError, RuntimeError)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except EXPECTED as error:
        print(f"\n{error}")
        return 1


def _report_framing(target: Path) -> int:
    """What the double ring will be worth at the model's input.

    Sampled across a session rather than measured once. The camera does not
    move within one -- that is what makes it a session -- so the stills agree,
    and where they do not the spread is itself the answer.
    """
    from dartvision.capture.framing import measure_image
    from dartvision.model.spec import MIN_RING_PX

    if target.is_dir():
        stills = sorted(target.glob("*.jpg"))[:5]
        if not stills:
            raise FileNotFoundError(f"no .jpg stills in {target}")
    else:
        stills = [target]

    measured = [measure_image(still) for still in stills]
    first = measured[0]
    across = sum(m.ring_across for m in measured) / len(measured)
    down = sum(m.ring_down for m in measured) / len(measured)
    squared = sum(m.ring_squared for m in measured) / len(measured)

    print(f"measured {len(stills)} still(s) from {target}")
    print(f"  frame            {first.source[0]} x {first.source[1]}")
    print(f"  board fills      {100 * first.fills:.0f}% of the frame's width")
    print()
    print("  the 10 mm double ring lands on, at the model's input:")
    print(f"    across         {across:5.1f} px")
    print(f"    down           {down:5.1f} px")
    print(f"    if squared up  {squared:5.1f} px   (cropping square around the "
          "board, which needs no re-shoot)")
    print(f"  it needs         {MIN_RING_PX:5.1f} px")
    print()

    worst = min(across, down)
    if worst >= MIN_RING_PX:
        print("  Good. The precision the model needs is present in these images.")
    elif squared >= MIN_RING_PX <= max(across, down):
        print("  Short as shot, but a square crop around the board recovers it — "
              "that is a change to\n  the training pipeline, not to how you "
              "record. Keep this footage.")
    else:
        print("  Too small. Nothing downstream recovers this: the ring is not in "
              "the image to be found.\n  Move closer until the board fills "
              "roughly 60% of the frame's width.")
    return 0


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:

    settings = ExtractionSettings(
        still=args.still,
        still_pixels=args.still_pixels,
        min_still_frames=args.min_still_frames,
        changed_pixels=args.changed_pixels,
        obstruction_fraction=args.obstruction_fraction,
        analysis_fps=args.analysis_fps,
    )

    if args.framing:
        return _report_framing(Path(args.framing))

    if args.diagnose:
        if not args.video:
            parser.error("--diagnose reads one video; pass --video")
        measured, frames = scan_video(args.video, settings)
        views = viewpoints(frames, settings, measured.fps)
        kept, _, _, _ = choose_frames(frames, settings)
        print(render(measured, sweep(frames, settings), settings, views,
                     state_jumps(frames, kept, settings)))
        return 0

    if not args.out:
        parser.error("--out is required unless you pass --diagnose")

    if args.video:
        found = {Path(args.video): extract_sessions(
            args.video, args.out, settings, args.prefix, args.min_states,
            args.overwrite,
        )}
    else:
        videos = find_videos(args.video_dir)
        if not videos:
            print(f"no videos in {args.video_dir}")
            return 1
        print(f"reading {len(videos)} video(s) from {args.video_dir}\n")
        found = extract_folder(
            args.video_dir, args.out, settings, args.prefix, args.min_states,
            args.overwrite,
        )

    stills = 0
    sessions = 0
    for video, produced in found.items():
        read = produced[0].frames_read if produced else 0
        minutes = read / max(args.analysis_fps, 1e-9) / 60
        print(f"  {video.name} — {read} frames (~{minutes:.1f} min), "
              f"{len(produced)} camera position(s)")
        print("\n".join(_describe(produced, args.analysis_fps)))
        stills += sum(len(s.files) for s in produced)
        sessions += len(produced)

    print(f"\n{stills} stills across {sessions} session folder(s) in {args.out}")
    if not stills:
        print("\nNothing kept. Run the same command with --diagnose instead of "
              "--out to see why.")
        return 1
    print("Annotate each folder separately — its landmarks are its own.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
