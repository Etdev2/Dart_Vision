"""``python -m dartvision.capture --video session.mp4 --out frames/``"""

from __future__ import annotations

import argparse
from pathlib import Path

from dartvision.capture.diagnose import render, scan_video, sweep
from dartvision.capture.frames import ExtractionSettings, extract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull one still per board state out of a recorded session",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--out", help="directory for the stills; not needed with --diagnose",
    )
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
             "in the way rather than a dart",
    )
    parser.add_argument(
        "--changed-pixels", type=int, default=ExtractionSettings.changed_pixels,
        help="pixels that must change before two stills count as different states",
    )
    args = parser.parse_args(argv)

    settings = ExtractionSettings(
        still=args.still,
        still_pixels=args.still_pixels,
        min_still_frames=args.min_still_frames,
        changed_pixels=args.changed_pixels,
        obstruction_fraction=args.obstruction_fraction,
        analysis_fps=args.analysis_fps,
    )

    if args.diagnose:
        measured, frames = scan_video(args.video, settings)
        print(render(measured, sweep(frames, settings), settings))
        return 0

    if not args.out:
        parser.error("--out is required unless you pass --diagnose")

    result = extract(args.video, args.out, settings, prefix=args.prefix)

    minutes = result.frames_read / max(args.analysis_fps, 1e-9) / 60
    print(f"scanned {result.frames_read} frames (~{minutes:.1f} minutes of video)")
    print(f"found {result.still_runs} still runs")
    print(f"dropped {result.dropped_as_duplicate} as the same board state")
    print(f"dropped {result.dropped_as_obstructed} as something in front of the board")
    print(f"kept {len(result.files)} stills in {Path(args.out)}")
    if not result.files:
        print("\nNothing kept. Either the camera never settled or a threshold is "
              "too tight for this footage.\nRun the same command with --diagnose "
              "instead of --out to see which.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
