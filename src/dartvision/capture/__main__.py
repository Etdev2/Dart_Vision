"""``python -m dartvision.capture --video session.mp4 --out frames/``"""

from __future__ import annotations

import argparse
from pathlib import Path

from dartvision.capture.frames import ExtractionSettings, extract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull one still per board state out of a recorded session",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="directory for the stills")
    parser.add_argument("--prefix", default="frame")
    parser.add_argument(
        "--still", type=float, default=ExtractionSettings.still,
        help="mean grey-level change below which the scene counts as still",
    )
    parser.add_argument(
        "--min-still-frames", type=int, default=ExtractionSettings.min_still_frames,
        help="how long a scene must hold to count as a board state",
    )
    parser.add_argument(
        "--changed-pixels", type=int, default=ExtractionSettings.changed_pixels,
        help="pixels that must change before two stills count as different states",
    )
    args = parser.parse_args(argv)

    result = extract(
        args.video, args.out,
        ExtractionSettings(
            still=args.still,
            min_still_frames=args.min_still_frames,
            changed_pixels=args.changed_pixels,
        ),
        prefix=args.prefix,
    )

    print(f"read {result.frames_read} frames")
    print(f"found {result.still_runs} still runs")
    print(f"dropped {result.dropped_as_duplicate} as the same board state")
    print(f"kept {len(result.files)} stills in {Path(args.out)}")
    if not result.files:
        print("\nNothing kept. Either the camera never settled, or --still is too low "
              "for this footage; try raising it.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
