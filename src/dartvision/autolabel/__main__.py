"""``python -m dartvision.autolabel --session <folder> [--out ...] [--against ...]``

Two things, from the same reading of the pixels: propose the tips for a session
that has not been labelled, or check the ones already placed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dartvision.autolabel.session import compare_to_labels, suggest_session
from dartvision.autolabel.tips import SuggestSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dartvision.autolabel",
        description="Propose dart tips from what changed between stills",
    )
    parser.add_argument("--session", required=True, help="a folder of stills, in order")
    parser.add_argument("--out", help="write a manifest of proposals to open in the annotator")
    parser.add_argument("--against", help="an existing manifest to check against instead")
    parser.add_argument("--level", type=float, default=SuggestSettings.level)
    parser.add_argument("--max-area", type=float, default=SuggestSettings.max_area)
    parser.add_argument("--tolerance", type=float, default=30.0,
                        help="pixels of disagreement to tolerate when checking")
    args = parser.parse_args(argv)

    folder = Path(args.session)
    if not folder.is_dir():
        print(f"\n{folder} is not a folder of stills")
        return 1
    stills = sorted(p for p in folder.glob("*.jpg") if p.name != "framing-check.jpg")
    if not stills:
        print(f"\nno .jpg stills in {folder}")
        return 1

    settings = SuggestSettings(level=args.level, max_area=args.max_area)
    suggestions = suggest_session(stills, settings)

    darts = sum(1 for s in suggestions if s.event == "dart")
    unsure = [s for s in suggestions if s.event == "dart" and not s.confident]
    print(f"read {len(stills)} stills: {darts} dart(s) arrived, "
          f"{sum(1 for s in suggestions if s.event == 'cleared')} board(s) cleared")
    if unsure:
        print(f"  {len(unsure)} are pointing at the camera and could not be told "
              "end from end:")
        for frame in unsure[:8]:
            print(f"    {frame.path.name}")

    if args.against:
        rows = [json.loads(line) for line in
                Path(args.against).read_text().splitlines() if line.strip()]
        labelled = {
            (r.get("meta") or {}).get("source_file") or str(r["image_id"]).split("/")[-1]:
                r.get("tips") or []
            for r in rows
        }
        size = tuple(rows[0]["image_size"]) if rows else (1, 1)
        disagreements = compare_to_labels(suggestions, labelled, size, args.tolerance)
        print(f"\nchecked against {args.against}")
        if not disagreements:
            print("  every hand label agrees with the pixels")
        else:
            print(f"  {len(disagreements)} frame(s) to look at:")
            for name, reason in disagreements[:25]:
                print(f"    {name:<22} {reason}")
            if len(disagreements) > 25:
                print(f"    ...and {len(disagreements) - 25} more")
        print("\n  Disagreement is not proof the label is wrong — it is where one "
              "of the two readings is.")
        return 0

    if not args.out:
        print("\nPass --out to write these as a manifest, or --against to check "
              "existing labels.")
        return 0

    template = {"setup_id": folder.parent.name or "setup", "session_id": folder.name}
    size = next((s.size for s in suggestions if s.size != (0, 0)), (0, 0))
    lines = []
    for frame in suggestions:
        lines.append(json.dumps({
            "image_id": f"{template['session_id']}/{frame.path.name}",
            "setup_id": template["setup_id"],
            "session_id": template["session_id"],
            "origin": "real-thrown",
            "landmarks": [None] * 8,
            "tips": [[x, y] for x, y in frame.tips],
            "image_size": list(size),
            "meta": {"source_file": frame.path.name, "suggested": True,
                     "event": frame.event},
        }, sort_keys=True))
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {len(lines)} proposal(s) to {args.out}")
    print("  Open it with the annotator's 'resume', click the 8 landmarks, and "
          "correct the tips.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
