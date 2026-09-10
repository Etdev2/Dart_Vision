"""``python -m dartvision.audit --manifest ... --image-root ...``

Exits non-zero when a check fails, so it can gate a run rather than merely
inform one. A corpus that fails here produces numbers that mean nothing, and
the cheapest moment to discover that is before the GPU is rented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dartvision.audit.checks import FAIL, PASS, SKIPPED, WARN, audit
from dartvision.data.labels import read_jsonl
from dartvision.data.splits import LeakageError, split_for_tier

_MARK = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIPPED: "----"}


def render(report) -> str:
    lines = []
    for check in report.checks:
        lines.append(f"[{_MARK[check.status]}] {check.name}")
        lines.append(f"       {check.summary}")
    lines.append("")
    lines.append(
        f"{report.status.upper()} -- "
        + (
            "the corpus is sound"
            if report.sound
            else "numbers computed on this corpus are not trustworthy"
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run #14's dataset verification checks over a manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--image-root",
        help="enables perceptual hashing; without it the duplicate checks are skipped",
    )
    parser.add_argument(
        "--split", default="none",
        choices=("none", "session", "cross_setup", "sim_to_real"),
    )
    parser.add_argument("--holdout-setup", default="")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hash-threshold", type=int, default=5,
        help="Hamming distance in bits below which two frames are near-duplicates",
    )
    parser.add_argument("--out", help="write the full report as JSON")
    parser.add_argument(
        "--quiet", action="store_true", help="print nothing; use the exit code"
    )
    args = parser.parse_args(argv)

    annotations = list(read_jsonl(args.manifest))

    assignment = None
    split_error = None
    if args.split != "none":
        try:
            assignment = split_for_tier(
                annotations, args.split, args.holdout_setup,
                args.val_fraction, args.seed,
            )
        except LeakageError as error:
            # A tier the corpus cannot support is a finding about the corpus.
            # Report it as a failed check and run the rest.
            split_error = f"cannot build the {args.split!r} tier: {error}"

    report = audit(
        annotations,
        image_root=args.image_root,
        assignment=assignment,
        hash_threshold=args.hash_threshold,
        split_error=split_error,
    )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if not args.quiet:
        print(render(report))
    return 0 if report.sound else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
