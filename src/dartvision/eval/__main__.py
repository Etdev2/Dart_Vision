"""``python -m dartvision.eval --checkpoint runs/x/last.pt --manifest ... --image-root ...``"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dartvision.eval.runner import evaluate_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a checkpoint against a tier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument(
        "--split", default="none",
        choices=("none", "session", "cross_setup", "sim_to_real"),
    )
    parser.add_argument("--holdout-setup", default="")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partition", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--ledger", help="holdout ledger path; a repeat read is refused without --override"
    )
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--override-reason")
    args = parser.parse_args(argv)

    report = evaluate_checkpoint(
        args.checkpoint, args.manifest, args.image_root,
        split=args.split, holdout_setup=args.holdout_setup,
        val_fraction=args.val_fraction, seed=args.seed, partition=args.partition,
        device=args.device, batch_size=args.batch_size, out_dir=args.out_dir,
        ledger_path=args.ledger, override=args.override,
        override_reason=args.override_reason,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
