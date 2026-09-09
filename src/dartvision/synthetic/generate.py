"""CLI: write a synthetic scene manifest (and previews) for a renderer to consume.

    python -m dartvision.synthetic.generate --out data/synthetic --seed 0

The manifest is the contract between label generation and rendering. Keeping
them separate means labels can be validated -- and a dataset's composition
reviewed -- before any GPU or renderer time is spent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dartvision.data.labels import write_jsonl
from dartvision.synthetic.dataset import DatasetSpec, generate_dataset, summarize
from dartvision.synthetic.preview import scene_to_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--setups", type=int, default=3)
    parser.add_argument("--sessions", type=int, default=4, help="sessions per setup")
    parser.add_argument("--images", type=int, default=50, help="images per session")
    parser.add_argument("--landmarks", type=int, choices=(4, 8), default=8)
    parser.add_argument("--min-coverage", type=float, default=0.12)
    parser.add_argument("--previews", type=int, default=6, help="SVG previews to write")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = DatasetSpec(
        setups=args.setups,
        sessions_per_setup=args.sessions,
        images_per_session=args.images,
        landmark_count=args.landmarks,
        min_board_coverage=args.min_coverage,
    )

    scenes = list(generate_dataset(np.random.default_rng(args.seed), spec))
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    write_jsonl((s.annotation for s in scenes), out / "manifest.jsonl")

    if args.previews > 0:
        preview_dir = out / "previews"
        preview_dir.mkdir(exist_ok=True)
        step = max(1, len(scenes) // args.previews)
        for scene in scenes[::step][: args.previews]:
            name = scene.annotation.image_id.replace("/", "_")
            (preview_dir / f"{name}.svg").write_text(scene_to_svg(scene), encoding="utf-8")

    report = {"seed": args.seed, "spec": vars(args) | {"out": str(out)}, **summarize(scenes)}
    report["spec"].pop("out", None)
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summarize(scenes), indent=2))
    print(f"\nwrote {len(scenes)} scenes to {out}/manifest.jsonl")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
