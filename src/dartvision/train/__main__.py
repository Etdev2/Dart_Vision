"""``python -m dartvision.train --manifest ... --image-root ...``"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from dartvision.train.config import TrainConfig
from dartvision.train.runner import train


def main(argv: list[str] | None = None) -> int:
    # `from __future__ import annotations` makes dataclass field types strings,
    # and fields without a default carry MISSING rather than a value -- so
    # neither the annotation object nor the default can be used directly to pick
    # an argparse type.
    TYPES = {"str": str, "int": int, "float": float, "bool": bool}

    parser = argparse.ArgumentParser(description="Train the Dart Vision Brain")
    parser.add_argument("--config", type=Path, help="JSON config; flags override it")
    for field in fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        annotation = field.type if isinstance(field.type, str) else field.type.__name__
        if annotation == "bool":
            parser.add_argument(flag, dest=field.name, action="store_true", default=None)
            parser.add_argument(
                "--no-" + field.name.replace("_", "-"),
                dest=field.name, action="store_false",
            )
        else:
            parser.add_argument(
                flag, dest=field.name, type=TYPES.get(annotation, str), default=None
            )
    args = parser.parse_args(argv)

    values: dict[str, object] = {}
    if args.config:
        values.update(json.loads(args.config.read_text(encoding="utf-8")))
    values.update(
        {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    )
    if "manifest" not in values or "image_root" not in values:
        parser.error("--manifest and --image-root are required")

    history = train(TrainConfig(**values))  # type: ignore[arg-type]
    print(json.dumps({
        "epochs": len(history),
        "first_loss": round(history[0].train_loss, 6),
        "final_loss": round(history[-1].train_loss, 6),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
