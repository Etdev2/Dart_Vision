# Dart Vision

Dart Vision is a mobile-first automatic steel-tip darts scoring project. The goal is to let a phone camera observe a dartboard, detect throws, calculate scores, maintain game state, and connect to other dart ecosystems through authorized integrations.

## Current phase: Wayfinder planning

We are intentionally resolving the major product and architecture decisions before building production code.

- **Canonical Wayfinder map:** https://github.com/Etdev2/Dart_Vision/issues/1
- **Expert team and routing:** [`AGENTS.md`](./AGENTS.md)
- **Issue tracker / Wayfinder conventions:** [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md)
- **Initial system hypothesis:** [`docs/architecture/initial-system-hypothesis.md`](./docs/architecture/initial-system-hypothesis.md)
- **Data strategy decision:** [`docs/architecture/data-strategy-decision.md`](./docs/architecture/data-strategy-decision.md)
- **Canonical throw event (#9):** [`docs/architecture/throw-event.md`](./docs/architecture/throw-event.md) — the contract between the Brain and the app, with the TypeScript definition.

## Dart Vision Brain (AI engine)

The vision model is planned as a standalone, versioned AI engine rather than training logic buried in the app. It is trained on **synthetic imagery plus data Dart Vision captures itself** - no third-party datasets ([decision](./docs/architecture/data-strategy-decision.md)).

- **Brain Wayfinder map:** https://github.com/Etdev2/Dart_Vision/issues/12
- **DeepDarts provenance & rights (#13):** [`docs/research/13-deepdarts-provenance.md`](./docs/research/13-deepdarts-provenance.md)
- **Training compute & experiment stack (#16):** [`docs/research/16-training-compute-and-stack.md`](./docs/research/16-training-compute-and-stack.md)
- **Leakage-safe benchmark design (#14):** [`docs/research/14-leakage-safe-benchmark.md`](./docs/research/14-leakage-safe-benchmark.md)
- **Model representation (#15):** [`docs/research/15-model-representation.md`](./docs/research/15-model-representation.md)
- **Ship gates (#21):** [`docs/research/21-ship-gates.md`](./docs/research/21-ship-gates.md)
- **Alternative dart data sources (#24):** [`docs/research/24-alternative-dart-data-sources.md`](./docs/research/24-alternative-dart-data-sources.md)
- **DeepDarts rights enquiry draft (#23):** [`docs/research/23-mcnally-enquiry-draft.md`](./docs/research/23-mcnally-enquiry-draft.md)

## Getting started

**[`docs/runbook.md`](./docs/runbook.md)** — day-one sequence from clone to a rendered dataset.

**[`docs/capture-protocol.md`](./docs/capture-protocol.md)** — how to photograph and annotate your own board.

```bash
pip install -e '.[dev]' && pytest && python -m dartvision.doctor
```

## Code

First implementation, on the `prototype/` track for #17:

- `src/dartvision/geometry/board.py` — board geometry and deterministic scoring. Clean-room from published BDO dimensions, standard library only.
- `src/dartvision/geometry/calibration.py` — calibration landmarks and the image-to-board homography.
- `src/dartvision/data/labels.py` — the annotation contract, shared by synthetic (#25) and captured (#26) data.
- `src/dartvision/events.py` — the canonical `ThrowEvent`, calibration status, and the confirmation policy.
- `src/dartvision/stream/` — the temporal layer ([#3](https://github.com/Etdev2/Dart_Vision/issues/3)): per-frame detections in, throws out. Its job is memory, not detection — a dart hidden behind a later dart was seen when it landed, and [#2](https://github.com/Etdev2/Dart_Vision/issues/2) measured that this happens to about a fifth of tightly grouped darts at every camera angle.
- `src/dartvision/data/splits.py` — leakage-safe splits for [#14](https://github.com/Etdev2/Dart_Vision/issues/14)'s tiers, plus a ledger enforcing holdout discipline.
- `src/dartvision/data/torch_dataset.py` — training dataset over a scene manifest, with label-safe photometric jitter.
- `src/dartvision/audit/` — [#14](https://github.com/Etdev2/Dart_Vision/issues/14) §7's dataset verification checks: schema conformance, session grouping, perceptual-hash near-duplicates, effective dataset size, dart-count balance and the margin-to-boundary distribution.
- `src/dartvision/eval/` — tip matching, the checkpoint evaluation CLI, and the [#21](https://github.com/Etdev2/Dart_Vision/issues/21) gate report. Rectifies with the *predicted* landmarks, so landmark error is measured where it actually hurts.
- `src/dartvision/train/` — the training loop, run configuration, and the provenance every run records ([#16](https://github.com/Etdev2/Dart_Vision/issues/16)'s reproducibility contract).
- `src/dartvision/model/net.py` — the two-head Brain from [#15](https://github.com/Etdev2/Dart_Vision/issues/15): heatmap landmarks + point-detection tips on a shared `timm` backbone. Needs `pip install -e '.[train]'`.
- `src/dartvision/model/targets.py` — target encoding and decoding (heatmaps, soft-argmax, sub-cell offsets). No training framework required.
- `src/dartvision/metrics/scoring.py` — [#14](https://github.com/Etdev2/Dart_Vision/issues/14)'s metrics and [#21](https://github.com/Etdev2/Dart_Vision/issues/21)'s gates, including the risk–coverage curve and error concentration.
- `src/dartvision/geometry/camera.py` — pinhole camera over the board plane.
- `src/dartvision/annotate/` — [#26](https://github.com/Etdev2/Dart_Vision/issues/26) capture tooling: session landmark propagation, burst expansion, and placed-dart capture plans.
- `tools/annotator.html` — browser annotator, no install.
- `src/dartvision/doctor.py` — reports what this machine can run.
- `src/dartvision/synthetic/` — [#25](https://github.com/Etdev2/Dart_Vision/issues/25) scene sampling: camera poses, dart placement at controlled distances from scoring boundaries, exact labels, structured dataset generation, and a wireframe SVG preview.

Generate a scene manifest for a renderer to consume:

```bash
python -m dartvision.synthetic.generate --out data/synthetic --setups 3 --sessions 4 --images 40
```

469 tests.

Train a smoke run once a manifest is rendered:

```bash
python -m dartvision.train --manifest data/synthetic/manifest.jsonl \
    --image-root data/synthetic/images --out-dir runs/smoke \
    --epochs 3 --limit 64 --input-height 256 --input-width 256 --device cpu
```

Check a corpus before trusting a number computed on it:

```bash
python -m dartvision.audit --manifest data/synthetic/manifest.jsonl \
    --image-root data/synthetic/images --split session
```

Exits non-zero when the corpus is unsound, so it can gate a run rather than
merely inform one. It answers the question that decides whether an accuracy
figure means anything: are there darts near a wire at all?

Score a checkpoint against a benchmark tier:

```bash
python -m dartvision.eval --checkpoint runs/smoke/last.pt \
    --manifest data/synthetic/manifest.jsonl --image-root data/synthetic/images \
    --split session --partition test --ledger runs/holdout.json
```

There is no argument for *which* images to score: the tier and seed define the
test partition, the evaluator derives it, and `--ledger` refuses a second read
of it without a logged reason.

```bash
pip install -e '.[dev]' && pytest
```

## Core idea

```text
Phone camera -> dart/board detection -> calibrated board coordinates
             -> deterministic score -> ThrowEvent -> game engine/UI/integrations
```

The first planning milestone is complete when the Wayfinder map has no material unresolved decisions and can be converted into an implementation spec/backlog.