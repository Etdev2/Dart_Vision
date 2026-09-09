# Dart Vision

Dart Vision is a mobile-first automatic steel-tip darts scoring project. The goal is to let a phone camera observe a dartboard, detect throws, calculate scores, maintain game state, and connect to other dart ecosystems through authorized integrations.

## Current phase: Wayfinder planning

We are intentionally resolving the major product and architecture decisions before building production code.

- **Canonical Wayfinder map:** https://github.com/Etdev2/Dart_Vision/issues/1
- **Expert team and routing:** [`AGENTS.md`](./AGENTS.md)
- **Issue tracker / Wayfinder conventions:** [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md)
- **Initial system hypothesis:** [`docs/architecture/initial-system-hypothesis.md`](./docs/architecture/initial-system-hypothesis.md)
- **Data strategy decision:** [`docs/architecture/data-strategy-decision.md`](./docs/architecture/data-strategy-decision.md)

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

## Code

First implementation, on the `prototype/` track for #17:

- `src/dartvision/geometry/board.py` — board geometry and deterministic scoring. Clean-room from published BDO dimensions, standard library only.
- `src/dartvision/geometry/calibration.py` — calibration landmarks and the image-to-board homography.

64 tests.

```bash
pip install -e '.[dev]' && pytest
```

## Core idea

```text
Phone camera -> dart/board detection -> calibrated board coordinates
             -> deterministic score -> ThrowEvent -> game engine/UI/integrations
```

The first planning milestone is complete when the Wayfinder map has no material unresolved decisions and can be converted into an implementation spec/backlog.