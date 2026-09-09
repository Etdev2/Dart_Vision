# Dart Vision

Dart Vision is a mobile-first automatic steel-tip darts scoring project. The goal is to let a phone camera observe a dartboard, detect throws, calculate scores, maintain game state, and connect to other dart ecosystems through authorized integrations.

## Current phase: Wayfinder planning

We are intentionally resolving the major product and architecture decisions before building production code.

- **Canonical Wayfinder map:** https://github.com/Etdev2/Dart_Vision/issues/1
- **Expert team and routing:** [`AGENTS.md`](./AGENTS.md)
- **Issue tracker / Wayfinder conventions:** [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md)
- **Initial system hypothesis:** [`docs/architecture/initial-system-hypothesis.md`](./docs/architecture/initial-system-hypothesis.md)

## Dart Vision Brain (AI engine)

The vision model is planned as a standalone, versioned AI engine rather than training logic buried in the app.

- **Brain Wayfinder map:** https://github.com/Etdev2/Dart_Vision/issues/12
- **DeepDarts provenance & rights (#13):** [`docs/research/13-deepdarts-provenance.md`](./docs/research/13-deepdarts-provenance.md)
- **Training compute & experiment stack (#16):** [`docs/research/16-training-compute-and-stack.md`](./docs/research/16-training-compute-and-stack.md)

## Core idea

```text
Phone camera -> dart/board detection -> calibrated board coordinates
             -> deterministic score -> ThrowEvent -> game engine/UI/integrations
```

The first planning milestone is complete when the Wayfinder map has no material unresolved decisions and can be converted into an implementation spec/backlog.