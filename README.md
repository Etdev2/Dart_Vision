# Dart Vision

Dart Vision is a mobile-first automatic steel-tip darts scoring project. The goal is to let a phone camera observe a dartboard, detect throws, calculate scores, maintain game state, and connect to other dart ecosystems through authorized integrations.

## Current phase: Wayfinder planning

We are intentionally resolving the major product and architecture decisions before building production code.

- **Canonical Wayfinder map:** https://github.com/Etdev2/Dart_Vision/issues/1
- **Expert team and routing:** [`AGENTS.md`](./AGENTS.md)
- **Issue tracker / Wayfinder conventions:** [`docs/agents/issue-tracker.md`](./docs/agents/issue-tracker.md)
- **Initial system hypothesis:** [`docs/architecture/initial-system-hypothesis.md`](./docs/architecture/initial-system-hypothesis.md)

## Core idea

```text
Phone camera -> dart/board detection -> calibrated board coordinates
             -> deterministic score -> ThrowEvent -> game engine/UI/integrations
```

The first planning milestone is complete when the Wayfinder map has no material unresolved decisions and can be converted into an implementation spec/backlog.