# Dart Vision — Initial System Hypothesis

> **Status: hypothesis, not an architecture decision.** The Wayfinder map in GitHub issue #1 exists to validate or replace these assumptions before implementation.

## Product loop

1. Player mounts/positions a phone with the dartboard fully visible.
2. Dart Vision detects board landmarks and validates calibration.
3. The camera watches for a new throw.
4. Vision estimates the dart impact/tip position and confidence.
5. Geometry maps the normalized impact point to a board segment and multiplier.
6. A canonical `ThrowEvent` is emitted.
7. A deterministic game engine applies the throw to X01/Cricket/etc.
8. The phone updates the score immediately.
9. Low-confidence or wrong detections can be corrected in one or two taps.
10. Authorized integration adapters may forward normalized score/throw events to supported third-party systems.

## Candidate component boundaries

```text
Phone Camera
    |
    v
Capture / Frame Sampler
    |
    v
Throw Change Detector
    |
    v
CV Dart + Board Landmark Detector
    |
    v
Calibration / Homography / Board Coordinates
    |
    v
Deterministic Segment Mapper
    |
    v
Canonical ThrowEvent
   /        |         \
  v         v          v
Game     Storage   Integration Adapters
Engine      |       (authorized APIs only)
  |         |          |
  +---------+----------+
            |
            v
      Mobile-first UI
```

## Why this boundary is promising

- The ML system answers **where the dart appears to have landed**, not the rules of the game.
- Board geometry converts coordinates to a segment deterministically, making scoring testable.
- A canonical event prevents DartConnect/DartCounter/Autodarts-specific concepts from leaking into computer vision.
- Confidence and correction can be preserved end-to-end.
- The inference runtime can move between browser, native wrapper, edge/server GPU, or hybrid without rewriting the game engine.

## Candidate `ThrowEvent` shape

This is intentionally provisional until Wayfinder issue #9 is resolved.

```ts
type ThrowEvent = {
  id: string;
  sessionId: string;
  playerId?: string;
  turnId?: string;
  capturedAt: string;

  impact: {
    xNormalized: number;
    yNormalized: number;
  };

  detected: {
    segment: number | 25;
    multiplier: 0 | 1 | 2 | 3;
    score: number;
    notation: string; // e.g. T20, D16, S5, DB, MISS
    confidence: number;
  };

  calibrationId: string;
  source: "camera" | "manual-correction";
  correctionOf?: string;
};
```

## Most important unknowns

1. Can a single mounted phone hit the eventual accuracy contract across realistic dart/lighting/board conditions?
2. Is dart-tip keypoint detection sufficient, or is temporal/multi-view information required for close groupings and occlusion?
3. Can inference stay on-device/mobile-web at acceptable frame rate and battery cost, or does the MVP need a server inference service?
4. How much calibration friction will players tolerate?
5. What official third-party score-ingest surfaces are actually available?
6. Which existing datasets/models are commercially usable?

Those unknowns map directly to Wayfinder issues #2–#10.