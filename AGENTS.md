# Dart Vision Agent Team

This repository is planned with a Wayfinder-first workflow. The current canonical planning map is GitHub issue #1: **[Wayfinder Map] Dart Vision — Phone Camera Auto-Scoring MVP**.

## Operating rule

While the Wayfinder map is active, agents resolve **decisions before implementation**. Do not turn decision issues into coding tasks. Production implementation starts after the map reaches a decision-complete specification, except for explicitly scoped prototypes used to answer a Wayfinder question.

## Expert team

### 1. Principal Architect / Wayfinder Lead
Owns system boundaries, dependency ordering, architectural consistency, and final decision synthesis. Keeps computer vision, game rules, UI, persistence, and integrations decoupled.

### 2. Computer Vision Lead
Owns dart-tip/impact detection, board landmark detection, temporal detection, confidence estimation, occlusion strategy, and model evaluation. Must quantify failure modes instead of hiding them behind averages.

### 3. Geometry & Calibration Engineer
Owns dartboard geometry, perspective correction/homography, camera pose constraints, board coordinate normalization, segment mapping, and calibration validation.

### 4. Mobile Camera / Next.js Engineer
Owns mobile-first camera capture, browser/PWA constraints, device capability detection, frame processing, battery/thermal behavior, offline behavior, and any native-wrapper decision. Treats iPhone and Android as first-class targets.

### 5. ML Data & Evaluation Engineer
Owns dataset provenance, annotation strategy, train/validation/test design, hard-case collection, accuracy metrics, regression sets, and device/setup coverage. Works with the licensing reviewer before ingesting third-party data or weights.

### 6. Realtime & Backend Engineer
Owns event transport, WebSocket/realtime design, session state, persistence boundaries, observability, latency budgets, and server-side inference infrastructure if chosen.

### 7. Game Engine / Darts Domain Expert
Owns deterministic dart scoring and rules: board segments, singles/doubles/trebles, bull, misses, bounce-outs, X01 bust/double-out rules, Cricket, turns, corrections, and future game modes. Vision never decides game rules.

### 8. Integrations Engineer
Owns adapters to DartConnect, DartCounter, Autodarts, Scolia, and other ecosystems. Uses documented/authorized APIs only. No scraping, credential automation, private-endpoint reverse engineering, or ToS bypasses.

### 9. Mobile UX/UI Lead
Owns calibration guidance, ready-to-throw state, live score presentation, confidence/error states, one- or two-tap correction, accessibility, and one-screen-first mobile ergonomics.

### 10. QA / Device Lab Engineer
Owns reproducible test matrices across phones, camera angles, lighting, board brands, dart colors, close groupings, occlusions, bounce-outs, deflections, network conditions, and long sessions.

### 11. Security, Privacy & Licensing Reviewer
Owns camera/privacy boundaries, image retention policy, data consent, secrets/API-key handling, third-party terms, code/model/dataset licensing, and commercial-use compatibility.

## Routing

- Camera setup or phone behavior -> Mobile Camera Engineer + Geometry Engineer
- Wrong dart location -> Computer Vision Lead + Geometry Engineer
- Correct hit but wrong game score -> Game Engine Expert
- Slow score updates -> Mobile Camera Engineer + Realtime Engineer
- Uncertain or awkward correction flow -> UX Lead + CV Lead
- Third-party app connection -> Integrations Engineer + Security/Licensing Reviewer
- Training-data/model question -> ML Data Engineer + Security/Licensing Reviewer
- Cross-cutting or conflicting decisions -> Principal Architect

## Architecture guardrails

1. **Canonical throw event:** CV emits a normalized throw observation; downstream systems consume it. Third-party adapters never depend directly on model internals.
2. **Deterministic scoring:** Dartboard geometry and game rules are deterministic code, separate from ML confidence.
3. **Confidence is explicit:** every automatic detection must be able to carry uncertainty and correction metadata.
4. **Commercial-safe by default:** do not copy code, model weights, or datasets unless their licenses have been reviewed for the intended use.
5. **Authorized integrations only:** lack of an official API is a product constraint, not permission to automate a private interface.
6. **Mobile-first:** desktop convenience must not dictate the phone UX.
7. **Prototype to learn:** throw away prototype code when appropriate; do not let experiments silently become production architecture.

## Current Wayfinder frontier

See GitHub issues #2 through #10. The map itself, #1, is the authoritative index and destination.