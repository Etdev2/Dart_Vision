# The canonical throw event

> Resolves **[Wayfinder Research] Define the canonical throw-event and game-engine boundary** (issue #9).
>
> Authoritative definition: [`src/dartvision/events.py`](../../src/dartvision/events.py). The TypeScript below mirrors it, and a test asserts the serialized field set matches, so the two cannot drift silently.

## 1. The boundary

`AGENTS.md` sets the guardrail this implements. Four responsibilities, kept apart:

| Stage | Answers | Deterministic? |
| --- | --- | --- |
| **Vision** | Where is the tip? How sure am I? | No — this is the model |
| **Geometry** | Which segment is that point in? How close to a boundary? | **Yes** — tested, no ML |
| **The event** | What did this dart score, and how much should you trust it? | Carries both |
| **Game engine** | Bust? Checkout? Whose turn? | **Yes** — rules, no camera |

An event says *"this dart is a treble 20, worth 60"*. It never says whether that busted the leg.

That separation is what makes scoring testable without a rulebook, rules testable without a camera, and third-party adapters (#7) independent of model internals. A test asserts no rule vocabulary — bust, checkout, remaining, leg — appears anywhere in the serialized event.

## 2. Two corrections to the earlier draft

The draft in `initial-system-hypothesis.md` has been superseded in two ways by what the Brain work established.

**Board millimetres are canonical, not image-normalized coordinates.** An image coordinate means nothing without the calibration that produced it, so it cannot be the field a game engine, a database, or an integration adapter depends on. Image coordinates are still carried, optionally, for drawing an overlay — but the rectified board point is the contract.

**Calibration is a separate concern, not a field on a throw.** Below four landmarks there is **no score for any dart** — a cliff, not a slope (#13). The application must be able to distinguish *"the board is not calibrated"* from *"no darts detected"*; collapsing them produces a frozen scoreboard with no explanation, which is the worst state the UI can present. Hence `CalibrationStatus` as its own object.

## 3. TypeScript

```ts
type Source = "camera" | "manual" | "correction";
type CalibrationQuality = "good" | "marginal" | "unusable";

interface Detection {
  segment: number;        // 1–20, 25 for either bull, 0 for a miss
  multiplier: 0 | 1 | 2 | 3;
  score: number;          // always segment × multiplier
  notation: string;       // "T20" | "D16" | "S5" | "DB" | "SB" | "MISS"
  confidence: number;     // 0–1, the model's own
  margin_mm: number;      // distance to the nearest score-changing boundary
  boundary: string | null; // e.g. "outer treble wire", "20|1 wire"
}

interface ThrowEvent {
  id: string;
  session_id: string;
  captured_at: string;    // ISO 8601
  calibration_id: string;
  source: Source;

  board_x_mm: number;     // canonical: rectified board plane, origin at the bull
  board_y_mm: number;
  image_x: number | null; // display only, meaningless without calibration_id
  image_y: number | null;

  detected: Detection;

  player_id: string | null;
  turn_id: string | null;
  sequence: number | null; // 1–3 within the turn
  corrects: string | null; // id of the event this replaces
  image_ref: string | null;
}

interface CalibrationStatus {
  calibration_id: string;
  session_id: string;
  established_at: string;
  landmarks_found: number;
  landmarks_expected: 4 | 8;
  landmark_confidence: number[];
  board_coverage: number;     // fraction of the frame's shorter side
  scoring_possible: boolean;  // landmarks_found >= 4
  quality: CalibrationQuality;
}
```

## 4. Example

```json
{
  "calibration": {
    "calibration_id": "cal_A4",
    "session_id": "ses_7Q2",
    "established_at": "2026-09-09T20:02:55.000Z",
    "landmarks_found": 8,
    "landmarks_expected": 8,
    "landmark_confidence": [
      0.99,
      0.98,
      0.99,
      0.97,
      0.96,
      0.98,
      0.97,
      0.95
    ],
    "board_coverage": 0.211,
    "scoring_possible": true,
    "quality": "good"
  },
  "throw": {
    "id": "thr_01J8Z3",
    "session_id": "ses_7Q2",
    "captured_at": "2026-09-09T20:14:07.412Z",
    "calibration_id": "cal_A4",
    "source": "camera",
    "board_x_mm": 0.0,
    "board_y_mm": 107.0,
    "image_x": 0.5124,
    "image_y": 0.3311,
    "detected": {
      "segment": 20,
      "multiplier": 3,
      "score": 60,
      "notation": "T20",
      "confidence": 0.93,
      "margin_mm": 0.4,
      "boundary": "outer treble wire"
    },
    "player_id": "ply_alex",
    "turn_id": "trn_19",
    "sequence": 2,
    "corrects": null,
    "image_ref": null
  },
  "needs_confirmation": true
}
```

## 5. Confidence alone cannot decide what to confirm

#5 set the budget: flag roughly the least-confident 15% of darts, about two confirmations a leg. But **model confidence is the wrong signal on its own.**

Confidence answers *"how sure am I where the tip is"*. It does not answer *"does that uncertainty change the score"*. Those come apart constantly:

- 2 mm of uncertainty, 20 mm from any boundary → certain, whatever the model says.
- The same 2 mm, 0.5 mm from a treble wire → a coin flip.

So `ConfirmationPolicy` combines both: a throw is auto-scored only when the model is confident **and** the dart is far enough from a boundary that its residual error cannot cross one. Near a boundary, only a very confident call stands alone.

This costs nothing — the margin falls out of the geometry for free — and it is a strictly better routing signal than confidence alone. It is the decision #10's correction flow is built around.

## 6. What the game engine owns

Everything the event deliberately omits: X01 bust and double-out rules, remaining score, checkout detection, turn and player rotation, Cricket marks and closures, leg and set structure, and undo semantics beyond the single-event `corrects` link.

The engine consumes an ordered stream of `ThrowEvent`s and is a pure function of them, which is what makes it exhaustively testable without a camera and lets a corrected event replay the whole leg deterministically.
