# Vision and board-calibration pipeline

> Research record for **[Wayfinder Research] Select the vision and board-calibration pipeline** (issue #3), parent map #1.
>
> Much of this ticket was answered by building it. The Brain map (#12) selected the single-frame representation in #15, proved it trains in #17, and gated it in #21. What that work did *not* address is the question the product actually asks — *did a dart just land?* — which is where the new design in §3 comes from.

## 1. Recommendation

Three stages, each doing one thing:

1. **Detect the whole board, every settled frame.** A decomposed two-head model on a shared backbone: 8 landmark heatmaps with soft-argmax, and an anchor-free point head for dart tips (#15). Not a detector of *new* darts — a detector of *all* darts.
2. **Rectify through a homography estimated from the predicted landmarks**, into board millimetres, then score deterministically from geometry (`dartvision.geometry`). No learning in the scoring step at all.
3. **Remember across frames.** `dartvision.stream.ThrowSegmenter` holds the darts it has committed and decides which observations mean something. Its purpose is memory, not detection.

Frame differencing is in the pipeline, but only as a **gate**: it says whether the scene is still. It never says where a dart is.

## 2. Why not the obvious design

The standard approach — subtract consecutive frames, find the blob that appeared, call it the dart — is the first thing anyone tries, and it fails in four ways that this project can now be specific about.

**Camera shake, lighting change and a hand each produce a blob.** A difference image cannot tell them apart from a dart. Every one of those is routine during a visit.

**A missed frame corrupts the state permanently.** The difference is taken against a history, so once the history is wrong it stays wrong for the rest of the visit. Whole-board detection re-establishes the full truth every frame, so one bad frame costs exactly one frame.

**A dart that lands partly hidden is never detected.** There is no blob to find, and no later frame will produce one.

**And the decisive one: occlusion is not rare.** #2 measured that roughly **21% of tightly grouped darts have their tip hidden behind another dart**, and that the rate is flat from 10° to 75° of elevation — it is set by the barrel's width, not the camera angle. A treble-20 grouping is exactly the throw the product exists to score, and exactly the case where a single frame cannot see all three darts.

What saves the score is that the hidden dart *was* visible when it landed, before the dart now covering it arrived. That is a fact about time, not about pixels, and no single-frame model of any quality recovers it. It is the entire reason stage 3 exists.

## 3. The temporal layer

`ThrowSegmenter` consumes `FrameObservation`s — tips already rectified into board millimetres, plus two flags from the caller — and emits events: `throw`, `bounce_out`, `visit_complete`, `board_cleared`, `calibration_lost`, `calibration_restored`.

Four rules carry all of the behaviour.

**A dart must be seen repeatedly before it is committed.** One frame is noise; the model will occasionally fire on a shadow or a wire. Two consecutive settled frames is an event.

**A committed dart is never removed by absence alone.** Absence is weak evidence — hidden and gone look identical. Removal requires that the absence be *unexplained*.

**A later dart is a standing explanation.** When a dart lands while an earlier one is out of sight, the earlier one's absence stops counting against it, permanently. Not for a few frames — the new dart does not stop hiding it.

**An unsettled frame is dropped entirely, including its absences.** This is the rule that stops a hand reaching for the darts from reading as three simultaneous bounce-outs. Dropping detections during motion is obvious; dropping *absences* during motion is the part that is easy to miss and produces the most spectacular failure.

### 3.1 What that buys, case by case

| Hard case from the ticket | How the design handles it |
| --- | --- |
| **Close groupings** | Temporal memory. The darts are committed as they land, while each is still visible. |
| **Dart occlusion** | Same mechanism. #2 quantified it at ~21% for tight groups, so this is the common case, not the exotic one. |
| **Bounce-outs** | A committed dart vanishing while its neighbours stay visible, with no later dart to explain it, is a bounce-out. All darts vanishing together is retrieval — the board clearing, not three simultaneous bounce-outs. |
| **Deflections** | A deflected dart that lands anywhere on the board is an ordinary throw; nothing special is needed. One that leaves the board is never observed at all, and the application must offer manual entry — see §5. |
| **Camera shake** | Landmarks are re-detected every frame. When they move more than a few millimetres from the set the current calibration was built on, the segmenter emits `calibration_lost` and stops scoring. A stale homography produces confident, wrong scores, which is worse than no score (#13's cliff, and why `CalibrationStatus` is a separate object). |
| **Lighting variation** | Domain randomization in #25's renderer plus label-safe photometric jitter in the training dataset. Not a pipeline feature — a data feature. |
| **Board and dart colours** | Same: randomized in synthetic, varied deliberately across sessions in #26's capture protocol. The landmarks are wire *intersections*, which are geometric features rather than colour features, precisely so that a worn or differently-coloured board does not move them. |

## 4. The approaches not chosen

**Segmentation (per-pixel masks).** A mask must still be reduced to a point, and a barrel's mask centroid is not its tip — it sits somewhere along the shaft that moves with the viewing angle. That reduction throws away the sub-millimetre precision #15's budget is denominated in, at considerably higher cost per frame.

**Classical CV for calibration** — Hough circles for the rings, colour thresholding for the red and green beds. This is the traditional route and it is genuinely cheap, but it keys on exactly the properties that vary most: ring colour on a worn board, and lighting. The landmarks chosen in #15 are wire intersections for the same reason. Worth keeping as a fallback initialiser if the learned landmark head ever proves the slower half of the model, but not as the baseline.

**Unified keypoints-as-objects** — one representation for landmarks and tips, YOLO-style. This is not rejected; it is **Challenger A** in #15, and the comparison is a deliverable of #17 rather than a decision to take here. The baseline is decomposed because landmarks and tips have opposite structure: exactly 8 landmarks, always the same 8, in a known order, versus 0–3 tips with no identity at all.

## 5. What the pipeline still does not cover

Stated plainly, because a gap named is a gap the UI can handle.

**A dart that misses the board entirely is never seen.** There is no tip to detect and no event to emit. The application must allow manual entry of a miss; the throw-event contract already carries `Source.MANUAL` for exactly this.

**A bounce-out from behind a later dart is undetectable.** Once dart 3 stands in front of dart 1, dart 1's absence carries no information. #2 established that a second viewpoint is the only thing that separates hidden from gone. The segmenter therefore keeps the dart, and a test pins that behaviour deliberately: believing the absence would lose a genuinely scored dart from roughly a fifth of tight groupings, which is far more common than a bounce-out from behind another dart. It is the cheaper of two errors, not a free one.

## 6. What must be measured before committing

Three of the five have been measured. The other two need data that does not exist yet.

| Measurement | Status |
| --- | --- |
| Landmark and tip error in board millimetres, on a leakage-safe split | **Built** — `python -m dartvision.eval` produces #21's gate report |
| Precision achievable at a given camera angle | **Measured** — #2, `tools/camera_topology.py` |
| Occlusion rate for tight groupings | **Measured** — #2, ~21%, flat across elevation |
| **Cluster-case accuracy** (≥90% per-dart on 2+ darts within 15 mm) | Needs the training campaign. This is #2's escalation trigger as well as #21's gate. |
| **Segmenter thresholds against real video** | Not measured. `match_mm=15.0` and `confirm_frames=2` are reasoned from #21's p95 tip-error gate and from what a phone's frame rate makes cheap, not observed. They need a pass over real capture footage. |

The second of those is the honest weak point of this ticket, and it is cheap to close once #26 produces video rather than stills.

## 7. A consequence for #4

The gate is not only a correctness device. Running a full forward pass on every frame of a 30 fps stream is the expensive way to do this, and it is unnecessary: a visit contains a handful of *settled* moments, and everything between them is a dart in flight or a hand in the way. Gating on stillness cuts inference from ~30 evaluations per second to a few per visit.

That changes the shape of #4 entirely. The question stops being "can a phone run this model in real time" and becomes "can a phone run this model a few times per visit" — which almost any runtime can. **#4 should be decided against the gated rate, not the frame rate.**
