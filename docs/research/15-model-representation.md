# Model Representation for Board Landmarks and Dart Tips

> Research record for **[Brain Research] Choose the model representation for board landmarks and dart tips** (issue #15), parent map #12.

## 1. The framing error worth avoiding

DeepDarts' contribution was **modeling keypoints as objects** — detecting each keypoint as a small bounding box rather than regressing a heatmap. The motivation was sound: heatmap regression assumes **one instance per channel**, and dart tips are up to three instances of the *same* class, sometimes clustered within millimetres. Heatmaps genuinely cannot separate those.

But the reference implementation then applied that one representation to **all seven points** (#13: `bbox_size: 0.025`, a ~20 px box at 800 px input, for calibration points and dart tips alike). That is the error worth not repeating, because the two sub-problems have **opposite structure**:

| | Calibration landmarks | Dart tips |
| --- | --- | --- |
| Count | Exactly 4, always | 0–3, variable |
| Class | 4 **semantically distinct** points | 1 class, repeated |
| Instances per class | Exactly one | Up to three, possibly clustered |
| Spatial context | Large, rigid, planar structure | Small, thin, self-occluding |
| What limits accuracy | Sub-pixel precision → homography quality | Instance separation and occlusion |
| Best-fit representation | **Heatmap + soft-argmax** | **Keypoints-as-objects / point detection** |

Heatmaps are a near-perfect fit for the landmarks — four distinct classes, exactly one instance each, and soft-argmax yields **sub-pixel** localization plus a natural confidence signal from peak sharpness. Using a coarse box representation there sacrifices precision for no benefit, since the multi-instance problem that motivated boxes does not exist for the landmarks.

## 2. The precision budget — why landmark precision is the whole game

From #13's board constants: the double and treble rings are **`w_double_treble` = 10 mm** wide, on a board of radius 225.5 mm. The ring is therefore **~4.4% of the board radius**.

At a typical framing where the board spans most of an 800 px crop, 10 mm is on the order of **10–20 px**. So:

- **Tip localization error must be a few millimetres** to resolve single/double/treble reliably. #14's margin-to-boundary metric exists precisely to measure this.
- **Landmark error is worse than it looks**, because it propagates through the homography and is *amplified* away from the calibration points. A small landmark error becomes a larger board-coordinate error at the edge of the board — where doubles live.

> These figures are derived from board geometry and framing assumptions, not measured. #17 should confirm the px-per-mm relationship on real captures.

**Consequence:** landmark sub-pixel accuracy is not a refinement, it is the precision floor for every dart in the image. A representation choice that trades landmark precision for architectural uniformity is trading away the thing that matters most.

## 3. The data strategy changes the answer

The [data strategy decision](../architecture/data-strategy-decision.md) — synthetic plus our own capture — makes decomposition *more* attractive, not just cleaner:

- **Synthetic data is strongest at geometry.** Board landmarks are pure projective geometry on a rigid, dimensionally standardised object. A renderer produces exact landmark ground truth in unlimited pose and lighting diversity. This is the best case for sim-to-real.
- **Synthetic data is weakest at appearance.** Dart tips depend on metal specularity, thin barrels, translucent flights, motion blur — precisely where a renderer's material realism is hardest and the domain gap is widest.

A decomposed architecture lets each stage be trained on the data that suits it: **synthetic-heavy for landmarks, real-heavy for tips.** A single unified head forces one data mixture on both sub-problems and gets the worst of each. This also gives #25 a sharper hypothesis to test — its landmark/tip split reporting should show a much smaller sim-to-real gap on landmarks.

## 4. Candidates

| Option | Verdict |
| --- | --- |
| **Decomposed: heatmap landmarks + point-detection tips** (shared backbone, two heads) | **Recommended baseline** |
| **Unified keypoints-as-objects** (YOLOX-style, modern clean-room reimplementation of DeepDarts) | **Challenger** — the published approach; a fair comparison is required, not optional |
| **Rectify-then-detect** (landmarks → homography → detect tips in canonical rectified space) | **Second challenger** — highest upside, see §4.1 |
| Pure heatmap for all 7 points | **Rejected** — cannot separate clustered same-class tips. This is DeepDarts' original, correct objection. |
| Segmentation → derive tips from masks | **Rejected** — tip localization from a mask boundary is imprecise, and precision is the binding constraint (§2) |
| Transformer set prediction (RT-DETR / DETR-family, Apache-2.0) | **Deferred** — naturally handles variable counts with no NMS, but data-hungry, and we are generating our own corpus. Revisit if #26 produces a large set. |
| Direct 8-DoF homography regression | **Rejected as primary** — loses per-landmark confidence, which #13 showed is operationally required (the sub-4-landmark cliff). Possible auxiliary loss. |

### 4.1 Rectify-then-detect deserves a real trial

Detect the 4 landmarks, compute the homography, warp to a canonical face-on board, then detect tips **in rectified space**.

**Why it is attractive:** it removes perspective variation from the tip problem entirely. Every tip detection then happens in one canonical frame at a fixed scale. That directly attacks the failure this whole project is exposed to — the reference model's drop from 94.7% face-on to 84.0% at varied angles (#13). It also means the tip detector can be smaller, since it no longer has to be invariant to viewpoint.

**Why it is a challenger and not the baseline:** errors compound. A poor homography corrupts the rectification and the tip stage inherits it, with no recovery path. It stands or falls on landmark reliability — which is exactly what §3 says synthetic data should be able to deliver. If #25 confirms strong synthetic landmark transfer, this becomes the favourite.

### 4.2 A geometric prior — corrected, and better than first stated

> **Correction.** An earlier version of this section claimed that landmark sets "inconsistent with a valid homography" are provably wrong and could be rejected on a residual. **That is false for exactly four landmarks.** Any four points in general position admit an *exact* homography onto any other four, so the residual is identically zero regardless of how wrong the predictions are. There is no such signal to extract from four points. Verified in `tests/test_calibration.py::test_four_landmarks_have_no_residual_by_construction`.

What four landmarks *can* reveal is narrower:

- **Crossed or non-convex ordering.** A perspective projection preserves the convex position and cyclic order of four coplanar points, so a crossed quadrilateral is genuinely impossible and rejectable.
- **Near-degeneracy** — collinear triples, coincident points, an ill-conditioned homography.

Both are implemented in `assess_landmarks`, and neither needs the network's own confidence — so they remain a useful independent second opinion for the "board not calibrated" state.

**The better fix is to stop using only four landmarks.** With more than four correspondences the homography becomes over-determined, which buys two real things:

1. **A genuine least-squares residual**, which *is* the self-consistency signal the original claim wanted. Measured: with 8 landmarks, 2 px of injected noise produces a residual over 1 mm, while 4 landmarks report nothing.
2. **Noise averaging.** Measured over 200 trials at 1.5 px landmark noise, eight landmarks give lower board-coordinate error than four (`test_eight_landmarks_beat_four_under_noise`). Given §2's precision budget, that is a direct accuracy gain.

DeepDarts' four-point contract was *its* choice. Since Dart Vision generates and annotates its own data (#25, #26), we are not bound by it. The implemented eight-point set adds the same four wire-intersection angles on the **outer treble wire** — identical feature type, no new annotation ambiguity, twice the constraints.

**Recommendation: predict 8 landmarks, not 4.** Keep the 4-point path for compatibility with the #13 contract and for ablation.

## 5. Non-negotiables carried in

- **Permissive licences only** (#16). Backbone from `timm` (Apache-2.0); YOLOX (Apache-2.0) for the challenger. **No Ultralytics** — AGPL-3.0 reaching trained weights.
- **Clean-room** (#13) — DeepDarts' approach may be reimplemented, its code may not be copied.
- **Per-landmark and per-tip confidence are required outputs**, not optional. #13 showed the reference scorer returns *no score at all* below 4 valid landmarks, so the app must be able to distinguish "board not calibrated" from "no darts detected" (#10, #21).
- **Emit the #13 contract:** 7 normalized points, fixed order.
- **Mobile export is a first-class criterion** (#4). Prefer standard ops; heatmap + soft-argmax and anchor-free NMS-free point detection both export cleanly to Core ML. Avoid custom ops and anything requiring dynamic shapes.
- **Target to beat:** `dart-sense`'s 0.884 image-level PCS (#24) as external context, on our own benchmark tiers (#14).

## 6. Recommendation

1. **Baseline: a decomposed two-head model on a shared `timm` backbone** — 4-channel heatmap with soft-argmax for landmarks, anchor-free point detection for tips. Matches each sub-problem's structure, preserves sub-pixel landmark precision, and lets §3's data split work for us.
2. **Challenger A: unified keypoints-as-objects (YOLOX-based).** The published approach, reimplemented clean-room. If it wins, we learn something real; if it loses, the decomposition argument is evidenced rather than asserted.
3. **Challenger B: rectify-then-detect**, contingent on #25 showing strong synthetic landmark transfer. Highest ceiling on the oblique-angle failure mode.
4. **Predict 8 calibration landmarks rather than 4** (§4.2), giving an over-determined homography with a real residual and measurably lower board-coordinate error. Add convexity/ordering and conditioning checks as an independent inference-time rejection signal.
5. **Report landmarks and tips as separate metrics throughout** — #14 already requires it, and §3 makes it the diagnostic that tells us whether the data strategy is working.
6. **Defer transformer set prediction** until #26 reports corpus size.

### What #17 should build first

The shared-backbone two-head baseline, with the heads behind a small interface so Challenger A can swap in without touching the data pipeline, the geometry code, or the metrics. The comparison is the deliverable — not the baseline's own score.
