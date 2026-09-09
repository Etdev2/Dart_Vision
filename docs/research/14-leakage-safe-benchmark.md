# Leakage-Safe Benchmark for Unseen Single-Camera Setups

> Research record for **[Brain Research] Design a leakage-safe benchmark for unseen single-camera setups** (issue #14), parent map #12.
>
> This is the benchmark **design**. Implementation and the verification checks in §7 belong to #17, which has the dataset in hand.

## 1. The problem in one paragraph

#13 established that DeepDarts' published splits hold out **capture dates**, not **boards**. Every D1 folder is the same board, room, and camera, so the widely quoted 94.7% is a *within-setup* number. The dataset contains exactly **two** physical setups. Any benchmark built only on re-splitting DeepDarts therefore cannot answer the question the product lives on — *will this work on a board and phone it has never seen?* — no matter how carefully the split is drawn. The benchmark must be honest about that ceiling rather than disguising it.

## 2. Three tiers, reported separately, never averaged

A single accuracy number cannot carry this. Report three, always together.

| Tier | Train on | Evaluate on | What it answers | Status |
| --- | --- | --- | --- | --- |
| **T0 — within-setup** | D1 minus held-out sessions | Held-out D1 sessions | Regression harness; comparability with published 94.7% | Sanity only — **not** a generalization claim |
| **T1 — cross-setup (LOSO)** | **D1 only, zero D2 exposure** | All of D2 | Best unseen-board signal available from DeepDarts | The early read |
| **T2 — independent holdout** | Anything | Dart Vision's own capture (#19) | The real answer | The gate for #21 |

### 2.1 T1 is a number DeepDarts never reported

The reference implementation initialises its D2 model **from trained D1 weights** and fine-tunes on D2 (`weights_path: 'models/deepdarts_d1/weights'`, per #13). So its 84.0% is a *transfer-learned, D2-exposed* figure — not a measure of generalizing to an unseen board.

Training on D1 alone and evaluating on D2 with **zero D2 gradient updates** gives a genuine leave-one-setup-out estimate. It costs one training run and can be measured before #19's capture set exists. **This should be the first real experiment after #17's harness works** — it is the cheapest available answer to the project's central risk.

Run the mirror (train D2 → test D1) for completeness, but weight it lightly: 830 training images is too few for the result to mean much.

### 2.2 State the ceiling explicitly

With two setups, leave-one-setup-out has **n = 1** per direction. That yields a point estimate with no confidence interval. The benchmark's own documentation must say so:

> *The DeepDarts benchmark is a necessary regression harness and an insufficient generalization claim.*

#19 is not a nice-to-have that improves this number. It is the only thing that produces a defensible one.

## 3. Grouping rules

- **Group unit is `img_folder`** — one capture session. #13 confirmed folder names are dates, giving session IDs for free.
- **No folder may span splits.** Ever. This is the single rule that prevents the most likely leak.
- **Splits are assigned once**, written to a committed file (`splits/v1.json`), hashed, and referenced by hash in every run's metadata. A run whose split hash does not match the declared benchmark version is not comparable and must be labelled as such.
- **Augment after splitting**, never before. Augmenting a pooled set and then splitting puts transformed siblings of training images into test.
- **Setup identity is a first-class field.** Tag every image with its setup (`d1`/`d2`, later our own capture IDs) so T1 can be computed mechanically and future data slots in without redesign.

### 3.1 Near-duplicate frames

15,000 images of one board almost certainly contains burst sequences of near-identical frames. Session-level grouping handles the *leakage* risk, since whole sessions move together. Two residual effects to measure in #17:

1. **Effective dataset size is smaller than 15,000.** Near-duplicates inflate epoch length without adding information — relevant to #18's augmentation conclusions and #20's epoch budget.
2. **Cross-folder duplicates would break grouping.** Run a perceptual-hash pass over the whole dataset; any near-duplicate pair spanning two folders is a real leak and must be resolved by merging those folders into one group.

Also flagged by #13 and worth resolving while parsing `labels.pkl`: `d2_test` contains `d2_03_03_2020`, a 2020 date among otherwise-2021 D2 folders. Confirm it is a genuine session before trusting any D2 number.

## 4. Holdout discipline

Distinguish the two roles sharply, because conflating them is how research projects fool themselves:

- **Validation** — used freely for model selection, hyperparameters, augmentation choices, early stopping. Any number you optimise against is a validation number, by definition.
- **Test** — read **once per candidate checkpoint**, at most. Never used to choose anything.

Enforce it mechanically, not by good intentions: test evaluation is a separate command that records `(checkpoint hash, split hash, timestamp)` and refuses a repeat run on the same pair without an explicit `--i-know-what-im-doing` override that is logged. If the test set has been read fifty times during tuning, it is a validation set and the project has no holdout.

**T2 (#19) is stricter still: those images never enter training or validation, under any circumstance, for the life of the project.**

## 5. Metrics

### 5.1 Calibration landmarks

- **All-four detection rate** — the fraction of images where all 4 landmarks are found. #13 showed the reference scorer returns *no score at all* below 4 valid points, so this is a hard cliff, not a gradual degradation. Report it prominently.
- **Per-landmark localization error** — median, p95, max, in normalized image coordinates.
- **Induced homography error** — more meaningful than raw point error: reproject known board geometry through the estimated homography and measure the resulting radial and angular distortion. Two landmark sets with identical mean error can yield very different board rectification.

### 5.2 Dart-tip localization

Measure in **normalized board coordinates after homography**, not image pixels — image-pixel error means different things at different perspectives and is not comparable across camera angles.

Report median / p95 / max, and convert to **millimetres on the physical board** using the BDO radii from #13 (`r_double` = 0.170 m). Millimetres are interpretable; normalized units are not.

### 5.3 The metric that actually matters: margin to the nearest boundary

A localization error only matters if it **crosses a wire**. A 3 mm error in the middle of the 20 bed is invisible; the same error 1 mm from the treble wire flips T20 to 20 and costs 40 points.

So for every dart, record both:

- the localization error, and
- the **true tip's distance to the nearest scoring boundary** (sector wire, double/treble ring edge, bull rings, board edge).

Cross-tabulating them separates two very different diagnoses:

| | Error < margin | Error > margin |
| --- | --- | --- |
| **Large margin** | Fine | Model is genuinely imprecise |
| **Small margin** | Lucky | Unavoidable at this precision |

This is what tells you whether to improve the model or accept a physical limit — exactly the "improve the model vs. single-camera physics is the limiting factor" judgement #21 has to make. Without it, a raw error number cannot distinguish the two.

### 5.4 Score accuracy at three granularities

| Granularity | Definition | Use |
| --- | --- | --- |
| **Per-dart** | Was this dart's notation correct? | Diagnostic; isolates the model from turn composition |
| **Per-image / per-turn** | Were *all* darts in the image correct? | Comparable to the published 94.7% / 84.0% |
| **Per-leg (simulated)** | Fraction of simulated 501 legs with zero scoring errors | **The product metric** |

Per-leg is the headline for #21. From #13's analysis, 94.7% per-turn leaves only ~52% of legs error-free over ~12 visits. Per-image accuracy flatters the system by hiding compounding; a player experiences legs, not images. Simulate over a realistic visit-count distribution and report the whole curve, not one point.

### 5.5 Confidence calibration

- **Reliability diagram + ECE** — standard, necessary, not sufficient.
- **Risk–coverage curve** — the operationally decisive one. If we auto-score only above confidence threshold *t*: what is the error rate among auto-scored throws, and what fraction of throws get auto-scored?

That curve *is* the answer to #5's "should uncertain detections auto-score or ask for confirmation?" and it sizes #10's correction flow. Frame the gate as: **at what coverage can we hold auto-scored error at or below the agreed rate?** A model with mediocre raw accuracy but well-calibrated confidence can still make a good product; a model with high accuracy and useless confidence cannot.

### 5.6 Failure taxonomy

Counted categories, each with a saved example gallery — averages hide the failures that matter, and #12's core principle requires quantifying them rather than burying them:

1. Missing landmark(s) → no score possible
2. Dart missed entirely (false negative)
3. Phantom dart (false positive)
4. Correct dart, **wrong sector** (angular error)
5. Correct dart, **wrong multiplier** (radial error — S/D/T confusion)
6. **Cluster failure** — two or more darts within a threshold distance
7. **Occlusion** — tip hidden by another dart or flight
8. Out-of-board / bounce-out misclassification

Categories 4 and 5 should be reported separately: angular and radial errors have different causes and different fixes (sector confusion often points at homography quality; multiplier confusion at raw tip precision).

## 6. Benchmark versioning

The benchmark is an artifact with a version, not a script that drifts. Each version pins: split assignment file + hash, metric definitions and thresholds, the dedup pass result, and the dataset manifest hash from #16's reproducibility contract. Changing any of these mints a new version. **Numbers from different benchmark versions are never compared in the same table.**

## 7. Verification checks for #17 to run

These require the data, so they are #17's to execute and report back:

1. Confirm no `img_folder` spans splits under `splits/v1.json`.
2. Perceptual-hash the full dataset; report cross-folder near-duplicates and merge affected folders into single groups.
3. Quantify within-session near-duplication to estimate effective dataset size.
4. Resolve the `d2_03_03_2020` anomaly.
5. Confirm the annotation schema matches #13's contract across all rows, and count images with fewer than 4 valid calibration points.
6. Report the class balance that matters: darts per image (0/1/2/3), and the distribution of margin-to-nearest-boundary — a benchmark with almost no near-wire darts cannot measure the thing that breaks scoring.

## 8. Recommendation

1. **Adopt the three-tier structure** (§2) and never report a single blended accuracy number.
2. **Group by `img_folder`; no folder spans splits.** Pin splits to a hashed file referenced by every run.
3. **Run T1 (train D1 only → test D2, zero D2 exposure) as the first real experiment** after #17's harness works. It is the cheapest honest read on the project's central risk, and it is a number the original paper never reported.
4. **Make margin-to-boundary a first-class metric**, not a derived curiosity — it is what separates "improve the model" from "single-camera physics".
5. **Report per-leg accuracy as the headline**, with per-image kept only for comparability.
6. **Use the risk–coverage curve as the bridge to product decisions** #5 and #10.
7. **Enforce test-set discipline mechanically**, and treat #19's holdout as untouchable for the life of the project.
8. **Write the ceiling into the benchmark's own docs.** Two setups cannot support a generalization claim; say so in the artifact rather than in a footnote someone will drop.
