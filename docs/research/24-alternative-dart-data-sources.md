# Alternative Dart Data Sources

> Research record for **[Brain Research] Survey and qualify dart data sources beyond DeepDarts** (issue #24), parent map #12.
>
> Motivated by #13: DeepDarts contains only **two physical setups**, so it cannot support an unseen-board claim. If other usable data exists, it attacks the project's central risk directly.

## 1. Headline finding

**There is no second substantial, commercially usable public dart dataset.**

DeepDarts (2021) remains the only large annotated set. Everything else is small, encumbered, unreleased, or all three. And the pattern is more interesting than the individual results: **essentially the entire open dart-CV ecosystem is copyleft, non-commercial, or unlicensed.**

That is not bad luck. It explains why Autodarts and Scolia are proprietary hardware businesses — there is no permissive commons to build on, so every commercial entrant builds a private dataset. It also means Dart Vision's data position is a real moat if built properly, and a real blocker if not.

## 2. Public datasets

| Source | Size | Setups | Rights | Usable? |
| --- | --- | --- | --- | --- |
| **DeepDarts** | 16,050 images / 32,027 darts | **2** | Unresolved — #23 | Pending |
| **dart-sense's additional 8,000** | 8,000 (YouTube + personal play) | Many | **CC BY-NC 4.0**, and **not released as a dataset** | **No** |
| **Roboflow Universe community sets** | ~105–660 each; one dartboard-detection set ~500 | Few each | Uploader-asserted; unreliable provenance (see #13 §2.2b) | Only if originally captured by the uploader — must be verified case by case |
| **New academic datasets since 2021** | — | — | — | **None found** |

Searches for post-DeepDarts benchmarks surfaced only dartboard *patents* (no data) and unrelated papers using "DART" as an acronym for general object-detection pipelines. No successor benchmark exists.

## 3. Code ecosystem — licences verified from `LICENSE` files

| Project | Approach | Licence | Commercial use |
| --- | --- | --- | --- |
| `wmcnally/deep-darts` | Single camera, keypoints-as-objects | **None** | ❌ all rights reserved |
| `bnww/dart-sense` | Single camera, YOLOv8n | **CC BY-NC 4.0** | ❌ non-commercial |
| `lksmlr/autodarts` | **Triple** camera, YOLO keypoints + triangulation | **Apache 2.0** | ✅ — but no data, and wrong topology |
| `OpenDartboard/OpenDartboard` | 3 cameras, Raspberry Pi | **GPL-3.0** | ❌ for a proprietary product |
| `hanneshoettinger/opencv-steel-darts` | 2 webcams, OpenCV | **GPL-3.0** | ❌ |
| `vassdoki/opencv-darts` | OpenCV | **GPL-3.0** | ❌ |
| `LarsG21/Darts_Project` | CV scoring | **None** | ❌ all rights reserved |

Exactly one permissive repository, and it is a student project (Datenfusion module, TH Nürnberg) with a 425-byte README, no released dataset, and a three-camera topology that does not match our single-camera target. Useful as a triangulation reference for a future multi-camera accuracy tier; not useful now.

## 4. dart-sense deserves attention despite being unusable

`bnww/dart-sense` is the closest prior art to Dart Vision: single camera, smartphone video, real-time scoring. Its numbers and method are informative even though we cannot touch its code or data.

**What it did:** took DeepDarts' ~16k images and added **~8,000 of its own, harvested from YouTube videos and personal gameplay**, reaching ~24,000 images spanning varied board types, dart styles, lighting, and camera angles. Trained YOLOv8n with genetic hyperparameter search. Reported image-level Percentage Correct Score **0.884**, precision 0.977, recall 0.942, 4.0% missed detections, 0.4% extra detections.

**Three things this tells us:**

1. **Harvesting existing footage is a validated route to setup diversity.** 8,000 images across many boards is exactly what DeepDarts lacks, and it was assembled by one person. The diversity problem is solvable without a warehouse of dartboards.
2. **But that route carries its own rights problem** — YouTube's terms prohibit downloading content without permission, and every video is separately copyrighted. Their **CC BY-NC** licence is very plausibly a *consequence* of that provenance: you cannot license out what you did not clear. Copying the method naively walks into the same trap as #13 and #16, for the third time.
3. **0.884 image-level PCS on a more diverse test set is a useful reality check** on DeepDarts' 94.7%. More diversity, lower headline number — consistent with #13's finding that 94.7% is a within-setup figure. It also used Ultralytics YOLOv8, so its weights carry AGPL exposure on top of the NC licence.

## 5. The option nobody in this ecosystem is using: synthetic data

A dartboard is a **rigid, planar, dimensionally standardised** object. That makes it unusually well suited to synthetic generation — far better than most CV targets.

**Why it fits this problem specifically:**

- **Ground truth is free and exact.** You placed the dart; you know the sub-millimetre tip position and the true landmark coordinates. No annotation cost, no annotation error.
- **Unlimited diversity on the axes that matter:** camera pose across the full hemisphere, focal length, lighting and colour temperature, board brand and wear, wire glare, dart and flight styles, shadows.
- **The hard cases can be generated to order** — near-wire impacts, tight three-dart clusters, partial occlusion, extreme obliquity. #14 flagged that a benchmark with few near-wire darts cannot measure what actually breaks scoring; synthetic data can guarantee their presence at any density.
- **Zero licensing exposure.** We own it outright, which resolves the constraint that blocks every other option in this document.
- **The domain gap is smaller here than usual.** For a rigid planar object with known geometry, the model largely needs to learn projective geometry and local tip appearance — not organic variation. This is close to the best case for sim-to-real.

**The risk is material texture:** wire specularity, worn sisal, translucent flights, motion blur. Mitigate with a **hybrid schedule** — synthetic for landmark/geometry pretraining where it is strongest, real data for fine-tuning tip appearance — and measure the residual gap honestly against #19's real holdout.

**This is the only data avenue that is entirely unblocked by licensing**, and it is complementary to #19 rather than competing with it: synthetic supplies volume and hard-case coverage, #19 supplies ground truth about reality.

## 6. Recommendation

1. **Do not count on a second public dataset.** None exists. Plan as though DeepDarts (pending #23) plus our own data is the whole picture.
2. **Treat #19 as the strategic centre of the Brain, not a QA chore.** The ecosystem survey confirms that proprietary data is what every serious dart-scoring product actually competes on.
3. **Add a synthetic-data prototype to the map.** Highest-leverage unblocked lever available: no licensing risk, exact labels, and controllable coverage of the exact failure modes #14 says we must measure.
4. **Study `bnww/dart-sense` as prior art; take nothing from it.** CC BY-NC plus Ultralytics AGPL on the weights. Its 0.884 PCS is a sane target to beat, and its diversity result is the encouraging part.
5. **Do not harvest YouTube footage.** It is a proven method with an unresolved rights position, and it would recreate exactly the problem #23 is currently stuck on.
6. **Qualify Roboflow community sets individually, or skip them.** Only original captures by the uploader carry grantable rights; a few hundred images of unverified provenance is poor value against the effort of verifying it.
7. **Keep `lksmlr/autodarts` (Apache 2.0) bookmarked** for the multi-camera accuracy tier that map #1 lists as a later option — it is the only permissive triangulation reference found.
