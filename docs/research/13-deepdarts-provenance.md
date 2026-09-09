# DeepDarts — Provenance, Rights, and Annotation Contract

> Research record for **[Brain Research] Verify DeepDarts dataset provenance, license, and annotation contract** (issue #13), parent map #12.
>
> **Status: partial. The rights question is NOT resolved.** Everything below marked *Verified* was read from a primary source. The dataset licence itself could not be verified from this environment and requires one human action (see [Blocked](#blocked-requires-a-human-with-browser-access)).

## 1. Identity of the work

| Item | Value | Confidence |
| --- | --- | --- |
| Paper | McNally et al., *DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera* | Verified |
| Venue | CVSports workshop, CVPR 2021 | Verified |
| Preprint | `arXiv:2105.09880` | Verified |
| Code | `https://github.com/wmcnally/deep-darts` | Verified |
| Data + weights | IEEE DataPort, `deepdarts-dataset` | Verified (from repo README) |
| **Dataset DOI** | **`10.21227/05e7-xs69`** | **Verified** (DataPort record) |
| Citation author | William McNally, University of Waterloo | Verified |
| Record created / last updated | 2021-06-12 / **2026-06-18** | Verified |
| Data format | `*.jpg` | Verified |

The DOI is the citable identifier and will be required for attribution under any licence outcome — record it in the model card for #22.

> The DataPort record was **last updated 2026-06-18**, only months ago. Whatever licence or terms are read from the page should be captured with that date noted, since the record is not static.

## 2. Rights matrix

Code licence and data licence are separate questions and must stay separate.

| Asset | Source | Licence | Commercial use | Confidence |
| --- | --- | --- | --- | --- |
| **Code** (`wmcnally/deep-darts`) | GitHub | **None declared** | **No — assume all rights reserved** | **Verified** |
| **Dataset** (`images.zip`, `cropped_images.zip`) | IEEE DataPort | Unverified | Unknown | **Unverified** |
| **Pretrained weights** (`models.zip`) | IEEE DataPort | Unverified; also a derivative of the unlicensed code | Unknown | **Unverified** |
| **Board geometry constants** | Repo configs (BDO standard measurements) | Facts / standard, not protected expression | Yes | Verified |
| **Third-party re-uploads** (forks, Roboflow copies) | Various | Inherit the same unresolved rights | **No** | Verified by inspection |

### 2.1 The code finding is decisive

`wmcnally/deep-darts` has **no `LICENSE`, `LICENSE.md`, `LICENSE.txt`, or `COPYING` file** — all four paths return HTTP 404 while other files in the same tree fetch normally, so this is a real absence, not a fetch failure. GitHub's terms grant other users the right to *view and fork* a public repo; they do not grant a licence to use, modify, or redistribute it. The default is therefore **all rights reserved**.

**Consequence:** Dart Vision may **study** the DeepDarts code and **reimplement** its ideas, but must not copy its source into the Brain. Algorithms, mathematical methods, and physical board measurements are not copyrightable; the specific source code is. Our training/eval pipeline (#17) must be written clean-room.

### 2.2 The data finding is unresolved and contradictory

Secondary sources disagree, which is itself the finding:

- One source states the dataset is CC BY 4.0.
- Another describes CC-BY only as IEEE DataPort's *general* policy for open-access datasets, which is not a statement about this dataset.
- A third states the dataset "requires an IEEE DataPort Subscription", which sits awkwardly with the `open-access/` path in its own URL.

None of these is authoritative. **Do not record a licence for the data until the DataPort page itself has been read.** A CC BY 4.0 result would be a good outcome (commercial use permitted with attribution); anything non-commercial, or terms attached to the *download* rather than the data, would invalidate the current plan for #20 and #22.

## 3. Data manifest

| Archive | Contents | Notes |
| --- | --- | --- |
| `images.zip` | Full-resolution source images | Cropped locally via `crop_images.py --size 800` |
| `cropped_images.zip` | Pre-cropped 800×800 images | Optional shortcut; what the models actually train on |
| `models.zip` | Pretrained TF checkpoints (`deepdarts_d1`, `deepdarts_d2`) + `yolov4-tiny.h5` init | Derivative of unlicensed code |
| `dataset/labels.pkl` | Annotations (in the git repo, not the DataPort archive) | Pandas pickle, keyed by `img_folder` |

**Scale — now confirmed from the paper abstract (via the IEEE DataPort record):**

| | D1 | D2 |
| --- | --- | --- |
| Images | ~15,000 | ~1,050 total, **830 training** |
| Capture | Smartphone, **face-on** | Various camera angles |
| Setup | Board setup A | Board setup B |
| Method | Trained from scratch | Transfer from D1 + extensive augmentation |
| **Reported test accuracy** | **94.7%** | **84.0%** |

Total ~16,050 images across the two setups (~32,027 darts, per secondary sources). The paper states "the code and datasets are available" — an availability statement, **not** a licence grant.

### 3.1 What those accuracy numbers mean for the product

The reported metric is **per-image total-score accuracy** — an image holds a whole throw state (up to 3 darts), so 94.7% means roughly **1 in 19 turns scored wrong**, and 84.0% means roughly **1 in 6**.

Compounded over a 501 leg of ~12 visits per player:

| Condition | Per-turn accuracy | Chance of an error-free leg |
| --- | --- | --- |
| D1 — face-on, **board already seen in training** | 94.7% | **~52%** |
| D2 — varied angles, seen board | 84.0% | **~12%** |

So even the flattering number means about **half of all legs contain at least one scoring error**, and that is measured on a board the model trained on. Unseen boards will be worse by an unknown margin (see §5.1).

Two conclusions:

1. **Correction UX is not a safety net, it is a core feature.** At these rates a player hits a wrong score roughly once a leg. Issue #10's one-or-two-tap correction flow, and #5's decision on whether low-confidence throws auto-score or ask, are load-bearing product decisions — not polish.
2. **The angle penalty is the headline risk.** Dropping 94.7% → 84.0% when the camera moves off-axis is the single most product-relevant number in the paper, because "mount your phone wherever" is the promise. It sharpens #2 (camera topology) and #6 (calibration UX): constraining the mounting position is a legitimate lever for buying accuracy back.

*(Caveat: D2 also had far less training data (830 images), so the 94.7 → 84.0 drop confounds camera-angle difficulty with data scarcity. Both hurt us, but they have different fixes — more angled data vs. a better model — and #18/#20 should try to separate them.)*

## 4. Annotation contract — *Verified* from `dataset/annotate.py`

Exactly **7 points maximum per image**, in fixed order, each stored as `[x/width, y/height]` normalized to `[0, 1]`:

| Index | Name | Meaning |
| --- | --- | --- |
| 0–3 | `cal_1`–`cal_4` | Board calibration landmarks |
| 4–6 | `dart_1`–`dart_3` | Dart tip locations (may be fewer than 3) |

Annotation was **manual mouse-click** labelling through an OpenCV window. A missing calibration point is encoded as a non-positive coordinate.

### 4.1 Geometry and scoring pipeline

The reference implementation converts points to a score exactly as our own system hypothesis proposes:

1. Take the 4 calibration points; centre `c` = their mean, radius `r` = mean distance from `c`.
2. `cv2.getPerspectiveTransform` maps the 4 points onto their ideal positions → **homography**, rectifying perspective.
3. Translate to centre, convert dart tips to polar: `angle = atan2(-y, x)`, `distance = ‖xy‖`.
4. Sector = `BOARD_DICT[int(angle / 18)]` — twenty 18° sectors, ordered `13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6`.
5. Radial bands decide the multiplier.

Board constants used (BDO standard, metres):

| Constant | Value |
| --- | --- |
| `r_board` | 0.2255 |
| `r_double` | 0.1700 |
| `r_treble` | 0.1074 |
| `r_outer_bull` | 0.0159 |
| `r_inner_bull` | 0.00635 |
| `w_double_treble` | 0.0100 |

Score notation emitted: `0` (miss), `B` (outer bull, 25), `DB` (inner bull, 50), `D<n>`, `T<n>`, or bare `<n>` for a single.

### 4.2 Two contract observations for Dart Vision

1. **Notation differs from our draft `ThrowEvent`.** Ours proposes `T20 / D16 / S5 / DB / MISS`; DeepDarts uses `T20 / D16 / 5 / DB / 0` and adds `B` for outer bull. Ours is better (explicit `S` prefix, explicit `MISS`) — keep it, and treat the mapping as an adapter concern. Feeds #9.
2. **Scoring collapses entirely if any calibration point is missing.** The reference `get_dart_scores` returns an empty list — no score at all — when fewer than 4 calibration points are valid. For a live scorer this is a hard failure mode, not a degraded one. The Brain should therefore emit **per-landmark confidence** so the application can distinguish "board not calibrated" from "no darts detected", and the UX (#10) needs a defined state for it. Feeds #15, #21.

## 5. Split structure — *Verified* from `dataloader.py`

Splits are assigned **by image folder, and folders are capture dates**:

```
d1_val  = d1_02_06_2020, d1_02_16_2020, d1_02_22_2020
d1_test = d1_03_03_2020, d1_03_19_2020, d1_03_23_2020, d1_03_27_2020,
          d1_03_28_2020, d1_03_30_2020, d1_03_31_2020
d2_val  = d2_02_03_2021, d2_02_05_2021
d2_test = d2_03_03_2020, d2_02_10_2021, d2_02_03_2021_2
train   = every folder not listed above
```

### 5.1 This is session-level separation, not board-level separation

The published split holds out *dates*, but every D1 folder is the **same board, same room, same camera**. A model evaluated this way is tested on a setup it has already seen thousands of images of. The headline accuracy is therefore **not** evidence of generalization to a new board — it is a within-setup number.

Two consequences, both of which harden decisions already on the map:

- **The dataset contains only two physical setups in total.** No re-splitting can manufacture a third. A genuine "unseen board and unseen phone" measurement is **impossible from DeepDarts alone**. This promotes issue #19 (independent real-world holdout) from a good idea to a **precondition** for any accuracy claim, and pushes map #12's open question "whether a second Dart Vision-owned dataset is required" strongly toward *yes*.
- **The folder naming does give us free session IDs**, so leakage-safe grouping for #14 is mechanically easy — group by `img_folder`, never split within one. The limit is dataset diversity, not tooling.

*(Minor oddity worth noting when we parse the labels: `d2_test` contains `d2_03_03_2020`, a 2020 date among otherwise-2021 D2 folders. Confirm whether it is a genuine session or a transcription slip before trusting D2 test numbers.)*

## 6. Reference baseline — *Verified* from `configs/`

Useful as a starting point for #15/#17, and as evidence about #18.

| Aspect | Value |
| --- | --- |
| Architecture | YOLOv4-**tiny**, keypoints modelled as objects |
| Input | 800×800 |
| Box size for a keypoint | 0.025 of input (~20 px) |
| Loss | CIoU |
| Schedule | 100 epochs, lr 1e-3, seed 0; batch 16 (D1) / 4 (D2) |
| Transfer | D2 initialises from the trained D1 weights |
| Stack | TensorFlow 2.3, Python 3.7, `yolov4==2.0.3` |

**Augmentation actually used** — directly relevant to #18:

| Aug | D1 | D2 |
| --- | --- | --- |
| Flip LR / UD | 0.5 / 0.5 | 0.5 / 0.5 |
| Rotation (36° steps) | 0.5 | 0.5 |
| Small rotation (±2°) | 0.5 | 0.5 |
| Jitter (0.02) | 0.5 | 0.5 |
| **Cutout (occlusion)** | **0 — off** | **0 — off** |
| **Perspective warp** | **0 — off** | 0.5 |

The reference model was trained with **occlusion augmentation disabled**, and with **perspective augmentation disabled for the 15k-image D1 set**. Both are plausible reasons its performance degrades on clustered darts and off-axis cameras — and both are cheap, high-value experiments for #18.

**The stack is legacy.** TF 2.3 / Python 3.7 / `yolov4==2.0.3` is not a reasonable base for new work in 2026. Reproducing the original numbers would mean resurrecting a dead toolchain; combined with the code being unlicensed, this argues for a **modern clean-room reimplementation** rather than a fork. Feeds #16 and #17.

## 7. Blocked — requires a human with browser access

This session's network egress policy blocks `ieee-dataport.org`, `arxiv.org`, `openaccess.thecvf.com`, and `huggingface.co`. The following could not be verified here:

1. **The dataset licence and terms of use** — *still the gating question.* The DataPort record's description and load instructions have been read and contributed §3; they contain **no licence statement**. The licence lives in a separate metadata field on that page (near the file list / access box), and that field is what must be recorded verbatim, along with any terms accepted at download and whether access requires a subscription.
2. ~~Exact image counts and D1/D2 composition~~ — **resolved**, see §3. Archive sizes on disk still unconfirmed.
3. **Whether trained derivative weights may be used commercially** — if the licence is CC BY 4.0 this is straightforwardly yes with attribution; if it is CC BY-NC or has bespoke DataPort terms, the Brain cannot ship commercially on DeepDarts alone and #20/#22 must be re-planned around our own captured data.

> Note for #22: `huggingface.co` is also blocked from this environment, so publishing the Brain will need either an egress-policy change or a different execution context.

## 8. Recommendation

1. **Treat the DeepDarts code as reference-only.** Do not vendor it. Reimplement clean-room on a modern stack.
2. **Freeze commercial planning until the data licence is read from the DataPort page.** Research and prototyping on the dataset may continue; shipping decisions may not.
3. **Reuse the board constants and the homography→polar method freely** — these are standard measurements and mathematics, not protected expression.
4. **Adopt the folder-as-session grouping for #14**, and treat the two-setup ceiling as the binding constraint it is.
5. **Escalate #19.** Without our own independent capture on an unseen board and phone, Dart Vision has no defensible accuracy claim, regardless of how the licence resolves.
