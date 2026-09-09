# Dart Vision Brain v0.1 — Ship Gates

> Research record for **[Brain Research] Define the Dart Vision Brain v0.1 ship gates** (issue #21), parent map #12.

## 1. The gate is not raw accuracy

Per-image accuracy is the wrong currency. #14 established that players experience **legs**, not images, and compounding is brutal. This document's job is to make that arithmetic explicit so the product target becomes a derived number rather than an argued one.

**Two gates, not one:**

- **Gate R (research)** — the approach works and is worth continuing. Model-side only.
- **Gate P (product)** — good enough to connect to the app and let a player trust it.

A model can pass R and fail P by a wide margin. Conflating them is how a promising prototype gets shipped too early and destroys user trust on day one.

## 2. The compounding table — what a per-leg target actually costs

Required **per-dart** accuracy to achieve a given fraction of error-free 501 legs, by visits per player per leg:

| Legs error-free | v = 9 (~60 avg) | **v = 12 (~40 avg)** | v = 17 (~30 avg) |
| --- | --- | --- | --- |
| 50% | 97.465% (err 2.535%) | **98.093% (err 1.907%)** | 98.650% (err 1.350%) |
| 80% | 99.177% (err 0.823%) | **99.382% (err 0.618%)** | 99.563% (err 0.437%) |
| 90% | 99.611% (err 0.389%) | **99.708% (err 0.292%)** | 99.794% (err 0.206%) |
| 95% | 99.810% (err 0.190%) | **99.858% (err 0.142%)** | 99.899% (err 0.101%) |
| 99% | 99.963% (err 0.037%) | **99.972% (err 0.028%)** | 99.980% (err 0.020%) |

**Read the middle column.** Moving from "half your legs are clean" to "nine in ten" requires per-dart error to fall from **1.9% to 0.29%** — a **6.5× improvement**. That is the single most important number in this document.

For calibration: #13's reported 94.7% per-image implies roughly **1.9% per-dart error**, which lands almost exactly on the 50%-of-legs row. The published state of the art, on its *own* board, produces a scorer that gets through half a leg without a mistake. That is not a product.

> Assumptions: 3 darts per visit, independent per-dart errors, a scoring error anywhere in a leg spoils it. Independence is conservative — real errors correlate (a bad homography spoils all three darts in one image), which *concentrates* errors into fewer turns and makes true per-leg cleanliness somewhat **better** than this table shows. Treat the table as a pessimistic bound, and have #17 measure the real correlation.

## 3. Why confidence is a ship gate, not a feature

If the model flags its least-confident throws for a one-tap confirmation, those darts stop being errors. The gate should therefore be **accuracy at a chosen coverage**, not accuracy overall.

Effect of flagging the lowest-confidence **15%** of darts, assuming that 15% captures **80%** of all errors (v = 12):

| Raw per-dart error | Auto-scored error | Legs clean, no gating | **Legs clean, with gating** |
| --- | --- | --- | --- |
| 5.00% | 1.176% | 15.8% | **65.3%** |
| 3.00% | 0.706% | 33.4% | **77.5%** |
| **2.00%** | **0.471%** | **48.3%** | **84.4%** |
| 1.00% | 0.235% | 69.6% | **91.9%** |
| 0.50% | 0.118% | 83.5% | **95.9%** |

At a DeepDarts-like 2% raw per-dart error, informative confidence turns **48% of legs clean into 84%** — and costs the player roughly **one confirmation tap every seven darts, about two per leg.**

That is the difference between an unusable novelty and a product, obtained without improving the model at all. It has two direct consequences:

1. **Confidence calibration is a hard gate.** A model with worse raw accuracy and honest confidence beats a more accurate model with useless confidence. The metric that matters is not ECE alone but **what fraction of all errors falls inside the flagged set** — the concentration ratio.
2. **This is the quantitative input #5 needs.** #5 (map #1, "Define the MVP scoring success contract") is still open and HITL. It owns the product-facing choice: *how many legs must be clean, and how many taps will a player tolerate?* §2 and §3 convert whichever answer is chosen into a model requirement. **#5 should be decided with these tables in hand.**

## 4. Gate R — research milestone

Evaluated on #14's revised tiers, on an **unseen setup** (T2), never on synthetic validation data.

| Metric | Gate R |
| --- | --- |
| All-4 landmark detection rate | ≥ 98% |
| Landmark localization error (median) | ≤ 2 mm board-equivalent |
| Tip localization error (median / p95) | ≤ 4 mm / ≤ 12 mm |
| Per-dart notation accuracy | ≥ 96% |
| Missed darts / phantom darts | ≤ 3% / ≤ 1% |
| Sim-to-real gap | **Measured and reported**, landmarks and tips separately |

Gate R says the representation and pipeline work. It does not say the thing is shippable.

## 5. Gate P — product-ready

| Metric | Gate P | Rationale |
| --- | --- | --- |
| **Legs error-free** (confirmations counted as correct) | **≥ 90%** | Provisional pending #5 — the headline number |
| Auto-scored per-dart error | ≤ 0.30% | Derived from the row above (§2) |
| Auto-score coverage | ≥ 85% | ~2 confirmation taps per leg |
| **Error concentration** — share of all errors inside the flagged set | **≥ 75%** | Makes the coverage trade actually work (§3) |
| Confidence calibration (ECE) | ≤ 0.05 | Necessary but not sufficient; concentration is the real test |
| All-4 landmark detection rate | ≥ 99.5% | Below 4 landmarks there is **no score at all** (#13) — a cliff, not a slope |
| Tip localization error (p95) | ≤ 6 mm | Against a 10 mm ring width (#15) |
| Oblique-angle degradation | ≤ 3 pp per-dart vs face-on | The 94.7 → 84.0 drop (#13) is the risk this exists to bound |
| Cluster case (≥2 darts within 15 mm) | ≥ 90% per-dart | The known hard case |
| On-device inference latency | ≤ 150 ms | Within a ~300 ms impact-to-display budget (#4, #5) |
| Model size | ≤ 25 MB exported | Mobile deployment (#4, #22) |

**Every threshold in §4 and §5 is provisional.** They are calibrated against derived geometry (#15), published prior art (#13, #24) and the arithmetic above — not against measurements we have taken. Two things must revise them: **#5's product decision** (which sets the headline per-leg target and therefore everything derived from it) and **#17's measurements** (real pixels-per-millimetre, real per-epoch cost, real error correlation).

## 6. "Improve the model" versus "single-camera physics is the limit"

The map asks for a rule to tell these apart. #14's margin-to-boundary cross-tab supplies it.

For every misscored dart, record its localization error and the true tip's distance to the nearest scoring boundary:

| | Error < margin | Error > margin |
| --- | --- | --- |
| **Large margin** | Correct anyway | **Model problem** — genuinely imprecise, keep training |
| **Small margin** | Correct by luck | **Candidate physics limit** — see test below |

**The test:** if misscored darts are overwhelmingly small-margin *and* the localization error is at or near the precision achievable from the available pixels — i.e. p95 tip error approaches the quantization implied by pixels-per-millimetre at that framing (#17 measures this) — then more model capacity or more data will not help. The information is not in the image.

At that point the levers are physical, not statistical:

1. **More pixels on the board** — higher capture resolution, or constrain the mounting distance (#6).
2. **Constrain the viewing angle** — #13 showed obliquity costs ~10 pp. Narrowing the permitted mounting envelope buys accuracy back, at a cost in setup friction. A legitimate trade for #2 and #6 to make.
3. **A second view** — the escalation map #1 already contemplates, and where `lksmlr/autodarts` (Apache-2.0, #24) becomes a useful triangulation reference.

Declaring a physics limit requires the cross-tab as evidence. "The model plateaued" is not evidence; a plateau at an error floor **above** the pixel limit means the model is still the problem.

## 7. Recommendation

1. **Adopt two gates.** Gate R continues the research; Gate P connects to the product. Do not let R's numbers get quoted as though they were P's.
2. **Take §2 and §3 to #5 before finalising any product-facing figure.** The per-leg target is the owner's call; everything else is arithmetic from it. This is a cross-map dependency and it is the highest-value conversation on either board.
3. **Treat error concentration as a first-class gate metric** — the share of errors falling inside the flagged set. It is worth more than several points of raw accuracy and nobody's published numbers report it.
4. **Report every gate on an unseen setup**, per #14's T2, and never on synthetic validation data.
5. **Revisit these thresholds after #17** reports real pixels-per-millimetre and real error correlation. Provisional means provisional.
6. **Require the margin cross-tab before any "physics limit" claim.**
