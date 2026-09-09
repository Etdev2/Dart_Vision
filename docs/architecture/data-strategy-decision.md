# Decision: the Brain trains on synthetic and Dart Vision's own data

> **Status: decided** (2026-09-09). Supersedes the working assumption that DeepDarts would be the Brain's training corpus.
> Recorded against map #12. Consequences are tracked per-ticket in §4.

## 1. Decision

**Dart Vision's Brain will be trained on synthetically generated imagery plus data we capture ourselves. DeepDarts will not be used as training or evaluation data.**

The published DeepDarts *paper* remains a reference — reading and citing published research needs no licence, and its methodological findings are already recorded in [`../research/13-deepdarts-provenance.md`](../research/13-deepdarts-provenance.md). What we drop is any use of **its images, annotations, or weights**.

## 2. Why

- The dataset licence could not be established. Every route was exhausted (§2.1–2.5 of the provenance record), and an identical enquiry has sat unanswered on the project's GitHub since January 2026. Under IEEE's estoppel clause, silence grants nothing.
- **Waiting had no defined end.** A dependency with no resolution date at the base of the architecture is worse than a larger, bounded amount of work.
- DeepDarts was never sufficient anyway: **two physical setups** (#13), and #24 confirmed no usable public alternative exists. Our own data was always required for a defensible accuracy claim.
- Owning the corpus removes an entire category of ongoing risk — no attribution obligations to track, no redistribution questions for #22, no possibility of a rights position changing after we have shipped.

## 3. What this costs — stated plainly

This is not a free win, and the plan must account for it.

| Cost | Mitigation |
| --- | --- |
| **~16k images of free training data given up** | Synthetic generation supplies volume at near-zero marginal cost (#25) |
| **The sim-to-real gap is unmeasured** | #25's four-arm comparison measures it early, before scale-up |
| **Real capture is now on the critical path**, not just a QA step | New ticket for training capture; auto-labelling strategies in §5 cut the cost sharply |
| **We lose the published-benchmark comparison** | The paper's numbers remain valid *context* (94.7% within-setup, 0.884 for `dart-sense`); we simply cannot reproduce them on identical data |
| **Multiple physical setups now needed by us** | See §4.3 — leave-one-setup-out has to run on our own data |

**The principal risk is that synthetic data does not transfer well and real capture has to carry more weight than planned.** That is exactly what #25 is designed to find out, and it is why #25 should run before large-scale capture rather than after.

## 4. Consequences per ticket

### 4.1 #23 — DeepDarts rights: moot

No longer needed. Closed as not-planned rather than resolved. Reopen only if this decision is revisited; the enquiry draft is kept in case the upside is ever wanted.

### 4.2 #20 / #22 — unblocked from rights

Both lose their #23 dependency. #22 also loses the dataset-redistribution question entirely: we own everything we publish, so a curated dataset representation becomes ours to release if we choose.

### 4.3 #14 — the benchmark tiers must be revised

**This is the largest knock-on effect.** #14's three tiers were built on DeepDarts: T0 was within-setup on D1, T1 was cross-setup D1→D2. Both disappear.

The replacement structure:

| Tier | Train on | Evaluate on | Answers |
| --- | --- | --- | --- |
| **T0 — synthetic-only** | Synthetic | Held-out synthetic | Pipeline sanity check. Never a capability claim. |
| **T1 — sim-to-real** | Synthetic | Real captured data | The transfer gap — the new central question |
| **T2 — cross-setup (LOSO)** | Synthetic + real, minus one setup | The held-out setup | Unseen-board generalization, on data we own |
| **T3 — untouchable holdout** | Anything | #19's holdout | The ship gate |

Everything else in #14 survives intact and is now *more* important, not less: session-level grouping, margin-to-nearest-boundary as a first-class metric, per-leg accuracy as the headline, the risk–coverage curve, mechanical test-set discipline, and the failure taxonomy.

**New hard requirement:** T2 needs **at least three distinct physical setups** of our own — two-plus to train on and one to hold out, with #19's holdout separate again. This is the single biggest change to the capture plan and it must be designed in from the first session, not retrofitted.

### 4.4 #19 — stays the untouchable holdout

#19 is *not* the training capture. Keeping them separate is the whole point: its images must never enter training or validation. A separate ticket covers the training corpus.

### 4.5 #25 — promoted to the critical path

No longer an exploratory "is this worth trying". Synthetic data is now the Brain's primary source of training volume. Its measurement discipline stays exactly as specified — the four-arm comparison and the real-data-budget curve are how we learn whether the strategy is working, and its Arm D result sizes the capture effort.

### 4.6 #15 / #17 / #18 — largely unaffected

Architecture choice, harness construction and augmentation tuning proceed unchanged. Two small revisions: #18 loses DeepDarts as an augmentation baseline (its disabled-cutout finding remains a useful *hypothesis* to test, drawn from the paper rather than the data), and #17's harness now ingests synthetic output as its first data source.

## 5. Making our own capture affordable

Real capture is the new cost centre, so the capture design should exploit the structure of the problem. Three levers, all worth building into the training-capture ticket:

1. **Landmarks are per-session, not per-image.** The board and camera are static within a session, so the 4 calibration points need annotating **once per session** and can be propagated to every frame in it. That removes the majority of the annotation labour outright.
2. **Dart tips can be derived by differencing.** Capture incrementally — throw one dart, capture; throw the second, capture; throw the third, capture. Each new tip is the difference between consecutive frames, which localizes it automatically and needs only confirmation rather than blind annotation.
3. **Placed darts give free ground truth.** Pushing darts in by hand at chosen positions means the intended segment is known before the shutter opens, and near-wire margins can be dialled in deliberately — exactly the controlled-margin cases #14 requires and that thrown darts supply only by luck.

We also need our own annotation tool regardless, since DeepDarts' `annotate.py` is unlicensed. Building it around (1) and (2) rather than naive per-image clicking is the difference between a weekend and a month.

## 6. Bottom line

We traded an unbounded licensing dependency for a bounded engineering one. The work is larger and entirely under our control, the resulting corpus is an asset rather than a liability, and #24 established that owning the data is what this market actually competes on.
