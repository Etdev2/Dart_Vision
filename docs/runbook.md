# Day-one runbook

Everything below was built without the machine, so this should be a sequence of
commands rather than a debugging session. Work top to bottom; each step's output
tells you whether to continue.

## 0. Find out what your machine is

```bash
git clone https://github.com/Etdev2/Dart_Vision.git && cd Dart_Vision
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m dartvision.doctor
```

This prints your exact chip (`Apple M1`, `Apple M1 Pro`, `Apple M1 Max`), memory,
free disk, and a checklist of which pipeline stages this machine can run. It
answers the "which M1 do I have" question so nothing downstream has to guess.

Expected on a Mac with nothing else installed: manifests yes, rendering no
(Blender missing), training no (no CUDA), Core ML yes.

## 1. Confirm the build is sound

```bash
pytest
```

All tests should pass with no network access and no GPU. If they do, the
geometry, scoring, calibration, label contract, metrics and scene generation are
all working on your machine.

## 2. Generate a scene manifest

```bash
python -m dartvision.synthetic.generate \
    --out data/synthetic --setups 3 --sessions 4 --images 40 --seed 42
```

Writes `manifest.jsonl`, six SVG previews and a `summary.json`. **Open the
previews first** -- they show the board projected through each sampled camera
with landmarks and dart tips marked. If those look like dartboards from
plausible angles, the labels are right.

Check `summary.json` for the composition: roughly 35% near-boundary darts, 20%
clusters, a spread of 0-3 darts per image, and a board-coverage range.

## 3. Install Blender and verify labels *before* rendering

Blender runs natively on Apple Silicon with Metal acceleration. Download the
Apple Silicon build from blender.org, then:

```bash
python -m dartvision.doctor          # should now find Blender
```

Now run the renderer in **verify-only** mode. This projects every scene's
landmarks through Blender's own camera and compares them against the manifest,
without rendering anything:

```bash
blender --background --python tools/render_blender.py -- \
    --manifest data/synthetic/manifest.jsonl \
    --out data/synthetic/images \
    --verify-only
```

Expect `worst_label_error_px` well under 1. **If this fails, stop.** It means
the images would not match their labels, which is the one failure that poisons
everything downstream while looking fine at every later stage.

This check exists because our camera model and Blender's disagree in exactly
one place -- the sign of camera roll, since our image frame has +y down and
Blender's camera has +y up. That is already fixed and unit-tested, but the
renderer re-checks against the real Blender camera rather than trusting the
unit test.

## 4. Render a small batch

```bash
blender --background --python tools/render_blender.py -- \
    --manifest data/synthetic/manifest.jsonl \
    --out data/synthetic/images \
    --engine EEVEE --limit 20
```

**Start with EEVEE.** It is a rasterizer and dramatically faster than Cycles;
Apple Silicon lacks the dedicated ray-traversal hardware that makes Cycles quick
on RTX cards. Compare a handful of EEVEE and Cycles frames -- the thing to look
at is **wire specularity**, since scoring depends on resolving which side of a
wire a dart tip sits, and that is exactly where the two renderers differ most.

Once a batch looks right, drop `--limit` and render the lot.

## 4b. Capturing your own data (#26)

Two tools, both usable the moment you have a board and a phone.

**A capture plan for hand-placed darts** — thrown darts land near a wire only
by luck, and #14 needs them deliberately:

```bash
python -m dartvision.annotate --margins 0.25,0.5,1,2 --per-margin 5 --out plan.txt
```

Prints rows like `T8 | 0.25 mm inside the outer treble wire, about 107 mm out
from the bull`. Place, photograph, move on.

**A browser annotator** — open `tools/annotator.html` in any browser, no install:

1. Enter setup and session ids, load a session's images (sorted by filename).
2. Click the 8 landmarks **once**. They apply to every frame in the session.
3. Switch to Tips, click up to 3 tips in throw order on the **last** frame of a
   visit, then "Expand burst" backwards over the preceding frames.
4. Export JSONL, which matches the same contract the synthetic generator emits.

A magnifier follows the cursor, because the precision that matters here is
sub-millimetre on the board (#15).

The header shows clicks spent against clicks if every frame were labelled
independently. That ratio is the reason this is a weekend rather than a month.

## 5. Rent a GPU for training

Only training needs rented hardware. RTX 4090 on RunPod Community is about
**$0.34/hr**; a full run is a few dollars.

> **Bill discipline:** you pay for every hour the pod *exists*, not hours it
> computes. A 4090 left running overnight costs about $8 for nothing. Make the
> container exit when the run finishes, and check for live pods before closing
> the laptop.

The Mac drives the rented box over SSH; its own GPU is irrelevant to that.

**Never debug on MPS and train on CUDA.** PyTorch's Apple backend differs
numerically and silently falls back to CPU for some operators, so you would
chase bugs that exist on neither. The Mac authors and inspects; the rented box
measures (#16).

## 6. Train

```bash
python -m dartvision.train \
    --manifest data/synthetic/manifest.jsonl \
    --image-root data/synthetic/images \
    --out-dir runs/arm-a --split session --epochs 20 --device cuda
```

Every flag mirrors a field of `TrainConfig`, so `--config run.json` plus
overrides reproduces a run exactly. The run directory gets `manifest.json`
(git SHA, config digest, seeds, split digest) written **before** the first
step, `metrics.jsonl` appended per epoch, and `last.pt` rewritten per epoch.

Training never sees the test partition. `apply_split` has no code path that
returns it -- not a convention, a missing branch.

Sanity check before committing to a long run: `--limit 32 --epochs 30` on a
handful of images should drive the loss most of the way to zero. If a model
cannot overfit 32 images it will not learn 3,000, and you have found a bug for
the price of two minutes.

## 7. Evaluate

```bash
python -m dartvision.eval \
    --checkpoint runs/arm-a/last.pt \
    --manifest data/synthetic/manifest.jsonl \
    --image-root data/synthetic/images \
    --split session --partition test \
    --ledger runs/holdout.json --out-dir runs/arm-a/eval
```

Prints and writes #21's gate report: calibratable rate, landmark and tip error
in **board millimetres** (not pixels), per-dart accuracy, legs error-free with
and without flagging, and #14's margin cross-tab separating "the model was
imprecise" from "the dart was on the wire".

Three things about this command are deliberate:

- **There is no argument for which images to score.** The tier and seed define
  the test partition; the evaluator derives it the same way training derived
  what to avoid. It cannot be widened by the caller.
- **`--ledger` refuses a second read** of the same checkpoint and split without
  `--override` and a written reason, which is logged. A test set read fifty
  times during tuning is a validation set, and the project then has no holdout.
- **The homography comes from the *predicted* landmarks**, never the true ones.
  Rectifying with ground truth hides the largest error source in the system and
  reports a number the product will never see.

Use `--partition val` while tuning. It touches nothing in the ledger.

## What is already built

| Component | State |
| --- | --- |
| Board geometry and deterministic scoring | Done, tested |
| Calibration landmarks and homography | Done, tested |
| Annotation contract and leakage-safe splits | Done, tested |
| Metrics (#14) and ship gates (#21) | Done, tested |
| Scene sampling and dataset generation (#25) | Done, tested |
| Blender renderer | Done -- verified against Blender's own camera to 0.003 px |
| Model heads and losses (#15, #17) | Done, tested; overfits a small set |
| Training loop, config, provenance (#16, #17) | Done, tested |
| Evaluation CLI and gate report (#14, #21) | Done, tested |
| Real capture (#26) | Waiting on photographs |
| Core ML export and the app (#1) | Not started |

Everything above the capture row runs today. What the project is short of is
data, not code.

## What to expect to go wrong

1. **Blender API drift.** Engine names move between releases; the renderer
   tries `BLENDER_EEVEE_NEXT` then `BLENDER_EEVEE`. If a future build rejects
   both, that list is the fix.
2. **Materials look wrong before they look broken.** Colours and roughness are
   first guesses, not matched to a real board. Geometry correctness is what
   step 3 verifies; appearance is a tuning pass afterwards.
3. **Dart geometry is crude** -- a cylinder, a point and a flat flight. Good
   enough to occlude and cast shadows, not good enough to fool anyone.
4. **A loss that falls while nothing is learned.** The tip head puts every dart
   in one channel, so anything deciding "positive" per channel supervises all
   but the strongest tip as background -- and the total loss still drops. Twelve
   unit tests passed with exactly that bug present; only the overfit check
   caught it. Trust the overfit check, not the loss curve.
