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

## What is already built

| Component | State |
| --- | --- |
| Board geometry and deterministic scoring | Done, tested |
| Calibration landmarks and homography | Done, tested |
| Annotation contract | Done, tested |
| Metrics (#14) and ship gates (#21) | Done, tested |
| Scene sampling and dataset generation (#25) | Done, tested |
| Blender renderer | Written; **unrun** -- step 3 is its first real test |
| Model heads (#15, #17) | Not started -- needs PyTorch |
| Training loop, config, container (#16, #17) | Not started |

## What to expect to go wrong

The renderer is the only substantial piece never executed. Likely snags, in
order of probability:

1. **Blender API drift.** `BLENDER_EEVEE_NEXT` is the 4.2+ engine name; older
   builds use `BLENDER_EEVEE`. If the engine string is rejected, that is the fix.
2. **Materials look wrong before they look broken.** Colours and roughness are
   first guesses, not matched to a real board. Geometry correctness is what step
   3 verifies; appearance is a tuning pass afterwards.
3. **Dart geometry is crude** -- a cylinder, a point and a flat flight. Good
   enough to occlude and cast shadows, not good enough to fool anyone. Refine
   after the pipeline runs end to end, not before.
