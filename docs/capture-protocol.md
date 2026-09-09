# Capture protocol

> The procedure for #26 (training data) and #19 (the untouchable holdout).
> Follow it consistently and later sessions extend the corpus rather than
> fragmenting it.

## Why not photos found online

Every image online is someone's copyright, and screenshotting does not change
that. `dart-sense` harvested YouTube frames and is CC BY-NC as a result (#24) —
you cannot license out what you never cleared, and avoiding exactly this is why
the Brain trains on synthetic plus our own data at all.

It is also technically worse, which matters more day to day:

- **No session structure.** Every image becomes its own setup, so landmarks need
  annotating per image — 11 clicks each rather than 8 per session plus 3 per
  visit. The saving that makes #26 affordable disappears.
- **#14's cross-setup tier becomes impossible.** A setup cannot be held out when
  every image is its own.
- **No control over margins.** The deliberate near-wire cases that #14 requires
  cannot be obtained from found photographs at all.
- **Unknown provenance** for the model card in #22.

Using real photographs as *visual reference* while tuning the renderer's
materials is fine and useful. As training data they are not.

## What to buy

One thing: a **phone tripod or gooseneck clamp mount**. Anything that holds the
phone still and lets you set an angle. This is the only purchase the whole
capture plan needs.

## Per-session procedure

A **session** is one fixed phone position. It ends the moment the phone moves.

1. Mount the phone. Frame the board so it fills most of the frame — #25 found
   that a board occupying too little of the frame carries no recoverable
   precision at any model quality.
2. **Lock focus and exposure.** On iPhone, tap and hold on the board until
   AE/AF LOCK appears. This is not optional: if autofocus hunts between frames,
   the board's apparent geometry shifts slightly and the one-landmark-set-per-session
   assumption quietly breaks.
3. **Turn HDR off.** It alters geometry subtly and changes appearance frame to
   frame.
4. Photograph the **empty board**.
5. Throw or place dart 1 → photograph. Dart 2 → photograph. Dart 3 →
   photograph.
6. Remove the darts and repeat from step 4.

**Use a timer or remote shutter** so the phone is never touched — earbud volume
buttons or a watch both work. A nudge mid-session is a new session whether you
meant it or not.

That sequence is what makes annotation cheap: one visit yields **four labelled
images for three clicks**, because frame *k* holds the first *k* darts.

## Across sessions: vary deliberately

The corpus is only as good as its diversity, and within a session everything is
fixed by design. So vary between sessions:

| Axis | Vary across |
| --- | --- |
| Phone position | Face-on, off to one side, low, high, deliberately oblique |
| Distance | Near and far, within the framing rule above |
| Lighting | Daylight, room light, mixed, dim, a lamp throwing glare |
| Darts | Different barrels and flight colours if you have them |
| Board state | Fresh and worn, if you have access to more than one |

**Three physically distinct setups are a hard requirement** for #14's
cross-setup tier — different board *and* room *and* camera position, not one
board from three angles. Two setups train, one is held out, and #19's holdout is
separate again. This cannot be retrofitted: a model trained on two setups cannot
be honestly evaluated on either of them.

## The placed-dart pass

The highest-value hour of the whole plan. Thrown darts land near a scoring wire
only by luck, and #14 established that a benchmark with few near-wire darts
cannot measure what actually breaks scoring.

```bash
python -m dartvision.annotate --margins 0.25,0.5,1,2 --per-margin 5 --out plan.txt
```

Prints rows like `T8  |  0.25 mm inside the outer treble wire, about 107 mm out
from the bull`. Push a dart in by hand at each, photograph, move on. Record the
row number with each photo.

Mix both kinds: thrown darts for realistic distributions, placed darts for
deliberate coverage of the hard cases.

## Getting files onto the Mac

- **Shoot JPEG, not HEIC** where the option exists — one less conversion step.
- AirDrop or cable into a folder per session.
- **Filenames must sort in capture order.** iPhone's sequential `IMG_1234.JPG`
  does this naturally. The annotator sorts by filename and burst expansion
  depends on that order being right.

## Annotating

Open `tools/annotator.html` in any browser — no install.

1. Enter the setup and session ids. Use something structured and stable, e.g.
   `garage` / `garage/2026-01-14-a`.
2. Load the session's images.
3. Click the **8 landmarks once**. They apply to every frame in the session.
4. Switch to Tips. On the **last** frame of a visit, click the tips **in throw
   order**, then "Expand burst" back over the frames that preceded it.
5. Export JSONL. It matches the same contract the synthetic generator emits, so
   real and synthetic data are interchangeable inputs to the same pipeline.

A magnifier follows the cursor because the precision that matters here is
sub-millimetre on the board (#15). Clicking by eye at full-frame zoom throws
away the accuracy the model is trying to achieve.

The header shows clicks spent against clicks if every frame were labelled
independently. That ratio is why this is a weekend rather than a month.

## Sizing

Start with **one setup and about 300 images** — roughly 75 visits, about 90
minutes of throwing. That is enough to run the sim-to-real measurement, and
**that measurement tells you whether the full corpus needs 2,000 images or
20,000**.

Capturing more before it runs means guessing at the number the experiment exists
to produce.
