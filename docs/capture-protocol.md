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

## Recording video instead of photographs

Easier, and slightly *better*: the app reads video frames, so training on video
frames removes a domain gap rather than adding one.

**Resolution is not the constraint; framing is.** This paragraph used to say
1080p puts a 10 mm ring on about 20 px and that anything above the model's 768 px
input is discarded anyway. Both halves are true only of a board that fills a
square frame, and the first real session was neither. The model stretches a whole
frame onto a 768 px square, so a portrait 1080×1920 recording is squashed 2.5×
vertically and 1.4× horizontally before the model sees anything, and the double
ring lands here:

| How the board is framed | ring across | ring down |
| --- | --- | --- |
| 43% of a portrait frame's width (session-01, as shot) | 9.7 px | 3.0 px |
| filling a portrait frame's width | 21.5 px | 6.6 px |
| filling a **square** frame | 21.5 px | 11.8 px |

Against a floor of 10 px (`model.spec.MIN_RING_PX`). Measure it rather than
estimate it — eyeballing a board's share of a frame is unreliable enough that a
first reading off one photograph came out nearly twice the truth:

```bash
./capture --framing data/captures/garage/img_0632-01
```

It finds the board with no landmarks and no model, from the one thing reliably
true of a photograph of a dartboard: it is much the darkest large object in the
frame, and it is roughly as wide as it is tall. The estimate is crude on purpose
and scores nothing; it only has to separate a ring landing on four pixels from
one landing on twelve.

**It writes `framing-check.jpg` beside the stills with a red box around what it
measured. Look at it.** Every number above rests on an assumption about what a
photograph of a dartboard looks like, and one photograph is the cheapest way to
find out whether the assumption held on yours.

The assumption fails in one common case: **a board mounted against something
dark that touches it** — the box many boards hang on — is one dark region with
the board, and the bounding box spans both. The tool says so, and still answers,
because a merge can only make a region *larger*. Whichever side is shorter is
the side the merge did not inflate, and on a round object that side is the
diameter. It errs low for a board seen at a steep angle, which is the safe
direction when the decision is whether to re-shoot. `--board-width` overrides it
with a measurement made by hand.

Two things follow. **Fill
the frame with the board** — it is the single largest factor, worth a factor of
two. And **the portrait-to-square stretch costs more than framing can recover**:
even a perfectly framed portrait video lands at 6.6 px. Shoot so the board fills
the frame in *both* directions — stand closer and turn the phone whichever way
puts the board across the short side — or accept that the vertical precision is
roughly half the horizontal. Squaring up the frame around the board at training
time would remove this entirely and is not yet built.

Three settings matter, and the first will ruin a session silently:

1. **Turn video stabilization off.** Electronic stabilization warps and shifts
   every frame to cancel shake, which breaks the one-landmark-set-per-session
   assumption invisibly. Nothing downstream would flag it; the labels would
   simply be slightly wrong everywhere.
2. **Lock focus and exposure** before recording. The phone adjusts continuously
   in video, and focus hunting changes the board's apparent geometry.
3. **Highest bitrate, 4K over 1080p** — not for the pixels, for the
   compression. Artifacts cluster on thin high-contrast edges, which is exactly
   what a wire is.

Then **throw, pause about two seconds, throw**. The pause is what the extractor
looks for.

All the commands below are written as `./capture`, a wrapper in the repository
root that finds the virtualenv for you. Capture happens away from the keyboard
and the tool gets run from whichever Terminal window is open, which is rarely
one with `.venv` activated; the wrapper makes that the script's problem rather
than yours, creating and populating the environment on first run. `python -m
dartvision.capture` with the environment active does exactly the same thing.

Scan the first video before committing a session to it:

```bash
./capture --video session.mov --diagnose
```

That writes nothing. It reports how much changed between frames, how many times
the camera itself moved, and what each candidate threshold would have kept, so
the settings are chosen against your footage rather than against the reasoning
in `capture/frames.py`. **Aim for about four kept frames per visit.** Fewer means
whole visits are collapsing into one still; far more means noise is reading as a
dart. Then run it for real:

```bash
./capture --video session.mov --out data/captures/garage
```

It keeps one still per board state — the empty board, then each dart as it
lands — by finding runs of frames where nothing moves, taking the sharpest
frame of each run, dropping runs that are a body standing at the board, and
dropping runs where nothing actually landed. Four frames a visit, 75 visits,
300 images: the same target, without 300 shutter presses.

`--out` names a **parent** directory, not a session folder. Output goes to one
subfolder per camera position — `garage/img_0624-01`, `garage/img_0624-02` —
because landmarks are annotated once per folder and applied to everything in it.
Two camera positions sharing a folder means one of them is labelled against a
viewpoint it was never shot from, in every frame, with nothing downstream to
flag it. That is why the split is automatic rather than a flag to remember.

A viewpoint yielding fewer than two stills is dropped: eight landmark clicks to
gain one image is not a trade worth making, and a one-state viewpoint is usually
the instant *during* a move rather than a position anybody threw from. Change it
with `--min-states` if you disagree.

## A folder of recordings made without a protocol

Footage shot before any of this existed is still usable. Point it at the folder:

```bash
./capture --video-dir ~/Desktop/Darts --out data/captures/garage
```

Every video in the folder is read in filename order — which for phone footage is
chronological — and each is split into its own camera positions. A file that
cannot be decoded is reported at the end rather than stopping the run.

Re-running into a directory that already holds an earlier extraction is refused,
naming every recording that conflicts, before anything is decoded. A second run
can split the same recording differently — a threshold changed, or the code did
— and the leftovers would sit beside the new folders under names that do not
tell them apart. Annotating that mixture means labelling the same frames twice
under two landmark sets, which is a corpus problem found long after the effort
is spent. Pass `--overwrite` to replace that recording's folders, or `--out`
somewhere new to keep both.

## A folder of photographs

Shot from several angles, the way anyone tries a new setup:

```bash
./capture --stills ~/Desktop/Darts/T2 --out data/captures/t2
```

Sorted into one folder per camera position, by the same measure a recording is
split by, and for the same reason: landmarks are annotated once per folder.
Nothing about the input being files rather than frames changes that. A change of
orientation splits on its own — a portrait photograph and a landscape one were
not taken from the same position. Originals are copied, never moved.

**Photographs are expensive to annotate compared with video.** A folder of one
shot per angle is eight landmark clicks to gain one image; a video of a visit is
eight clicks to gain a hundred. Photographs earn their place as *diversity* — a
deliberately awkward angle, a lighting case, a near-wire dart placed by hand —
not as bulk.

iPhones shoot HEIC by default and ffmpeg does not read it. Settings > Camera >
Formats > **Most Compatible** shoots JPEG; existing photographs export as JPEG
from the Photos app.

Note that `--video-dir` reads one folder and not the ones inside it, and says so
when it finds videos nested below. Each subfolder is its own run, which is the
right shape anyway: a folder of trials is a different setup from the one before
it.

Nothing about this needs the camera to have been set up deliberately, because
there is no pre-capture calibration step anywhere in this system: the 8 landmarks
are clicked **afterwards**, on the images, and the board geometry is recovered
from those clicks. A recording made by propping the phone and pressing record is
exactly as usable as one made to the protocol above. What varies is the
annotation cost — one landmark set per camera position — and how much of the
precision budget the framing left behind.

The report ends with **how many viewpoints the recording contains**, and that
is the number that decides how the footage may be used. A session is one fixed
phone position; landmarks are annotated once per session; so a recording with
three viewpoints is three sessions, and extracting it to one folder puts wrong
labels on two thirds of it. Extract each stretch separately when it reports more
than one.

It reports movement across the shot separately, and that number is expected to
be large — you walking to the board is movement, twice a visit. It costs a
second of footage each time and nothing else. Only a view that does not come
back is a session boundary.

The split asks whether the view *came back*, not how big the change was. Pulling
darts returns the board to how it looked at the start of the visit; a camera
moved somewhere new resembles nothing seen before. That distinction matters more
the better the framing gets: the threshold is a share of the frame and a dart is
a share of the board, so filling the frame properly walks a visit's worth of
darts towards a line meant to sit far above them. Measured on a real session,
clearing a closely framed board cleared that line by 1.17x, where a phone
actually picked up cleared it by 11x.

One limit worth knowing: the check finds a phone that was picked up and put
down, not a phone that was knocked. A shift of a few pixels changes too little
to tell from a dart, and separating those needs the board located, which is the
model's job rather than the extractor's. Lock the mount — and if it slipped
anyway, the annotator shows it, because the session's single landmark set stops
sitting on the board.

Long recordings are the expected case, so the video is scanned at 5 frames a
second rather than its own rate: a pause of a second still spans five frames,
while a twenty-minute session costs about 124 MB instead of 746. Nothing is
buffered — frames stream out of ffmpeg as they decode.

Check the result with the audit afterwards. If it reports 300 images carrying
40 images' worth of information, the extraction kept too much.

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
   **[`docs/landmarks.svg`](landmarks.svg) shows exactly which points**, in click
   order: four on the outer edge of the double ring, then four on the outer edge
   of the treble ring, at the four radial wires that sit 90° apart (`13|6`,
   `20|5`, `8|11`, `17|3`). Drawn from `geometry.board` rather than by hand, and
   a test asserts the drawn points are the ones the annotator asks for.
4. Switch to Tips. On the **last** frame of a visit, click the tips **in throw
   order**, then "Expand burst" back over the frames that preceded it.
5. Export JSONL. It matches the same contract the synthetic generator emits, so
   real and synthetic data are interchangeable inputs to the same pipeline.

A magnifier follows the cursor because the precision that matters here is
sub-millimetre on the board (#15). Clicking by eye at full-frame zoom throws
away the accuracy the model is trying to achieve.

**The click gets the point close; the arrow keys place it.** A canvas has to
fit on a screen, so a 1080×1920 photograph is shown at about half size and every
click lands within a couple of source pixels of where it was aimed — on a task
whose whole point is sub-millimetre placement. Arrow keys move the last point
one *source* pixel, shift moves five, and the magnifier follows the point rather
than the cursor while they do. The point list reads back both the fraction and
the pixel, because the pixel is the unit the keys move in.

The header shows clicks spent against clicks if every frame were labelled
independently. That ratio is why this is a weekend rather than a month.

## Sizing

Start with **one setup and about 300 images** — roughly 75 visits, about 90
minutes of throwing. That is enough to run the sim-to-real measurement, and
**that measurement tells you whether the full corpus needs 2,000 images or
20,000**.

Capturing more before it runs means guessing at the number the experiment exists
to produce.
