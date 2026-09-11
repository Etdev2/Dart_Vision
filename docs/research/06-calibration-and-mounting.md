# Calibration and mounting experience

> Prototype record for **[Wayfinder Prototype] Design the camera calibration and mounting experience** (issue #6), parent map #1. Blocked by #2 and #5, both resolved.
>
> Prototype: `src/dartvision/calibrate/`.

## 1. The setup screen never asks for an angle

#2 concluded that the phone belongs at 45–65° from the board plane. The obvious way to deliver that is to tell the player so.

Nobody knows what angle their phone is at. A protractor is not part of the product, and "about 45 degrees" is a sentence people nod at and then ignore.

It turns out not to be necessary. Everything the guidance needs can be **measured** from the eight landmarks the model already detects, in the frame the player is already pointing at the board — including the angle itself.

## 2. Elevation comes out of the homography, with no intrinsics at all

A plane seen face-on scales equally in every direction. Tilted by `el` out of the image plane, it compresses along the tilt by `sin(el)`. So the two singular values of the homography's local Jacobian at the bull stand in ratio `1/sin(el)`, and the angle follows from an `asin`.

The important word is **intrinsics**. This needs no focal length, no sensor size, no distance from the board — none of which a browser can be trusted to report, and all of which vary by device. It needs only the eight points.

Measured against the synthetic camera, where the true pose is known:

| | Result |
| --- | --- |
| Exact poses, any azimuth, any distance | **exact to < 0.1°** |
| With 1 px of landmark noise | ±0.3° |
| With 3 px of landmark noise | ±1° (±3.5° at 80°, where `asin` flattens) |

Three pixels is worse landmark error than #21 gates for, and the guidance band is twenty degrees wide. The estimate is far better than it needs to be.

This is the finding that makes the whole screen work. Everything below follows from being able to measure rather than instruct.

## 3. What the screen measures

`assess_framing(landmarks, image_size)` returns a verdict, a single piece of guidance, and the numbers behind it:

- **`elevation_deg`** — §2.
- **`ring_px`** — how many pixels a 10 mm scoring ring will occupy *at the model's input size*, median around the double ring. Measured after the resize, because resizing is where the resolution is actually lost, and measured through the player's actual perspective rather than face-on.
- **`ring_px_worst`** — the most compressed point on the ring, for diagnosis.
- **`board_fill`** — the board's extent in the frame, which is the thing the player actually adjusts.

Three bands, and each threshold traces to a measurement rather than a preference:

| Threshold | Value | Where it comes from |
| --- | --- | --- |
| Minimum elevation | 30° | #2: below this, #21's 4 mm gate needs sub-pixel precision the representation cannot deliver |
| Ideal elevation | 45–65° | #2 |
| Maximum elevation | 70° | Not accuracy — face-on is the *best* view #2 measured. It is where the darts are |
| Minimum ring width | 10 px | #15's precision budget |

## 4. One instruction at a time, and the right one

The screen says one thing. Which one is a real design question, because the same measurement can have two causes.

**Being in the throwing line outranks everything.** It is the only failure that costs a phone rather than a score, so it is said first even when the framing is also wrong.

**A short ring measurement has two causes and only one is fixed by moving closer.** An oblique view compresses the rings by roughly `sin(elevation)`, so a player already filling the frame at 36° *cannot* move close enough — the rings are squashed, not small. Telling them to move closer is the most annoying advice the screen could give: it is impossible to comply with and it never converges. So when the angle is shallow, the guidance is to move around toward the front of the board, which fixes the angle and the ring width together.

The same reasoning applies one tier softer: at 40° with the frame nearly full, the advice is still about the angle, not the distance.

## 5. Degenerate landmark sets have to be caught here

Any four points in general position admit an exact homography onto any other four. #15 established this while correcting an earlier claim, and it has a consequence for this screen: a detector that reports four landmarks lying along one wire produces a perfectly valid-looking homography and completely meaningless numbers. Nothing downstream complains.

So `assess_framing` checks the landmark set for convexity and ordering before trusting it, and refuses rather than answering. A test pins the specific case — the four points on one diameter — because it is exactly what a partial detection on a half-visible board produces.

## 6. Re-calibration during play

Handled, and not by this screen. #3's `ThrowSegmenter` re-detects landmarks every frame and emits `calibration_lost` when they move more than a few millimetres from the set the current calibration was built on. A stale homography produces confident, wrong scores, which is worse than no score at all.

The setup screen and the in-play check are therefore the same measurement at two moments, and the recovery flow is the setup screen again.

## 7. Lighting

Deliberately not a check.

The instinct is a "lighting looks bad" warning, and the honest position is that we do not yet know what bad looks like to this model. #25 randomizes illumination and #26's protocol varies it deliberately, precisely so that lighting stops being a variable the user has to manage. A warning invented before the model has been evaluated under varied light would be guessing, and a setup screen that cries wolf is worse than one that stays quiet.

If the training campaign shows a genuine lighting failure mode, the threshold to add it here is one measurement and one branch. Until then, the screen judges what it can actually measure.

## 8. What this leaves for the UI

The prototype is the decision logic, not the screen. What the app still owns:

- A live preview with the board outline drawn through the current homography — the fastest possible feedback, and it costs nothing extra since the landmarks are already detected.
- Holding a `ready` verdict for a second or two before committing, so the state does not flicker while the player is still moving the phone.
- The "ready to throw" state itself, which #10 owns.
- Whether the setup flow insists on `ready` or lets a player proceed from `marginal`. The recommendation is to let them proceed with the reason shown: a player who cannot reach `ready` in their room needs to know *why*, not be blocked by a screen that will never go green.
