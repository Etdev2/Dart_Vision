# MVP camera topology

> Research record for **[Wayfinder Research] Choose the MVP camera topology** (issue #2), parent map #1.
>
> The first decision on the product map. It was written before the Brain existed; it can now be answered with measurements rather than intuition, because #15's precision budget, #21's gates and the geometry code all exist.

## 1. Recommendation

**One mounted phone, at 45–65° from the board plane, off to one side or above.** Not handheld, not two devices, not a synchronized rig.

The escalation condition is specific and measurable: **if the cluster case in #21 cannot reach 90% per-dart accuracy after the training campaign, occlusion is the cause and a second viewpoint is the fix.** Nothing else about the single-camera design is the limiting factor, and no amount of model work substitutes for the missing viewpoint.

Everything below is why, and each claim is a number this repository can reproduce.

## 2. The two things a single camera actually costs

Only two properties of the topology matter. Both were measured rather than assumed.

### 2.1 Precision: how many millimetres one pixel of error buys

The board is planar, so the camera collapses to a homography — and the local scale of that homography is what converts the model's pixel error into a scoring error. At a grazing angle the board's image is compressed, so the same pixel of error costs far more board millimetres.

Measured on the **double ring**, where the errors are most expensive, at 80% board fill:

| Elevation | mm per pixel (worst point) | px precision needed for 4 mm | for 6 mm |
| --- | --- | --- | --- |
| 10° | 3.18 | 1.26 | 1.89 |
| 15° | 2.13 | 1.88 | 2.82 |
| 20° | 1.60 | 2.50 | 3.75 |
| 30° | 1.08 | 3.69 | 5.54 |
| **45°** | **0.75** | **5.36** | **8.04** |
| **60°** | **0.59** | **6.78** | **10.18** |
| 75° | 0.51 | 7.87 | 11.81 |
| 90° | 0.47 | 8.51 | 12.77 |

The right-hand columns are the ones to read. They say how precise the tip head must be, in pixels, to hit #21's Gate R (4 mm median) and Gate P (6 mm p95).

**Below about 30° the requirement stops being achievable.** A stride-4 heatmap with soft-argmax refinement lands around a pixel; asking it for 1.26 px at 10° elevation is asking for a number the representation cannot produce. Those poses are not merely harder — they are outside the budget before the model makes its first mistake.

Above 45° the requirement relaxes to 5–7 px, which is comfortably inside what #15's decomposition was designed to deliver.

### 2.2 Occlusion: the part that cannot be mounted away

A dart is a 150 mm object standing out of the board. It hides the board behind it, including another dart's tip.

Share of darts whose board position is hidden by another dart, three darts per throw, barrels never closer than touching:

| Elevation | 12 mm grouping | 20 mm | 35 mm |
| --- | --- | --- | --- |
| 10° | 21.5% | 15.6% | 9.5% |
| 20° | 21.4% | 15.8% | 9.7% |
| 30° | 21.4% | 15.9% | 9.9% |
| 45° | 20.6% | 15.3% | 10.1% |
| 60° | 20.4% | 15.4% | 10.2% |
| 75° | 19.9% | 15.2% | 9.6% |
| 90° | 10.5% | 8.2% | 5.2% |

**The rate is flat from 10° to 75°.** That is the finding, and it is not what intuition predicts — the obvious move when a camera cannot see something is to raise it, and here that does nothing at all.

The mechanism is worth stating because it also says what *would* help. A dart hides a strip of board behind it. That strip's **length** depends strongly on the angle:

| Elevation | Length of hidden strip |
| --- | --- |
| 10° | 737 mm |
| 30° | 225 mm |
| 60° | 75 mm |
| 75° | 35 mm |

But a tight grouping is 12 mm across. Every one of those strips is longer than the cluster, so the strip's length never binds — only its **width** does, and its width is the barrel's width, which the camera angle does not change. Confirming that directly: holding elevation fixed and varying the barrel radius moves the occlusion rate (15.1% → 21.1% → 25.9% for 2.0, 3.5 and 5.0 mm), while holding the barrel fixed and varying elevation from 20° to 60° does not (21.1% → 20.0%).

So: **occlusion is set by how fat a dart is, not by where the camera sits.** One camera has one viewpoint, and roughly a fifth of tightly grouped darts are behind something from that viewpoint. The only fix is another viewpoint.

The 90° row is an idealization, not an option — see §3.

## 3. Why not simply mount it face-on

Face-on is the best row in both tables. It is also **where the darts are.**

Elevation here is the angle out of the board's own plane, so 90° means the camera sits on the board's normal axis — directly in the throwing line, at roughly the height a dart travels. That position takes a dart eventually, and a phone mount in the flight path is a product that damages its user's phone.

The dart's approach is a narrow cone about the board normal. Staying at or below roughly 65–70° clears it from any direction — to the side, above, or below. That, plus §2.1's floor at 30°, is where the recommended band comes from:

> **45–65° elevation, any azimuth, out of the throwing cone.**

It is a genuine Goldilocks band rather than a preference. Lower and the pixel budget is unachievable; higher and the mount is in the way of the darts.

## 4. The topologies not chosen

**Handheld.** Rejected for a reason the capture work already ran into: a session is one fixed phone position, and every landmark set is per-session. A moving camera means re-solving calibration continuously, and a hand shakes at exactly the scale being measured — 1 px at 60° is 0.59 mm. It also eliminates the annotation economy that makes #26 affordable at all.

**Two devices.** This is the correct answer to §2.2 and the wrong answer for an MVP. Two phones require time synchronization, a second mount, a second calibration, and a pairing flow — each a setup-friction cost paid by every user on every session, to fix a failure mode that affects a fifth of *tightly grouped* darts. #14's risk–coverage curve and #10's correction flow already cover uncertain detections at a fraction of the cost. Revisit it when, and only when, the escalation condition in §1 fires.

**Synchronized multi-camera.** This is what the Autodarts-class systems do, and it is a different product: dedicated hardware, fixed installation, no phone. Out of scope for an MVP whose entire premise is that the camera is already in the user's pocket.

## 5. What this changes elsewhere

**#25's pose sampling wastes corpus below 30°.** `PoseRanges.elevation_deg` currently samples `(8.0, 75.0)`. By §2.1, everything under about 30° is outside the achievable precision budget, so roughly a third of the sampled poses train the model on framings the product will not use and cannot gate on. Narrowing it to something like `(35.0, 70.0)` concentrates the corpus on the band the mount will actually occupy. **This is #25's decision, not this ticket's** — recorded here because the measurement implies it.

**#26's capture protocol does not mention elevation.** It says to vary the phone position "face-on, off to one side, low, high, deliberately oblique." The oblique captures are the ones that cannot meet the gate. The protocol should name the 45–65° band.

**#21's cluster gate becomes the escalation trigger.** "≥90% per-dart on clusters of 2+ darts within 15 mm" is no longer just a quality bar — it is the specific measurement that decides whether the MVP topology holds. §2.2 predicts roughly 15–21% of those darts arrive occluded, so the tip head must recover most of them from the visible barrel axis. Whether it can is an empirical question that the training campaign answers.

**#4 (where inference runs) inherits a resolved input.** One camera, one stream, one homography per session — the cheapest case for every runtime option, and it removes multi-stream synchronization from that decision entirely.

## 6. Reproducing these numbers

`python tools/camera_topology.py` reproduces every table above. Both experiments use only `dartvision.geometry` and `dartvision.synthetic`. §2.1 takes the singular values of the numerical Jacobian of the image→board homography, sampled around the double ring. §2.2 models each dart as a 130 mm cylinder of 3.5 mm radius, tilted 20° from the board normal to reflect a descending throw, and tests whether one tip's projection falls inside another dart's silhouette while that dart is nearer the camera. Groupings are area-uniform with barrels never overlapping.

The simplifications are stated because they bound the result: real darts tilt by varying amounts, flights are wider than barrels further from the board, and a real grouping is not isotropic. Each of those makes occlusion **worse**, not better, so §2.2's rate is a floor.
