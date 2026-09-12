# Where real-time inference runs

> Research record for **[Wayfinder Research] Decide where real-time inference runs** (issue #4), parent map #1. Blocked by #2 and #3, both now resolved.

## 1. Recommendation

**On the device, in the browser.** The model ships as a versioned asset the PWA fetches and caches, not as part of the JavaScript bundle. **No server inference, at any tier.**

A native wrapper stays available as a known escape hatch. The thing that would force it is *camera reliability*, not inference speed — see §6.

## 2. The question changed shape

#4 was written as "can a phone run this model in real time." #3 dissolved that question.

Inference runs only on **settled** frames — the scene still, no dart in flight, no hand over the board. Everything between those moments is gated out by a frame difference costing a few thousand operations. A 501 leg is about 24 visits, and a visit contains roughly five settled moments: the empty board, three darts, and the board being cleared.

| | Inferences per leg |
| --- | --- |
| Every frame at 30 fps, 15 minutes | 27,000 |
| Gated on stillness | ~120 |

**225× fewer.** That is about **0.13 inferences per second**, and it is the number #4 has to be decided against. Almost every argument for moving inference off the device evaporates at that rate.

## 3. What one forward pass actually costs

Measured on this repository's own model, unoptimised fp32 on four CPU threads — deliberately a pessimistic proxy, since it uses no NPU, no quantisation and no graph optimisation:

| Backbone | Input | ms/frame | Compute per leg | GMACs | int8 size |
| --- | --- | --- | --- | --- | --- |
| resnet18 | 640 | 96 | 11.5 s | 17.85 | 11.3 MB |
| **resnet18** | **768** | **162** | **19.5 s** | 25.70 | 11.3 MB |
| resnet18 | 896 | 200 | 24.0 s | 34.99 | 11.3 MB |
| efficientnet_b0 | 768 | 171 | 20.5 s | 6.67 | 3.7 MB |
| mobilenetv3_small | 768 | 27 | 3.3 s | 1.71 | 1.0 MB |

At the recommended configuration — resnet18 at 768, the slowest credible choice — a leg costs about **20 seconds of compute spread over fifteen minutes. A 2% duty cycle**, on the CPU, with none of the acceleration a phone actually has. Even the largest row is under 3%.

There is no latency problem to solve here. There is barely a latency question.

One thing in that table is worth flagging for anyone tuning it later: efficientnet_b0 does a quarter of resnet18's multiply-accumulates and takes slightly *longer*. Depthwise separable convolutions are arithmetic-efficient and memory-bound, so MACs badly mispredict their speed. Pick a backbone on measured latency, never on GMACs.

## 4. The input resolution is the real cost lever — and it has a floor

Cost scales with input area, so shrinking the input is the obvious economy. #15's precision budget forbids it, and by more than a face-on calculation suggests.

A 10 mm scoring ring must land on 10–20 px for the tip head to have anything to work with. The easy way to check that is `ring_width × fill × input / board_diameter` — but **that is the face-on number, and nobody mounts a phone face-on** (§2 of #2: that is where the darts are). A board tilted out of the image plane is compressed along the tilt by roughly `sin(elevation)`, and its rings compress with it.

Measured through the perspective a player in #2's recommended 45–65° band actually has, at 85% board fill — median ring width around the double ring:

| Model input | Face-on arithmetic | 45° mount | 55° | 65° |
| --- | --- | --- | --- | --- |
| 512 | 9.6 px | 6.8 | 7.9 | 8.7 |
| 640 | 12.1 px | 8.5 | 9.8 | 10.9 |
| **768** | 14.5 px | **10.2** | **11.8** | **13.1** |
| 896 | 16.9 px | 11.9 | 13.8 | 15.3 |

**768 is the smallest input that clears #15's floor across the mount band**, and 896 is the first that clears it with margin at the shallow end. The current training default of 512 puts a ring on 7–9 px — well under the floor, before the model makes a single mistake.

> **Correction.** An earlier version of this record recommended 640, computed from the face-on arithmetic in the second column. That overstates the resolution by the `sin(elevation)` factor plus perspective across the board, and 640 does not in fact clear the floor at a realistic mount angle. The correction *raises* the cost and does not change the decision: resnet18 at 896 is still a 2.7% duty cycle.

> **Done (2026-09-12).** `TrainConfig` now defaults to 768, and the reasoning lives in `dartvision.model.spec` rather than as a number typed into three config files — `train.config`, `data.torch_dataset` and `eval.runner` each said 512 independently, which is how defaults drift apart. A test fails if it is lowered below what #15's budget supports, and says why.

This is also why the answer is not simply "use MobileNet and stop worrying." The backbone is cheap to shrink; the input is not.

## 5. Why on-device wins, axis by axis

| Axis | On-device | Server | Why |
| --- | --- | --- | --- |
| **Latency** | ~100 ms | 100 ms + round trip | Irrelevant at 0.13/s either way |
| **Bandwidth** | none | ~12 MB/leg | 120 frames × ~100 KB |
| **Offline** | works | dead | Dartboards live in garages, basements, sheds and pubs. Bad connectivity is the normal case, not the edge case |
| **Privacy** | frames never leave | frames of someone's home on a server | The decisive one — see below |
| **Hosting cost** | zero | scales per user per leg | No GPU fleet, no autoscaling, no cost model at all |
| **Battery** | ~14 s compute/leg | upload instead | Neither matters — see below |
| **Model updates** | needs a fetch | instant | The one axis the server wins (§6) |

**Privacy is not a tiebreaker here, it is the argument.** These are camera frames from inside someone's home, and people walk through them. Sending them to a server creates a consent flow, a retention policy, a breach surface, and a regulatory position — for a feature that does not need any of it. Keeping inference local means the honest claim "your camera never leaves your phone", which is both a better product and a smaller legal surface. `AGENTS.md` gives camera and privacy boundaries their own owner for a reason.

**Battery deserves a correction.** The instinct is that inference drains the phone and offloading it helps. It does not: 14 seconds of compute per leg is nothing. What drains the battery is **holding the camera open for fifteen minutes** — the sensor, the ISP, the preview surface — and that cost is identical in every topology. Server inference adds upload on top of it. The battery argument runs the opposite way from the intuition.

## 6. What the server wins, and how to get it anyway

Model updateability is real. A server can swap the model for every user instantly; a shipped model cannot.

Most of that is recoverable by **not baking the model into the bundle**. Ship it as a versioned asset — `brain-v3.onnx` on a CDN — fetched on first run and held in the Cache API. A new model is then a manifest change: users pick it up on next launch, and the app still works offline afterwards. That is the standard pattern and it costs nothing to adopt from the start. Adopting it later means shipping an app update to change how apps update, so decide it now.

What stays lost is *forced* immediate updates. For a dart scorer that is not worth a GPU fleet.

## 7. One narrow server path worth keeping

Inference should not go to a server. **Corrections should — with consent, and only corrections.**

When a player corrects a score (#10), that is a labelled hard case: an image the model got wrong, with the right answer attached, from a real board this project has never seen. #26 will spend real hours capturing far less valuable data than that. A handful of frames per leg, opt-in, is a training flywheel that costs almost nothing and does not compromise the privacy story — because the user is choosing, per throw, to send one frame.

This is a product decision as much as an architectural one, and it belongs to #10 and to the privacy reviewer. Recorded here because the architecture has to leave room for it, and retrofitting a consented upload path into a strictly-local app is painful.

## 8. Browser or native wrapper

**Browser first.** A PWA over `getUserMedia` has no install step, which for a scorer someone sets up once in a garage is a genuine product advantage over an app-store download.

A native shell (Capacitor or similar around the same Next.js app) buys Core ML on iOS and NNAPI on Android, and therefore the NPU. Per §3, **the NPU is not needed** — the arithmetic clears comfortably without it.

So the trigger for going native is not speed. It is whether a browser can hold a camera open reliably for a fifteen-minute leg: screen sleep, wake lock, backgrounding, and iOS PWA lifecycle behaviour. That is a device-testing question, not an architecture question, and it should be answered early because it is the one finding that would change this decision.

Next.js deployment is unaffected either way: no GPU, no inference endpoint, static assets plus a CDN.

## 9. To verify at implementation time

Browser capabilities move quickly, and this record should not be trusted on them without a check on real devices.

| To verify | Why it matters |
| --- | --- |
| Sustained `getUserMedia` + wake lock across a 15-minute leg, iOS and Android | The single finding that would force a native wrapper |
| WebGPU availability and fallback to WASM on target devices | Sets the real per-inference latency; §3 says even the fallback is sufficient |
| Measured latency of the exported model in-browser, at 768 input | §3 is a CPU proxy, not a browser measurement |
| Cache API behaviour for a ~12 MB model asset, including eviction | Eviction mid-leg must degrade gracefully, not crash |
| Thermal behaviour with the camera open for a full session | The dominant battery and thermal cost, and it is not the model |

None of these can change the *direction* of the decision. Their job is to pick between browser and wrapper, and to size the model.
