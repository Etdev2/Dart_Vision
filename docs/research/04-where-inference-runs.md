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

| Backbone | Input | 10 mm ring | GMACs | ms/frame | Params | int8 size |
| --- | --- | --- | --- | --- | --- | --- |
| resnet18 | 512 | 9.1 px | 11.42 | 60 | 11.3 M | 11.3 MB |
| resnet18 | **640** | **11.4 px** | 17.85 | 115 | 11.3 M | 11.3 MB |
| resnet18 | 768 | 13.6 px | 25.70 | 156 | 11.3 M | 11.3 MB |
| efficientnet_b0 | 640 | 11.4 px | 4.63 | 113 | 3.7 M | 3.7 MB |
| mobilenetv3_small | 640 | 11.4 px | 1.19 | 33 | 1.0 M | 1.0 MB |

At the recommended configuration — resnet18 at 640, the slowest credible choice — a leg costs `120 × 115 ms ≈ 14 seconds` of compute spread over fifteen minutes. **A 1.5% duty cycle, on the CPU, with none of the acceleration a phone actually has.**

There is no latency problem to solve here. There is barely a latency question.

One thing in that table is worth flagging for anyone tuning it later: efficientnet_b0 does a quarter of resnet18's multiply-accumulates and takes the same wall-clock time. Depthwise separable convolutions are arithmetic-efficient and memory-bound, so MACs badly mispredict their speed. Pick a backbone on measured latency, never on GMACs.

## 4. The input resolution is the real cost lever — and it has a floor

Cost scales with input area, so shrinking the input is the obvious economy. #15's precision budget forbids it.

A 10 mm scoring ring must land on 10–20 px for the tip head to have anything to work with. At 80% board fill:

| Model input | 320 | 448 | 512 | **640** | 768 |
| --- | --- | --- | --- | --- | --- |
| Ring width | 5.0 px | 7.9 px | **9.1 px** | **11.4 px** | 13.6 px |

**The current training default of 512 is under-resolved**, at 9.1 px — below #15's own floor before the model makes a single mistake. 640 is the smallest input that satisfies the budget with any margin, and it is the number this recommendation assumes.

> **Consequence for #17:** `TrainConfig.input_height/input_width` default to 512. That should be reviewed against this table before the training campaign runs, because a campaign at 512 measures a model the precision budget already rules out. Flagged here rather than changed — it is #17's call, and it costs GPU hours.

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
| Measured latency of the exported model in-browser, at 640 input | §3 is a CPU proxy, not a browser measurement |
| Cache API behaviour for a ~12 MB model asset, including eviction | Eviction mid-leg must degrade gracefully, not crash |
| Thermal behaviour with the camera open for a full session | The dominant battery and thermal cost, and it is not the model |

None of these can change the *direction* of the decision. Their job is to pick between browser and wrapper, and to size the model.
