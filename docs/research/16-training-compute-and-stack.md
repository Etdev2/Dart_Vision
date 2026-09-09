# Training Compute and Experiment Stack

> Research record for **[Brain Research] Choose the training compute and experiment stack** (issue #16), parent map #12.
>
> Audience: a first-time ML project owner who needs serious experiments to be reproducible by an agent, without becoming an ML engineer.

## 1. Headline finding

**Compute is not this project's constraint, and it is not close.**

Dart Vision's workload is a small keypoint model on ~16k images. A full training run costs single-digit dollars on rented hardware. The entire v0.1 experiment campaign fits comfortably inside a **~$150–300** budget.

The real constraints are the ones #13 already surfaced: **dataset diversity** (only two board setups) and **licensing**. Do not over-invest in compute. Buy hours, not hardware, and spend the saved effort on #19's capture set.

## 2. Sizing the workload

| Property | Value | Source |
| --- | --- | --- |
| Training images | ~15,000 (D1) + ~830 (D2) | #13 |
| Image size | 800×800 | #13 |
| Dataset on disk | ~3.35 GB cropped | #13 |
| Task | 7 keypoints/image (4 landmarks + ≤3 tips) | #13 |
| Reference model | YOLOv4-tiny class — small | #13 |
| Reference schedule | 100 epochs, batch 16 | #13 |

**Estimated** requirements for a modern equivalent at 800px, batch 16:

- **VRAM: 16 GB workable, 24 GB comfortable.** A 24 GB card removes batch-size fiddling entirely.
- **Wall clock: roughly 4–8 hours** for a 100-epoch run on a 4090-class GPU.

> Both figures are estimates from the model class and dataset size, not measurements. **Issue #17's small-sample harness should measure them first** — that is precisely what it is for. Do not commit to a compute plan before #17 reports real seconds-per-epoch.

## 3. Options compared

| Option | Cost | Verdict |
| --- | --- | --- |
| **Rented hourly GPU** (RunPod Community, Vast.ai) | RTX 4090 **$0.34/hr** RunPod Community (vs $0.69 Secure); Vast.ai spot from ~$0.11–0.39; A100 80GB ~$1.19–1.39/hr | **Recommended.** Pay only for run time, pick your VRAM, full root access, reproducible via a pinned container. |
| **Rented, SLA-backed** (Lambda, RunPod Secure) | A100 ~$2.06/hr; H100 ~$2.69–2.99/hr | Overkill. Reserve for a final long run if one is ever needed. |
| **Google Colab Pro / Pro+** | $9.99/mo (100 compute units) / higher tier (500 CU). A100 burns up to ~13 CU/hr | **Exploration only — not for runs of record.** |
| **Local GPU** | Hardware capex | Only sensible above ~100 GPU-hr/month sustained. We are far below that. |

### 3.1 Why not Colab for the real runs

Colab's compute-unit model buys a *budget*, not a *reservation* — a paid tier can still hand you a T4, GPU assignment depends on availability and usage patterns, and sessions can be interrupted. Issue #20 requires saving "all configs, seeds, checkpoints, metrics and experiment logs" for a defensible model-selection decision. A runtime that may silently change hardware under you, or die mid-run, is the wrong substrate for that.

Colab is genuinely good for looking at data, sanity-checking annotations, and rendering predictions. Use it there.

### 3.2 Local hardware: MacBook Pro — *resolved*

The product owner's machine is a **MacBook Pro**, so there is no NVIDIA GPU and no CUDA. This **confirms the rent-hourly recommendation**: local full training is not on the table.

It does not make the Mac useless — it makes its role specific.

| Runs on the Mac | Runs on the rented Linux/CUDA box |
| --- | --- |
| Writing code, configs, tests | **Every run that produces a number anyone cites** |
| Dataset inspection, annotation review | #17 harness runs, #18 sweeps, #20 full campaign |
| Visualizing predictions and failure cases | Anything feeding #21's ship gates |
| Smoke tests on ~50 images (does loss go down?) | |
| **Core ML export and on-device benchmarking** | |

**Keep the split strict.** PyTorch's Apple `mps` backend is fine for smoke tests, but its numerics differ from CUDA and some operators fall back to CPU or are missing outright. Debugging on `mps` while training on CUDA is a good way to chase ghosts that do not exist on either. Rule: **the Mac authors, the rented box measures.** Same repo, same pinned container; the container only ever runs CUDA.

### 3.3 What the Mac can do *now* — including all of the rendering

Confirmed September 2026: **Blender runs natively on Apple Silicon with Metal GPU acceleration.** The Cycles Metal backend arrived in Blender 3.1, EEVEE viewport support in 3.5, and GPU-accelerated ray tracing and denoising are available on Apple Silicon (macOS 13+ for full feature support).

That materially changes #25. The synthetic renderer does **not** need rented hardware — it runs on the MacBook. So the entire synthetic pipeline, label generation *and* image rendering, is local and free.

Caveat on speed: Apple Silicon lacks the dedicated BVH-traversal hardware that RTX cards expose through OptiX, so Cycles is slower per frame than an equivalent NVIDIA card. For this workload that matters less than it sounds — a dartboard is simple geometry at modest resolution — and **EEVEE, a rasterizer, is dramatically faster than Cycles** and very likely sufficient. Wire specularity is the one thing worth comparing between the two before committing.

Practical note: base M1 ships with 8 GB unified memory, M1 Pro/Max with 16–64 GB. Rendering is comfortable either way; the memory question only bites for training, which is rented regardless.

### 3.4 The Mac becomes essential later, for a different reason

Core ML conversion and real on-device iPhone latency/thermal benchmarking **require macOS**. So the MacBook is the right and necessary tool for the edge-export path — the Edge Inference Engineer's work, issue #4's runtime decision, and the export artifacts in #22. It is an asset for deployment, just not for training.

### 3.5 Working loop

1. Author code and configs on the Mac; commit.
2. Push to git.
3. Rented pod pulls the pinned container and the repo at a specific SHA, runs one config, pushes checkpoints and metrics to object storage.
4. Pull results back to the Mac for inspection and plotting.

This satisfies §6's reproducibility contract, needs no GPU on the Mac, and is directly runnable by an agent without supervision.

## 4. Framework licensing — the same trap as #13, and easy to walk into

**The default choice a newcomer would make is a commercial landmine.**

**Ultralytics YOLO (YOLOv8/YOLO11) is AGPL-3.0.** Critically, that licence covers **the training code *and* the models produced by it** — so weights trained with Ultralytics are themselves encumbered. Using it in a proprietary product requires open-sourcing the project, or buying an Ultralytics Enterprise Licence.

This is the single most likely way Dart Vision accidentally poisons its own Brain: the obvious "just fine-tune YOLO11-pose" path would make the published Hugging Face artifact (#22) unusable commercially, independently of however the DeepDarts data licence resolves.

### Permissive alternatives (all Apache-2.0)

| Option | Notes |
| --- | --- |
| **Custom head on a `timm` backbone** (plain PyTorch) | **Recommended baseline.** The task is narrow — 7 points, fixed order. Full control, zero licence ambiguity, small surface to reason about, and it satisfies the clean-room requirement from #13. |
| **YOLOX** (Megvii) | Closest permissive analogue to DeepDarts' "keypoints as objects" framing. Good challenger for #15. |
| **MMPose / MMDetection** (OpenMMLab) | Batteries included, strong keypoint support, more framework to learn. |
| **Detectron2** (Meta) | Mature, permissive, keypoint-capable; heavier than needed. |

> **Rule for #15 and #17: every model, weight, and package that touches the Brain must have its licence checked before use, not after.** Pretrained *weights* often carry different terms from the code that trained them — check both. Route through the Licensing/Data Governance Reviewer.

## 5. Recommended stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ | Reference stack's Python 3.7 is end-of-life |
| Framework | **PyTorch 2.x** | The reference TF 2.3 stack is dead (#13); do not resurrect it |
| Model code | Permissive only — see §4 | Commercial-safe by default |
| Config | YAML + typed dataclasses (or Hydra) | Every run reconstructible from one file |
| Tracking | Weights & Biases free tier, or self-hosted MLflow | Must log config, git SHA, seed, and metrics together |
| Checkpoints | S3-compatible object storage (Cloudflare R2 / Backblaze B2) | Cheap, outlives the rented box |
| Environment | **Pinned Docker image** | The reproducibility guarantee; agent-runnable |
| Data integrity | Manifest of file hashes committed to git | Detects silent data drift without DVC overhead |

Deliberately excluded for now: DVC, Kubeflow, multi-node orchestration. At this scale they add ceremony without buying anything.

## 6. Reproducibility contract

Any run that informs a #20 or #21 decision must record, automatically:

1. Git SHA of the training repo (refuse to run on a dirty tree, or record the diff)
2. The complete resolved config
3. All random seeds (Python, NumPy, PyTorch, CUDA) with deterministic flags where affordable
4. Dataset manifest hash and the exact split assignment (#14)
5. Container image digest
6. Full metric history, not just the final number
7. Checkpoint URI

**Non-negotiable:** the rented box is ephemeral. Anything not pushed to object storage or git is gone when the pod stops. The run command should be a single container invocation reading one config file — that is what makes an agent able to reproduce it without the owner supervising.

## 7. Cost envelope

| Phase | Estimate |
| --- | --- |
| #17 small-sample harness | < $10 — short runs, mostly debugging |
| #18 augmentation sweep (~10–20 short runs) | $30–80 |
| #20 full campaign (~10–20 full runs) | $50–150 |
| Storage + tracking | ~$5/month |
| **v0.1 total** | **~$150–300** |

For context, a single Ultralytics Enterprise Licence would very likely exceed this entire budget — which is another reason the permissive path in §4 is the right default, not merely the safe one.

## 8. Recommendation

1. **Rent hourly on a 4090-class GPU (24 GB)** from RunPod Community or Vast.ai. No commitment, no hardware purchase.
2. **PyTorch 2.x, permissive model code only.** Start with a custom keypoint head on a `timm` backbone; carry YOLOX as the challenger for #15.
3. **Never use Ultralytics** unless someone has bought an Enterprise Licence and written it down.
4. **Colab for looking at data, never for runs of record.**
5. **Containerize before the first real run**, so #17's harness is reproducible from the outset rather than retrofitted.
6. **Local hardware is a MacBook Pro** — no CUDA, so rent for all real runs; the Mac authors code and later handles Core ML export (§3.2–3.5).
7. **Let #17 measure the real per-epoch cost** and correct §2's estimates.

> Note for #22: `huggingface.co` is blocked from the current agent environment. Whichever compute is chosen must have outbound access to the Hub, or publishing needs a separate step.

## 8.1 The real financial risk is an idle pod, not training

Rented GPUs bill **for every hour the pod exists, not for hours it computes.** A 4090 left running overnight costs about $8 for nothing. Over a campaign that dwarfs the cost of the training itself.

Two habits remove it: make the container exit when the run finishes rather than dropping to a shell, and check for running pods before closing the laptop. This is the mistake everyone makes exactly once.

## 9. Rates cited

GPU pricing moves week to week and Vast.ai spot rates move by the minute. Re-check live rates before committing. Figures above were gathered September 2026 from public comparison sources and should be treated as indicative, not quoted.
