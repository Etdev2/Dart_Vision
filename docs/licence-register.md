# Licence register

> Every third-party component that ships in, or is used to build, Dart Vision.
> Required by #8's governance plan. `tests/test_licences.py` fails if a
> dependency appears in `pyproject.toml` without an entry here.
>
> **Status column** follows #13's convention: *Verified* means someone read the
> primary source (the package's own `LICENSE`) and recorded the date. *Declared*
> means the licence is the one the project publishes and is widely relied upon,
> but nobody on this project has read it from source yet. Declared is not a
> blocker for development; it is a blocker for a commercial release.

## Runtime and build dependencies

| Distribution | Used for | Licence | Commercial use | Status |
| --- | --- | --- | --- | --- |
| `numpy` | Geometry, targets, metrics — everywhere | BSD-3-Clause | ✅ | Declared |
| `torch` | Training and inference | BSD-3-Clause | ✅ | Declared |
| `timm` | Backbone architectures | Apache-2.0 | ✅ | Declared |
| `pillow` | Image loading in the dataset | MIT-CMU (HPND) | ✅ | Declared |
| `pytest` | Tests only; does not ship | MIT | ✅ | Declared |

## Tools used but not distributed

| Component | Used for | Licence | Notes |
| --- | --- | --- | --- |
| Blender | Rendering the synthetic corpus (#25) | GPL-2.0-or-later | GPL covers Blender itself. **Renders are the artist's own work** and carry no GPL obligation — the same reason images made in GIMP are not GPL. `tools/render_blender.py` is our own code, run *inside* Blender via its Python API. |

## Model weights

| Component | Status |
| --- | --- |
| Backbone initialisation | **Trained from scratch.** `pretrained=False` is the default in `TrainConfig`, `DartVisionBrain` and the eval loader. See #8 §4 — this is a live decision, not a settled one. |
| Published Dart Vision weights | Ours outright. Trained on synthetic data we generated and captures we own (#26). No third-party data, no third-party weights. |

## Explicitly excluded

| Component | Licence | Why excluded |
| --- | --- | --- |
| **Ultralytics YOLO** (v8, v11, any) | AGPL-3.0 | The licence reaches **the weights trained with it**, not just the code. Using it would make the published artifact (#22) unusable commercially. #16 calls this the single most likely way the project poisons its own Brain. |
| **DeepDarts** images, annotations, weights | None declared / unverified | No `LICENSE` in the repository; the dataset licence could not be established (#13). Default is all rights reserved. The *paper* remains citable prior art. |
| **`bnww/dart-sense`** | CC BY-NC 4.0 | Non-commercial. Its weights also carry Ultralytics AGPL exposure on top. Study only (#24). |
| **Re-uploads and mirrors** of any of the above | n/a | A re-uploader cannot grant rights they do not hold (#13 §2.2). |

## Bookmarked, not used

| Component | Licence | Why kept |
| --- | --- | --- |
| `lksmlr/autodarts` | Apache-2.0 | The only permissive triangulation reference found (#24). Relevant only if #2's escalation condition fires and a second viewpoint is needed. |
