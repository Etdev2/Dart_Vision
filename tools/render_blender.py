"""Render a Dart Vision synthetic manifest with Blender.

    blender --background --python tools/render_blender.py -- \
        --manifest data/synthetic/manifest.jsonl \
        --out data/synthetic/images \
        --engine EEVEE

Runs inside Blender's Python, so it imports ``bpy``. The pure geometry it
relies on lives in ``dartvision.synthetic.blender_math`` and is unit-tested
without Blender present.

**Board geometry is generated from the published BDO dimensions**, not from any
third-party asset, which keeps it clean-room per #13. Any HDRI or texture added
later must be CC0 for the same reason (#25).

Self-verification
-----------------
Before rendering, every scene projects its calibration landmarks through
*Blender's own* camera and compares them against the manifest. A disagreement
means the render would not match its label, and the script refuses to continue.
This is the failure that matters: mislabelled images train happily and poison
everything downstream in ways no later metric attributes to the renderer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import bpy  # noqa: E402  (only importable inside Blender)
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from dartvision.data.labels import read_jsonl  # noqa: E402
from dartvision.geometry.board import BDO_BOARD, BoardSpec  # noqa: E402
from dartvision.geometry.camera import CameraPose  # noqa: E402
from dartvision.synthetic.blender_math import (  # noqa: E402
    MM_PER_METRE,
    camera_world_matrix,
    lens_mm_from_focal_px,
)

SECTOR_ARC_DEG = 18.0
FIRST_BOUNDARY_DEG = 99.0


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------

def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights):
        for block in list(collection):
            collection.remove(block)


def make_material(name: str, rgba, roughness: float = 0.6, metallic: float = 0.0):
    material = bpy.data.materials.new(name)
    # Blender 5.x materials always use nodes and deprecate the setter; 4.x needs it.
    if getattr(material, "node_tree", None) is None:
        material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return material


def annular_sector(r_inner_m, r_outer_m, angle_from, angle_to, z, steps=8):
    """Vertices and faces for one annular sector lying flat at height ``z``."""
    verts, faces = [], []
    angles = [
        math.radians(angle_from + (angle_to - angle_from) * i / steps)
        for i in range(steps + 1)
    ]
    for a in angles:
        verts.append((r_inner_m * math.cos(a), r_inner_m * math.sin(a), z))
        verts.append((r_outer_m * math.cos(a), r_outer_m * math.sin(a), z))
    for i in range(steps):
        base = 2 * i
        faces.append((base, base + 1, base + 3, base + 2))
    return verts, faces


def add_mesh(name: str, verts, faces, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def build_board(board: BoardSpec, style: str) -> None:
    """The scoring surface, built from published dimensions."""
    worn = style == "worn-sisal"
    cream = make_material("cream", (0.86, 0.80, 0.62, 1.0) if not worn else (0.72, 0.66, 0.50, 1.0), 0.85)
    black = make_material("black", (0.045, 0.045, 0.05, 1.0), 0.9)
    red = make_material("red", (0.62, 0.06, 0.06, 1.0), 0.7)
    green = make_material("green", (0.04, 0.35, 0.16, 1.0), 0.7)
    wire_mat = make_material("wire", (0.62, 0.62, 0.65, 1.0), 0.25, metallic=1.0)
    surround = make_material("surround", (0.03, 0.03, 0.035, 1.0), 0.95)

    m = MM_PER_METRE
    bands = [
        (board.r_outer_bull / m, board.r_treble_inner / m, "single"),
        (board.r_treble_inner / m, board.r_treble / m, "ring"),
        (board.r_treble / m, board.r_double_inner / m, "single"),
        (board.r_double_inner / m, board.r_double / m, "ring"),
    ]

    for i in range(20):
        angle_to = FIRST_BOUNDARY_DEG - SECTOR_ARC_DEG * i
        angle_from = angle_to - SECTOR_ARC_DEG
        dark = i % 2 == 0
        for band_index, (r0, r1, kind) in enumerate(bands):
            if kind == "single":
                material = black if dark else cream
            else:
                material = red if dark else green
            verts, faces = annular_sector(r0, r1, angle_from, angle_to, 0.0)
            add_mesh(f"bed-{i}-{band_index}", verts, faces, material)

    # Bull rings and the surround, each a full annulus.
    for r0, r1, material, name in (
        (0.0, board.r_inner_bull / m, red, "inner-bull"),
        (board.r_inner_bull / m, board.r_outer_bull / m, green, "outer-bull"),
        (board.r_double / m, board.r_board / m, surround, "surround"),
    ):
        verts, faces = annular_sector(r0, r1, 0.0, 360.0, 0.0, steps=160)
        add_mesh(name, verts, faces, material)

    # Wires sit fractionally proud of the surface, as real ones do.
    wire_z = 0.0012
    wire_radius = 0.0006
    for radius in (
        board.r_inner_bull, board.r_outer_bull,
        board.r_treble_inner, board.r_treble,
        board.r_double_inner, board.r_double,
    ):
        bpy.ops.mesh.primitive_torus_add(
            major_radius=radius / m, minor_radius=wire_radius, location=(0, 0, wire_z),
            major_segments=192, minor_segments=8,
        )
        bpy.context.object.data.materials.append(wire_mat)

    for i in range(20):
        angle = math.radians(FIRST_BOUNDARY_DEG - SECTOR_ARC_DEG * i)
        r0, r1 = board.r_outer_bull / m, board.r_double / m
        length = r1 - r0
        mid = (r0 + r1) / 2.0
        bpy.ops.mesh.primitive_cylinder_add(
            radius=wire_radius, depth=length,
            location=(mid * math.cos(angle), mid * math.sin(angle), wire_z),
            rotation=(0.0, math.pi / 2.0, angle),
        )
        bpy.context.object.data.materials.append(wire_mat)


def add_dart(tip_mm, index: int, rng) -> None:
    """A dart whose point sits exactly on the annotated board coordinate."""
    x, y = tip_mm[0] / MM_PER_METRE, tip_mm[1] / MM_PER_METRE
    barrel_mat = make_material(f"barrel-{index}", (0.34, 0.34, 0.36, 1.0), 0.3, metallic=1.0)
    flight_mat = make_material(f"flight-{index}", (0.75, 0.12, 0.12, 1.0), 0.8)

    tilt = math.radians(rng.uniform(4.0, 16.0))
    swing = rng.uniform(0.0, 2.0 * math.pi)
    axis = Vector((math.sin(tilt) * math.cos(swing), math.sin(tilt) * math.sin(swing), math.cos(tilt)))

    point_len, barrel_len = 0.022, 0.050
    for length, radius, material, start in (
        (point_len, 0.0011, barrel_mat, 0.0),
        (barrel_len, 0.0035, barrel_mat, point_len),
    ):
        centre = Vector((x, y, 0.0)) + axis * (start + length / 2.0)
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=centre)
        obj = bpy.context.object
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = axis.to_track_quat("Z", "Y")
        obj.data.materials.append(material)

    flight_centre = Vector((x, y, 0.0)) + axis * (point_len + barrel_len + 0.012)
    bpy.ops.mesh.primitive_cube_add(size=0.024, location=flight_centre)
    flight = bpy.context.object
    flight.scale = (0.06, 1.0, 1.0)
    flight.rotation_mode = "QUATERNION"
    flight.rotation_quaternion = axis.to_track_quat("Z", "Y")
    flight.data.materials.append(flight_mat)


def add_lighting(style: str, rng) -> None:
    presets = {
        "daylight-soft": (4.5, 5600, 0.55),
        "daylight-hard": (7.0, 6200, 0.12),
        "tungsten": (3.0, 3000, 0.30),
        "mixed-dim": (1.6, 4200, 0.45),
        "overhead-glare": (9.0, 5200, 0.06),
    }
    energy, kelvin, softness = presets.get(style, presets["daylight-soft"])

    bpy.ops.object.light_add(type="AREA", location=(
        rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(1.4, 2.4)
    ))
    key = bpy.context.object.data
    key.energy = energy * 60.0
    key.size = max(0.05, softness * 2.0)
    key.color = kelvin_to_rgb(kelvin)

    bpy.ops.object.light_add(type="AREA", location=(-1.5, 1.0, 1.0))
    fill = bpy.context.object.data
    fill.energy = energy * 12.0
    fill.size = 2.0
    fill.color = kelvin_to_rgb(kelvin + 400)


def kelvin_to_rgb(kelvin: float):
    """Rough blackbody tint; exact colour science is not the point here."""
    t = max(1000.0, min(12000.0, kelvin)) / 100.0
    red = 1.0 if t <= 66 else min(1.0, 1.292 * (t - 60) ** -0.1332)
    green = min(1.0, (0.3900 * math.log(t) - 0.6318) if t <= 66 else 1.129 * (t - 60) ** -0.0755)
    blue = 0.0 if t <= 19 else (1.0 if t >= 66 else min(1.0, 0.5432 * math.log(t - 10) - 1.1963))
    return (max(0.0, red), max(0.0, green), max(0.0, blue))


def setup_camera(pose: CameraPose):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.matrix_world = Matrix(camera_world_matrix(pose).tolist())
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = lens_mm_from_focal_px(pose.focal_px, pose.image_size[0], 36.0)
    bpy.context.scene.camera = camera
    return camera


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify_labels(camera, annotation, board: BoardSpec, tolerance_px: float) -> float:
    """Project landmarks through Blender's camera and compare to the manifest.

    Returns the largest disagreement in pixels. Blender reports normalized
    coordinates with the origin at the *bottom* left, so the vertical axis is
    flipped relative to our labels -- a mismatch that would otherwise show up
    as a perfectly plausible but vertically mirrored dataset.
    """
    from dartvision.geometry.calibration import calibration_points_4, calibration_points_8

    scene = bpy.context.scene
    width, height = annotation.image_size
    expected = (
        calibration_points_4(board)
        if len(annotation.landmarks) == 4
        else calibration_points_8(board)
    )

    worst = 0.0
    for (bx, by), label in zip(expected, annotation.landmarks):
        if label is None:
            continue
        world = Vector((bx / MM_PER_METRE, by / MM_PER_METRE, 0.0))
        u, v, _ = world_to_camera_view(scene, camera, world)
        rendered = (u * width, (1.0 - v) * height)
        stored = (label[0] * width, label[1] * height)
        worst = max(worst, math.dist(rendered, stored))

    if worst > tolerance_px:
        raise SystemExit(
            f"label mismatch on {annotation.image_id}: landmarks disagree by "
            f"{worst:.2f}px (tolerance {tolerance_px}px). Refusing to render a "
            "dataset whose images and labels do not match."
        )
    return worst


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a Dart Vision manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=("EEVEE", "CYCLES"), default="EEVEE")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="render at most N scenes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-px", type=float, default=1.0)
    parser.add_argument("--verify-only", action="store_true",
                        help="check labels against the camera without rendering")
    return parser.parse_args(argv)


# The EEVEE identifier has changed across Blender releases -- "BLENDER_EEVEE" up
# to 4.1, "BLENDER_EEVEE_NEXT" through the 4.2 series, and "BLENDER_EEVEE" again
# from 5.0. Rather than track that, try the candidates and keep the one the build
# accepts.
EEVEE_CANDIDATES = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")


def select_engine(scene, engine: str) -> str:
    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        return "CYCLES"
    for name in EEVEE_CANDIDATES:
        try:
            scene.render.engine = name
            return name
        except TypeError:
            continue
    raise SystemExit(
        "no EEVEE engine found in this Blender build; tried "
        f"{', '.join(EEVEE_CANDIDATES)}. Use --engine CYCLES instead."
    )


def enable_gpu_compute() -> str | None:
    """Point Cycles at the best available GPU backend, Metal included."""
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return None
    prefs = addon.preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        try:
            prefs.get_devices()
        except Exception:  # noqa: BLE001 -- older builds lack this entirely
            pass
        for device in getattr(prefs, "devices", []):
            device.use = True
        return backend
    return None


def configure_render(engine: str, samples: int, image_size) -> str:
    scene = bpy.context.scene
    resolved = select_engine(scene, engine)
    scene.render.resolution_x, scene.render.resolution_y = image_size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92

    if resolved == "CYCLES":
        scene.cycles.samples = samples
        backend = enable_gpu_compute()
        try:
            scene.cycles.device = "GPU" if backend else "CPU"
        except (AttributeError, TypeError):
            pass
        return f"CYCLES ({backend or 'CPU'})"

    # taa_render_samples is EEVEE's sample count; the attribute name has moved
    # around, so set it only if this build exposes it.
    eevee = getattr(scene, "eevee", None)
    if eevee is not None and hasattr(eevee, "taa_render_samples"):
        eevee.taa_render_samples = samples
    return resolved


def main() -> int:
    import random

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    annotations = list(read_jsonl(args.manifest))
    if args.limit:
        annotations = annotations[: args.limit]

    rng = random.Random(args.seed)
    worst_overall = 0.0
    resolved_engine = args.engine

    for index, annotation in enumerate(annotations, start=1):
        meta = annotation.meta
        pose = CameraPose(
            elevation_deg=float(meta["elevation_deg"]),
            azimuth_deg=float(meta["azimuth_deg"]),
            distance_mm=float(meta["distance_mm"]),
            focal_px=float(meta["focal_px"]),
            roll_deg=float(meta.get("roll_deg", 0.0)),
            image_size=tuple(annotation.image_size),
        )

        clear_scene()
        build_board(BDO_BOARD, str(meta.get("board_style", "classic-black")))

        width, height = annotation.image_size
        for tip_index, (nx, ny) in enumerate(annotation.tips):
            # Labels are normalized image coordinates; the renderer needs the
            # board-space point, so invert the projection we generated from.
            from dartvision.geometry.calibration import estimate_homography
            board_pt = estimate_homography(annotation.landmarks).to_board(
                [(nx, ny)]
            )[0]
            add_dart((float(board_pt[0]), float(board_pt[1])), tip_index, rng)

        add_lighting(str(meta.get("lighting", "daylight-soft")), rng)
        camera = setup_camera(pose)
        resolved_engine = configure_render(args.engine, args.samples, (width, height))

        worst_overall = max(worst_overall, verify_labels(camera, annotation, BDO_BOARD, args.tolerance_px))

        if not args.verify_only:
            name = annotation.image_id.replace("/", "_") + ".jpg"
            bpy.context.scene.render.filepath = str(args.out / name)
            bpy.ops.render.render(write_still=True)

        if index % 25 == 0 or index == len(annotations):
            print(f"[{index}/{len(annotations)}] worst label error so far: {worst_overall:.3f}px",
                  flush=True)

    print(json.dumps({
        "scenes": len(annotations),
        "rendered": 0 if args.verify_only else len(annotations),
        "worst_label_error_px": round(worst_overall, 4),
        "engine": resolved_engine,
        "blender": bpy.app.version_string,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
