"""Report what this machine can and cannot do for the Dart Vision pipeline.

    python -m dartvision.doctor

Written because the answer differs sharply by machine and the differences are
easy to get wrong from memory: an Apple Silicon Mac can generate labels and
render images but must not be used for training runs of record (#16), while a
CUDA box can train but cannot export Core ML. Rather than guess, ask.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

__all__ = ["MachineReport", "inspect_machine", "capability_table", "main"]


@dataclass
class MachineReport:
    python_version: str
    platform: str
    machine: str
    processor: str
    is_apple_silicon: bool
    cpu_count: int | None
    total_memory_gb: float | None
    free_disk_gb: float | None
    numpy_version: str | None
    torch_version: str | None
    torch_mps_available: bool
    torch_cuda_available: bool
    blender_path: str | None
    blender_version: str | None
    notes: list[str] = field(default_factory=list)


def _total_memory_gb() -> float | None:
    try:  # Linux and most Unixes
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, AttributeError, OSError):
        pass
    try:  # macOS
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return round(int(out.stdout.strip()) / 1e9, 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _apple_chip() -> str | None:
    """The specific chip, e.g. 'Apple M1 Pro' -- which ``platform`` will not tell you."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _blender() -> tuple[str | None, str | None]:
    path = shutil.which("blender")
    if path is None:
        for candidate in (
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender/Blender.app/Contents/MacOS/Blender",
        ):
            if os.path.exists(candidate):
                path = candidate
                break
    if path is None:
        return None, None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
        version = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        version = None
    return path, version


def inspect_machine() -> MachineReport:
    processor = _apple_chip() or platform.processor() or "unknown"
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    numpy_version = torch_version = None
    mps = cuda = False
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        pass
    try:
        import torch
        torch_version = torch.__version__
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda = bool(torch.cuda.is_available())
    except (ImportError, AttributeError, RuntimeError):
        pass

    blender_path, blender_version = _blender()
    try:
        free_disk_gb = round(shutil.disk_usage(".").free / 1e9, 1)
    except OSError:
        free_disk_gb = None

    report = MachineReport(
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        processor=processor,
        is_apple_silicon=is_apple_silicon,
        cpu_count=os.cpu_count(),
        total_memory_gb=_total_memory_gb(),
        free_disk_gb=free_disk_gb,
        numpy_version=numpy_version,
        torch_version=torch_version,
        torch_mps_available=mps,
        torch_cuda_available=cuda,
        blender_path=blender_path,
        blender_version=blender_version,
    )
    report.notes = _notes(report)
    return report


def _notes(report: MachineReport) -> list[str]:
    notes: list[str] = []
    if report.is_apple_silicon:
        notes.append(
            "Apple Silicon: generate labels and render here, but run training on a "
            "rented CUDA GPU. Do not debug on MPS and train on CUDA -- the numerics "
            "differ and some operators fall back to CPU (#16)."
        )
        if report.total_memory_gb is not None and report.total_memory_gb < 12:
            notes.append(
                f"{report.total_memory_gb} GB unified memory is tight for Cycles. "
                "Prefer the EEVEE engine and modest resolutions."
            )
    if report.blender_path is None:
        notes.append("Blender not found: needed to render synthetic images (#25).")
    if report.torch_version is None:
        notes.append("PyTorch not installed: needed only for training and export.")
    if report.torch_cuda_available:
        notes.append("CUDA available: this machine can run training of record.")
    if report.free_disk_gb is not None and report.free_disk_gb < 20:
        notes.append(f"Only {report.free_disk_gb} GB free; rendered datasets grow quickly.")
    return notes


def capability_table(report: MachineReport) -> list[tuple[str, bool, str]]:
    """Which pipeline stage can run here, and why not when it cannot."""
    has_numpy = report.numpy_version is not None
    return [
        ("run the test suite", has_numpy, "needs numpy"),
        ("generate scene manifests (#25)", has_numpy, "needs numpy"),
        ("render synthetic images (#25)", report.blender_path is not None, "needs Blender"),
        ("train the model (#17, #20)", report.torch_cuda_available,
         "needs a CUDA GPU -- rent one, see docs/research/16"),
        ("smoke-test on MPS only", report.torch_mps_available, "needs PyTorch with MPS"),
        ("export Core ML (#4, #22)", report.platform.startswith("Darwin"), "needs macOS"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report this machine's capabilities")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    report = inspect_machine()
    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return 0

    print("Dart Vision environment\n")
    print(f"  python    {report.python_version}")
    print(f"  platform  {report.platform} ({report.machine})")
    print(f"  processor {report.processor}")
    if report.total_memory_gb:
        print(f"  memory    {report.total_memory_gb} GB")
    if report.free_disk_gb:
        print(f"  disk free {report.free_disk_gb} GB")
    print(f"  numpy     {report.numpy_version or '-'}")
    print(f"  torch     {report.torch_version or '-'}"
          f"{'  (MPS)' if report.torch_mps_available else ''}"
          f"{'  (CUDA)' if report.torch_cuda_available else ''}")
    print(f"  blender   {report.blender_version or '-'}")

    print("\nCan this machine:")
    for label, ok, reason in capability_table(report):
        print(f"  [{'x' if ok else ' '}] {label}" + ("" if ok else f"  ({reason})"))

    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
