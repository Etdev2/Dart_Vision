"""Tests for the environment report."""

from __future__ import annotations

import json

from dartvision.doctor import MachineReport, capability_table, inspect_machine, main


def base_report(**overrides) -> MachineReport:
    defaults = dict(
        python_version="3.11.0", platform="Darwin 23.0", machine="arm64",
        processor="Apple M1 Pro", is_apple_silicon=True, cpu_count=8,
        total_memory_gb=16.0, free_disk_gb=200.0, numpy_version="2.0",
        torch_version=None, torch_mps_available=False, torch_cuda_available=False,
        blender_path=None, blender_version=None,
    )
    defaults.update(overrides)
    return MachineReport(**defaults)


def test_inspects_the_real_machine_without_raising():
    report = inspect_machine()
    assert report.python_version and report.platform and report.machine
    assert isinstance(report.notes, list)


def test_capabilities_track_what_is_installed():
    caps = dict((label, ok) for label, ok, _ in capability_table(base_report()))
    assert caps["generate scene manifests (#25)"] is True
    assert caps["render synthetic images (#25)"] is False
    assert caps["train the model (#17, #20)"] is False
    assert caps["export Core ML (#4, #22)"] is True


def test_a_cuda_box_can_train_but_not_export_core_ml():
    report = base_report(
        platform="Linux 6.1", machine="x86_64", processor="x86_64",
        is_apple_silicon=False, torch_version="2.5", torch_cuda_available=True,
    )
    caps = dict((label, ok) for label, ok, _ in capability_table(report))
    assert caps["train the model (#17, #20)"] is True
    assert caps["export Core ML (#4, #22)"] is False


def test_blender_presence_flips_the_render_capability():
    report = base_report(blender_path="/Applications/Blender.app/Contents/MacOS/Blender")
    caps = dict((label, ok) for label, ok, _ in capability_table(report))
    assert caps["render synthetic images (#25)"] is True


def test_apple_silicon_is_warned_off_training():
    from dartvision.doctor import _notes

    notes = " ".join(_notes(base_report()))
    assert "rented CUDA GPU" in notes
    assert "MPS" in notes


def test_low_memory_apple_machine_is_steered_to_eevee():
    from dartvision.doctor import _notes

    assert "EEVEE" in " ".join(_notes(base_report(total_memory_gb=8.0)))
    assert "EEVEE" not in " ".join(_notes(base_report(total_memory_gb=32.0)))


def test_missing_tools_are_reported():
    from dartvision.doctor import _notes

    notes = " ".join(_notes(base_report()))
    assert "Blender not found" in notes and "PyTorch not installed" in notes


def test_cli_runs_in_both_modes(capsys):
    assert main([]) == 0
    assert "Can this machine:" in capsys.readouterr().out

    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "python_version" in payload and "notes" in payload
