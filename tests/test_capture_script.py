"""The ./capture wrapper.

Not a unit test of behaviour -- it is four lines of shell whose whole job is to
be runnable from a Terminal that has nothing set up. What is worth asserting is
that it stays runnable: executable, syntactically valid, and passing its
arguments through rather than interpreting them.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "capture"


def test_the_script_exists_and_is_executable():
    assert SCRIPT.exists(), "the documented entry point must be present"
    assert SCRIPT.stat().st_mode & stat.S_IXUSR, (
        "a script the protocol tells people to run as ./capture has to be "
        "executable in the checkout, not after a chmod they were never told about"
    )


def test_the_script_is_valid_shell():
    result = subprocess.run(["sh", "-n", str(SCRIPT)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_it_passes_arguments_through_untouched():
    """`"$@"` and not `$*`: a path with a space in it is one argument, and
    Desktop folders have spaces in them."""
    text = SCRIPT.read_text()
    assert 'exec "$python" -m dartvision.capture "$@"' in text


def test_it_finds_its_environment_without_changing_directory():
    """Relative paths in the arguments have to keep the meaning the caller gave
    them. A `cd` into the checkout would re-root them silently: run from a
    recordings folder, `--out captures/` would land inside the repository."""
    text = SCRIPT.read_text()

    assert 'root="$(cd "$(dirname "$0")" && pwd)"' in text
    assert '"$python" -m dartvision.capture' in text
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("cd "), (
            f"changing directory re-roots the caller's relative paths: {stripped}"
        )


def test_it_stops_on_the_first_failure():
    assert "set -e" in SCRIPT.read_text(), (
        "a failed venv creation must not fall through to a confusing import error"
    )


def test_it_verifies_the_install_worked_rather_than_assuming():
    """pip can report success having installed this package but not a
    dependency -- no network, most often -- and the failure then surfaces as a
    traceback out of numpy, which reads like a bug in the tool."""
    text = SCRIPT.read_text()
    assert text.count("import dartvision.capture") == 2, (
        "importability is checked before installing and again after"
    )
    assert "still not importable" in text


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell")
@pytest.mark.skipif(
    not (SCRIPT.parent / ".venv").exists(),
    reason="no .venv in this checkout; creating one needs the network",
)
def test_it_forwards_the_help_text(tmp_path):
    """End to end, where an environment already exists: the wrapper reaches the
    module from an unrelated working directory and the parser answers.

    Run from ``tmp_path`` on purpose -- the tool is used from wherever the
    recordings are, and a wrapper that only works from the repository root
    would pass every other test here.
    """
    result = subprocess.run(
        [str(SCRIPT), "--help"], capture_output=True, cwd=tmp_path, timeout=300
    )
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    assert b"--video-dir" in result.stdout
