"""The parity fixtures must describe the Python as it is now.

`web/lib/` is a port, and `web/test/parity.test.mjs` checks it against fixtures
generated from this package. That only means anything while the fixtures are
current: if the Python's behaviour changes and the fixtures do not, the
JavaScript keeps passing against a specification that no longer exists.

So regeneration is checked here. A behavioural change to scoring, the
homography or the framing assessment fails this test, and the fix is to
regenerate and re-run the JavaScript suite -- which is exactly the moment to
find out the port needs updating too.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "make_parity_fixtures.py"
FIXTURES = ROOT / "web" / "fixtures" / "parity.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not FIXTURES.exists():
        pytest.fail(f"{FIXTURES} is missing; run `python {GENERATOR.relative_to(ROOT)}`")
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_the_fixtures_match_what_the_python_computes_today(committed, tmp_path):
    """Regenerating must reproduce the committed file byte for byte."""
    regenerated = tmp_path / "parity.json"
    result = subprocess.run(
        [sys.executable, str(GENERATOR), str(regenerated)],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert result.returncode == 0, result.stderr

    assert json.loads(regenerated.read_text(encoding="utf-8")) == committed, (
        "web/fixtures/parity.json is stale: the Python now computes something "
        f"different. Run `python {GENERATOR.relative_to(ROOT)}` and re-run the "
        "JavaScript suite (`node --test web/test/parity.test.mjs`) -- the port "
        "probably needs the same change."
    )


def test_the_fixtures_are_substantial_enough_to_mean_something(committed):
    """A fixture file that quietly shrank would pass vacuously."""
    assert len(committed["scoring"]) > 400
    assert len(committed["homography"]) >= 8
    assert len(committed["framing"]) >= 60
    assert {c["expected"]["verdict"] for c in committed["framing"]} == {
        "ready", "marginal", "unusable",
    }


def test_the_port_exists_and_is_wired_to_the_fixtures():
    """Guards against the fixtures outliving the thing they test."""
    for module in ("board.js", "homography.js", "framing.js", "match.js"):
        assert (ROOT / "web" / "lib" / module).exists(), module
    suite = (ROOT / "web" / "test" / "parity.test.mjs").read_text(encoding="utf-8")
    assert "../fixtures/parity.json" in suite
