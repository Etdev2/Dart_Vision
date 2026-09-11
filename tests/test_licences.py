"""Governance, enforced rather than intended.

#8 asks for a clean data-governance and attribution plan. A plan in a document
is a plan somebody forgets on the afternoon they add a dependency in a hurry,
and licence problems are cheapest at exactly that moment and most expensive
later -- #13 is a whole research record about a licence nobody wrote down.

So the register is checked by a test, in the same spirit as the holdout ledger
in #14: the discipline is mechanical, and adding a dependency without recording
its licence fails the build.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "docs" / "licence-register.md"
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_JSON = ROOT / "web" / "package.json"

# Names that must never appear as a dependency. #16 established that
# Ultralytics' AGPL reaches the weights trained with it, so a stray import
# would encumber the published model (#22), not merely the source.
FORBIDDEN = ("ultralytics", "yolov5", "yolov8")


def declared_dependencies() -> set[str]:
    """Every distribution named in pyproject, required or optional."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    specifiers = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specifiers.extend(extra)

    names = set()
    for specifier in specifiers:
        # "timm>=1.0" -> "timm"; also handles extras and environment markers.
        name = re.split(r"[<>=!~\[;\s]", specifier, maxsplit=1)[0]
        if name:
            names.add(name.strip().lower())
    return names


def declared_node_dependencies() -> set[str]:
    """Everything the browser bundle pulls in.

    These ship to the user exactly as the Python ones ship to the trainer, so
    they belong under the same discipline. Dev-only tooling is excluded: it
    never reaches a user.
    """
    if not PACKAGE_JSON.exists():
        return set()
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    return {name.lower() for name in data.get("dependencies", {})}


def registered() -> str:
    return REGISTER.read_text(encoding="utf-8").lower()


def test_every_dependency_has_a_licence_recorded():
    """The moment to record a licence is when the dependency is added."""
    missing = sorted(d for d in declared_dependencies() if f"`{d}`" not in registered())
    assert not missing, (
        f"{missing} appear in pyproject.toml but not in docs/licence-register.md. "
        "Record the licence before merging — checking afterwards is how #13 happened."
    )


def test_every_browser_dependency_has_a_licence_recorded():
    """The app ships to users, so its dependencies are not a lesser category."""
    missing = sorted(d for d in declared_node_dependencies() if f"`{d}`" not in registered())
    assert not missing, (
        f"{missing} appear in web/package.json but not in docs/licence-register.md."
    )


def test_the_ported_logic_has_no_dependencies_of_its_own():
    """`web/lib/` is the decision logic. It should not acquire a supply chain:
    everything it needs is arithmetic, and a dependency there would ship a
    third party's code into the scoring path."""
    for module in (ROOT / "web" / "lib").glob("*.js"):
        for line in module.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*import\s+.*from\s+['\"]([^'\"]+)['\"]", line)
            if match:
                assert match.group(1).startswith("."), (
                    f"{module.name} imports {match.group(1)!r}; web/lib must stay dependency-free"
                )


def test_the_register_is_not_empty_of_the_things_we_actually_use():
    """Guards the guard: a register that lost its table would pass vacuously."""
    text = registered()
    for expected in ("numpy", "torch", "timm", "pillow"):
        assert f"`{expected}`" in text


@pytest.mark.parametrize("name", FORBIDDEN)
def test_agpl_encumbered_packages_are_not_dependencies(name):
    """#16: Ultralytics' AGPL covers the models produced by it. A dependency
    added for convenience would encumber the published weights (#22)."""
    assert name not in declared_dependencies()


def test_no_source_file_imports_a_forbidden_package():
    """pyproject is not the only way a package gets used."""
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for name in FORBIDDEN:
            if re.search(rf"^\s*(import|from)\s+{name}\b", text, re.MULTILINE):
                offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert not offenders, offenders


def test_the_excluded_list_still_names_the_things_that_matter():
    """These exclusions were each expensive to establish (#13, #16, #24). A
    register that quietly dropped one would lose the reason with it."""
    text = registered()
    for expected in ("ultralytics", "deepdarts", "dart-sense", "agpl", "cc by-nc"):
        assert expected in text, expected


def test_weights_are_recorded_as_trained_from_scratch():
    """Pretrained weights carry terms separate from the code that trained them
    (#16 §4). While the default is scratch, the register must say so."""
    from dartvision.train.config import TrainConfig

    assert TrainConfig(manifest="m", image_root="i").pretrained is False
    assert "from scratch" in registered()
