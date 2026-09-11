"""Inline `web/lib/*.js` into one self-contained HTML page.

The Next app is the real front end. This exists so the same *logic* can be
opened from a file, shared as a link, or handed to a phone with no toolchain --
and the point of generating it is that the logic is not copied. `web/lib/` is
the single source, checked against the Python by `web/test/parity.test.mjs`,
and this script embeds it rather than reimplementing it.

The markup and the event wiring are genuinely a second implementation of the
UI, and that is a real cost. It is bounded: the part that decides a score, a
bust or a verdict is shared, and the part that is duplicated is the part a
screenshot would have shown anyway.

Modules cannot simply be concatenated -- `DEG`, `READY` and `DARTS_PER_VISIT`
are each declared in more than one -- so each becomes an IIFE returning its
exports, and imports become destructuring from those namespaces.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "web" / "lib"
TEMPLATE = ROOT / "tools" / "standalone.template.html"
OUT = ROOT / "web" / "public" / "standalone.html"

# Dependency order. Short enough to state; a cycle here would be a design bug,
# not a build problem.
MODULES = ["board", "homography", "framing", "match", "x01"]
NAMESPACE = {name: f"__{name}" for name in MODULES}

_EXPORT_DECL = re.compile(
    r"^export\s+(?:async\s+)?(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)", re.M)
_EXPORT_LIST = re.compile(r"^export\s*\{([^}]*)\}", re.M)
_IMPORT = re.compile(
    r"^import\s*\{([^}]*)\}\s*from\s*['\"]\./(\w+)\.js['\"];?\s*$", re.M)


def exported_names(source: str) -> list[str]:
    names = list(dict.fromkeys(_EXPORT_DECL.findall(source)))
    for group in _EXPORT_LIST.findall(source):
        for entry in group.split(","):
            entry = entry.strip()
            if entry:
                names.append(entry.split(" as ")[-1].strip())
    return list(dict.fromkeys(names))


def to_iife(name: str, source: str) -> str:
    """One module, as an expression returning its exports."""
    names = exported_names(source)
    if not names:
        raise SystemExit(f"{name}.js exports nothing; the inliner would drop it")

    def rewrite_import(match: re.Match[str]) -> str:
        wanted = ", ".join(part.strip() for part in match.group(1).split(",") if part.strip())
        other = match.group(2)
        if other not in NAMESPACE:
            raise SystemExit(f"{name}.js imports unknown module {other!r}")
        return f"const {{ {wanted} }} = {NAMESPACE[other]};"

    body = _IMPORT.sub(rewrite_import, source)
    body = re.sub(r"^export\s+", "", body, flags=re.M)
    returned = ", ".join(names)
    return (
        f"/* ---- lib/{name}.js ---- */\n"
        f"const {NAMESPACE[name]} = (() => {{\n{body}\nreturn {{ {returned} }};\n}})();"
    )


# The Artifact host supplies the document skeleton -- doctype, charset,
# viewport, and the favicon as a parameter -- so an artifact build drops ours
# rather than emitting a second set for the browser to reconcile.
_HOST_PROVIDED = re.compile(
    r"^\s*(<!doctype html>|<meta charset[^>]*>|<meta name=\"viewport\"[^>]*>|<link rel=\"icon\"[^>]*>)\s*$",
    re.M | re.I)


def main(argv: list[str] | None = None) -> int:
    argv = [a for a in (sys.argv[1:] if argv is None else argv)]
    for_artifact = "--artifact" in argv
    argv = [a for a in argv if a != "--artifact"]
    out = Path(argv[0]) if argv else OUT

    missing = [m for m in MODULES if not (LIB / f"{m}.js").exists()]
    if missing:
        raise SystemExit(f"missing modules: {missing}")

    bundle = "\n\n".join(
        to_iife(name, (LIB / f"{name}.js").read_text(encoding="utf-8")) for name in MODULES
    )
    html = TEMPLATE.read_text(encoding="utf-8")
    if "/*__LIB__*/" not in html:
        raise SystemExit("the template has no /*__LIB__*/ marker")
    page = html.replace("/*__LIB__*/", bundle)
    if for_artifact:
        page = _HOST_PROVIDED.sub("", page).lstrip("\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")

    where = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"wrote {where} — {len(MODULES)} modules, "
          f"{len(out.read_text(encoding='utf-8')) // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
