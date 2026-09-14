"""The compiler must contain NO workload-specific code path.

Delete every spec file and the compiler still compiles. This test is the
enforcement; the rule is not kept by intent (spec section 9).

I4 (final review): `SRC = Path("src/tsu_compiler")` was CWD-relative. Run from anywhere
other than the repo root (e.g. `pytest` invoked from `tests/`), `rglob` silently
yields nothing, every `hits`/`offenders` list stays empty, and all three tests
in this file pass having read ZERO files -- the sole mechanical enforcement of
the example-independence and backend-isolation rules, passing vacuously.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "tsu_compiler"
FORBIDDEN = ("wfc", "minecraft", "maxcut", "codon", "tile", "voxel",
             "placement_kernel", "exclusion")

# Matches `import thrml`, `from thrml import ...`, `from thrml.x import ...`
# regardless of internal whitespace width, plus `importlib.import_module("thrml")`
# / `importlib.import_module('thrml.x')` -- a dynamic import that a plain
# `startswith(("import thrml", ...))` check cannot see at all. Same for torx.
_VENDOR_IMPORT = re.compile(
    r"^\s*(?:import\s+(thrml|torx)\b|from\s+(thrml|torx)\b)"
    r"|importlib\s*\.\s*import_module\(\s*['\"](thrml|torx)(?:\.[\w.]+)?['\"]")


def _py_files():
    files = [p for p in SRC.rglob("*.py") if "__pycache__" not in str(p)]
    # A scan that silently reads zero files is indistinguishable from a scan
    # that read everything and found nothing -- the exact way this file's own
    # SRC-path bug passed vacuously. Never let that happen again unnoticed.
    assert files, (
        f"no .py files found under {SRC}; this scan would pass having checked "
        f"nothing, which is exactly the bug this assertion exists to catch")
    return files


def test_no_workload_identifier_appears_in_the_package():
    hits = []
    for p in _py_files():
        text = p.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN:
            if re.search(rf"\b{word}\b", text):
                hits.append(f"{p}: {word}")
    assert not hits, (
        "workload identifiers leaked into the compiler; workloads are spec "
        f"files, not features: {hits}")


def test_only_backends_import_thrml_or_torx():
    offenders = []
    for p in _py_files():
        if p.parent.name == "backends":
            continue
        text = p.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _VENDOR_IMPORT.search(line):
                offenders.append(f"{p}:{lineno}: {line.strip()}")
    assert not offenders, (
        "the two stacks are disjoint and must stay isolated in backends/: "
        f"{offenders}")


def test_every_module_in_the_package_imports_without_any_spec_file_present():
    """Renamed from `test_deleting_every_spec_leaves_the_package_importable`,
    which did not delete any spec file and only imported a hand-picked subset of
    modules -- it tested neither the thing its name claimed nor the full
    package. This discovers and imports EVERY module actually under src/tsu_compiler
    (skipping nothing), which is what "the compiler contains no workload-
    specific code path that only spec-driven execution would exercise" (spec
    section 9) actually requires checking."""
    import importlib

    root = SRC.parent    # .../src, so module names come out as "tsu...."
    for p in _py_files():
        rel = p.relative_to(root).with_suffix("")
        parts = rel.parts[:-1] if rel.parts[-1] == "__init__" else rel.parts
        importlib.import_module(".".join(parts))
