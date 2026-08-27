"""The compiler must contain NO workload-specific code path.

Delete every spec file and the compiler still compiles. This test is the
enforcement; the rule is not kept by intent (spec section 9).
"""
import re
from pathlib import Path

SRC = Path("src/tsu")
FORBIDDEN = ("wfc", "minecraft", "maxcut", "codon", "tile", "voxel",
             "placement_kernel", "exclusion")


def _py_files():
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in str(p)]


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
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("import thrml", "from thrml", "import torx", "from torx")):
                offenders.append(f"{p}: {s}")
    assert not offenders, (
        "the two stacks are disjoint and must stay isolated in backends/: "
        f"{offenders}")


def test_deleting_every_spec_leaves_the_package_importable():
    import importlib
    for mod in ("tsu.ir", "tsu.spec", "tsu.target", "tsu.gates",
                "tsu.passes.lower", "tsu.passes.encode", "tsu.passes.analyse",
                "tsu.passes.place", "tsu.passes.program", "tsu.passes.search",
                "tsu.receipt", "tsu.viz", "tsu.cli"):
        importlib.import_module(mod)
