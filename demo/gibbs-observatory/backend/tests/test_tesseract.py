"""A four-dimensional cube, as an ordinary Ising model.

Q4 is 16 vertices with an edge wherever two binary indices differ in exactly one
bit. That makes it 4-regular and bipartite, so it compiles with zero mediators
and an exact two-colour schedule.

It ships for two reasons. Nobody hardcoded this shape anywhere in the viewer, so
it is a real test of whether an arbitrary topology can be loaded and drawn. And
it is honest about what "four-dimensional" means here: the fourth dimension is
in the connectivity, not in the hardware and not in the physics. The chip is
flat and the spins are ordinary binary spins.

The structural tests below check the hypercube property directly rather than
trusting the script that generated the file.
"""
import itertools
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

SPEC = ROOT / "programs" / "tesseract_16.yaml"


def _edges() -> list[tuple[int, int]]:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    out = []
    for term in spec["terms"]:
        if term.get("kind") != "product":
            continue
        a = next(iter(term["a"]))
        b = next(iter(term["b"]))
        out.append((int(a[1:]), int(b[1:])))
    return out


def test_the_file_exists():
    assert SPEC.is_file(), f"{SPEC} is missing"


def test_it_has_exactly_the_edges_a_4_cube_has():
    edges = _edges()
    assert len(edges) == 32, f"Q4 has 32 edges, found {len(edges)}"
    expected = {
        tuple(sorted(pair)) for pair in itertools.combinations(range(16), 2)
        if bin(pair[0] ^ pair[1]).count("1") == 1
    }
    assert {tuple(sorted(e)) for e in edges} == expected


def test_every_edge_joins_vertices_differing_in_one_coordinate():
    """The definition of a hypercube, checked rather than asserted in prose."""
    for a, b in _edges():
        assert bin(a ^ b).count("1") == 1, (
            f"v{a}-v{b} differ in more than one coordinate, so this is not a "
            f"hypercube")


def test_it_is_four_regular():
    degree: dict[int, int] = {}
    for a, b in _edges():
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    assert set(degree) == set(range(16)), "some vertex has no edges"
    assert set(degree.values()) == {4}, (
        f"every vertex of a 4-cube has four neighbours, got {sorted(set(degree.values()))}")


def test_it_compiles_with_no_mediators():
    """Bipartite, so the two-colour schedule is exact and costs nothing."""
    from tsu_compiler.preflight.check import preflight
    from tsu_compiler.preflight.model import load_model

    rep = preflight(load_model(spec=str(SPEC)))
    assert rep.verdict == "ok", rep.verdict
    assert rep.placed is True
    assert rep.n_spins == 16
    assert rep.max_degree == 4
    assert rep.mediators == 0, (
        "a hypercube is bipartite, so it should need no helper spins")
    assert not [g.name for g in rep.gates if g.status == "fail"]


def test_it_is_bipartite_by_parity_of_the_index():
    """Why it needs no mediators: colour by popcount parity and no edge is
    monochromatic. This is what makes every hypercube bipartite."""
    for a, b in _edges():
        assert bin(a).count("1") % 2 != bin(b).count("1") % 2, (
            f"v{a} and v{b} have the same index parity, so the parity "
            f"2-colouring is not valid and the graph is not bipartite")
