import numpy as np
import pytest

from tsu.target import Z1, IDEAL
from tsu.passes.lower import IsingModel
from tsu.passes.analyse import analyse
from tsu.passes.place import place
from tsu.failures import CompileError


def ising(n, edges, w=None):
    return IsingModel(tuple(f"x{i}" for i in range(n)), tuple(edges),
                      np.array(w if w is not None else [1.0] * len(edges), dtype=float),
                      np.zeros(n), 1.0, 0.0)


def test_a_short_path_places_on_z1():
    im = ising(3, [(0, 1), (1, 2)])
    p = place(im, analyse(im), Z1)
    assert len(p.coords) == 3
    assert p.unrealized == ()


def test_placement_on_ideal_is_trivially_satisfied():
    edges = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    im = ising(20, edges)
    p = place(im, analyse(im), IDEAL)
    assert p.unrealized == ()


def test_degree_failure_is_classified_as_degree_exceeded():
    edges = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    im = ising(20, edges)
    with pytest.raises(CompileError) as e:
        place(im, analyse(im), Z1)
    f = e.value.failures[0]
    assert f.failure_class == "degree_exceeded"
    assert f.measured == 19
    assert f.offending, "must name the offending nodes, not just a count"


def test_odd_cycle_failure_is_classified_as_parity_conflict_and_is_routable():
    """A triangle cannot sit on a bipartite lattice. That is parity, not geometry."""
    im = ising(3, [(0, 1), (1, 2), (0, 2)])
    with pytest.raises(CompileError) as e:
        place(im, analyse(im), Z1)
    f = e.value.failures[0]
    assert f.failure_class == "parity_conflict"
    assert any(r.action == "route through mediator" for r in f.remediations)


def test_failure_carries_the_provenance_of_the_limit_it_violated():
    edges = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    im = ising(20, edges)
    with pytest.raises(CompileError) as e:
        place(im, analyse(im), Z1)
    assert e.value.failures[0].assumed is False   # degree is F-14
