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


def _grid_edges(side):
    edges = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                edges.append((i, i + 1))
            if r + 1 < side:
                edges.append((i, i + side))
    return edges


@pytest.mark.parametrize("side", [8, 12, 16])
def test_square_grids_place_via_the_structured_embedding_not_the_annealer(side):
    """C1 (final review, most important finding): Z1's offsets include (1,0) and
    (0,1), so the identity map v(x,y) -> (x,y) realizes EVERY edge of a square
    grid -- an embedding provably exists. Before the fix, place() reported the
    12x12 and 16x16 cases as `geometry_unreachable` (a categorical claim about
    the substrate) purely because its seeded annealer did not happen to find the
    embedding within its fixed budget. All three sizes below must place cleanly,
    and must do so via the structured (deterministic) embedding, not by getting
    lucky with the annealer -- restarts=0 disables the annealer fallback entirely,
    so a pass here is only possible through `_try_grid_embed`."""
    edges = _grid_edges(side)
    im = ising(side * side, edges)
    p = place(im, analyse(im), Z1, restarts=0, iters=0)
    assert p.unrealized == ()
    assert len(p.realized) == len(edges)
    assert len(p.coords) == side * side


def test_effort_exhaustion_is_never_reported_as_geometry_unreachable():
    """C1: a graph that IS embeddable in principle (bipartite, degree within Z1's
    cap, not a subgraph of the grid so the structured embed does not apply) but
    that needs more search than a deliberately tiny budget provides must raise
    `placement_effort_exhausted`, never `geometry_unreachable` -- this pass
    cannot cheaply prove no legal offset spans a required edge, so it must never
    claim that just because a small budget ran out. The same graph placing
    cleanly under the production budget proves the tiny-budget failure was a
    search limitation, not a fact about Z1."""
    import networkx as nx

    G = nx.bipartite.random_graph(30, 30, 0.15, seed=7)
    G = nx.convert_node_labels_to_integers(G)
    assert nx.is_bipartite(G) and G.number_of_edges() > 0
    assert max(d for _, d in G.degree()) <= Z1.degree.value
    im = ising(G.number_of_nodes(), list(G.edges()))
    report = analyse(im)

    with pytest.raises(CompileError) as e:
        place(im, report, Z1, restarts=1, iters=50)
    f = e.value.failures[0]
    assert f.failure_class == "placement_effort_exhausted"
    assert any(r.action == "increase placement effort" for r in f.remediations)

    p = place(im, report, Z1)   # production budget: the same graph places cleanly
    assert p.unrealized == ()
