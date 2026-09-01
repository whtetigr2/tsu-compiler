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


def test_odd_cycle_is_mediated_and_places_cleanly():
    """Task 6: a triangle cannot sit on a bipartite lattice DIRECTLY, but
    `place` now attempts hidden-spin mediation before ever reporting
    `parity_conflict` (spec 5.3.6) -- subdividing the triangle's own
    within-side edge through one mediator spin makes it bipartite, and that
    mediated (4-node) graph places on Z1 with nothing left unrealized."""
    im = ising(3, [(0, 1), (1, 2), (0, 2)])
    p = place(im, analyse(im), Z1)
    assert p.unrealized == ()
    assert p.mediation is not None
    assert p.mediation.mediator_count == 1
    assert p.mediation.bipartite_after is True
    assert p.mediated_ising is not None
    assert len(p.mediated_ising.nodes) == 4          # 3 original + 1 mediator
    assert len(p.coords) == 4


def test_parity_conflict_is_still_raised_when_mediation_itself_cannot_fix_it():
    """Defensive path: `insert_mediators`' own construction proves every
    within-side edge becomes cross-side once subdivided, so a mediated graph
    failing to be bipartite should be unreachable in practice -- but `place`
    checks this rather than assuming it (spec 5.3.6), and that check must
    still classify the failure as `parity_conflict` with a routable
    remediation, exactly as before Task 6, if it is ever hit."""
    import tsu.passes.place as place_mod
    from tsu.passes.route import MediationReport

    from tsu.passes.route import insert_mediators as real_insert_mediators

    def _broken_insert_mediators(ising, report):
        med, rep = real_insert_mediators(ising, report)
        return med, MediationReport(rep.mediator_count, rep.partition_method,
                                    False, rep.beta_used)

    im = ising(3, [(0, 1), (1, 2), (0, 2)])
    orig = place_mod.insert_mediators
    place_mod.insert_mediators = _broken_insert_mediators
    try:
        with pytest.raises(CompileError) as e:
            place(im, analyse(im), Z1)
    finally:
        place_mod.insert_mediators = orig
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


def test_a_large_odd_ring_beyond_exact_maxcut_is_mediated_and_places_cleanly():
    """Task 6: a 25-node odd ring exceeds MAXCUT_EXACT_LIMIT (20), so
    `report.mediators` (the theoretical floor) is never computed -- but
    `insert_mediators` needs no max-cut computation at all (it is O(V+E)
    regardless), so `place` mediates and places this graph exactly as it
    would a small one. Before Task 6 this raised `parity_conflict`
    immediately; see
    `test_parity_conflict_is_still_raised_when_mediation_itself_cannot_fix_it`
    for the (now defensive-only) I5 sentinel-safety check that test used to
    exercise via this same graph."""
    from tsu.passes.analyse import MAXCUT_EXACT_LIMIT

    n = MAXCUT_EXACT_LIMIT + 5
    edges = [(i, (i + 1) % n) for i in range(n)]
    im = ising(n, edges)
    report = analyse(im)
    assert report.mediators == -1

    p = place(im, report, Z1)
    assert p.unrealized == ()
    assert p.mediation is not None
    assert p.mediation.bipartite_after is True


def test_parity_conflict_remediation_never_publishes_the_uncomputed_sentinel():
    """I5: report.mediators == -1 means 'not computed' (the graph exceeded
    MAXCUT_EXACT_LIMIT), not zero. The parity_conflict remediation must not
    publish it as a numeric extra_spins estimate -- that is the same
    sentinel-as-a-published-fact error C1 fixes for placement's own verdict,
    one field over. Task 6 makes this remediation defensive-only (mediation
    itself resolves parity in every practical case -- see
    `test_parity_conflict_is_still_raised_when_mediation_itself_cannot_fix_it`),
    so the failing-mediation path is forced here the same way that test
    forces it, on the same 25-node odd ring the original review found this
    on."""
    import tsu.passes.place as place_mod
    from tsu.passes.analyse import MAXCUT_EXACT_LIMIT
    from tsu.passes.route import MediationReport

    from tsu.passes.route import insert_mediators as real_insert_mediators

    def _broken_insert_mediators(ising, report):
        med, rep = real_insert_mediators(ising, report)
        return med, MediationReport(rep.mediator_count, rep.partition_method,
                                    False, rep.beta_used)

    n = MAXCUT_EXACT_LIMIT + 5
    edges = [(i, (i + 1) % n) for i in range(n)]
    im = ising(n, edges)
    report = analyse(im)
    assert report.mediators == -1

    orig = place_mod.insert_mediators
    place_mod.insert_mediators = _broken_insert_mediators
    try:
        with pytest.raises(CompileError) as e:
            place(im, report, Z1)
    finally:
        place_mod.insert_mediators = orig
    f = e.value.failures[0]
    assert f.failure_class == "parity_conflict"
    rem = next(r for r in f.remediations if r.action == "route through mediator")
    assert "extra_spins" not in rem.estimated_cost, \
        "the mediator count was never computed; -1 must not appear as a spin count"
    assert rem.estimated_cost, "the remediation must still say WHY the count is missing"
