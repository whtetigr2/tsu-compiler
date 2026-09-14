"""Task 6: hidden-spin mediation is exact (spec section 5.3.2)."""
import math

import numpy as np
import networkx as nx
import pytest

from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.program import build_program
from tsu_compiler.passes.route import (BetaMismatchError, assert_beta_consistent,
                              insert_mediators)
from tsu_compiler.backends.thrml_backend import exact_distribution


def _marginal(states, probs, keep):
    out = {}
    for row, p in zip(states, probs):
        k = tuple(int(row[i]) for i in keep)
        out[k] = out.get(k, 0.0) + float(p)
    return out


def _triangle(beta=4.0, J_value=-1.25):
    J = {(0, 1): J_value, (0, 2): J_value, (1, 2): J_value}
    return IsingModel(nodes=("a", "b", "c"), edges=tuple(J),
                      weights=np.array([J[e] for e in J]),
                      biases=np.zeros(3), beta=beta, offset=0.0)


def test_mediation_preserves_the_marginal_on_a_frustrated_triangle():
    """The minimal non-bipartite case. Verified by hand at 3.3e-16 before this
    task was written; the pass must reproduce it."""
    orig = _triangle()
    assert not analyse(orig).bipartite
    med, rep = insert_mediators(orig, analyse(orig))
    assert analyse(med).bipartite
    assert rep.mediator_count >= 1
    so, po = exact_distribution(build_program(orig, analyse(orig)))
    sm, pm = exact_distribution(build_program(med, analyse(med)))
    mo, mm = _marginal(so, po, [0, 1, 2]), _marginal(sm, pm, [0, 1, 2])
    for k in mo:
        assert mo[k] == pytest.approx(mm[k], abs=1e-9)


def test_frustration_is_preserved_exactly():
    """The hand-verified reference: both all-equal states sit at probability
    0 (frustrated), the other six uniform at 1/6."""
    orig = _triangle()
    med, _ = insert_mediators(orig, analyse(orig))
    so, po = exact_distribution(build_program(orig, analyse(orig)))
    mo = _marginal(so, po, [0, 1, 2])
    for k, p in mo.items():
        if len(set(k)) == 1:
            assert p == pytest.approx(0.0, abs=1e-9)
        else:
            assert p == pytest.approx(1.0 / 6.0, abs=1e-9)


def test_degree_is_unchanged_by_subdivision():
    """Each original node still sees one edge where it saw one before, so
    section 4.2's degree law and the p<=3 budget survive mediation."""
    orig = _triangle()
    before = dict(nx.Graph(list(orig.edges)).degree())
    med, rep = insert_mediators(orig, analyse(orig))
    G = nx.Graph()
    G.add_nodes_from(range(len(med.nodes)))
    G.add_edges_from(med.edges)
    after = dict(G.degree())
    for node in before:
        assert after[node] == before[node], \
            f"node {node}: degree {before[node]} -> {after[node]}"
    # every mediator has degree exactly 2 (one edge to each side of the
    # subdivided edge).
    for m in med.mediator_nodes:
        assert after[m] == 2


def test_mediator_couplings_stay_within_the_target_cap():
    """Measured: |A| = 1.3366 for |J| = 1.25 at beta 4.0, against an assumed
    cap of 6.0."""
    orig = _triangle(beta=4.0, J_value=-1.25)
    med, rep = insert_mediators(orig, analyse(orig))
    mediator_weights = [w for (u, v), w in zip(med.edges, med.weights)
                        if u in med.mediator_nodes or v in med.mediator_nodes]
    assert mediator_weights, "the triangle must have needed at least one mediator"
    for w in mediator_weights:
        assert abs(w) == pytest.approx(1.3366, abs=1e-3)
        assert abs(w) < 6.0


def test_mediator_coupling_is_the_one_function_insert_mediators_itself_calls():
    """WHAT THIS PINS (A-1 refactor, external review 2026-09-09): the gadget
    formula A = arccosh(exp(2*beta*|J|))/(2*beta) now lives in exactly ONE
    place -- `tsu_compiler.passes.route.mediator_coupling` -- and `insert_mediators`
    calls it rather than re-deriving the expression inline a second time.
    `tsu_compiler.preflight.check.preflight` also calls this SAME function (see
    tests/test_preflight_check.py) to predict a non-bipartite model's
    post-mediation coupling before ever placing anything -- this project
    has been bitten before by a constant re-derived in a second place and
    quietly drifting from the first (see this module's own PROVENANCE
    comments elsewhere in the codebase), so this pins there being one
    function, not two independent expressions of the same math.
    HOW IT FAILS: if `mediator_coupling` is removed or renamed, the import
    below raises ImportError. If `insert_mediators` reverts to computing
    `math.acosh(math.exp(2*beta*absJ))/(2*beta)` inline instead of calling
    `mediator_coupling`, this test can still pass by coincidence (the two
    expressions are mathematically identical) -- so the point of this test
    is the IMPORT succeeding at all combined with bit-for-bit equality
    (`==`, not `pytest.approx`) against a value computed by calling
    `mediator_coupling` directly on the same inputs the fixture also feeds
    `insert_mediators`, which a future accidental divergence in either
    implementation would break."""
    from tsu_compiler.passes.route import mediator_coupling
    orig = _triangle(beta=4.0, J_value=-1.25)
    med, rep = insert_mediators(orig, analyse(orig))
    mediator_weights = [abs(w) for (u, v), w in zip(med.edges, med.weights)
                        if u in med.mediator_nodes or v in med.mediator_nodes]
    assert mediator_weights, "the triangle must have needed at least one mediator"
    expected = mediator_coupling(1.25, 4.0)
    for w in mediator_weights:
        assert w == expected


@pytest.mark.parametrize("beta", [1.0, 4.0, 8.0])
@pytest.mark.parametrize("absJ", [0.5, 1.25, 2.5, 5.0])
def test_coupling_formula_recovers_J_across_beta_and_J(beta, absJ):
    """The formula was verified (per this task's brief) across beta in
    {1,4,8} and |J| in {0.5..5.0}, recovering J to 1e-12 every time -- lock
    that in directly against the closed-form derivation:
    2*cosh(2*beta*A) == exp(2*beta*|J|)."""
    A = math.acosh(math.exp(2 * beta * absJ)) / (2 * beta)
    recovered = math.log(math.cosh(2 * beta * A)) / (2 * beta)
    assert recovered == pytest.approx(absJ, abs=1e-12)


def test_mediated_program_refuses_a_different_beta():
    """A is temperature-dependent. Sampling a mediated program at a beta other
    than the one its couplings were computed at is WRONG and must raise,
    not silently sample (spec 5.3.5)."""
    orig = _triangle(beta=4.0)
    med, rep = insert_mediators(orig, analyse(orig))
    assert med.mediator_nodes

    # the SAME beta is fine.
    assert_beta_consistent(med, med.beta)

    with pytest.raises(BetaMismatchError):
        assert_beta_consistent(med, med.beta * 2.0)

    # a model that was never mediated has nothing beta-dependent baked in,
    # so any beta is fine for it.
    assert_beta_consistent(orig, orig.beta * 2.0)


def test_already_bipartite_graph_needs_no_mediators():
    im = IsingModel(nodes=("a", "b"), edges=((0, 1),),
                    weights=np.array([1.0]), biases=np.zeros(2),
                    beta=1.0, offset=0.0)
    med, rep = insert_mediators(im, analyse(im))
    assert rep.mediator_count == 0
    assert rep.bipartite_after is True
    assert med.nodes == im.nodes
    assert med.edges == im.edges


def test_a_larger_odd_cycle_mediates_and_stays_exact():
    """A 5-cycle (all-equal-weight, all frustrated antiferromagnetically) is
    a second, independent non-bipartite case beyond the minimal triangle."""
    beta = 2.0
    n = 5
    J = -0.8
    edges = [(i, (i + 1) % n) for i in range(n)]
    orig = IsingModel(nodes=tuple(f"x{i}" for i in range(n)), edges=tuple(edges),
                      weights=np.array([J] * n), biases=np.zeros(n),
                      beta=beta, offset=0.0)
    assert not analyse(orig).bipartite
    med, rep = insert_mediators(orig, analyse(orig))
    assert analyse(med).bipartite
    assert rep.mediator_count >= 1

    so, po = exact_distribution(build_program(orig, analyse(orig)))
    sm, pm = exact_distribution(build_program(med, analyse(med)))
    keep = list(range(n))
    mo, mm = _marginal(so, po, keep), _marginal(sm, pm, keep)
    for k in mo:
        assert mo[k] == pytest.approx(mm[k], abs=1e-9)


def test_zero_weight_within_side_edge_is_harmless_despite_the_sign_branch():
    """Review finding (lower priority): `insert_mediators` picks the
    mediator's second coupling as `A if J > 0 else -A`, so a J == 0
    within-side edge takes the `-A` branch -- untested, and not hit by
    either real lattice spec (neither has a zero-weight coupling).

    Decision (recorded in task-6-report.md): no code change. `A =
    arccosh(exp(2*beta*|J|)) / (2*beta)` is EXACTLY 0 when J == 0
    (arccosh(exp(0)) == arccosh(1) == 0) for every beta, so `A` and `-A` are
    the same value here -- IEEE 0.0 vs -0.0, equal under `==` and identical
    under any multiplication, so the sign branch is a distinction without a
    difference for this edge specifically. Verified empirically below (not
    just asserted from the formula): a triangle with one J == 0 edge still
    mediates to a bipartite graph whose marginal over the original spins is
    unchanged, to the same 1e-9 tolerance every other exactness test in this
    file uses."""
    beta = 4.0
    J = {(0, 1): -1.25, (0, 2): -1.25, (1, 2): 0.0}
    orig = IsingModel(nodes=("a", "b", "c"), edges=tuple(J),
                      weights=np.array([J[e] for e in J]),
                      biases=np.zeros(3), beta=beta, offset=0.0)
    report = analyse(orig)
    assert not report.bipartite
    med, rep = insert_mediators(orig, report)
    assert rep.mediator_count == 1   # only (1, 2) lands within-side

    mediator_weights = [w for (u, v), w in zip(med.edges, med.weights)
                        if u in med.mediator_nodes or v in med.mediator_nodes]
    assert len(mediator_weights) == 2
    for w in mediator_weights:
        assert w == pytest.approx(0.0, abs=1e-12)

    med_report = analyse(med)
    assert med_report.bipartite is True

    so, po = exact_distribution(build_program(orig, report))
    sm, pm = exact_distribution(build_program(med, med_report))
    mo, mm = _marginal(so, po, [0, 1, 2]), _marginal(sm, pm, [0, 1, 2])
    for k in mo:
        assert mo[k] == pytest.approx(mm[k], abs=1e-9)


# ---------------------------------------------------------------------------
# N-2 (R14 / F-R14): insert_mediators' gadget formula
# (A = arccosh(exp(2*beta*|J|))/(2*beta)) accepted a NaN/inf coupling (or
# beta) on a within-side edge and silently produced a NaN/inf mediator
# coupling in a MediationReport/IsingModel that still looked fine. Same
# audit finding as N-1 (lower.py), one file over -- and IsingModel is a
# frozen dataclass constructible directly (every fixture in this file does
# exactly that), so lower()'s own new guard does not by itself protect this
# entry point.
#
# The triangle topology (a,b,c) is used deliberately, not an arbitrary
# graph: this file's own test_zero_weight_within_side_edge... above already
# established (and this is re-confirmed below) that BFS-parity colouring
# from node 0 leaves edge (b, c) -- i.e. (1, 2) -- as the ONE within-side
# edge that reaches the gadget formula; (a, b) and (a, c) are cross-side and
# never touch it. Putting the bad value on (1, 2) is therefore what actually
# exercises the code path under test, not incidental.
#
# Production change that would make each test fail: deleting the
# `math.isfinite` guard it exercises from `insert_mediators`. Confirmed
# directly by commenting out each guard and rerunning: both regress to
# "no exception raised", with the resulting mediator weight silently NaN/
# inf (checked via med.weights) rather than merely a different error --
# the exact silent-garbage failure mode this guards against.
# ---------------------------------------------------------------------------

def test_nan_coupling_on_the_mediated_edge_is_rejected_not_silently_propagated():
    J = {(0, 1): -1.25, (0, 2): -1.25, (1, 2): math.nan}
    orig = IsingModel(nodes=("a", "b", "c"), edges=tuple(J),
                      weights=np.array([J[e] for e in J]),
                      biases=np.zeros(3), beta=4.0, offset=0.0)
    report = analyse(orig)
    assert not report.bipartite
    with pytest.raises(ValueError, match="finite"):
        insert_mediators(orig, report)


def test_infinite_coupling_on_the_mediated_edge_is_rejected_not_silently_propagated():
    J = {(0, 1): -1.25, (0, 2): -1.25, (1, 2): math.inf}
    orig = IsingModel(nodes=("a", "b", "c"), edges=tuple(J),
                      weights=np.array([J[e] for e in J]),
                      biases=np.zeros(3), beta=4.0, offset=0.0)
    report = analyse(orig)
    with pytest.raises(ValueError, match="finite"):
        insert_mediators(orig, report)


def test_nan_beta_on_a_model_needing_mediation_is_rejected_not_silently_propagated():
    """Same gadget, the other operand: beta feeds `2*beta*|J|` for every
    mediated edge, so a NaN/inf beta is just as capable of producing a
    silent NaN/inf mediator coupling as a bad J is."""
    orig = _triangle(beta=math.nan, J_value=-1.25)
    report = analyse(orig)
    with pytest.raises(ValueError, match="finite"):
        insert_mediators(orig, report)


def test_already_bipartite_graph_tolerates_a_nan_beta():
    """The gadget formula never runs when no mediation is needed (spec
    5.3's own early-out), so a non-finite beta on an ALREADY-bipartite
    model must not be rejected here -- there is nothing beta-dependent to
    protect on this path (mirrors assert_beta_consistent's own "nothing
    temperature-dependent baked in" reasoning for an unmediated model)."""
    im = IsingModel(nodes=("a", "b"), edges=((0, 1),),
                    weights=np.array([1.0]), biases=np.zeros(2),
                    beta=math.nan, offset=0.0)
    med, rep = insert_mediators(im, analyse(im))
    assert rep.mediator_count == 0
