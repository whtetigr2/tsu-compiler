import pytest
from tsu.spec import load_spec
from tsu.target import Z1
from tsu.failures import CompileError, PlacementFailure
from tsu.passes import search as search_mod
from tsu.passes.route import insert_mediators
from tsu.passes.search import compile_spec
from tsu.states import CandidateState


def test_toy_compiles_on_z1():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verdict == "COMPILED"
    assert c.repset.selected.state == CandidateState.SELECTED


def test_broken_spec_fails_ideal_and_reports_LOGICAL_not_hardware():
    """The whole point of the ideal control."""
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    assert c.hardware_evaluated is False, \
        "z1 must not be evaluated when the ideal control fails"


def test_too_dense_passes_ideal_then_fails_z1_as_a_hardware_constraint():
    c = compile_spec(load_spec("specs/too_dense.yaml"), Z1)
    assert c.verdict == "HARDWARE"
    assert c.ideal_passed is True
    assert c.repset.selected is None


def test_ideal_control_always_runs_and_is_recorded():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.ideal_passed is True
    assert c.ideal_report is not None


def test_too_strong_fails_on_an_assumed_gate_and_compiles_with_override():
    s = load_spec("specs/too_strong.yaml")
    assert compile_spec(s, Z1).verdict == "HARDWARE"
    assert compile_spec(s, Z1, allow_assumed=True).verdict == "COMPILED"


def test_mediated_candidate_carries_the_post_mediation_report_on_placement_failure(
        monkeypatch):
    """Review finding C2. `_try` already swaps `ising`/`report` to the
    post-mediation model on the placement-SUCCESS path (the moment
    `place()` returns a `Placement` with `mediation is not None`), but
    before this fix the placement-FAILURE path (`except CompileError`)
    left `report` at its PRE-mediation value, because `place()` raises
    before ever returning the `Placement` that carries `mediated_ising`/
    `mediated_report`. This is exactly what happened in production: L0
    (both encodings) and L1 one_hot all mediate successfully and THEN hit
    `placement_effort_exhausted`, and the committed receipts showed
    `bipartite: False` for a graph the mediation pass itself had already
    proven bipartite -- a factual misrepresentation in the audit record.

    Forces the placement-failure path CHEAPLY (no minutes-long annealer
    run) by monkeypatching `place` to raise `placement_effort_exhausted`
    carrying a REAL `MediationReport`, computed by a real `insert_mediators`
    call against this spec's real non-bipartite ising -- exactly the
    mediation `place()` itself would have performed before its geometric
    search ran out of budget, just without spending that budget here.
    """
    spec = load_spec("specs/lattice_l1_16x16.yaml")

    def fake_place(ising, report, target):
        assert not report.bipartite, \
            "fixture assumption: this spec's pre-mediation ising is non-bipartite"
        _, mediation = insert_mediators(ising, report)
        raise CompileError(
            "placement failed: placement effort exhausted",
            [PlacementFailure(
                failure_class="placement_effort_exhausted",
                offending=(), measured=1, limit=0, assumed=False,
                remediations=(), mediation=mediation)])

    monkeypatch.setattr(search_mod, "place", fake_place)

    cand, art = search_mod._try(spec, Z1, "one_hot", False,
                                coefficient_scale=0.25)

    assert cand.state == CandidateState.HARDWARE_INFEASIBLE
    assert cand.failure.mediation is not None
    assert cand.report.bipartite is True, \
        ("the returned report must describe the POST-mediation graph "
         "place() actually embedded against, not the pre-mediation one "
         "it started from")
    assert cand.report.n_nodes == 3778   # 1280 logical + 2498 mediators
    assert art["report"] is cand.report
