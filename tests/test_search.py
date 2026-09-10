import numpy as np
import pytest
from tsu.spec import load_spec
from tsu.target import Z1
from tsu.failures import CompileError, GateFailure, PlacementFailure
from tsu.passes import search as search_mod
from tsu.passes.analyse import analyse
from tsu.passes.lower import IsingModel
from tsu.passes.place import place as real_place
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


def test_too_strong_bias_fails_on_an_assumed_gate_and_compiles_with_override():
    """P-3/F-A5 + I-9a/F-R7: was test_too_strong_fails_on_an_assumed_gate_
    and_compiles_with_override, using specs/too_strong.yaml -- but that
    spec's coupling violation is no longer an "assumed gate" scenario at
    all (max_abs_coupling is now Extropic-documented; see the test below).
    specs/too_strong_bias.yaml is the replacement fixture: a bias, not a
    coupling, past the cap, so it still exercises the genuinely-assumed
    max_abs_bias field."""
    s = load_spec("specs/too_strong_bias.yaml")
    assert compile_spec(s, Z1).verdict == "HARDWARE"
    assert compile_spec(s, Z1, allow_assumed=True).verdict == "COMPILED"


def test_too_strong_coupling_violation_is_not_overridable_now_that_it_is_sourced():
    """P-3/F-A5 + I-9a/F-R7: |J| <= 6.0 is now Extropic-documented (not
    assumed) -- specs/too_strong.yaml's coupling violation must stay
    HARDWARE even with --allow-assumed, the opposite of the bias violation
    above and of this spec's own pre-fix behaviour."""
    s = load_spec("specs/too_strong.yaml")
    assert compile_spec(s, Z1).verdict == "HARDWARE"
    assert compile_spec(s, Z1, allow_assumed=True).verdict == "HARDWARE"


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


# ---------------------------------------------------------------------------
# A-1 (external review, 2026-09-09): `_try` gated the PRE-mediation model
# (search.py's own `gate_checks`/`check_gates` call above) then swapped to the
# post-mediation `ising`/`report` the moment `place()` succeeded with
# `mediation is not None` -- but nothing re-ran the gates against THAT model.
# Mediation's own gadget coupling A = arccosh(exp(2*beta*|J|))/(2*beta) is
# STRICTLY GREATER than |J| for every J != 0 (route.py's own derivation), so
# a model that legitimately clears `coupling_cap` pre-mediation (|J| <= 6.0)
# can still need a mediator coupling the hardware cannot represent: solving
# A(|J|) <= 6.0 at beta=1 gives |J| <= ln(cosh(12))/2 = 5.653426..., so any
# mediated edge with 5.653426 < |J| <= 6.0 passed the cap gate and then
# needed an unprogrammable coupling. Before this fix, `_try` proceeded straight
# to `route`/`build_program`/`regime` and reported HARDWARE_FEASIBLE for
# exactly this case -- a "compiles clean" verdict for a model carrying a
# coupling beyond the sourced (non-overridable) Extropic hardware cap.
# ---------------------------------------------------------------------------

def _over_cap_mediated_placement():
    """The reviewer's own reproduction, run against the REAL `place()` (no
    mock at this layer): a frustrated triangle at |J|=6.0, beta=1.0 clears
    `check_gates` pre-mediation (0 failures -- 6.0 is exactly at, not over,
    the cap) but real hidden-spin mediation needs A(6.0, beta=1) =
    6.346573590275254 > 6.0 for its one within-side edge. Returns the real,
    already-computed `Placement` so the test below only exercises whether
    `_try` re-gates it -- never a fabricated Placement standing in for one
    `place()` would not actually produce."""
    triangle = IsingModel(nodes=("a", "b", "c"), edges=((0, 1), (1, 2), (0, 2)),
                          weights=np.full(3, 6.0), biases=np.zeros(3),
                          beta=1.0, offset=0.0)
    triangle_report = analyse(triangle)
    assert not triangle_report.bipartite, "fixture assumption: a triangle is an odd cycle"
    from tsu.gates import check_gates as _check_gates
    assert _check_gates(triangle, triangle_report, Z1, False) == (), \
        "fixture assumption: |J|=6.0 must clear coupling_cap PRE-mediation"
    placement = real_place(triangle, triangle_report, Z1)
    assert placement.mediation is not None
    over_cap = float(np.abs(placement.mediated_ising.weights).max())
    assert over_cap == pytest.approx(6.346573590275254)
    assert over_cap > Z1.max_abs_coupling.value, \
        "fixture assumption: mediation must have pushed |J| past the cap"
    return placement


def test_try_re_gates_the_mediated_model_and_rejects_an_over_cap_coupling(monkeypatch):
    """WHAT THIS PINS: `_try` must re-run the hardware gates against the
    POST-mediation model the moment `place()` returns `mediation is not
    None`, and reject the candidate (HARDWARE_INFEASIBLE, carrying a real
    `coupling_cap` `GateFailure`) when that re-gate fails -- through the
    SAME `except CompileError` reporting path a placement failure already
    uses below in this file, not a second, parallel failure shape.

    `place` is monkeypatched to return the REAL, already-computed
    over-cap `Placement` from `_over_cap_mediated_placement()` above
    regardless of what `_try`'s own spec/encoding would have placed --
    isolating the one thing under test (does `_try` re-gate the swapped
    model) from needing a real spec whose real geometry happens to mediate
    past the cap.
    HOW IT FAILS: before this fix, `_try` never re-runs `check_gates` after
    the `ising, report = placement.mediated_ising, placement.mediated_report`
    swap, so it proceeds straight through `route`/`build_program`/`regime`
    and returns `CandidateState.HARDWARE_FEASIBLE` here -- every assertion
    below on `cand.state`/`cand.failure` fails against that unfixed
    behaviour, since `cand.failure` stays `None` on the feasible path.
    PROVENANCE: reviewer reproduction; A(6.0, beta=1)=6.346573590275254
    confirmed independently via `place()` itself before this test was
    written (see `_over_cap_mediated_placement`'s own assertions)."""
    over_cap_placement = _over_cap_mediated_placement()

    def fake_place(ising, report, target, **kwargs):
        return over_cap_placement

    monkeypatch.setattr(search_mod, "place", fake_place)

    spec = load_spec("specs/toy.yaml")
    cand, art = search_mod._try(spec, Z1, "domain_wall", False)

    assert cand.state == CandidateState.HARDWARE_INFEASIBLE, \
        "a mediated coupling beyond the hardware cap must not report feasible"
    assert cand.failure is not None
    assert isinstance(cand.failure, GateFailure)
    assert cand.failure.gate == "coupling_cap"
    assert cand.failure.measured == pytest.approx(6.346573590275254)
    assert cand.failure.limit == pytest.approx(6.0)
    assert cand.failure.assumed is False, \
        "max_abs_coupling is Extropic-documented, not a project assumption"
    # the candidate's own report must still be the POST-mediation one (C2's
    # own rule, extended here): a receipt for a REJECTED-at-re-gate
    # candidate must still describe the model that was actually evaluated.
    assert cand.report.bipartite is True
    assert cand.report.n_nodes == 4   # 3 original + 1 mediator

    post_checks = {c.gate: c for c in art["gate_checks"]}
    assert post_checks["coupling_cap"].passed is False
    assert post_checks["coupling_cap"].measured == pytest.approx(6.346573590275254)


def test_compile_spec_reports_hardware_not_compiled_when_mediation_exceeds_the_cap(
        monkeypatch):
    """WHAT THIS PINS: the SAME defect, one layer up -- `compile_spec`'s
    overall verdict must be HARDWARE (never COMPILED) for a spec whose
    every candidate mediates past the coupling cap, exactly as it already
    is for any other unrecoverable gate failure. This is the behaviour
    change the task brief warns about: a model that previously reported
    COMPILED (because nothing re-gated the mediated model) now correctly
    reports HARDWARE, because it always needed a coupling the hardware
    cannot represent -- the fix does not change the model, only whether
    this compiler tells the truth about it.
    HOW IT FAILS: before this fix, both `domain_wall` and `one_hot`
    candidates reach `place()` (mocked here to always return the same
    over-cap `Placement`), both swap to the mediated report and proceed to
    HARDWARE_FEASIBLE unchecked, and `compile_spec` reports `verdict ==
    "COMPILED"` -- the assertion below fails against that unfixed
    behaviour."""
    over_cap_placement = _over_cap_mediated_placement()

    def fake_place(ising, report, target, **kwargs):
        return over_cap_placement

    monkeypatch.setattr(search_mod, "place", fake_place)

    c = compile_spec(load_spec("specs/toy.yaml"), Z1)

    assert c.verdict == "HARDWARE"
    assert c.ideal_passed is True
    assert c.repset.selected is None
    for cand in c.repset.candidates:
        assert cand.state == CandidateState.HARDWARE_INFEASIBLE
        assert cand.failure.gate == "coupling_cap"


# ---------------------------------------------------------------------------
# C-5 (external review, 2026-09-10): `_try`'s try/except around the
# place()/route() block only catches `CompileError` -- the encode/lower steps
# just above it already catch broad `Exception` and turn a failure into a
# rejected candidate (see the `except Exception as e:` clauses under
# "encode failed"/"lower failed"), but place()/route() did not, so a genuine
# bug or any other unexpected exception out of either one propagated as a raw
# traceback all the way out of `compile_spec`, past the CLI's own clean-
# refusal standard (preflight/regime already refuse cleanly on a bad input --
# see test_cli.py). Fixed by adding a matching `except Exception` clause that
# reports a HARDWARE_INFEASIBLE candidate naming the pass ("place"/"route")
# and the exception's own type -- never swallowed, and a genuine CompileError
# still goes through the SAME `except CompileError` branch as before,
# unchanged.
# ---------------------------------------------------------------------------

def test_try_surfaces_an_unexpected_place_exception_as_a_candidate_not_a_traceback(
        monkeypatch):
    """WHAT THIS PINS: an exception from `place()` that is NOT a `CompileError`
    (a genuine bug, not a documented hardware-infeasibility cause) must come
    back from `_try` as a normal `HARDWARE_INFEASIBLE` candidate whose
    `reason` names both the pass that raised ("place") and the exception's
    own type and message -- enough to debug from the printed reason alone,
    never silently swallowed and never a raw traceback.
    HOW IT FAILS: before this fix, `_try`'s try/except around the
    place()/route() block has only an `except CompileError` clause, which
    does not match a plain `RuntimeError` -- `search_mod._try(...)` below
    raises that `RuntimeError` straight out of this test instead of
    returning, so every assertion after the call is never reached (the test
    errors out, not fails cleanly, on the unfixed code).
    PROVENANCE: reviewer's own C-5 finding; reproduced by monkeypatching
    `place` (the same technique `_over_cap_mediated_placement`'s own callers
    already use above) to raise a plain `RuntimeError` instead of returning a
    `Placement` or raising `CompileError`."""
    def fake_place(ising, report, target, **kwargs):
        raise RuntimeError("boom: unexpected placement bug")

    monkeypatch.setattr(search_mod, "place", fake_place)

    spec = load_spec("specs/toy.yaml")
    cand, art = search_mod._try(spec, Z1, "domain_wall", False)

    assert cand.state == CandidateState.HARDWARE_INFEASIBLE
    assert "RuntimeError" in cand.reason
    assert "place" in cand.reason
    assert "boom: unexpected placement bug" in cand.reason
    assert art is not None and "gate_checks" in art


def test_try_surfaces_an_unexpected_route_exception_as_a_candidate_not_a_traceback(
        monkeypatch):
    """WHAT THIS PINS: the SAME fix, for `route()` instead of `place()` --
    both are named together in the review finding ("an unexpected exception
    out of place/route propagates as a raw traceback"), and the fix must
    cover both, distinguishing which one actually raised in `reason`.
    HOW IT FAILS: same as the `place()` test above -- before this fix a
    plain `KeyError` from `route()` is not caught by the `except CompileError`
    clause and propagates straight out of `search_mod._try(...)` below,
    erroring the test out before its assertions run.
    PROVENANCE: reviewer's own C-5 finding, same reproduction technique as
    the `place()` test above, applied to `route` instead."""
    def fake_route(ising, report, target):
        raise KeyError("boom: unexpected route bug")

    monkeypatch.setattr(search_mod, "route", fake_route)

    spec = load_spec("specs/toy.yaml")
    cand, art = search_mod._try(spec, Z1, "domain_wall", False)

    assert cand.state == CandidateState.HARDWARE_INFEASIBLE
    assert "KeyError" in cand.reason
    assert "route" in cand.reason
    assert "boom: unexpected route bug" in cand.reason


def test_compile_spec_reports_hardware_not_a_traceback_on_an_unexpected_place_bug(
        monkeypatch):
    """WHAT THIS PINS: the SAME defect, one layer up (matching the existing
    style of `test_compile_spec_reports_hardware_not_compiled_when_mediation_
    exceeds_the_cap` above) -- `compile_spec` itself must return a normal
    `Compilation` with verdict HARDWARE, not raise, when every candidate's
    `place()` call hits an unexpected exception. This is what lets the CLI's
    existing `if comp.verdict != "COMPILED": ... print(cand.reason)` path
    (cli.py) report it cleanly, with no try/except needed at the CLI layer
    at all.
    HOW IT FAILS: before this fix, `compile_spec(...)` below raises
    `RuntimeError` instead of returning -- the call itself fails, before
    `c.verdict` can even be read.
    PROVENANCE: reviewer's own C-5 finding, same fixture pattern as the
    mediation-cap test above."""
    def fake_place(ising, report, target, **kwargs):
        # The mandatory ideal control (spec 5.1) also calls `place()` --
        # real `place()` short-circuits it trivially (IDEAL.offsets.value is
        # `()`, so it never reaches the geometric search a placement bug
        # would actually live in); call through to the REAL `place()` for it
        # so this test exercises "z1's place() call hits an unexpected bug",
        # not "the ideal control itself is broken" (a different, already-
        # covered scenario -- see test_broken_spec_fails_ideal_and_reports_
        # LOGICAL_not_hardware above).
        if target.name == "ideal":
            return real_place(ising, report, target, **kwargs)
        raise RuntimeError("boom: unexpected placement bug")

    monkeypatch.setattr(search_mod, "place", fake_place)

    c = compile_spec(load_spec("specs/toy.yaml"), Z1)

    assert c.verdict == "HARDWARE"
    assert c.ideal_passed is True
    assert c.repset.selected is None
    for cand in c.repset.candidates:
        assert cand.state == CandidateState.HARDWARE_INFEASIBLE
        assert "RuntimeError" in cand.reason
