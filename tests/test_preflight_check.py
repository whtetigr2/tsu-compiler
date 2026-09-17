"""Task 2: the pre-flight report. Compiler passes only -- no sampling."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.preflight.check import Gate, PreflightReport, preflight, WARN_FRACTION
from tsu_compiler.target import PROFILES


def grid(n: int, j: float = 0.4, b: float = 0.0) -> IsingModel:
    """A 4-neighbour grid: bipartite, and a direct subgraph of Z1."""
    idx = {(x, y): y * n + x for y in range(n) for x in range(n)}
    e = []
    for (x, y), i in idx.items():
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in idx:
                e.append((i, idx[(x + dx, y + dy)]))
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n * n)),
                      edges=tuple(e), weights=np.full(len(e), j),
                      biases=np.full(n * n, b), beta=1.0, offset=0.0)


def triangle() -> IsingModel:
    """An odd cycle: the smallest non-bipartite graph."""
    return IsingModel(nodes=("a", "b", "c"), edges=((0, 1), (1, 2), (0, 2)),
                      weights=np.full(3, 0.4), biases=np.zeros(3),
                      beta=1.0, offset=0.0)


def _bigbias() -> IsingModel:
    """A small bipartite grid whose peak |b| (9.0) exceeds z1's
    max_abs_bias cap (6.0, source="assumed" -- see target.py) while |J|
    stays well under its own, separately-sourced (Extropic-documented,
    NOT assumed) cap. The exact live repro from the branch review's F2:
    `preflight --edges .../bigbias.json` -> verdict FAIL on "|b| against
    the hardware's bias cap" for a value target.py's own Sourced field
    marks "assumed", not a hardware fact."""
    n = 4
    idx = {(x, y): y * n + x for y in range(n) for x in range(n)}
    e = []
    for (x, y), i in idx.items():
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in idx:
                e.append((i, idx[(x + dx, y + dy)]))
    biases = np.zeros(n * n)
    biases[0] = 9.0
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n * n)),
                      edges=tuple(e), weights=np.full(len(e), 0.5),
                      biases=biases, beta=1.0, offset=0.0)


def test_a_grid_is_reported_bipartite_and_directly_embeddable():
    r = preflight(grid(8))
    assert r.bipartite is True
    assert r.mediators == 0
    assert r.embedding == "grid_embed"
    assert r.n_spins == 64


def test_an_odd_cycle_is_reported_non_bipartite():
    """The decisive line in the whole report. A non-bipartite graph cannot be a
    direct subgraph of a chessboard lattice, whatever else is true of it."""
    r = preflight(triangle())
    assert r.bipartite is False
    assert r.embedding != "grid_embed"


def test_a_coupling_over_the_cap_fails_its_gate():
    r = preflight(grid(6, j=9.0))
    g = {x.name: x for x in r.gates}["max_abs_coupling"]
    assert g.status == "fail"
    assert g.limit == PROFILES["z1"].max_abs_coupling.value
    assert r.verdict == "fail"


def test_a_coupling_near_the_cap_warns_but_does_not_fail():
    """A model at 90% of the cap still compiles, but a user should know it has
    no headroom left for any later scaling."""
    cap = PROFILES["z1"].max_abs_coupling.value
    r = preflight(grid(6, j=cap * 0.9))
    g = {x.name: x for x in r.gates}["max_abs_coupling"]
    assert g.status == "warn"
    assert r.verdict != "fail"


def test_gate_statuses_use_the_stated_warn_fraction():
    """The boundary itself is asserted, not points near it: `_gate` uses
    `value >= limit * WARN_FRACTION`, so a value exactly at the boundary must
    warn. Sampling only either side would let a `>=` become a `>` unnoticed."""
    cap = PROFILES["z1"].max_abs_coupling.value
    at = preflight(grid(4, j=cap * WARN_FRACTION))
    assert {x.name: x for x in at.gates}["max_abs_coupling"].status == "warn"
    below = preflight(grid(4, j=cap * WARN_FRACTION * 0.99))
    assert {x.name: x for x in below.gates}["max_abs_coupling"].status == "ok"


def test_the_node_budget_gate_reports_the_share_used():
    r = preflight(grid(8))
    g = {x.name: x for x in r.gates}["node_budget"]
    assert g.value == 64
    assert g.limit == PROFILES["z1"].node_budget.value
    assert g.status == "ok"


def test_every_gate_carries_its_limit_and_a_note():
    """A gate without its limit is a number the reader cannot act on.

    UPDATED (F2, branch review): `limit > 0` loosened to `limit >= 0` --
    preflight now carries a delegated `colouring` gate (sourced from
    tsu_compiler.gates.gate_checks(), which the review's own comparison table found
    missing from preflight entirely), a structural binary check ("any
    violation at all fails") whose natural limit is 0, not some positive
    threshold. `limit == 0` is still fully actionable (a reader knows
    'zero violations tolerated'), so the test's actual intent -- every
    gate carries a limit a reader can act on, and a note -- is preserved;
    only the illustrative `> 0` bound no longer covers every gate shape
    this tool reports."""
    for g in preflight(grid(6)).gates:
        assert g.limit >= 0
        assert g.note.strip()


def test_preflight_imports_no_sampler_at_all():
    """Pre-flight must be fast enough to run on every edit, and the guarantee is
    structural rather than behavioural: this module must not import the sampler
    under ANY spelling. A monkeypatch on the backend module would miss
    `from ... import sample` at module scope, which binds the real function
    before the patch runs -- the footgun this project documents in
    audit/findings/R1.md."""
    import ast
    from pathlib import Path
    # Resolved from this file, not the CWD: a string path that only works
    # from the repo root is how a package rename silently disarms a guard.
    src = (Path(__file__).resolve().parents[1]
           / "src" / "tsu_compiler" / "preflight" / "check.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert "thrml" not in a.name, a.name
        elif isinstance(node, ast.ImportFrom):
            assert "thrml" not in (node.module or ""), node.module
            for a in node.names:
                assert a.name != "sample", "check.py must not import a sampler"


def test_a_degree_violation_is_REPORTED_not_raised():
    """place() raises CompileError on a degree violation. A tool meant to run on
    every edit must report that as a failed gate, not hand the user a traceback."""
    n = 20
    e = [(i, j) for i in range(n) for j in range(i + 1, n)]
    im = IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=tuple(e),
                    weights=np.full(len(e), 0.1), biases=np.zeros(n),
                    beta=1.0, offset=0.0)
    r = preflight(im)
    assert r.verdict == "fail"
    assert {x.name: x for x in r.gates}["max_degree"].status == "fail"
    assert r.placed is False and r.place_error


def test_a_bipartite_graph_that_is_not_a_grid_subgraph_is_not_called_grid_embed():
    """Bipartiteness is necessary and NOT sufficient for a direct embed. This
    fixture is bipartite and is not a subgraph of the lattice; reporting it as
    grid_embed would claim a deterministic millisecond placement for a run that
    used the annealer."""
    import networkx as nx
    g = nx.bipartite.random_graph(30, 30, 0.15, seed=7)
    e = tuple(sorted((min(u, v), max(u, v))) for u, v in g.edges())
    im = IsingModel(nodes=tuple(f"n{i}" for i in range(60)), edges=e,
                    weights=np.full(len(e), 0.1), biases=np.zeros(60),
                    beta=1.0, offset=0.0)
    r = preflight(im)
    assert r.bipartite is True
    assert r.embedding != "grid_embed"


def test_the_mediated_fixture_reports_its_mediator_count():
    """The triangle is the only mediated fixture and nothing asserted what it
    cost, so a mediator-accounting change would pass unnoticed."""
    r = preflight(triangle())
    assert r.mediators > 0
    assert {x.name: x for x in r.gates}["node_budget"].value == r.n_spins + r.mediators


def test_a_placement_failure_carries_the_compiler_s_own_remediations():
    """CompileError already computes what the user should DO about a failure.
    A pre-flight report that says only "degree exceeded" throws away the actionable
    half of an answer the compiler had already worked out."""
    n = 20
    e = [(i, j) for i in range(n) for j in range(i + 1, n)]
    im = IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=tuple(e),
                    weights=np.full(len(e), 0.1), biases=np.zeros(n),
                    beta=1.0, offset=0.0)
    r = preflight(im)
    assert r.remediations, "a failed placement must carry its remediations"
    assert any("degree" in s.lower() for s in r.remediations)
    assert preflight(grid(4)).remediations == ()


# ---------------------------------------------------------------------------
# F2 (branch review): preflight's gates now DELEGATE to tsu_compiler.gates.gate_checks()
# -- the same evaluation the real compile pipeline (passes/search.py) uses --
# instead of an independent reimplementation that (a) never carried
# target.py's `assumed` provenance, so an Extropic-documented fact
# (max_abs_coupling) and a genuine project guess (max_abs_bias) rendered
# IDENTICALLY even though target.py's own docstring records fixing that
# exact conflation once already; (b) had no colouring gate; (c) had no
# --allow-assumed downgrade path. node_budget is the one gate DELIBERATELY
# left un-delegated (see its own note/test below) -- gate_checks() measures
# it as n_nodes alone (correct for the real pipeline, which runs it BEFORE
# placement/mediators are known), while preflight folds in analyse()'s
# pre-placement mediator ESTIMATE up front; the two serve different
# purposes and are not reconciled, per this task's explicit allowance to
# report such a discrepancy rather than force a merge.
# ---------------------------------------------------------------------------

def test_max_abs_bias_gate_is_marked_assumed_and_max_abs_coupling_is_not():
    """WHAT THIS PINS: the max_abs_bias gate (target.py:
    Sourced(6.0, "assumed", ...)) carries assumed=True; max_abs_coupling
    (Sourced(6.0, "2608.01615v1.pdf...", ...), an Extropic-documented
    fact) carries assumed=False -- the two currently share a numeric value
    (6.0) but must never render identically (target.py's own docstring:
    "Never conflate these two fields again just because their VALUES
    happen to agree").
    HOW IT FAILS: a Gate built without sourcing `assumed` from
    target.is_assumed(...) (this branch's shipped behaviour) leaves the
    dataclass default (False) for both, so `g["max_abs_bias"].assumed is
    True` fails."""
    r = preflight(grid(6))
    g = {x.name: x for x in r.gates}
    assert g["max_abs_bias"].assumed is True
    assert g["max_abs_coupling"].assumed is False


def test_max_abs_bias_note_never_claims_it_is_a_hardware_fact():
    """WHAT THIS PINS: the max_abs_bias gate's note must never POSSESSIVELY
    claim to be "the hardware's" anything -- target.py marks this field
    assumed ("project working value; NOT a sourced Extropic figure"), and
    the shipped note ("|b| against the hardware's bias cap") rendered
    IDENTICALLY to max_abs_coupling's (an actual Extropic-documented
    fact), making the two indistinguishable to a reader on the exact FAIL
    verdict the branch review's own bigbias repro produces. The fix's own
    ASSUMED marker text ("...NOT a sourced hardware figure") legitimately
    mentions the word "hardware" while NEGATING it -- that is the
    opposite of the defect, so this checks for the possessive claim
    specifically, not the bare word.
    HOW IT FAILS: the shipped literal note string in check.py says "the
    hardware's bias cap" for max_abs_bias too, so
    `"the hardware's" not in note` fails against the unfixed defect."""
    r = preflight(_bigbias())
    g = {x.name: x for x in r.gates}
    note = g["max_abs_bias"].note.lower()
    assert "the hardware's" not in note
    assert "assumed" in note or "not a sourced" in note
    assert g["max_abs_bias"].status == "fail"


def test_a_bigbias_model_fails_on_the_assumed_bias_cap_not_a_hardware_fact():
    """Full F2 reproduction: |b|=9 > cap=6 (assumed) fails the gate and the
    overall verdict, while |J| stays comfortably under its own (real,
    sourced) cap -- the two gates must reach OPPOSITE statuses despite
    sharing a numeric cap value."""
    r = preflight(_bigbias())
    g = {x.name: x for x in r.gates}
    assert g["max_abs_bias"].status == "fail"
    assert g["max_abs_coupling"].status == "ok"
    assert r.verdict == "fail"


def test_preflight_gates_agree_with_gates_gate_checks_for_degree_coupling_and_bias():
    """WHAT THIS PINS: preflight's degree/max_abs_coupling/max_abs_bias
    gates are SOURCED from tsu_compiler.gates.gate_checks() -- the same evaluation
    the real compile pipeline uses -- not a second, independently-computed
    set of numbers that could silently drift from it (the branch review's
    own comparison table found exactly this kind of drift for
    node_budget/colouring/assumed/--allow-assumed).
    HOW IT FAILS: a reimplementation that recomputes peak |J|/|b|/degree by
    hand (this branch's shipped check.py) can still numerically AGREE by
    coincidence on an ordinary fixture, so this uses `_bigbias` (which
    exercises a FAILING gate) and checks every field gate_checks()
    computes -- measured, limit, assumed -- not just the pass/fail
    outcome."""
    from tsu_compiler.gates import gate_checks
    from tsu_compiler.passes.analyse import analyse
    model = _bigbias()
    rep = analyse(model)
    delegated = {gc.gate: gc for gc in gate_checks(model, rep, PROFILES["z1"], False)}
    r = preflight(model)
    g = {x.name: x for x in r.gates}
    assert g["max_degree"].value == delegated["degree"].measured
    assert g["max_degree"].limit == delegated["degree"].limit
    assert g["max_abs_coupling"].value == delegated["coupling_cap"].measured
    assert g["max_abs_coupling"].limit == delegated["coupling_cap"].limit
    assert g["max_abs_bias"].value == delegated["field_cap"].measured
    assert g["max_abs_bias"].limit == delegated["field_cap"].limit
    assert g["max_abs_bias"].assumed == delegated["field_cap"].assumed


def test_preflight_now_carries_a_colouring_gate():
    """WHAT THIS PINS: the colouring gate gates.py already evaluates (a
    structural check that analyse()'s own colouring never puts two
    adjacent nodes in the same colour block) is present in preflight's
    gate set too -- the branch review's comparison table found it present
    in gates.py and absent from preflight."""
    r = preflight(grid(6))
    names = {x.name for x in r.gates}
    assert "colouring" in names
    assert {x.name: x for x in r.gates}["colouring"].status == "ok"


def test_allow_assumed_downgrades_a_failing_assumed_gate_but_warns_not_oks():
    """WHAT THIS PINS: --allow-assumed (threaded through to
    gates.gate_checks) downgrades a FAILING assumed gate (max_abs_bias
    here) instead of failing the whole preflight on it -- gates.py already
    supports this; the branch's shipped preflight had no such flag at
    all. The verdict must still visibly reflect that something was
    downgraded (silently reporting "ok" would hide from a reader that a
    limit was overridden, not cleared) -- this project's convention is
    "warn", not "fail" and not silently "ok".
    HOW IT FAILS: a preflight() with no allow_assumed parameter at all
    (this branch's shipped signature) makes the first call raise
    TypeError; one that accepts the flag but never threads it into
    gate_checks() leaves max_abs_bias failing (status=='fail',
    verdict=='fail') regardless of allow_assumed."""
    r = preflight(_bigbias(), allow_assumed=True)
    g = {x.name: x for x in r.gates}
    assert g["max_abs_bias"].status == "downgraded"
    assert g["max_abs_bias"].downgraded is True
    assert r.verdict != "fail"
    r_off = preflight(_bigbias(), allow_assumed=False)
    assert r_off.verdict == "fail"


def test_node_budget_note_distinguishes_its_estimate_from_placements_actual_count():
    """WHAT THIS PINS (F9): node_budget's gate value folds in analyse()'s
    pre-placement mediator ESTIMATE (deliberately NOT delegated to
    gates.gate_checks(), which measures node_budget as n_nodes alone --
    see check.py's own comment for why the two definitions serve
    different purposes and are not reconciled). Without a note explaining
    this, a reader sees two different mediator-related numbers on the
    page (this gate's value, and the `mediators` line reporting
    placement's ACTUAL count) with no explanation of which is which --
    exactly the branch review's F9 finding on a 25-node odd cycle
    (node_budget showed 25, one line above showed '1 mediators', with no
    explanation of the discrepancy).
    HOW IT FAILS: the branch's SHIPPED note ("spins required, including
    estimated mediators, against the die's node budget") already contains
    the substring "estimat[ed]" -- so a weak check for that word ALONE
    would pass against the unfixed defect too. This additionally requires
    the word "actual" (pointing a reader at the `mediators` line above),
    which the shipped note does not contain."""
    r = preflight(triangle())
    note = {x.name: x for x in r.gates}["node_budget"].note.lower()
    assert "estimate" in note
    assert "actual" in note, \
        "the note must point a reader at placement's ACTUAL count, not " \
        "just name this gate's own estimate"


# ---------------------------------------------------------------------------
# A-1 task 3 (external review, 2026-09-09): preflight exists to answer "will
# it fit" BEFORE an expensive placement search. Mediation's own gadget
# coupling A = arccosh(exp(2*beta*|J|))/(2*beta) is a closed form -- a
# non-bipartite model's post-mediation coupling is therefore predictable
# from the model's own beta and |J|max alone, with no placement (and no
# annealer run) required at all. `mediator_coupling` (route.py) is the ONE
# implementation of that formula (see test_mediator_insertion.py's own
# reuse test); preflight calls it directly rather than re-deriving it.
# ---------------------------------------------------------------------------

def _near_cap_triangle(j: float = 6.0, beta: float = 1.0) -> IsingModel:
    """A triangle whose pre-mediation |J| clears `max_abs_coupling` (j <=
    6.0) but whose predicted MEDIATED coupling does not -- the reviewer's
    own reproduction (A-1): |J|=6.0 at beta=1.0 clears the pre-mediation
    cap gate (6.0 is not > 6.0) but real hidden-spin mediation needs
    A(6.0, beta=1) = 6.346573590275254 > 6.0."""
    return IsingModel(nodes=("a", "b", "c"), edges=((0, 1), (1, 2), (0, 2)),
                      weights=np.full(3, j), biases=np.zeros(3),
                      beta=beta, offset=0.0)


def test_a_bipartite_model_carries_no_mediated_coupling_prediction():
    """WHAT THIS PINS: mediation never runs on an already-bipartite model
    (route.py: `insert_mediators` early-outs on `report.bipartite`), so
    preflight must not report a predicted mediated coupling for one -- there
    is nothing to predict.
    HOW IT FAILS: an unconditional gate (added for every model regardless of
    bipartiteness) makes `"mediated_coupling_cap" in names` true here, which
    this test's `assert ... not in` catches."""
    r = preflight(grid(8))
    names = {x.name for x in r.gates}
    assert "mediated_coupling_cap" not in names


def test_a_non_bipartite_model_reports_its_predicted_mediated_coupling():
    """WHAT THIS PINS: for a non-bipartite model, preflight reports a
    `mediated_coupling_cap` gate BEFORE any placement runs -- the coupling
    mediation would require, computed from this model's own |J|max and
    beta via the closed form, gated against the same `max_abs_coupling`
    cap `coupling_cap` uses.
    HOW IT FAILS: if this gate is never added, `by_gate["mediated_coupling_
    cap"]` raises KeyError. If preflight guessed instead of computing (e.g.
    reused |J|max itself, or a hardcoded beta), the value would not match
    0.7191781980478423 -- the real A(0.4, beta=1.0)."""
    r = preflight(triangle())
    by_gate = {x.name: x for x in r.gates}
    g = by_gate["mediated_coupling_cap"]
    assert g.value == pytest.approx(0.7191781980478423)
    assert g.limit == PROFILES["z1"].max_abs_coupling.value
    assert g.status == "ok"
    assert g.assumed is False


def test_the_predicted_mediated_coupling_gate_uses_the_models_own_beta():
    """WHAT THIS PINS: the closed form must use the MODEL's own beta, never
    beta=1.0 hardcoded -- A is temperature-dependent (route.py's own
    BetaMismatchError/spec 5.3.5). Cross-checked against the SAME
    (beta=4.0, |J|=1.25) pair test_mediator_insertion.py's own
    `test_mediator_couplings_stay_within_the_target_cap` already pins to
    1.3366 (abs=1e-3) -- reusing that trusted reference number, not a new
    one invented for this test.
    HOW IT FAILS: a hardcoded `beta=1.0` in the prediction would compute
    A(1.25, 1.0) = 1.5668... instead, failing the `pytest.approx` check
    below (a ~17% difference, not a rounding-tolerance miss)."""
    r = preflight(_near_cap_triangle(j=1.25, beta=4.0))
    g = {x.name: x for x in r.gates}["mediated_coupling_cap"]
    assert g.value == pytest.approx(1.336643397505582)


def test_the_predicted_mediated_coupling_gate_matches_route_mediator_coupling_exactly():
    """WHAT THIS PINS: preflight DELEGATES to `tsu_compiler.passes.route.
    mediator_coupling` -- the SAME function `insert_mediators` itself
    calls (see test_mediator_insertion.py) -- rather than re-deriving the
    formula a second time. Checked with bit-for-bit `==`, not
    `pytest.approx`: two independently-written but mathematically
    equivalent expressions would still pass an approx check, which is
    exactly the kind of drift this project has been bitten by before.
    HOW IT FAILS: any independently re-derived expression of the same
    formula that is not EXACTLY `mediator_coupling`'s own floating-point
    result (e.g. a different but equivalent grouping of the same ops, which
    floating point does not guarantee agrees bit-for-bit) fails `==` even
    though it would pass a tolerance-based check."""
    from tsu_compiler.passes.route import mediator_coupling
    model = _near_cap_triangle(j=6.0, beta=1.0)
    r = preflight(model)
    g = {x.name: x for x in r.gates}["mediated_coupling_cap"]
    expected = mediator_coupling(6.0, 1.0)
    assert g.value == expected


def test_a_model_that_passes_coupling_cap_can_still_fail_the_mediated_prediction():
    """WHAT THIS PINS: the exact scenario A-1 exists for -- a model whose
    pre-mediation |J|max clears `coupling_cap` (6.0 is not > 6.0 -- it only
    WARNS, at exactly WARN_FRACTION*limit, per
    test_gate_statuses_use_the_stated_warn_fraction's own boundary rule; it
    never FAILS) but whose PREDICTED post-mediation coupling exceeds the
    same cap outright (6.346573590275254 > 6.0 -- a hard fail, not a
    warn). Both gates must be visible SIMULTANEOUSLY with their OWN,
    different statuses, so a reader sees the exact discrepancy this task's
    fix closes -- a model that reads as merely "tight" on the existing
    gate is actually INFEASIBLE once mediation is accounted for.
    HOW IT FAILS: before this feature, `max_abs_coupling` (preflight's
    display name for the `coupling_cap` gate) alone would read 'warn' --
    never 'fail' -- and preflight would say nothing about mediation
    raising the coupling past the cap; the overall verdict would be 'warn'
    for a model that, per A-1's own compile-time fix, cannot actually
    compile on Z1 at all. `g["mediated_coupling_cap"].status == 'fail'`
    catches the missing prediction; `g["max_abs_coupling"].status ==
    'warn'` (never 'fail') pins that the EXISTING gate genuinely does not
    catch this failure on its own (i.e. this is not a redundant check) --
    and `r.verdict == 'fail'` pins that the new gate's failure, not just
    its presence, actually drives the overall verdict."""
    model = _near_cap_triangle(j=6.0, beta=1.0)
    r = preflight(model)
    g = {x.name: x for x in r.gates}
    assert g["max_abs_coupling"].status == "warn"
    assert g["mediated_coupling_cap"].status == "fail"
    assert g["mediated_coupling_cap"].value == pytest.approx(6.346573590275254)
    assert r.verdict == "fail"


def test_the_predicted_mediated_coupling_note_explains_the_discrepancy():
    """WHAT THIS PINS: the gate's own note must plainly state that
    mediation RAISES every coupling it touches, and that this is WHY a
    model can pass `max_abs_coupling` before placement and still not fit
    once mediated -- a reader must not have to already know this project's
    internals to understand what the gate means.
    HOW IT FAILS: a generic/reused note (e.g. `coupling_cap`'s own
    "|J| against the hardware's coupling cap") that never mentions
    mediation raising the coupling at all fails the substring checks
    below."""
    model = _near_cap_triangle(j=6.0, beta=1.0)
    r = preflight(model)
    note = {x.name: x for x in r.gates}["mediated_coupling_cap"].note.lower()
    assert "mediat" in note
    assert "raise" in note or "increase" in note or "greater" in note
    assert "coupling_cap" in note or "max_abs_coupling" in note


def test_predicted_mediated_coupling_gate_needs_no_placement_to_compute():
    """WHAT THIS PINS: the prediction is available even when placement
    itself FAILS (or would be prohibitively slow) -- it is a closed form
    over the pre-placement model alone, per this feature's whole reason to
    exist (report the risk before waiting on the slower placement search).
    Forces a placement failure cheaply (a 20-node complete graph blows the
    degree gate outright, `test_a_degree_violation_is_REPORTED_not_raised`'s
    own fixture) on a model ALSO made non-bipartite, to prove the gate is
    populated regardless.
    HOW IT FAILS: if the prediction were computed FROM `placement`/
    `placement.mediated_ising` (rather than independently, from `ising`
    itself, before `place()` ever runs), it would be missing entirely here
    (`r.placed is False`) -- `by_gate` would KeyError."""
    from tsu_compiler.passes.analyse import analyse
    from tsu_compiler.passes.route import mediator_coupling
    n = 20
    e = [(i, j) for i in range(n) for j in range(i + 1, n)]
    im = IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=tuple(e),
                    weights=np.full(len(e), 0.1), biases=np.zeros(n),
                    beta=1.0, offset=0.0)
    assert not analyse(im).bipartite, "fixture assumption: a complete graph K20 is odd-cycled"
    r = preflight(im)
    assert r.placed is False
    g = {x.name: x for x in r.gates}["mediated_coupling_cap"]
    assert g.value == pytest.approx(mediator_coupling(0.1, 1.0))


def test_a_small_model_says_nothing_about_the_connected_fabric_assumption():
    """WHAT THIS PINS: the NEGATIVE case, and it is the one that matters. A
    model that fits inside one core places the same way however the cores are
    wired, so the assumption cannot affect it and mentioning it would be noise.
    A disclosure printed on every model is one a reader learns to skip, and
    then it is absent exactly when it counts.
    HOW IT FAILS: drop the `n_spins <= core` guard in `_fabric_note` and this
    fires on every 4-spin fixture in the suite.
    PROVENANCE: Z1 is documented as 269,568 pbits in 8 cores (billion, Fig.
    05), so one core is 33,696 -- computed here from the profile, not pasted."""
    from tsu_compiler.preflight.check import _core_nodes
    r = preflight(grid(4))
    assert r.n_spins < _core_nodes(PROFILES["z1"])
    assert r.fabric_note is None


def test_a_model_larger_than_one_core_says_its_placement_rests_on_an_assumption():
    """WHAT THIS PINS: above one core's worth of spins, a placement verdict
    depends on the cores being one connected lattice -- which the published
    material never states. It must say so.
    HOW IT FAILS: return None unconditionally from `_fabric_note`, or mark
    `connected_fabric` as a sourced fact rather than "assumed" in target.py,
    and this goes quiet while the tool keeps asserting placement.
    PROVENANCE: the threshold is the profile's own node_budget // 8; the
    assumption's status comes from `target.is_assumed`, not from a literal
    here. Exercised through the helper rather than by compiling a 33,697-node
    model, which would cost minutes for no extra coverage."""
    from tsu_compiler.preflight.check import _core_nodes, _fabric_note
    z1 = PROFILES["z1"]
    core = _core_nodes(z1)
    assert _fabric_note(core, z1) is None, "at exactly one core, still silent"
    note = _fabric_note(core + 1, z1)
    assert note and "ASSUMED" in note and "connected_fabric" in note


def test_the_coupling_note_reports_distinct_values_not_edge_count():
    """WHAT THIS PINS: Z1 carries ~2,135,904 coupling EDGES against ~215,904
    coupling PARAMETERS (Extropic's own figures), so this compiler's
    one-independent-J-per-edge model is an idealisation. The measure that
    decides whether that bites is DISTINCT coupling values, not edge count: a
    model with thousands of edges but a handful of distinct |J| needs only a
    handful of parameters.

    HOW IT FAILS: count edges instead of distinct values in `_coupling_note`
    and this reads 3 instead of 1 on a fixture whose three edges share one
    coupling -- which would make every large model look near the budget when
    it is nowhere near it.

    PROVENANCE: target.coupling_parameters, sourced to the Z1T die callout;
    the edge figure to the same page's body prose. See audit/findings/R2.md."""
    from tsu_compiler.preflight.check import _coupling_note
    im = IsingModel(nodes=("a", "b", "c", "d"), edges=((0, 1), (1, 2), (2, 3)),
                    weights=np.full(3, 0.4), biases=np.zeros(4),
                    beta=1.0, offset=0.0)
    n, note = _coupling_note(im, PROFILES["z1"])
    assert n == 1, "three edges sharing one coupling value need ONE parameter"
    assert "MAPPING is unmodelled" in note


def test_the_coupling_note_is_absent_when_the_target_has_no_parameter_budget():
    """WHAT THIS PINS: the ideal control has no die and no parameter budget, so
    it must carry no note. A disclosure about physical sharing on a target that
    has no hardware would be noise pretending to be rigour.

    HOW IT FAILS: drop the infinite-budget guard and the ideal control starts
    reporting a sharing caveat about a substrate it does not have.

    PROVENANCE: PROFILES['ideal'].coupling_parameters is `inf`, sourced
    'control'."""
    from tsu_compiler.preflight.check import _coupling_note
    im = IsingModel(nodes=("a", "b"), edges=((0, 1),), weights=np.array([0.4]),
                    biases=np.zeros(2), beta=1.0, offset=0.0)
    assert _coupling_note(im, PROFILES["ideal"]) == (None, None)


# ---------------------------------------------------------------------------
# R27, on the preflight path.
#
# `compile_spec` distinguishes a search that ran out of budget (verdict EFFORT,
# hardware_evaluated False) from a gate that refused. `preflight` did not: both
# collapsed into the string "fail", so the most-used entry point in the project
# reported a timer and a hardware limit identically.
#
# This matters on the flagship model. codon_spike_full is 3,147 spins and every
# gate passes -- degree 12 of 16, |J| 2.5 of 6, |b| 4.05 of 6, 3,147 of 269,568
# nodes -- and it does not place inside the default budget. Reported as "fail",
# that reads as "Z1 cannot host this model", which is a claim about Extropic's
# silicon resting on how long a greedy search was allowed to run.
# ---------------------------------------------------------------------------

def test_an_exhausted_placement_search_is_not_reported_as_a_gate_failure():
    """Every gate passing plus no embedding found is a SEARCH result, and the
    verdict has to say which kind of no it is."""
    import networkx as nx
    from tsu_compiler.preflight.model import IsingModel

    # A 40-node expander: degree 4, well inside every Z1 cap, and not a
    # subgraph of the offset lattice, so the budgeted search will not place it.
    G = nx.random_regular_graph(4, 40, seed=7)
    edges = tuple(sorted((min(u, v), max(u, v)) for u, v in G.edges()))
    model = IsingModel(
        nodes=tuple(f"v{i}" for i in range(40)),
        edges=edges,
        weights=np.full(len(edges), 0.5),
        biases=np.zeros(40),
        beta=1.0, offset=0.0,
    )
    r = preflight(model, restarts=1, iters=200)

    if r.placed:
        pytest.skip("this graph placed; the test needs one that exhausts the budget")

    assert not [g for g in r.gates if g.status == "fail"], (
        "the premise of this test is that no gate refused")
    assert r.verdict != "fail", (
        "an exhausted search reported as 'fail' is indistinguishable from a "
        "hardware limit, which is the conflation R27 exists to prevent")
    assert r.verdict == "effort"


def test_a_real_gate_failure_still_verdicts_fail():
    """The other half. Widening the effort case must not soften a genuine
    hardware refusal into the same bucket."""
    n = 24
    edges = tuple((i, j) for i in range(n) for j in range(i + 1, n))
    model = IsingModel(
        nodes=tuple(f"v{i}" for i in range(n)),
        edges=edges,
        weights=np.full(len(edges), 0.5),
        biases=np.zeros(n),
        beta=1.0, offset=0.0,
    )
    r = preflight(model)
    assert [g for g in r.gates if g.status == "fail"], "a 24-clique must break degree"
    assert r.verdict == "fail"
