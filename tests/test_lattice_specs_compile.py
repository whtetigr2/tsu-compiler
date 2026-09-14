"""Task 3 (extended by Task 6): the LATTICE workload specs at their FULL
16x16 size, compiled to z1.

Deviation from the brief's own step-1 test, required by the measurement
itself: the brief's literal test asserted `comp.verdict == "COMPILED"` for
both lattice_l0_16x16.yaml and lattice_l1_16x16.yaml. Running it (as
written, against the real pipeline, no tuning) shows BOTH fail -- one of the
two explicitly acceptable outcomes both Task 3's and Task 6's briefs name --
and both briefs are equally explicit that the rule set must not be tuned to
force a COMPILED verdict. The tests below lock in the ACTUAL measured
verdicts and reasons, in the same style test_adjacency_specs.py already uses
for adjacency_4x4_k4.yaml's own measured HARDWARE outcome.

Task 6 changes WHERE each spec fails, not whether it compiles. Before Task
6, `place()` raised `parity_conflict` immediately for a non-bipartite graph
against z1 (a bipartite target) -- cheap, no geometric search. Task 6 makes
`place()` attempt hidden-spin mediation first; mediation always succeeds
(verified separately, `tests/test_mediator_insertion.py`), so the failure
that used to be `parity_conflict` is now `placement_effort_exhausted` --
mediation fixes PARITY, but the subsequent GEOMETRIC embedding search onto
Z1's lattice (place.py's own `_greedy_descent` -- renamed from `_anneal` by
external review C-4, 2026-09-10; it was never simulated annealing -- budget 6
restarts x 40,000 iters) is
a SEPARATE, much harder problem that this task does not attempt to solve,
and does not finish within that budget for either spec at this scale. This
is measured, not assumed -- both real `-m tsuc compile` invocations below
were run to completion.

**Wall-clock cost, and why these tests read a committed receipt instead of
calling `compile_spec` directly:** the annealer never reaches zero
unrealized edges for either graph at this scale (they are large and
irregular -- 1792/3838 physical nodes for L0, 3778 for L1's one_hot -- well
outside what a single-flip random-restart local search reliably embeds), so
every one of its 6 restarts runs its full 40,000-iteration budget every
time. Measured wall time: ~9-17 minutes PER spec. Re-running that inside
`pytest -q` on every invocation would take the whole suite from ~5 minutes
to ~35-40 minutes for a result that is, by construction, deterministic
(seeded) and does not change between runs absent a code change. The tests
below therefore read the ALREADY-PRODUCED, COMMITTED receipts at
`demo/receipts/l0/` and `demo/receipts/l1/` (produced by the literal
commands this task's brief specifies, in Step 8) -- the same "golden
receipt" pattern `demo/receipts/l1_infeasible/` already established in Task
3, just for a result expensive enough that regenerating it on every test run
is not a reasonable default. The mediation MATH itself (mediator count,
bipartiteness, degree preservation) is verified fast and directly against
the real spec topology below, with no placement search involved at all.
"""
import dataclasses
import json

import pytest

from tsu_compiler.gates import check_gates
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.encode import encode, spec_beta
from tsu_compiler.passes.lower import lower
from tsu_compiler.passes.route import insert_mediators
from tsu_compiler.passes.search import compile_spec, compare
from tsu_compiler.spec import load_spec
from tsu_compiler.target import PROFILES, Z1

L0_SCALE = 0.20     # Task 6: moved from 0.25 -- see the dedicated 0.25 test below
L1_SCALE = 0.25


def _receipt(name):
    d = f"demo/receipts/{name}"
    return {p: json.loads(open(f"{d}/{p}.json", encoding="utf-8").read())
           for p in ("passes", "candidates", "metrics")}


def _mediated(spec_path, encoding, scale):
    spec = load_spec(spec_path)
    enc = encode(spec, encoding, scale)
    ising = lower(enc.model)
    ising = dataclasses.replace(ising, beta=spec_beta(spec) / scale)
    report = analyse(ising)
    assert not report.bipartite
    med, rep = insert_mediators(ising, report)
    return report, med, rep


# -- mediation math against the real spec topology (fast: no placement) ----

def test_l1_one_hot_mediation_matches_the_measured_prototype_numbers():
    """The only L1 encoding that reaches mediation at all (domain_wall fails
    the degree gate outright, unaffected by this task -- see the receipt
    test below). Cross-checked against an independently measured standalone
    prototype: 2498 mediators, 1280->3778 spins, bipartite, degree still 16,
    2 colour blocks."""
    report, med, rep = _mediated("specs/lattice_l1_16x16.yaml", "one_hot", L1_SCALE)
    assert report.n_nodes == 1280 and report.n_edges == 8320
    assert report.max_degree == 16
    assert rep.mediator_count == 2498
    assert rep.bipartite_after is True
    med_report = analyse(med)
    assert len(med.nodes) == 3778
    assert med_report.bipartite is True
    assert med_report.max_degree == 16
    assert med_report.colour_blocks == 2


def test_l0_domain_wall_mediation_matches_the_measured_prototype_numbers():
    """The L0 encoding that reaches mediation at 0.20 (see the receipt test
    below for one_hot's own fate at this scale). Matches the independently
    measured standalone prototype exactly: 768 mediators, 1024->1792 spins,
    degree still 14. Mediator count/topology depends only on the GRAPH
    (edges), never on `coefficient_scale` -- verified identical at 0.20 and
    0.25 below."""
    report, med, rep = _mediated("specs/lattice_l0_16x16.yaml", "domain_wall", L0_SCALE)
    assert report.n_nodes == 1024 and report.n_edges == 5568
    assert report.max_degree == 14
    assert rep.mediator_count == 768
    med_report = analyse(med)
    assert len(med.nodes) == 1792
    assert med_report.bipartite is True
    assert med_report.max_degree == 14


def test_mediator_count_is_independent_of_coefficient_scale():
    """A is temperature-dependent (spec 5.3.5) but WHICH edges get mediated
    is pure graph structure -- `coefficient_scale` never touches the edge
    set, only weight magnitudes. Same graph, same BFS-parity partition, same
    greedy pass -- 0.20 and 0.25 must produce the identical mediator count."""
    _, _, rep_020 = _mediated("specs/lattice_l0_16x16.yaml", "domain_wall", 0.20)
    _, _, rep_025 = _mediated("specs/lattice_l0_16x16.yaml", "domain_wall", 0.25)
    assert rep_020.mediator_count == rep_025.mediator_count == 768


# -- the field-cap finding at 0.25 (Task 3's original, unchanged by Task 6,
#    cheap -- gate_checks alone, no placement) --------------------------

def test_l0_one_hot_fails_the_field_cap_at_0_25_which_is_why_l0_moved_to_0_20():
    """Task 3's original finding, reproduced directly against the gate layer
    alone (no placement search -- this gate runs BEFORE placement regardless
    of Task 6): |b|max=6.75 exceeds z1's assumed field cap of 6.0 at
    coefficient_scale=0.25. |b| is exactly linear in scale, so 0.20 gives
    6.75*0.8=5.4 < 6.0 -- see the receipt test below for what one_hot
    actually does once that gate clears."""
    spec = load_spec("specs/lattice_l0_16x16.yaml")
    enc = encode(spec, "one_hot", 0.25)
    ising = lower(enc.model)
    ising = dataclasses.replace(ising, beta=spec_beta(spec) / 0.25)
    report = analyse(ising)
    fails = {f.gate: f for f in check_gates(ising, report, Z1, False)}
    assert "field_cap" in fails
    assert fails["field_cap"].measured == pytest.approx(6.75)
    assert fails["field_cap"].limit == pytest.approx(6.0)


# -- the real, full-pipeline outcome (committed receipts; see module
#    docstring for why these read a receipt instead of calling
#    compile_spec directly) ------------------------------------------------

def test_l0_receipt_at_0_20_neither_encoding_places_within_budget():
    """`-m tsuc compile specs/lattice_l0_16x16.yaml --coefficient-scale 0.20
    --out demo/receipts/l0` (this task's Step 8, run for real). Both
    encodings now clear every GATE (domain_wall as at 0.25; one_hot's own
    field cap now clears too -- |b|max=5.4 < 6.0) and both now MEDIATE
    successfully (parity is fixed, unconditionally, by construction), but
    neither placement search reaches zero unrealized edges within the
    production budget -- `placement_effort_exhausted`, not `parity_conflict`."""
    r = _receipt("l0")
    assert r["passes"]["ideal_passed"] is True
    assert r["passes"]["verdict"] == "HARDWARE"

    rows = {row["encoding"]: row for row in r["candidates"]}
    for enc in ("domain_wall", "one_hot"):
        assert rows[enc]["state"] == "HARDWARE_INFEASIBLE"

    reasons = {c["encoding"]: c["reason"] for c in r["passes"]["candidates"]}
    assert "placement effort exhausted" in reasons["domain_wall"]
    assert "placement effort exhausted" in reasons["one_hot"]

    failures = {c["encoding"]: c["failure"] for c in r["passes"]["candidates"]}
    assert failures["domain_wall"]["failure_class"] == "placement_effort_exhausted"
    assert failures["domain_wall"]["mediation"] == {
        "mediator_count": 768,
        "partition_method": "bfs_depth_parity_with_greedy_local_search",
        "bipartite_after": True, "beta_used": pytest.approx(5.0)}
    assert failures["one_hot"]["failure_class"] == "placement_effort_exhausted"
    assert failures["one_hot"]["mediation"]["bipartite_after"] is True
    assert failures["one_hot"]["mediation"]["mediator_count"] > 0

    # one_hot's field cap genuinely cleared at 0.20 (contrast the dedicated
    # 0.25 test above) -- its own measured |b|max is on record here too.
    assert rows["one_hot"]["max_abs_b"] == pytest.approx(5.4)

    # C2: both candidates.json rows must describe the graph that was
    # ACTUALLY placed (post-mediation, bipartite) -- not the pre-mediation
    # one `place()` started from. domain_wall's exact post-mediation spin
    # count is measured independently above (test_l0_domain_wall_mediation_
    # matches_the_measured_prototype_numbers): 1024 -> 1792.
    assert rows["domain_wall"]["bipartite"] is True
    assert rows["domain_wall"]["logical_spins"] == 1792
    assert rows["one_hot"]["bipartite"] is True


def test_l1_receipt_at_0_25_domain_wall_fails_degree_one_hot_mediates_but_does_not_place():
    """`-m tsuc compile specs/lattice_l1_16x16.yaml --coefficient-scale 0.25
    --out demo/receipts/l1` (this task's Step 8, run for real). domain_wall
    still fails the DEGREE gate outright (18/16, unchanged by Task 6 --
    mediation never runs for it at all, since it never reaches placement).
    one_hot clears every gate, mediates (2498 mediators, matching the
    standalone prototype exactly), and is bipartite -- but the geometric
    placement search then exhausts its budget the same way L0's did."""
    r = _receipt("l1")
    assert r["passes"]["ideal_passed"] is True
    assert r["passes"]["verdict"] == "HARDWARE"

    reasons = {c["encoding"]: c["reason"] for c in r["passes"]["candidates"]}
    assert "ports" in reasons["domain_wall"]
    assert "18" in reasons["domain_wall"] and "16" in reasons["domain_wall"]
    assert "placement effort exhausted" in reasons["one_hot"]

    failures = {c["encoding"]: c["failure"] for c in r["passes"]["candidates"]}
    # domain_wall fails at the GATE layer -- `_try` rejects it (degree 18 >
    # 16) BEFORE `place()` ever runs (search.py's own gate_failures check
    # precedes the `place()` call), so its recorded failure is a serialised
    # `GateFailure` (`gate`/`cause`), never a `PlacementFailure`. A
    # `PlacementFailure`'s `failure_class`/`mediation` fields simply do not
    # exist on this schema -- asserting them was a test-authoring bug
    # (conflating the two failure schemas), not evidence of a code defect.
    assert failures["domain_wall"]["gate"] == "degree"
    assert "18" in failures["domain_wall"]["cause"]
    assert "16" in failures["domain_wall"]["cause"]
    assert "failure_class" not in failures["domain_wall"]
    assert "mediation" not in failures["domain_wall"]

    # one_hot reaches placement (clears every gate) and is the one that
    # actually exercises the PlacementFailure/mediation schema.
    assert failures["one_hot"]["failure_class"] == "placement_effort_exhausted"
    assert failures["one_hot"]["mediation"] == {
        "mediator_count": 2498,
        "partition_method": "bfs_depth_parity_with_greedy_local_search",
        "bipartite_after": True, "beta_used": pytest.approx(4.0)}

    # C2: one_hot's own candidates.json row must describe the graph that was
    # ACTUALLY placed (post-mediation, bipartite, 3778 spins) -- not the
    # pre-mediation one (1280 spins, non-bipartite) `place()` started from.
    rows = {row["encoding"]: row for row in r["candidates"]}
    assert rows["one_hot"]["bipartite"] is True
    assert rows["one_hot"]["logical_spins"] == 3778


def test_infeasible_instance_reports_hardware_not_a_crash():
    """The deliberately-infeasible instance (L1 + 2 extra hard rules pushing
    a value to a 4th partner) must fail cleanly with a named, measured
    reason on BOTH encodings -- domain_wall at 18/16 (unchanged from plain
    L1: the extra rules do not touch its own worst node), one_hot at 20/16
    (exactly the degree law's p=4 prediction: 4 + 4*4 = 20). Both fail the
    DEGREE gate, never reaching mediation/placement at all, so this stays
    cheap and calls `compile_spec` directly, unaffected by Task 6."""
    comp = compile_spec(load_spec("specs/lattice_l1_infeasible.yaml"),
                        PROFILES["z1"], coefficient_scale=L1_SCALE)
    assert comp.verdict == "HARDWARE"
    assert any("ports" in (c.reason or "") for c in comp.repset.candidates)

    reasons = {c.encoding: (c.reason or "") for c in comp.repset.candidates}
    assert "18" in reasons["domain_wall"]
    assert "20" in reasons["one_hot"]
