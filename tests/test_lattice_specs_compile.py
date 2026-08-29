"""Task 3: the LATTICE workload specs at their FULL 16x16 size, compiled to
z1 at coefficient_scale 0.25 -- the operating point derived (Task 2) by
measurement on a 3x3 grid, per this task's brief.

Deviation from the brief's own step-1 test, required by the measurement
itself: the brief's literal test asserted `comp.verdict == "COMPILED"` for
both lattice_l0_16x16.yaml and lattice_l1_16x16.yaml. Running it (as
written, against the real pipeline, no tuning) shows BOTH fail at 16x16 --
this is one of the two explicitly acceptable outcomes the brief names ("They
do not [compile]. Report the exact gate failure with its numbers.") and the
brief is equally explicit that the rule set must not be tuned to force a
COMPILED verdict. The tests below therefore lock in the ACTUAL measured
verdicts and reasons, in the same style test_adjacency_specs.py already uses
for adjacency_4x4_k4.yaml's own measured HARDWARE outcome, rather than
asserting something the pipeline does not produce. Full numbers (every gate,
every encoding) are in this task's report.

Summary of what was measured (coefficient_scale=0.25, target z1):

- lattice_l0_16x16 (6 hard rules, no soft term): domain_wall passes every
  GATE (degree 14/16, |J| 0.625/6.0, |b| 1.0/6.0) but fails at PLACE/ROUTE --
  the logical graph is non-bipartite, which z1's routing cannot map ("parity
  conflict"), a failure mode this workload's graph did not previously
  exhibit at smaller sizes. one_hot passes degree (16/16, exactly at the
  boundary) and coupling, but fails the field-cap gate: |b|max=6.75 exceeds
  the assumed cap of 6.0.
- lattice_l1_16x16 (4 hard + 4 soft rules): domain_wall fails the DEGREE
  gate outright -- 18 ports needed, 16 available -- 2 over budget even
  though the same rule set's degree law was verified size-independent at
  3x3/4x4/5x5 for ONE-HOT (whose measured degree, 16, matches the law
  `(k-1) + g*|partners|` = 4 + 4*3 exactly and clears the gate). one_hot
  clears every gate (degree 16/16, |b|max 5.0/6.0) but then fails the same
  non-bipartite placement failure L0's domain_wall hit.
- lattice_l1_infeasible (L1 + 2 extra hard rules, deliberately over budget):
  BOTH encodings fail the degree gate as designed -- domain_wall 18/16 (the
  same excess L1 already had; the 2 extra rules do not touch its own
  worst node), one_hot 20/16 (exactly the p=4 case the degree law predicts:
  4 + 4*4 = 20).
"""
import pytest
from tsu.passes.search import compile_spec, compare
from tsu.spec import load_spec
from tsu.target import PROFILES

SCALE = 0.25


def test_l0_ideal_passes_but_neither_encoding_is_hardware_feasible_at_16x16():
    """L0's rule set is logically sound (ideal control passes) but neither
    encoding survives z1 at the full 16x16 size and s=0.25: domain_wall
    clears every gate and fails at placement (non-bipartite graph -- a
    'parity conflict'), one_hot clears degree/coupling but fails the
    field-cap gate at |b|max=6.75 (cap 6.0)."""
    comp = compile_spec(load_spec("specs/lattice_l0_16x16.yaml"),
                        PROFILES["z1"], coefficient_scale=SCALE)
    assert comp.ideal_passed is True
    assert comp.verdict == "HARDWARE"

    rows = {r["encoding"]: r for r in compare(comp.repset)}
    assert rows["domain_wall"]["state"] == "HARDWARE_INFEASIBLE"
    assert rows["domain_wall"]["bipartite"] is False
    assert rows["one_hot"]["state"] == "HARDWARE_INFEASIBLE"
    assert rows["one_hot"]["max_abs_b"] == pytest.approx(6.75)

    reasons = {c.encoding: (c.reason or "") for c in comp.repset.candidates}
    assert "parity" in reasons["domain_wall"]
    assert "b|max" in reasons["one_hot"] and "6.75" in reasons["one_hot"]


def test_l1_ideal_passes_but_neither_encoding_is_hardware_feasible_at_16x16():
    """L1's rule set is logically sound (ideal control passes) but neither
    encoding survives z1 at the full 16x16 size and s=0.25: domain_wall
    needs 18 ports (2 over z1's 16-port budget) despite every partner count
    staying within the verified p<=3, and one_hot -- which DOES clear the
    degree gate at exactly 16/16, matching the degree law -- fails at
    placement with the same non-bipartite 'parity conflict' L0's domain_wall
    hit."""
    comp = compile_spec(load_spec("specs/lattice_l1_16x16.yaml"),
                        PROFILES["z1"], coefficient_scale=SCALE)
    assert comp.ideal_passed is True
    assert comp.verdict == "HARDWARE"

    rows = {r["encoding"]: r for r in compare(comp.repset)}
    assert rows["domain_wall"]["state"] == "HARDWARE_INFEASIBLE"
    assert rows["one_hot"]["state"] == "HARDWARE_INFEASIBLE"

    reasons = {c.encoding: (c.reason or "") for c in comp.repset.candidates}
    assert "ports" in reasons["domain_wall"]
    assert "18" in reasons["domain_wall"] and "16" in reasons["domain_wall"]
    assert "parity" in reasons["one_hot"]


def test_infeasible_instance_reports_hardware_not_a_crash():
    """The deliberately-infeasible instance (L1 + 2 extra hard rules pushing
    a value to a 4th partner) must fail cleanly with a named, measured
    reason on BOTH encodings -- domain_wall at 18/16 (unchanged from plain
    L1: the extra rules do not touch its own worst node), one_hot at 20/16
    (exactly the degree law's p=4 prediction: 4 + 4*4 = 20). This proves the
    failure path itself is real and legible, not a crash."""
    comp = compile_spec(load_spec("specs/lattice_l1_infeasible.yaml"),
                        PROFILES["z1"], coefficient_scale=SCALE)
    assert comp.verdict == "HARDWARE"
    assert any("ports" in (c.reason or "") for c in comp.repset.candidates)

    reasons = {c.encoding: (c.reason or "") for c in comp.repset.candidates}
    assert "18" in reasons["domain_wall"]
    assert "20" in reasons["one_hot"]
