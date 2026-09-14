"""Part C+D of the WFC stress test: the two adjacency-constrained
value-assignment specs (a `grid` shape + `product_over_edges` +
`forbid_value_pair_over_edges`, per A1-A3), compiled through the real,
unmodified pipeline against Z1. These tests lock in the STRUCTURAL findings
recorded in wfc-stress-report.md so they cannot silently drift:

- adjacency_2x2_k3: small enough for exact enumeration on both encodings;
  domain-wall compiles clean on Z1, one_hot is HARDWARE_INFEASIBLE there
  (its exactly-one penalty pushes |b|max past Z1's assumed cap).
- adjacency_4x4_k4: beyond exact enumeration on both encodings; BOTH
  encodings are HARDWARE_INFEASIBLE on Z1 under the same assumed field cap,
  domain-wall alone recovers under --allow-assumed, and task_validity is
  still a real number for it despite no exact reference existing.
"""
from tsu_compiler.spec import load_spec
from tsu_compiler.target import Z1
from tsu_compiler.passes.encode import encode
from tsu_compiler.passes.search import compile_spec, compare
from tsu_compiler.states import CandidateState
from tsu_compiler.backends.thrml_backend import EXACT_LIMIT


def test_2x2_k3_spin_counts_are_within_exact_enumeration_reach():
    s = load_spec("specs/adjacency_2x2_k3.yaml")
    assert len(s.variables) == 4
    assert len(s.edges) == 4
    dw = encode(s, "domain_wall")
    oh = encode(s, "one_hot")
    assert len(dw.model.variables) == 8 <= EXACT_LIMIT
    assert len(oh.model.variables) == 12 <= EXACT_LIMIT


def test_4x4_k4_spin_counts_exceed_exact_enumeration_on_both_encodings():
    s = load_spec("specs/adjacency_4x4_k4.yaml")
    assert len(s.variables) == 16
    assert len(s.edges) == 24
    dw = encode(s, "domain_wall")
    oh = encode(s, "one_hot")
    assert len(dw.model.variables) == 48 > EXACT_LIMIT
    assert len(oh.model.variables) == 64 > EXACT_LIMIT


def test_2x2_k3_domain_wall_compiles_clean_on_z1_one_hot_does_not():
    """Finding: one_hot's exactly-one penalty on a k=3 categorical produces a
    triangle per variable; even where that doesn't hit the parity gate first,
    its bias magnitude alone (|b|max=7.0) exceeds Z1's assumed field cap
    (6.0) -- domain_wall (|b|max=4.5) does not."""
    c = compile_spec(load_spec("specs/adjacency_2x2_k3.yaml"), Z1)
    assert c.verdict == "COMPILED"
    assert c.repset.selected.encoding == "domain_wall"

    rows = {r["encoding"]: r for r in compare(c.repset)}
    assert rows["domain_wall"]["state"] == "SELECTED"
    assert rows["one_hot"]["state"] == "HARDWARE_INFEASIBLE"
    assert "b|max" in rows["one_hot"]["reason"]


def test_2x2_k3_semantic_and_sampling_layers_are_real_numbers():
    c = compile_spec(load_spec("specs/adjacency_2x2_k3.yaml"), Z1)
    v = c.verification
    assert v.energy_tv == 0.0
    assert 0.0 <= v.task_validity <= 1.0
    assert v.execution_tv is not None
    # torx's exact route only applies to a single-bond model; this has 12
    # edges, so the honest, expected outcome is "unavailable", not a number
    # it cannot defend.
    assert v.cross_check_tv is None
    assert "Trotter" in v.cross_check_note


def test_4x4_k4_both_encodings_fail_z1s_assumed_field_cap():
    """Finding: scaling the SAME per-edge constraint weight to a 4x4 grid
    accumulates bias at interior variables (more edges touch them) past Z1's
    assumed |b| cap of 6.0, for BOTH encodings -- domain_wall's own bias
    (6.5) barely exceeds it, one_hot's (18.0) does so badly."""
    c = compile_spec(load_spec("specs/adjacency_4x4_k4.yaml"), Z1)
    assert c.verdict == "HARDWARE"
    assert c.ideal_passed is True   # logically fine; this is a hardware finding

    rows = {r["encoding"]: r for r in compare(c.repset)}
    assert rows["domain_wall"]["state"] == "HARDWARE_INFEASIBLE"
    assert rows["one_hot"]["state"] == "HARDWARE_INFEASIBLE"
    assert "b|max" in rows["domain_wall"]["reason"]
    assert "b|max" in rows["one_hot"]["reason"]


def test_4x4_k4_domain_wall_recovers_under_allow_assumed_one_hot_mediated_not_selected():
    """The tuned result: overriding the assumed field cap lets domain_wall
    through (its own excess was only 0.5 over the cap). one_hot's field-cap
    excess (18.0 vs 6.0) is ALSO overridden by --allow-assumed -- its
    remaining obstacle was never overridable field cap so much as its
    exactly-one clique per categorical not being bipartite; Task 6 makes
    `place` mediate that (32 within-side edges get a hidden spin each, since
    one_hot's per-cell K4 clique has 6 edges but a bipartite 2-partition of
    K4 realizes only 4 of them directly -- 2 mediated per cell x 16 cells).
    one_hot therefore now reaches HARDWARE_FEASIBLE too, just outranked by
    domain_wall's far smaller physical p-bit count (48 vs 96+32)."""
    c = compile_spec(load_spec("specs/adjacency_4x4_k4.yaml"), Z1, allow_assumed=True)
    assert c.verdict == "COMPILED"
    assert c.repset.selected.encoding == "domain_wall"

    rows = {r["encoding"]: r for r in compare(c.repset)}
    assert rows["one_hot"]["state"] == "VIABLE_NOT_SELECTED"
    assert rows["one_hot"]["bipartite"] is True
    assert rows["one_hot"]["mediators_inserted"] == 32


def test_4x4_k4_task_validity_is_measured_despite_no_exact_reference():
    """The whole point of including this instance (Part C): exact layers
    honestly unavailable, task layer still works because it needs only
    samples and the contract."""
    c = compile_spec(load_spec("specs/adjacency_4x4_k4.yaml"), Z1, allow_assumed=True)
    v = c.verification
    assert v.energy_tv is None
    assert "too large to enumerate" in v.energy_note
    assert v.task_validity is not None
    assert 0.0 <= v.task_validity <= 1.0
