"""Task 2: uniform coefficient scaling (spec section 4.9, 5.2).

Deviations from the original brief, both required by the controller's
rulings on this task:

- Ruling 1: `specs/lattice_l1_16x16.yaml` does not exist yet (it is a Task 3
  artefact). `test_scale_multiplies_every_coefficient_including_the_penalty`
  and `test_scale_leaves_topology_invariant` use `specs/adjacency_4x4_k4.yaml`
  instead -- it exists, is categorical (k=4), and exercises the same one-hot
  exactly-one penalty path.
- Ruling 4: `IsingModel` is a frozen dataclass, so
  `test_scaled_distribution_matches_unscaled_exactly` uses
  `dataclasses.replace(b, beta=b.beta / 0.25)` rather than the brief's
  `type(b)(**{**b.__dict__, ...})`.

A third, undirected adaptation (not covered by an explicit ruling):
`test_dominance_guard_is_evaluated_after_scaling` cannot use
`specs/too_strong.yaml` as the brief originally wrote it -- that spec has no
categorical variable, so `_guard_penalty_dominates` returns immediately
(`if not categorical: return`) regardless of scale, and the encode call would
never raise. `too_strong.yaml` exists to trip the *hardware* coupling_cap
gate (checked later, in `search.py`, against a target profile), not the
*encode-time* dominance guard this test is actually about. Replaced with
inline `WorkloadSpec` fixtures carrying a categorical variable, matching the
style already used by `tests/test_encode.py` and
`tests/test_encode_one_hot.py` for exactly this guard.

Also not in the brief's own test list, but required by note 2 of the task:
`encode(..., coefficient_scale=0)` (or negative) must raise a clear error --
a zero scale would flatten every energy to a uniform distribution while still
looking like a successful compile, the most dangerous silent failure
available here. Covered by `test_nonpositive_scale_is_rejected*` below.

Also not in the brief's own test list, but required by note 4 of the task:
the receipt must round-trip both the scale and the compensated beta, so a
later `tsu simulate` samples at the right temperature. Covered by
`test_receipt_program_beta_is_the_compensated_beta` below.
"""
import dataclasses
import json

import numpy as np
import pytest

from tsu.ir import Binary, Categorical, Linear, LinearForm, Product, Var, VarRef
from tsu.passes.encode import encode, spec_beta
from tsu.passes.lower import lower
from tsu.passes.program import SamplingProgram
from tsu.passes.search import compile_spec
from tsu.receipt import write_receipt
from tsu.report import render_explain, render_report
from tsu.spec import load_spec, TaskContract, WorkloadSpec
from tsu.target import PROFILES


def test_scale_multiplies_every_coefficient_including_the_penalty():
    """|b|max and |J|max must both scale linearly in s. If only the workload
    weights scaled, the one-hot exactly-one PENALTY would stay fixed and
    |b|max would not move by the full factor of s (spec section 4.8)."""
    spec = load_spec("specs/adjacency_4x4_k4.yaml")
    full = lower(encode(spec, "one_hot", coefficient_scale=1.0).model)
    quarter = lower(encode(spec, "one_hot", coefficient_scale=0.25).model)
    assert max(abs(quarter.biases)) == pytest.approx(
        0.25 * max(abs(full.biases)), rel=1e-9)
    assert max(abs(quarter.weights)) == pytest.approx(
        0.25 * max(abs(full.weights)), rel=1e-9)


def test_scale_leaves_topology_invariant():
    spec = load_spec("specs/adjacency_4x4_k4.yaml")
    full = lower(encode(spec, "one_hot", coefficient_scale=1.0).model)
    quarter = lower(encode(spec, "one_hot", coefficient_scale=0.25).model)
    assert full.nodes == quarter.nodes
    assert full.edges == quarter.edges


def test_dominance_guard_is_evaluated_after_scaling():
    """The guard's margin is RELATIVE: scaling the penalty AND the workload's
    own term weights by the same factor s cannot change which side
    dominates. A spec whose guard fails at s=1.0 must still fail at
    s=0.25, and one whose guard passes must still pass -- proving the guard
    runs on consistently-scaled values, not (say) a scaled penalty compared
    against an unscaled workload weight."""
    failing = WorkloadSpec(
        name="dominance_fails_both_scales",
        variables=(Var("c", Categorical(3)),),
        terms=(Linear(LinearForm({VarRef("c", 0): 1.0}), 1000.0),),
        contract=TaskContract(()))
    for s in (1.0, 0.25):
        with pytest.raises(ValueError, match="PENALTY"):
            encode(failing, "one_hot", coefficient_scale=s)
        with pytest.raises(ValueError, match="MONOTONE_PENALTY"):
            encode(failing, "domain_wall", coefficient_scale=s)

    passing = WorkloadSpec(
        name="dominance_passes_both_scales",
        variables=(Var("c", Categorical(3)),),
        terms=(Linear(LinearForm({VarRef("c", 0): 1.0}), 1.0),),
        contract=TaskContract(()))
    for s in (1.0, 0.25):
        encode(passing, "one_hot", coefficient_scale=s)          # must not raise
        encode(passing, "domain_wall", coefficient_scale=s)      # must not raise


def test_scaled_distribution_matches_unscaled_exactly():
    """The load-bearing claim: E -> sE with beta -> beta/s is the SAME
    distribution. Verified by exact enumeration on a model small enough to
    enumerate, not asserted.

    `exact_distribution` takes a `SamplingProgram`, not bare model output
    (Ruling 3), and returns `(states, probs)`, not `probs` alone -- checked
    directly against `thrml_backend.py` rather than assumed."""
    from tsu.backends.thrml_backend import exact_distribution
    spec = load_spec("specs/toy.yaml")
    a = lower(encode(spec, "domain_wall", coefficient_scale=1.0).model)
    b = lower(encode(spec, "domain_wall", coefficient_scale=0.25).model)
    b = dataclasses.replace(b, beta=b.beta / 0.25)

    states_a, pa = exact_distribution(SamplingProgram(a, blocks=()))
    states_b, pb = exact_distribution(SamplingProgram(b, blocks=()))
    assert states_a.tolist() == states_b.tolist()
    np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-10)


def test_scale_of_one_is_the_identity():
    """coefficient_scale=1.0 must reproduce encode()'s pre-existing default
    exactly -- no accidental drift for the unscaled case this whole codebase
    already depended on before this task existed."""
    spec = load_spec("specs/toy.yaml")
    default = lower(encode(spec, "domain_wall").model)
    explicit = lower(encode(spec, "domain_wall", coefficient_scale=1.0).model)
    assert default.nodes == explicit.nodes
    assert default.edges == explicit.edges
    np.testing.assert_array_equal(default.biases, explicit.biases)
    np.testing.assert_array_equal(default.weights, explicit.weights)
    assert default.beta == explicit.beta


def test_nonpositive_scale_is_rejected_by_encode():
    """A zero (or negative) scale would flatten every energy to a uniform (or
    sign-inverted) distribution while still looking like a successful
    compile -- the single most dangerous silent failure this task's own
    brief names. Must raise loudly at encode time, not downstream."""
    spec = load_spec("specs/toy.yaml")
    with pytest.raises(ValueError, match="coefficient_scale"):
        encode(spec, "domain_wall", coefficient_scale=0.0)
    with pytest.raises(ValueError, match="coefficient_scale"):
        encode(spec, "domain_wall", coefficient_scale=-1.0)


def test_nonpositive_scale_is_rejected_by_compile_spec():
    """Same guard, reachable through the public entry point -- `compile_spec`
    must fail fast and loud rather than let `_try`'s broad except swallow
    `encode`'s ValueError into an opaque SEMANTICALLY_INVALID candidate."""
    spec = load_spec("specs/toy.yaml")
    with pytest.raises(ValueError, match="coefficient_scale"):
        compile_spec(spec, PROFILES["z1"], coefficient_scale=0.0)


def test_receipt_records_scale_and_beta():
    spec = load_spec("specs/toy.yaml")
    comp = compile_spec(spec, PROFILES["z1"], coefficient_scale=0.25)
    assert comp.coefficient_scale == 0.25
    assert comp.scaled_beta == pytest.approx(spec_beta(spec) / 0.25)


def test_receipt_program_beta_is_the_compensated_beta(tmp_path):
    """Note 4: if a scaled compile's receipt does not carry the scale and the
    compensated beta, `tsu simulate` (which reads program.json's `beta`
    verbatim -- see simulate.py's `reconstruct_program`) would sample at the
    WRONG temperature later with nothing to flag it. The PHYSICAL program
    actually built by this compile -- not just the bookkeeping fields --
    must carry the compensated beta."""
    spec = load_spec("specs/toy.yaml")
    comp = compile_spec(spec, PROFILES["z1"], coefficient_scale=0.25)
    assert comp.verdict == "COMPILED"
    d = write_receipt(comp, tmp_path / "r")

    passes = json.loads((d / "passes.json").read_text())
    assert passes["coefficient_scale"] == 0.25
    assert passes["scaled_beta"] == pytest.approx(spec_beta(spec) / 0.25)

    program = json.loads((d / "program.json").read_text())
    assert program["beta"] == pytest.approx(spec_beta(spec) / 0.25)


def test_report_marks_a_scaled_sample_unmistakably(tmp_path):
    """Note 5: a scaled sample must never be mistakable for an unscaled one
    in any human-readable output. `render_report`/`render_explain` print the
    scale AND the beta compensation it implies, under SAMPLING/SAMPLER."""
    spec = load_spec("specs/toy.yaml")
    comp = compile_spec(spec, PROFILES["z1"], coefficient_scale=0.25)
    d = write_receipt(comp, tmp_path / "r")

    report_text = render_report(d)
    assert "Coefficient scale:" in report_text
    assert "0.25" in report_text
    assert "1 -> 4" in report_text.replace(".0", "")  # 1.0 -> 4.0, format-tolerant

    explain_text = render_explain(d)
    assert "Coefficient scale:" in explain_text
    assert "0.25" in explain_text


def test_report_marks_an_unscaled_sample_as_unscaled(tmp_path):
    """The converse of the above: coefficient_scale == 1.0 (the default, and
    every pre-Task-2 compile) must print unmistakably as unscaled, not as a
    bare '1.0' a reader could confuse for a scaled run that happened to land
    on 1.0."""
    spec = load_spec("specs/toy.yaml")
    comp = compile_spec(spec, PROFILES["z1"])
    d = write_receipt(comp, tmp_path / "r")

    report_text = render_report(d)
    assert "1.0 (unscaled)" in report_text
