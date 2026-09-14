"""Finding from the WFC stress test (adjacency_4x4_k4.yaml): `_verify` gated
task_validity behind the SAME `n > EXACT_LIMIT` early return the genuinely
exact-reference-dependent layers (energy_tv, execution_tv, cross_check_tv)
need -- even though task validity needs only SAMPLES and the task contract.
verify.py's own module docstring says exactly this: "Workload correctness is
measured on DECODED samples and is never inferred from energy." Task validity
does not need an exact reference at all; a spec big enough to exceed
EXACT_LIMIT should still get a real task_validity number, with only the truly
exact-dependent fields reading unavailable.
"""
import pytest

from tsu_compiler.ir import Binary, Product, LinearForm, Var, VarRef
from tsu_compiler.spec import TaskContract, WorkloadSpec
from tsu_compiler.target import IDEAL
from tsu_compiler.passes.search import compile_spec
from tsu_compiler.backends.thrml_backend import EXACT_LIMIT


def _oversized_spec(contract_rules=()):
    n = EXACT_LIMIT + 2
    variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
    terms = tuple(
        Product(LinearForm({VarRef(f"x{i}"): 1.0}),
               LinearForm({VarRef(f"x{i + 1}"): 1.0}), 1.0)
        for i in range(n - 1))
    return WorkloadSpec(name="oversized_task_layer", variables=variables, terms=terms,
                        contract=TaskContract(contract_rules),
                        source_text="name: oversized_task_layer\n")


def test_task_validity_is_a_real_number_even_when_too_large_to_enumerate_exactly():
    c = compile_spec(_oversized_spec(), IDEAL)
    assert c.verdict == "COMPILED"
    v = c.verification
    assert isinstance(v.task_validity, float)
    assert 0.0 <= v.task_validity <= 1.0
    assert isinstance(v.codeword_violation_rate, float)


def test_exact_reference_dependent_fields_stay_honestly_unavailable():
    """The fix must not blur the line: energy_tv/execution_tv/cross_check_tv
    genuinely DO need an exact reference, and must still read unavailable for
    a model this large."""
    c = compile_spec(_oversized_spec(), IDEAL)
    v = c.verification
    assert v.energy_tv is None
    assert "too large to enumerate" in v.energy_note
    assert v.execution_tv is None
    assert v.execution_noise_floor is None
    assert v.cross_check_tv is None


def test_task_validity_reflects_a_real_contract_not_just_a_trivial_empty_one():
    """A contract rule that is actually violatable must show up in
    task_validity < 1.0 for a large model too -- proving this is a genuine
    sampled measurement, not a hardcoded 1.0 stand-in for "no rules to check"."""
    n = EXACT_LIMIT + 2
    variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
    rules = (
        {"rule": "forbid_both", "vars": ["x0", "x1"], "message": "x0 and x1 clash"},
    )
    spec = WorkloadSpec(name="oversized_with_rule", variables=variables, terms=(),
                        contract=TaskContract(rules),
                        source_text="name: oversized_with_rule\n")
    c = compile_spec(spec, IDEAL)
    assert c.verdict == "COMPILED"
    v = c.verification
    assert isinstance(v.task_validity, float)
    # with no energetic preference either way, x0=x1=1 occurs with real
    # probability under a fair coin per spin -- task_validity must be
    # measurably less than 1.0, not silently 1.0 for a spec with a real rule
    assert v.task_validity < 1.0
