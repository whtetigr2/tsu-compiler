import pytest
from tsu.spec import load_spec, TaskContract, WorkloadSpec
from tsu.ir import Binary, Product, LinearForm, Var, VarRef
from tsu.target import IDEAL, Z1
from tsu.passes.search import compile_spec
from tsu.backends.thrml_backend import EXACT_LIMIT


def _oversized_spec(n) -> WorkloadSpec:
    """n binary variables in a path, n > EXACT_LIMIT -- too large to enumerate
    exactly. Compiled against IDEAL so it reaches COMPILED and _verify actually
    runs, taking the n > EXACT_LIMIT branch."""
    variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
    terms = tuple(
        Product(LinearForm({VarRef(f"x{i}"): 1.0}),
               LinearForm({VarRef(f"x{i + 1}"): 1.0}), 1.0)
        for i in range(n - 1))
    return WorkloadSpec(name="oversized", variables=variables, terms=terms,
                        contract=TaskContract(()), source_text="name: oversized\n")


def test_verification_reports_all_three_layers():
    v = compile_spec(load_spec("specs/toy.yaml"), Z1).verification
    assert v.energy_tv is not None
    assert v.task_validity is not None
    assert v.execution_tv is not None
    assert v.energy_tv == pytest.approx(0.0, abs=1e-6), \
        "lowering is exact; the model's own energy should reproduce it to float precision"
    assert 0.0 <= v.task_validity <= 1.0, \
        "task_validity is a fraction of valid decoded samples and must be a probability"


def test_execution_tv_is_within_its_own_noise_floor():
    """Regression: `execution_tv` must be computed against exact_distribution's
    OWN state ordering, never an independently-assumed bit order. An earlier
    version indexed the histogram LSB-first while `exact_distribution` (via
    itertools.product) enumerates MSB-first, which silently compared the
    sampled histogram against a permuted reference and inflated execution_tv
    from ~0.01 to ~0.52 -- a broken comparison indistinguishable from a broken
    sampler. A correct sampler's execution_tv should sit at or below its
    reported finite-sample noise floor."""
    v = compile_spec(load_spec("specs/toy.yaml"), Z1).verification
    assert v.execution_tv < max(5 * v.execution_noise_floor, 0.05)


def test_verification_says_unavailable_rather_than_guessing():
    """I8: this used to hand-construct a Verification and assert to_dict echoed
    it back -- it never called _verify, so it could not fail if the PRODUCTION
    path ever fabricated a number instead of reporting unavailability. Compiling
    a model too large to enumerate exactly (n > EXACT_LIMIT) exercises the real
    path: _verify's own `n > EXACT_LIMIT` branch.

    UPDATED (WFC-style stress test finding): task_validity and
    codeword_violation_rate need only samples and the task contract -- never
    an exact reference -- so `_verify` no longer gates them behind this same
    branch (see search.py's `_verify` for the fix). Only the genuinely
    exact-reference-dependent fields stay unavailable here; task_validity and
    codeword_violation_rate must instead be real, fabrication-free numbers."""
    n = EXACT_LIMIT + 2
    v = compile_spec(_oversized_spec(n), IDEAL).verification
    assert v is not None
    d = v.to_dict()
    for field in ("energy_tv", "execution_tv", "execution_noise_floor",
                 "cross_check_tv"):
        assert isinstance(d[field], str) and d[field].startswith("unavailable"), \
            f"{field} must read 'unavailable: <reason>', not a fabricated number " \
            f"or a bare 'unavailable' with no reason -- got {d[field]!r}"
    for field in ("task_validity", "codeword_violation_rate"):
        assert isinstance(d[field], float) and 0.0 <= d[field] <= 1.0, \
            f"{field} needs no exact reference and must be a real measured " \
            f"number even for an oversized model -- got {d[field]!r}"


def test_codeword_violation_rate_is_measured_and_excluded_from_task_validity():
    """C5: a sample that is not a valid codeword (a non-monotone domain-wall
    chain) must not be silently decoded and counted toward task_validity -- it
    must be tracked as a codeword violation instead, visibly, not folded into
    task_validity's denominator as if it had been a normal valid-or-invalid
    decoded sample."""
    v = compile_spec(load_spec("specs/toy.yaml"), Z1).verification
    assert v.codeword_violation_rate is not None
    assert 0.0 <= v.codeword_violation_rate <= 1.0
    # toy.yaml's MONOTONE_PENALTY genuinely dominates its term weights, so
    # violations should be rare, not absent by construction of the test
    assert v.codeword_violation_rate < 0.2


def test_regime_report_has_cheap_fields_and_unmeasured_elsewhere():
    r = compile_spec(load_spec("specs/toy.yaml"), Z1).regime
    assert r.coupling_utilisation is not None
    assert r.mixing_indicator is None
    assert r.regime in ("feasible", "precision_limited", "unmeasured")
