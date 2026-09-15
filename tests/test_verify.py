import pytest
from tsu_compiler.spec import load_spec, TaskContract, WorkloadSpec
from tsu_compiler.ir import Binary, Linear, Product, LinearForm, Var, VarRef
from tsu_compiler.target import IDEAL, Z1
from tsu_compiler.passes.search import compile_spec
from tsu_compiler.backends.thrml_backend import EXACT_LIMIT


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
        # `startswith("unavailable")` alone is satisfied BY the bare literal
        # the message above promises to reject, so the reason is checked
        # explicitly: a separator, and something after it. Found by a
        # claims-vs-evidence audit -- the assertion did not check what its
        # own failure message said it checked (cf. audit/findings/R20.md).
        assert d[field].startswith("unavailable:") and d[field][12:].strip(), (
            f"{field} is a bare 'unavailable' with no reason attached -- "
            f"saying WHY is the entire point of the field -- got {d[field]!r}")
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


def _always_violating_spec() -> WorkloadSpec:
    """A single occupancy variable strongly biased toward 1 (a large-magnitude
    preference), whose task contract forbids exactly that state -- built
    directly (not via YAML/sampling) so the guarantee is EXACT, not merely
    probable: no sampled codeword can ever pass this contract, exercising
    G2's 'no valid-and-task-valid sample exists' path deterministically.
    `vars: ["a", "a"]` makes `forbid_both` fire whenever a alone is
    occupied."""
    a = Var("a", Binary())
    term = Linear(LinearForm({VarRef("a"): 1.0}), weight=-50.0)
    contract = TaskContract(rules=(
        {"rule": "forbid_both", "vars": ["a", "a"],
         "message": "a must never be occupied"},))
    return WorkloadSpec(name="always_violating", variables=(a,), terms=(term,),
                        contract=contract, source_text="name: always_violating\n")


# ---------------------------------------------------------------------------
# G2: a concrete decoded sample must be persisted on every compile that
# actually ran verification -- the workload's own variable names/values
# (Encoded.decode's output), never raw physical spins, with enough
# provenance (seed, clamp) and honesty (is_codeword/task_valid/violations)
# to know what it actually demonstrates.
# ---------------------------------------------------------------------------

def test_compile_records_a_decoded_sample_in_the_workloads_own_vocabulary():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    s = c.sample
    assert s is not None
    assert isinstance(s.decoded, dict)
    assert set(s.decoded) == {"a", "b", "c"}, \
        "the decoded sample must be keyed by the WORKLOAD's own variable " \
        "names, not raw physical spin names"
    assert all(isinstance(v, int) for v in s.decoded.values())


def test_decoded_sample_prefers_a_codeword_that_also_passes_the_task_contract():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    s = c.sample
    assert s.is_codeword is True
    assert s.task_valid is True
    assert s.violations == ()
    # it must actually BE a real solution, not merely labelled one
    assert c.spec.contract.validate(s.decoded).ok is True


def test_decoded_sample_carries_its_seed_and_an_empty_clamp_when_unclamped():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    s = c.sample
    assert s.seed == 0     # _VERIFY_SAMPLE_PARAMS' fixed seed
    assert s.clamp == {}


def test_decoded_sample_records_the_active_clamp():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={"a": 1})
    s = c.sample
    assert s.clamp == {"a": 1}
    assert s.decoded["a"] == 1, \
        "a clamped node's decoded value must reflect the clamp on every draw"


def test_decoded_sample_persists_a_failing_one_when_no_sample_ever_passes_the_contract():
    """G2's honesty requirement: when no drawn sample is both a valid
    codeword AND task-valid, the compiler must say so and persist a FAILING
    sample with its violations -- never fabricate a passing one, never leave
    the field empty."""
    c = compile_spec(_always_violating_spec(), IDEAL)
    assert c.verdict == "COMPILED"
    assert c.verification.task_validity == pytest.approx(0.0), \
        "sanity: the contract must be genuinely unsatisfiable by this spec"
    s = c.sample
    assert s is not None
    assert s.is_codeword is True    # no categorical chains -- always a codeword
    assert s.task_valid is False
    assert s.violations, "a failing sample must carry the SPECIFIC violation(s)"
    assert "a must never be occupied" in s.violations


def test_regime_report_has_cheap_fields_and_measures_mixing_when_the_chain_supports_it():
    """toy.yaml's compiled chain mixes fast enough (tau ~ 0.9, close to the
    i.i.d. tau=1 floor -- a 4-spin chromatic-block-Gibbs chain has little to
    decorrelate) that N/tau clears tsu_compiler.ess's reliability threshold, so
    mixing_indicator is now a REAL measurement, not the old permanent None --
    the whole point of wiring tsu_compiler.ess in was to stop reporting "unmeasured"
    for a quantity the compiler can, in fact, measure here.

    `energy_scale` used to be a permanent None (regime.py's own docstring:
    "cheap fields only"); toy.yaml's 4 physical spins are trivially within
    EXACT_LIMIT, so it is now a real measured gap between the best
    task-satisfying and best task-violating physical state -- positive,
    because toy.yaml's contract is satisfiable and not vacuous (some legal
    codeword violates it: e.g. a=1,c=0)."""
    r = compile_spec(load_spec("specs/toy.yaml"), Z1).regime
    assert r.coupling_utilisation is not None
    assert r.mixing_indicator is not None
    assert r.mixing_indicator == pytest.approx(1.0, abs=0.5), \
        "a well-mixing 4-spin chain's IAT should sit close to the i.i.d. floor of 1"
    assert r.energy_scale is not None, \
        "toy.yaml is well within EXACT_LIMIT; energy_scale must now be measured"
    assert r.energy_scale > 0.0
    assert r.regime in ("feasible", "precision_limited", "unmeasured")


def test_beta_recommendation_is_a_window_in_units_of_the_energy_scale():
    """The project has measured an INTERIOR optimum for beta (regime.py's own
    justification): too low is noise, too high freezes the chain before it
    reaches the answer. So this must be a (lo, hi) window, not a point
    estimate, with lo < hi, both positive, and both expressed as an absolute
    beta (i.e. already divided through by energy_scale -- 'in units of' the
    gap, per the brief)."""
    r = compile_spec(load_spec("specs/toy.yaml"), Z1).regime
    assert r.energy_scale is not None
    assert r.beta_recommendation is not None
    lo, hi = r.beta_recommendation
    assert 0.0 < lo < hi
    # sanity: the window scales inversely with the gap -- a bigger gap needs
    # a SMALLER beta to reach the same beta*energy_scale crossover ratio
    assert lo == pytest.approx(1.0 / r.energy_scale)


def test_energy_scale_and_beta_recommendation_stay_none_beyond_the_enumeration_limit():
    """Never estimated: a model too large to enumerate exhaustively must
    report energy_scale as None (not a guess), and beta_recommendation --
    which is expressed in units of energy_scale -- must therefore also stay
    None rather than being derived from a number that was never measured."""
    n = EXACT_LIMIT + 2
    r = compile_spec(_oversized_spec(n), IDEAL).regime
    assert r is not None
    assert r.energy_scale is None
    assert r.beta_recommendation is None
    assert r.regime == "unmeasured"
