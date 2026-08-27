import pytest
from tsu.spec import load_spec
from tsu.target import Z1
from tsu.passes.search import compile_spec


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
    from tsu.passes.verify import Verification
    v = Verification(energy_tv=None, energy_note="unavailable: 2^n too large",
                     task_validity=0.9, execution_tv=None,
                     execution_note="unavailable: no exact reference",
                     execution_noise_floor=None,
                     cross_check_tv=None, cross_check_note="unavailable: n > 12")
    d = v.to_dict()
    assert d["energy_tv"] == "unavailable: 2^n too large"
    assert d["execution_tv"] == "unavailable: no exact reference"


def test_regime_report_has_cheap_fields_and_unmeasured_elsewhere():
    r = compile_spec(load_spec("specs/toy.yaml"), Z1).regime
    assert r.coupling_utilisation is not None
    assert r.mixing_indicator is None
    assert r.regime in ("feasible", "precision_limited", "unmeasured")
