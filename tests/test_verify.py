import pytest
from tsu.spec import load_spec
from tsu.target import Z1
from tsu.passes.search import compile_spec


def test_verification_reports_all_three_layers():
    v = compile_spec(load_spec("specs/toy.yaml"), Z1).verification
    assert v.energy_tv is not None
    assert v.task_validity is not None
    assert v.execution_tv is not None


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
