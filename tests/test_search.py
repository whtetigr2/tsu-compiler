import pytest
from tsu.spec import load_spec
from tsu.target import Z1
from tsu.passes.search import compile_spec
from tsu.states import CandidateState


def test_toy_compiles_on_z1():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verdict == "COMPILED"
    assert c.repset.selected.state == CandidateState.SELECTED


def test_broken_spec_fails_ideal_and_reports_LOGICAL_not_hardware():
    """The whole point of the ideal control."""
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    assert c.hardware_evaluated is False, \
        "z1 must not be evaluated when the ideal control fails"


def test_too_dense_passes_ideal_then_fails_z1_as_a_hardware_constraint():
    c = compile_spec(load_spec("specs/too_dense.yaml"), Z1)
    assert c.verdict == "HARDWARE"
    assert c.ideal_passed is True
    assert c.repset.selected is None


def test_ideal_control_always_runs_and_is_recorded():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.ideal_passed is True
    assert c.ideal_report is not None


def test_too_strong_fails_on_an_assumed_gate_and_compiles_with_override():
    s = load_spec("specs/too_strong.yaml")
    assert compile_spec(s, Z1).verdict == "HARDWARE"
    assert compile_spec(s, Z1, allow_assumed=True).verdict == "COMPILED"
