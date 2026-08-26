import pytest
from tsu.spec import load_spec
from tsu.passes.encode import encode
from tsu.ir import Binary


def test_categorical_expands_to_k_minus_one_binary_spins():
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    assert all(isinstance(v.domain, Binary) for v in enc.model.variables)
    # a, b, plus 2 chain spins for c (k=3)
    assert len(enc.model.variables) == 4
    assert "c__dw0" in enc.binary_names and "c__dw1" in enc.binary_names


def test_decode_round_trips_every_logical_assignment():
    """All 12 states of toy.yaml. If decode does not invert encode, nothing
    downstream means anything."""
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    seen = 0
    for asg in s.assignments():
        bits = enc.encode_assignment(asg)
        assert enc.decode(bits) == asg, f"round-trip failed for {asg}"
        seen += 1
    assert seen == 12


def test_energy_is_preserved_across_the_encoding():
    """Workload-equivalence: the encoded model must give the same energy as the
    logical one on every legal assignment, up to the monotonicity penalty which is
    zero on legal states."""
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    from tsu.ir import EnergyModel
    logical = EnergyModel(s.variables, s.terms, 1.0)
    for asg in s.assignments():
        bits = enc.encode_assignment(asg)
        assert enc.model.energy(bits) == pytest.approx(logical.energy(asg), abs=1e-9)


def test_illegal_chain_state_is_penalised():
    """A non-monotone chain is not a valid categorical value and must cost energy."""
    s = load_spec("specs/toy.yaml")
    enc = encode(s)
    illegal = {"a": 0, "b": 0, "c__dw0": 0, "c__dw1": 1}
    assert enc.model.energy(illegal) > 0.0
