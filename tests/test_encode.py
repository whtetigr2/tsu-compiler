import pytest
from tsu.spec import load_spec, WorkloadSpec, TaskContract
from tsu.passes.encode import encode
from tsu.ir import Binary, Categorical, Linear, LinearForm, Product, Var, VarRef


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


def test_monotone_penalty_not_dominating_workload_weights_is_rejected():
    """A categorical plus a term weight large enough that MONOTONE_PENALTY no
    longer exceeds MONOTONE_MARGIN_FACTOR * max|weight| must fail loudly at
    encode time, not silently let an illegal chain state win."""
    spec = WorkloadSpec(
        name="oversized_weight",
        variables=(Var("c", Categorical(3)),),
        terms=(Linear(LinearForm({VarRef("c", 0): 1.0}), 6.0),),
        contract=TaskContract(()),
    )
    with pytest.raises(ValueError, match="MONOTONE_PENALTY"):
        encode(spec)


def test_binary_only_spec_with_large_weight_still_encodes():
    """The monotonicity-margin guard must only fire when there is a chain to
    keep monotone; a purely binary workload has none and must not be rejected
    no matter how large its term weights are."""
    spec = WorkloadSpec(
        name="binary_only_big_weight",
        variables=(Var("x", Binary()), Var("y", Binary())),
        terms=(Product(LinearForm({VarRef("x"): 1.0}),
                        LinearForm({VarRef("y"): 1.0}), 1000.0),),
        contract=TaskContract(()),
    )
    enc = encode(spec)  # must not raise
    assert len(enc.model.variables) == 2


def test_generated_chain_name_colliding_with_declared_variable_is_rejected():
    """A categorical c (k=3) generates 'c__dw0' and 'c__dw1'. If the spec also
    declares a real variable literally named 'c__dw0', encode must fail loudly
    instead of silently producing a model with duplicate variable entries."""
    spec = WorkloadSpec(
        name="name_collision",
        variables=(Var("c", Categorical(3)), Var("c__dw0", Binary())),
        terms=(),
        contract=TaskContract(()),
    )
    with pytest.raises(ValueError, match="collides"):
        encode(spec)
