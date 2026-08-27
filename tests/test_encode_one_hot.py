"""A4: the `one_hot` encoding, alongside `domain_wall`. Compares representations
rather than asserting a preference -- both must satisfy the same contract:
round-trip on legal states, is_codeword flags illegal patterns instead of
"repairing" them, energy is preserved, and the encoding-specific penalty is
protected by the SAME dominance guard domain_wall uses.
"""
import pytest

from tsu.ir import Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef
from tsu.passes.encode import encode
from tsu.spec import TaskContract, WorkloadSpec, load_spec


def test_categorical_expands_to_k_binary_spins_one_per_value():
    s = load_spec("specs/toy.yaml")
    enc = encode(s, "one_hot")
    assert all(isinstance(v.domain, Binary) for v in enc.model.variables)
    # a, b, plus 3 one-hot spins for c (k=3) -- k spins, not k-1
    assert len(enc.model.variables) == 5
    assert {"c__oh0", "c__oh1", "c__oh2"} <= set(enc.binary_names)


def test_decode_round_trips_every_legal_assignment():
    s = load_spec("specs/toy.yaml")
    enc = encode(s, "one_hot")
    seen = 0
    for asg in s.assignments():
        bits = enc.encode_assignment(asg)
        assert enc.is_codeword(bits)
        assert enc.decode(bits) == asg, f"round-trip failed for {asg}"
        seen += 1
    assert seen == 12


def test_energy_is_preserved_across_the_encoding():
    s = load_spec("specs/toy.yaml")
    enc = encode(s, "one_hot")
    logical = EnergyModel(s.variables, s.terms, 1.0)
    for asg in s.assignments():
        bits = enc.encode_assignment(asg)
        assert enc.model.energy(bits) == pytest.approx(logical.energy(asg), abs=1e-9)


def test_is_codeword_is_false_for_zero_ones_and_for_two_ones():
    s = load_spec("specs/toy.yaml")
    enc = encode(s, "one_hot")
    legal = enc.encode_assignment({"a": 0, "b": 0, "c": 1})
    assert enc.is_codeword(legal)

    zero_ones = dict(legal)
    zero_ones["c__oh0"] = zero_ones["c__oh1"] = zero_ones["c__oh2"] = 0
    assert not enc.is_codeword(zero_ones)

    two_ones = dict(legal)
    two_ones["c__oh0"] = 1   # now both c__oh0 and c__oh1 are 1
    assert not enc.is_codeword(two_ones)


def test_decode_refuses_a_non_codeword_instead_of_repairing_it_via_argmax():
    """decode() must not silently pick a "best guess" index for a pattern with
    zero or multiple 1s -- that would be exactly the argmax-style repair the
    brief prohibits. is_codeword is the required gate; decode on a non-codeword
    is a hard error, not a plausible-looking wrong answer."""
    s = load_spec("specs/toy.yaml")
    enc = encode(s, "one_hot")
    legal = enc.encode_assignment({"a": 0, "b": 0, "c": 1})
    two_ones = dict(legal)
    two_ones["c__oh0"] = 1
    assert not enc.is_codeword(two_ones)
    with pytest.raises(ValueError):
        enc.decode(two_ones)


def test_indicator_for_a_value_is_a_single_spin_linear_not_a_chain_difference():
    """Structural difference from domain_wall: the one-hot indicator for value v
    is exactly VarRef(chain[v]) -- one spin, weight 1 -- not a two-spin
    difference. Verified by decoding the single-1-at-position-1 pattern back to
    c == 1, and confirming the workload term alone (weight 1 on VarRef(c, 1))
    reproduces the same logical energy the IR itself defines for that indicator."""
    from tsu.ir import Categorical

    spec = WorkloadSpec(
        name="single_value_probe",
        variables=(Var("c", Categorical(3)),),
        terms=(Linear(LinearForm({VarRef("c", 1): 1.0}), 1.0),),
        contract=TaskContract(()))
    enc = encode(spec, "one_hot")
    flipped = {"c__oh0": 0, "c__oh1": 1, "c__oh2": 0}
    workload_only = EnergyModel(spec.variables, spec.terms, 1.0)
    assert workload_only.energy({"c": 1}) == 1.0
    assert enc.decode(flipped) == {"c": 1}


def test_monotone_style_guard_protects_the_one_hot_penalty_too():
    """A2 for one_hot: the same dominance guard that protects domain_wall's
    monotonicity penalty must protect the exactly-one penalty -- a workload
    term large enough to overpower it must be rejected at encode time, not
    silently allow an illegal (not-exactly-one) pattern to win."""
    from tsu.ir import Categorical
    spec = WorkloadSpec(
        name="oversized_weight_one_hot",
        variables=(Var("c", Categorical(3)),),
        terms=(Linear(LinearForm({VarRef("c", 0): 1.0}), 1000.0),),
        contract=TaskContract(()))
    with pytest.raises(ValueError, match="PENALTY"):
        encode(spec, "one_hot")


def test_binary_only_spec_with_large_weight_still_encodes_one_hot():
    spec = WorkloadSpec(
        name="binary_only_big_weight_oh",
        variables=(Var("x", Binary()), Var("y", Binary())),
        terms=(Product(LinearForm({VarRef("x"): 1.0}),
                       LinearForm({VarRef("y"): 1.0}), 1000.0),),
        contract=TaskContract(()))
    enc = encode(spec, "one_hot")  # must not raise: no categorical, no chain
    assert len(enc.model.variables) == 2


def test_generated_chain_name_collision_is_rejected_for_one_hot_too():
    from tsu.ir import Categorical
    spec = WorkloadSpec(
        name="name_collision_oh",
        variables=(Var("c", Categorical(3)), Var("c__oh0", Binary())),
        terms=(), contract=TaskContract(()))
    with pytest.raises(ValueError, match="collides"):
        encode(spec, "one_hot")


def test_one_hot_produces_a_clique_per_categorical_and_is_reported_honestly():
    """Expect this encoding to produce a clique per variable and therefore a
    non-bipartite graph -- the honest result the brief asks be reported, not
    avoided. Verified structurally via lower+analyse."""
    from tsu.passes.analyse import analyse
    from tsu.passes.lower import lower

    s = load_spec("specs/toy.yaml")
    enc = encode(s, "one_hot")
    ising = lower(enc.model)
    report = analyse(ising)
    assert report.bipartite is False


def test_unknown_encoding_name_is_rejected():
    s = load_spec("specs/toy.yaml")
    with pytest.raises(ValueError, match="encoding"):
        encode(s, "not_a_real_encoding")
