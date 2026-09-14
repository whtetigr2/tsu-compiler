import pytest
from tsu_compiler.ir import Binary, Categorical, Var, VarRef, LinearForm, Linear, Product, EnergyModel


def toy_model():
    a = Var("a", Binary())
    b = Var("b", Binary())
    c = Var("c", Categorical(3))
    return EnergyModel(
        variables=(a, b, c),
        terms=(
            Product(LinearForm({VarRef("a"): 1.0}), LinearForm({VarRef("b"): 1.0}), 2.0),
            Linear(LinearForm({VarRef("a"): 1.0, VarRef("b"): 1.0}), -0.5),
            Product(LinearForm({VarRef("c", 0): 1.0}), LinearForm({VarRef("a"): 1.0}), 1.5),
        ),
        beta=1.0,
    )


def test_energy_of_a_hand_computed_assignment():
    """a=1,b=1,c=0 -> repulsion 2.0, preference -1.0, forbid 1.5 = 2.5"""
    m = toy_model()
    assert m.energy({"a": 1, "b": 1, "c": 0}) == pytest.approx(2.5)


def test_energy_zero_state():
    m = toy_model()
    assert m.energy({"a": 0, "b": 0, "c": 1}) == pytest.approx(0.0)


def test_categorical_indicator_is_one_only_for_its_value():
    m = toy_model()
    # a=1, b=0, c=0 -> preference -0.5, forbid 1.5
    assert m.energy({"a": 1, "b": 0, "c": 0}) == pytest.approx(1.0)
    # a=1, b=0, c=2 -> preference -0.5 only
    assert m.energy({"a": 1, "b": 0, "c": 2}) == pytest.approx(-0.5)


def test_linear_form_const_contributes():
    f = LinearForm({VarRef("a"): 2.0}, const=1.0)
    assert f.evaluate({"a": 1}) == pytest.approx(3.0)
    assert f.evaluate({"a": 0}) == pytest.approx(1.0)


def test_terms_are_frozen():
    with pytest.raises(Exception):
        Linear(LinearForm({VarRef("a"): 1.0}), 1.0).weight = 5.0
