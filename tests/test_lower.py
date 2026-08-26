import itertools
import numpy as np
import pytest

from tsu.ir import Binary, EnergyModel, LinearForm, Linear, Product, Var, VarRef
from tsu.passes.lower import lower, ThreeBodyError


def binary_model(terms, names=("a", "b")):
    return EnergyModel(tuple(Var(n, Binary()) for n in names), tuple(terms), 1.0)


def brute_energy(model, assignment):
    return model.energy(assignment)


def ising_energy(im, assignment):
    """Reconstruct E(x) from the lowered Ising form, in the SPEC's convention."""
    s = {n: 2 * assignment[n] - 1 for n in im.nodes}
    total = im.offset
    for i, n in enumerate(im.nodes):
        total += -im.biases[i] * s[n]
    for k, (u, v) in enumerate(im.edges):
        total += -im.weights[k] * s[im.nodes[u]] * s[im.nodes[v]]
    return total


def test_lowering_preserves_energy_on_every_assignment():
    """The sign flip is mandatory: sum b s + sum J s s must equal -E(x)."""
    m = binary_model([
        Product(LinearForm({VarRef("a"): 1.0}), LinearForm({VarRef("b"): 1.0}), 2.0),
        Linear(LinearForm({VarRef("a"): 1.0, VarRef("b"): 1.0}), -0.5),
    ])
    im = lower(m)
    for va, vb in itertools.product((0, 1), repeat=2):
        asg = {"a": va, "b": vb}
        assert ising_energy(im, asg) == pytest.approx(brute_energy(m, asg), abs=1e-12)


def test_repulsion_lowers_to_a_positive_J_under_thrml_convention():
    m = binary_model([Product(LinearForm({VarRef("a"): 1.0}),
                              LinearForm({VarRef("b"): 1.0}), 2.0)])
    im = lower(m)
    assert len(im.edges) == 1
    assert im.weights[0] == pytest.approx(-0.5)


def test_three_body_term_is_rejected_not_silently_dropped():
    """Product of two forms that SHARE no variable is fine; one that produces an
    order-3 monomial must raise."""
    three = Product(
        LinearForm({VarRef("a"): 1.0, VarRef("b"): 1.0}),
        LinearForm({VarRef("c"): 1.0}), 1.0)
    m = EnergyModel(tuple(Var(n, Binary()) for n in ("a", "b", "c")), (three,), 1.0)
    lower(m)  # order 2 only -> fine

    class Cubic:
        weight = 1.0
        def refs(self):
            return (VarRef("a"), VarRef("b"), VarRef("c"))
        def sympy_expr(self, sym):
            return sym["a"] * sym["b"] * sym["c"]

    m2 = EnergyModel(tuple(Var(n, Binary()) for n in ("a", "b", "c")), (Cubic(),), 1.0)
    with pytest.raises(ThreeBodyError):
        lower(m2)


def test_categorical_reaching_lower_is_an_error():
    from tsu.ir import Categorical
    m = EnergyModel((Var("c", Categorical(3)),), (), 1.0)
    with pytest.raises(ValueError, match="encode"):
        lower(m)
