import itertools
import math
import numpy as np
import pytest

from tsu_compiler.ir import Binary, EnergyModel, LinearForm, Linear, Product, Var, VarRef
from tsu_compiler.passes.lower import lower, ThreeBodyError


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
    order-3 monomial across 3 DISTINCT spins must raise."""
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


def test_four_distinct_spin_product_is_rejected_as_pairwise_violation():
    """A genuine order-4 term across 4 DISTINCT spins (a*b*c*d) is irreducible --
    no amount of s**2 == 1 folding collapses it -- and must still raise, same as
    the order-3 case."""
    class Quartic:
        weight = 1.0
        def refs(self):
            return (VarRef("a"), VarRef("b"), VarRef("c"), VarRef("d"))
        def sympy_expr(self, sym):
            return sym["a"] * sym["b"] * sym["c"] * sym["d"]

    m = EnergyModel(tuple(Var(n, Binary()) for n in ("a", "b", "c", "d")),
                     (Quartic(),), 1.0)
    with pytest.raises(ThreeBodyError):
        lower(m)


def test_categorical_reaching_lower_is_an_error():
    from tsu_compiler.ir import Categorical
    m = EnergyModel((Var("c", Categorical(3)),), (), 1.0)
    with pytest.raises(ValueError, match="encode"):
        lower(m)


def test_categorical_indicator_ref_reaching_lower_is_an_error():
    """The top-level isinstance(v.domain, Binary) guard only catches a variable
    DECLARED categorical. `_term_expr`'s `ref.value is not None` branch is the
    one that catches a categorical-indicator VarRef (name, value) surviving
    inside a term even when every declared variable is Binary -- exercise that
    branch directly."""
    m = EnergyModel(
        (Var("a", Binary()),),
        (Linear(LinearForm({VarRef("a", 0): 1.0}), 1.0),),
        1.0)
    with pytest.raises(ValueError, match="encode"):
        lower(m)


def test_single_spin_cubed_lowers_successfully_not_rejected():
    """For a binary spin, s**3 == s: a high power of a SINGLE spin is legitimate
    algebra (it reduces to a degree-1 bias term), not a pairwise violation. This
    replaces a prior (incorrect) version of this test that asserted ThreeBodyError
    here -- that locked in a false positive: s_a**3 is not a surviving order-3
    term, it genuinely equals s_a. The values below (biases=[-0.5, 0.0],
    offset=0.5) were verified by independent re-derivation, and the energy
    reconstruction is checked on every assignment, not just the coefficients."""
    class CubedSingle:
        weight = 1.0
        def refs(self):
            return (VarRef("a"),)
        def sympy_expr(self, sym):
            return sym["a"] ** 3
        def evaluate(self, assignment):
            # ground truth: occupancy(a)**3, matching sympy_expr above -- for a
            # binary occupancy in {0, 1} this equals occupancy(a) exactly, which
            # is exactly the identity the fix relies on.
            return self.weight * (float(assignment["a"]) ** 3)

    m = EnergyModel(tuple(Var(n, Binary()) for n in ("a", "b")), (CubedSingle(),), 1.0)
    im = lower(m)
    assert len(im.edges) == 0
    assert im.biases[im.nodes.index("a")] == pytest.approx(-0.5)
    assert im.biases[im.nodes.index("b")] == pytest.approx(0.0)
    assert im.offset == pytest.approx(0.5)
    for va, vb in itertools.product((0, 1), repeat=2):
        asg = {"a": va, "b": vb}
        assert ising_energy(im, asg) == pytest.approx(brute_energy(m, asg), abs=1e-12)


def test_single_spin_fourth_power_lowers_with_no_couplings():
    """s**4 == 1 for a binary spin: it folds entirely into the constant offset,
    contributing no bias and no coupling. Also not a pairwise violation."""
    class QuarticSingle:
        weight = 1.0
        def refs(self):
            return (VarRef("a"),)
        def sympy_expr(self, sym):
            return sym["a"] ** 4
        def evaluate(self, assignment):
            # ground truth: occupancy(a)**4, matching sympy_expr above -- for a
            # binary occupancy in {0, 1} this equals occupancy(a) exactly.
            return self.weight * (float(assignment["a"]) ** 4)

    m = EnergyModel(tuple(Var(n, Binary()) for n in ("a", "b")), (QuarticSingle(),), 1.0)
    im = lower(m)
    assert len(im.edges) == 0
    for va, vb in itertools.product((0, 1), repeat=2):
        asg = {"a": va, "b": vb}
        assert ising_energy(im, asg) == pytest.approx(brute_energy(m, asg), abs=1e-12)


# ---------------------------------------------------------------------------
# N-1 (R14 / F-R14): lower() validated neither term weights nor beta, so
# NaN/inf could propagate silently into a "successfully compiled" IsingModel
# -- the audit's ONLY compute-layer defect (everything else lived in the
# presentation layer). Mirrors passes/encode.py's validate_coefficient_scale,
# which already fixed the sibling hole for coefficient_scale and documents
# WHY `not (x > 0)` (or here, `not math.isfinite(x)`) is required instead of
# a bare `<=`/`==` comparison: `float('nan') <= 0` and `float('nan') == 0`
# are both `False` in Python, so a comparison-shaped guard lets NaN straight
# through.
#
# Production change that would make each test below fail: deleting the
# `_require_finite` call it exercises from `lower()`. Confirmed directly
# (not just argued) by commenting out each guard and rerunning -- see the
# fix-round report; both regressed to "no exception raised" (a NaN/inf
# biases array built successfully) rather than merely a different message,
# which is the exact silent-garbage-still-looks-compiled failure mode this
# guards against.
# ---------------------------------------------------------------------------

def test_nan_term_weight_is_rejected_not_silently_compiled():
    m = binary_model([Linear(LinearForm({VarRef("a"): 1.0}), math.nan)])
    with pytest.raises(ValueError, match="finite"):
        lower(m)


def test_infinite_term_weight_is_rejected_not_silently_compiled():
    m = binary_model([Product(LinearForm({VarRef("a"): 1.0}),
                              LinearForm({VarRef("b"): 1.0}), math.inf)])
    with pytest.raises(ValueError, match="finite"):
        lower(m)


def test_nan_beta_is_rejected_not_silently_compiled():
    """`EnergyModel.beta` unguarded: a NaN beta must raise here, not ride
    silently into IsingModel.beta and every downstream sampler call."""
    m = EnergyModel(tuple(Var(n, Binary()) for n in ("a", "b")), (), math.nan)
    with pytest.raises(ValueError, match="finite"):
        lower(m)


def test_infinite_beta_is_rejected_not_silently_compiled():
    m = EnergyModel(tuple(Var(n, Binary()) for n in ("a", "b")), (), math.inf)
    with pytest.raises(ValueError, match="finite"):
        lower(m)


def test_finite_weight_and_beta_still_compile_normally():
    """Sanity: the new guard must not reject ordinary finite models --
    otherwise every test above this one in the file would already have
    caught it, but this pins it explicitly against the exact boundary
    values (0.0, negative) a naive `not x` or `not (x > 0)`-style guard
    could wrongly reject."""
    m = binary_model([Linear(LinearForm({VarRef("a"): 1.0}), 0.0),
                      Product(LinearForm({VarRef("a"): 1.0}),
                              LinearForm({VarRef("b"): 1.0}), -3.5)])
    im = lower(EnergyModel(m.variables, m.terms, 0.5))
    assert im.beta == 0.5
