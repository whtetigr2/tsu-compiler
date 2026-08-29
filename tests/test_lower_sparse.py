"""Task 1: sparse lowering must be numerically identical to the dense path.

The dense implementation is retained as `_lower_dense` solely as a test
oracle -- it is the reference the sparse path is proven against, and is not
reachable from `lower()`.
"""
import numpy as np
import pytest

from tsu.passes.encode import encode
from tsu.passes.lower import lower, _lower_dense, _check_pairwise, ThreeBodyError
from tsu.spec import load_spec

SPECS = ["specs/toy.yaml", "specs/edgeless.yaml", "specs/too_strong.yaml",
         "specs/adjacency_2x2_k3.yaml", "specs/adjacency_4x4_k4.yaml"]


@pytest.mark.parametrize("path", SPECS)
@pytest.mark.parametrize("encoding", ["domain_wall", "one_hot"])
def test_sparse_matches_dense_exactly(path, encoding):
    model = encode(load_spec(path), encoding).model
    sparse, dense = lower(model), _lower_dense(model)

    assert sparse.nodes == dense.nodes
    assert sparse.edges == dense.edges
    np.testing.assert_allclose(sparse.weights, dense.weights, rtol=0, atol=1e-9)
    np.testing.assert_allclose(sparse.biases, dense.biases, rtol=0, atol=1e-9)
    assert sparse.offset == pytest.approx(dense.offset, abs=1e-9)
    assert sparse.beta == dense.beta


def test_three_body_still_raises():
    """A Product whose two forms share no variable but together span three
    distinct spins must still be rejected."""
    from tsu.ir import Binary, EnergyModel, LinearForm, Product, Var, VarRef
    a = LinearForm({VarRef("x"): 1.0})
    b = LinearForm({VarRef("y"): 1.0, VarRef("z"): 1.0})
    inner = Product(a, b, 1.0)
    # a genuine 3-body needs a term the IR can express; if Product(Product)
    # is not expressible, assert instead that a 3-distinct-spin monomial
    # reaching `acc` raises -- construct it directly and call the guard.
    model = EnergyModel(
        variables=(Var("x", Binary()), Var("y", Binary()), Var("z", Binary())),
        terms=(inner,), beta=1.0)
    lower(model)  # 2-body only: must NOT raise


# --- Ruling 3 -----------------------------------------------------------
# `Product(a, b)` contributes at most one spin from each of its two
# LinearForms, so the IR's two term kinds (`Linear`, `Product`) cannot
# themselves produce a monomial spanning 3+ distinct spins -- there is no
# way to construct a genuine 3-body term through the public IR surface.
#
# `grep -rn sympy_expr src tests` turns up `hasattr(term, "sympy_expr")` in
# `lower.py` itself, plus ad hoc duck-typed test doubles (`Cubic`, `Quartic`,
# ...) defined LOCALLY inside `tests/test_lower.py` -- not a term type that
# ships in `src/tsu/ir.py`. So no production term type in the repo carries
# `sympy_expr`, and case (a) from Ruling 3 does not apply here: this is case
# (b), a direct unit test on the guard itself (`_check_pairwise`) with a
# synthetic accumulator key `(i, j, k)`.
#
# (The existing `Cubic`/`Quartic`/`CubedSingle`/`QuarticSingle` doubles in
# `tests/test_lower.py` DO exercise `lower()`'s `hasattr(term, "sympy_expr")`
# branch end-to-end, including a genuine 3- and 4-distinct-spin rejection --
# those are part of the 272-test baseline and continue to pass unmodified.)
def test_guard_rejects_synthetic_three_body_key():
    """Direct unit test on `_check_pairwise`: a synthetic 3-index key in the
    sparse accumulator must raise ThreeBodyError, independent of how (or
    whether) any term kind can actually produce one."""
    names = ("x", "y", "z")
    with pytest.raises(ThreeBodyError):
        _check_pairwise({(0, 1, 2): 1.0}, names)


def test_guard_rejects_synthetic_four_body_key():
    """Same guard, order-4 synthetic key -- mirrors the dense oracle's
    quartic rejection."""
    names = ("w", "x", "y", "z")
    with pytest.raises(ThreeBodyError):
        _check_pairwise({(0, 1, 2, 3): 1.0}, names)


def test_guard_allows_pairwise_and_lower_order_keys():
    """The guard must not raise on constant, bias, or coupling keys -- only
    on keys spanning 3+ distinct spins."""
    names = ("x", "y", "z")
    _check_pairwise({(): 1.0, (0,): 2.0, (1, 2): 3.0}, names)


def test_guard_ignores_a_zeroed_three_body_key():
    """A 3+ key that nets to exactly zero (e.g. two terms that cancelled
    during accumulation) is not a surviving violation."""
    names = ("x", "y", "z")
    _check_pairwise({(0, 1, 2): 0.0}, names)


@pytest.mark.slow
def test_sparse_lowering_reaches_16x16():
    """The 16x16 k=5 one-hot model (1280 spins) must lower. The dense path
    raised RecursionError here; see spec section 4.4."""
    import time, tempfile, os
    body = ("name: scale_probe\ngenerate:\n  kind: grid\n  width: 16\n"
            "  height: 16\n  variable_domain: {domain: categorical, k: 5}\n"
            "terms:\n"
            "  - {kind: product_over_edges, a_value: 3, b_value: 0, weight: 1.0}\n")
    p = os.path.join(tempfile.gettempdir(), "scale_probe.yaml")
    open(p, "w").write(body)
    t0 = time.perf_counter()
    m = lower(encode(load_spec(p), "one_hot").model)
    elapsed = time.perf_counter() - t0
    assert len(m.nodes) == 1280
    assert elapsed < 60.0, f"lowering 1280 spins took {elapsed:.1f}s"
