"""Task 2: sampling one parameter field. These tests DO invoke the sampler, so
they are slower than the pure-logic tests -- they are kept to the smallest grid
that still exercises the real path."""
import sys

import numpy as np

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.fields import compile_layer, sample_field


def test_symmetric_rule_cancels_the_induced_field():
    """THE defect this whole design exists to avoid. A one-sided clumping rule
    compiles to a uniform field FOUR TIMES the coupling (measured |b|/|J| =
    4.000), which pins cells instead of clumping them, and is why every world
    this project sampled before now was structureless.

    The tolerance is relative and deliberately placed. The cancellation is exact
    in arithmetic, but a spec weight like 0.42 is not binary-representable, so a
    residue of float64 conversion survives -- measured at |b|/|J| = 1.98e-16,
    which is 0.89 of machine epsilon and identical at 8x8, 16x16 and 32x32.
    1e-12 sits twelve orders below the defect it guards against and four orders
    above that noise floor, so it can neither pass a real induced field nor fail
    on representation error."""
    _spec, _enc, ising, _prog = compile_layer(8, 0.42)
    j = float(np.abs(ising.weights).max())
    b = float(np.abs(ising.biases).max())
    assert b / j < 1e-12


def test_symmetric_rule_cancels_exactly_at_a_binary_exact_weight():
    """Proof the cancellation is EXACT in arithmetic rather than merely small:
    at 1.5, which IS representable in binary, the induced field is exactly 0.0
    with no tolerance at all. If this ever fails, the cancellation itself broke
    -- which the tolerance-based test above could not tell you."""
    _spec, _enc, ising, _prog = compile_layer(8, 1.5)
    assert float(np.abs(ising.biases).max()) == 0.0


def test_coupling_matches_the_requested_beta_j():
    """beta is 1.0, so beta*J is |J|. If the encoder's weight->J mapping ever
    changes, this test fails rather than silently sampling the wrong physics."""
    _spec, _enc, ising, _prog = compile_layer(8, 0.42)
    assert float(ising.beta) == 1.0
    assert abs(float(np.abs(ising.weights).max()) - 0.42) < 1e-9


def test_field_has_the_right_shape_and_range():
    f = sample_field(size=8, layers=3, beta_j=0.42, seed=0, warmup=200)
    assert f.shape == (8, 8)
    assert f.min() >= 0
    assert f.max() <= 3


def test_field_is_not_constant():
    """A field that came back all one value would pass every shape and range
    check while carrying no information at all."""
    f = sample_field(size=16, layers=3, beta_j=0.42, seed=0, warmup=200)
    assert len(np.unique(f)) > 1


def test_same_seed_reproduces_the_same_field():
    a = sample_field(size=8, layers=3, beta_j=0.42, seed=7, warmup=200)
    b = sample_field(size=8, layers=3, beta_j=0.42, seed=7, warmup=200)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_fields():
    a = sample_field(size=16, layers=3, beta_j=0.42, seed=1, warmup=200)
    b = sample_field(size=16, layers=3, beta_j=0.42, seed=2, warmup=200)
    assert not np.array_equal(a, b)
