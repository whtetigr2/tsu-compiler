import itertools
import numpy as np
import pytest

from tsu.ir import Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef
from tsu.passes.lower import lower
from tsu.passes.analyse import analyse
from tsu.passes.program import build_program
from tsu.backends.thrml_backend import exact_distribution, sample
from tsu.backends.torx_backend import torx_cross_check


def two_spin_program(w=2.0, bias_a=0.7):
    """I3: `bias_a` (nonzero by default) makes p(0,1) != p(1,0), so a test
    comparing per-state probabilities can actually detect a state-ordering bug
    -- a symmetric model (no bias) makes such a bug invisible, which is exactly
    why the MSB/LSB mismatch this fixture regression-tests survived once already
    (see `_verify`'s own comment in search.py)."""
    m = EnergyModel((Var("a", Binary()), Var("b", Binary())),
                    (Product(LinearForm({VarRef("a"): 1.0}),
                             LinearForm({VarRef("b"): 1.0}), w),
                     Linear(LinearForm({VarRef("a"): 1.0}), bias_a)), 1.0)
    im = lower(m)
    return build_program(im, analyse(im)), m


def test_exact_distribution_matches_a_hand_computed_boltzmann():
    prog, m = two_spin_program()
    states, probs = exact_distribution(prog)
    want = np.array([np.exp(-m.energy(dict(zip(("a", "b"), s)))) for s in states])
    want /= want.sum()
    assert np.abs(probs - want).max() == pytest.approx(0.0, abs=1e-10)


def test_sampling_reproduces_the_exact_distribution_within_noise():
    """I3: the histogram index MUST be derived from `exact_distribution`'s own
    `states` array, exactly as `_verify` does in search.py -- never a hand-rolled
    `(got * (1 << arange(n))).sum(axis=1))` LSB-first index, which silently
    assumes a specific bit order that `exact_distribution` (MSB-first via
    itertools.product) does not use. With the symmetric (no-bias) model this
    fixture used to build, that mismatch was invisible: p(0,1) == p(1,0) made a
    permuted comparison look correct by accident. With `two_spin_program`'s
    now-asymmetric bias, an LSB-first index reproducibly fails this assertion."""
    prog, _ = two_spin_program()
    states, probs = exact_distribution(prog)
    got = sample(prog, n_chains=32, n_samples=400, n_warmup=500,
                 steps_per_sample=2, seed=0)
    state_index = {tuple(int(x) for x in row): i for i, row in enumerate(states)}
    idx = np.array([state_index[tuple(int(x) for x in row)] for row in got])
    hist = np.bincount(idx, minlength=len(probs)).astype(float)
    hist /= hist.sum()
    assert np.abs(hist - probs).max() < 0.05


def test_torx_cross_check_agrees_with_thrml_by_a_DIFFERENT_route():
    """Two independent stacks, used SEPARATELY, never chained.

    The value of this control is that torx reaches the answer by driving PISING to
    stationarity, which is a different computation from thrml's energy softmax. A
    cross-check that recomputes the same formula is not a control.

    I3: this compared `tx` against `probs` element-wise on a SYMMETRIC model
    (no bias), which made a basis-order mismatch between the two routes
    invisible -- p(0,1) == p(1,0) means a permuted comparison looks correct by
    accident. `two_spin_program`'s now-asymmetric bias makes this a real check:
    torx agrees with thrml to within 1e-6 (verified to ~1e-16 in practice) with
    the asymmetric model, which is the actual evidence that the two routes use
    the same state ordering, not merely a coincidence of symmetry.
    """
    prog, _ = two_spin_program()
    _, probs = exact_distribution(prog)
    tx, note = torx_cross_check(prog.ising)
    assert tx is not None, f"torx route unavailable: {note}"
    assert np.abs(tx - probs).max() < 1e-6


def test_torx_cross_check_refuses_multi_edge_models_with_a_reason():
    """Composing PISING across edges is a Trotter splitting; saying so beats
    returning a number that cannot be defended."""
    import numpy as _np
    from tsu.passes.lower import IsingModel
    three = IsingModel(("a", "b", "c"), ((0, 1), (1, 2)),
                       _np.array([1.0, 1.0]), _np.zeros(3), 1.0, 0.0)
    tx, note = torx_cross_check(three)
    assert tx is None
    assert "Trotter" in note


def test_sampling_refuses_zero_warmup():
    """At n_warmup=0 thrml records BEFORE stepping and the run re-reports its init."""
    prog, _ = two_spin_program()
    with pytest.raises(AssertionError, match="n_warmup"):
        sample(prog, n_chains=4, n_samples=4, n_warmup=0, steps_per_sample=1, seed=0)


def one_spin_no_edges_program(b=1.5):
    """The simplest legal workload: one binary variable, one linear term, no
    products. Lowers to an IsingModel with zero edges."""
    m = EnergyModel((Var("a", Binary()),),
                    (Linear(LinearForm({VarRef("a"): 1.0}), b),), 1.0)
    im = lower(m)
    return build_program(im, analyse(im)), m


def test_exact_distribution_handles_a_zero_edge_model_instead_of_raising():
    """C3: thrml's IsingEBM.factors builds a SpinEBMFactor over the (empty) edge
    block unconditionally, and raises `IndexError: tuple index out of range`
    indexing `group.nodes[0]` on that empty block. Reserved for a model with real
    couplings, exact_distribution must handle the zero-edge case as the closed-
    form product of independent site marginals it actually is."""
    prog, m = one_spin_no_edges_program()
    states, probs = exact_distribution(prog)
    want = np.array([np.exp(-m.energy({"a": s[0]})) for s in states])
    want /= want.sum()
    assert np.abs(probs - want).max() == pytest.approx(0.0, abs=1e-10)


def test_sampling_handles_a_zero_edge_model_instead_of_raising():
    """Same vendor IndexError reaches `sample` via IsingSamplingProgram's own
    construction of ebm.factors -- not only exact_distribution's energy call."""
    prog, _ = one_spin_no_edges_program()
    _, probs = exact_distribution(prog)
    got = sample(prog, n_chains=64, n_samples=200, n_warmup=10,
                 steps_per_sample=1, seed=0)
    hist = np.bincount(got[:, 0], minlength=2).astype(float)
    hist /= hist.sum()
    assert np.abs(hist - probs).max() < 0.05
