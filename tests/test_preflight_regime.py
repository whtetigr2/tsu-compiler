"""Task 4: locating the transition by finite-size scaling.

The Binder cumulant's two limits are known exactly, so it is checked against
them rather than against our own sampler."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

from tsu.preflight.sweep import (RegimeRow, binder, crossing, susceptibility,
                                 usable_band, sweep, SATURATION)


def test_binder_of_a_gaussian_order_parameter_is_zero():
    """Deep in the disordered phase m is Gaussian about zero, where
    <m^4> = 3<m^2>^2 exactly, so U = 1 - 3/3 = 0."""
    m = np.random.default_rng(0).standard_normal(400_000)
    assert binder(m) == pytest.approx(0.0, abs=0.01)


def test_binder_of_a_two_delta_order_parameter_is_two_thirds():
    """Deep in the ordered phase m sits at +-m0, so <m^4> = m0^4 and
    <m^2> = m0^2, giving U = 1 - 1/3 = 2/3. These two limits are why the
    cumulant locates a transition without knowing anything about the model."""
    m = np.where(np.random.default_rng(1).random(400_000) < 0.5, -0.7, 0.7)
    assert binder(m) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_binder_of_a_constant_is_two_thirds():
    assert binder(np.full(1000, 0.4)) == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_binder_of_an_all_zero_order_parameter_is_defined():
    """A degenerate sample must not divide by zero -- it should report the
    disordered limit rather than a NaN that propagates into the band search."""
    assert binder(np.zeros(100)) == pytest.approx(0.0)


def test_susceptibility_of_a_gaussian_matches_its_closed_form():
    """chi = N(<m^2> - <|m|>^2). For m ~ Normal(0, sigma), <m^2> = sigma^2 and
    <|m|> = sigma*sqrt(2/pi), so chi = N*sigma^2*(1 - 2/pi) EXACTLY -- about
    0.3634*N*sigma^2. Pinned against that closed form rather than asserted to be
    merely 'bigger when the spread is bigger': an inequality that follows
    algebraically from the definition cannot fail, and proportionality to N is
    likewise true by construction, so neither would catch a wrong estimator."""
    n_spins, sigma = 100, 0.2
    m = np.random.default_rng(2).standard_normal(400_000) * sigma
    expected = n_spins * sigma ** 2 * (1.0 - 2.0 / np.pi)
    assert susceptibility(m, n_spins) == pytest.approx(expected, rel=0.02)


def _rows(size, us, couplings):
    return [RegimeRow(beta_j=b, size=size, abs_m=0.0, abs_m_err=0.0, chi=0.0,
                      binder=u, tau=1.0, n_eff=100.0, r_hat=1.0,
                      ess_reason="ok", provisional=False)
            for b, u in zip(couplings, us)]


def test_crossing_finds_where_two_binder_curves_meet():
    """Curves for different sizes cross at the critical coupling: below it the
    larger system is MORE disordered, above it more ordered. The crossing is the
    measurement -- 'where |m| looks like it jumps' is not."""
    cs = [0.1, 0.2, 0.3, 0.4]
    small = _rows(8, [0.10, 0.20, 0.40, 0.60], cs)
    large = _rows(16, [0.02, 0.15, 0.50, 0.64], cs)
    x = crossing(small, large)
    # The difference small-large is [+0.08, +0.05, -0.10, -0.04], so it flips
    # between 0.2 and 0.3 and linear interpolation gives
    # 0.2 + 0.1*0.05/(0.05+0.10) = 0.2 + 0.1/3 exactly. Asserting a range here
    # would pass for any implementation that merely lands in the right cell.
    assert x == pytest.approx(0.2 + 0.1 / 3.0, abs=1e-12)


def test_crossing_returns_none_when_the_curves_never_meet():
    """Reporting a transition that was not observed would be worse than
    reporting none, so a sweep that never brackets one says so."""
    cs = [0.1, 0.2, 0.3]
    assert crossing(_rows(8, [0.1, 0.2, 0.3], cs),
                    _rows(16, [0.05, 0.15, 0.25], cs)) is None


def test_usable_band_starts_at_ordering_and_ends_at_saturation():
    """Below the band the model is disordered; above it one state swallows the
    system, which is as useless as noise."""
    cs = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = [RegimeRow(beta_j=b, size=16, abs_m=m, abs_m_err=0.01, chi=1.0,
                      binder=u, tau=1.0, n_eff=100.0, r_hat=1.0,
                      ess_reason="ok", provisional=False)
            for b, m, u in zip(cs, [0.05, 0.12, 0.40, 0.80, 0.97],
                               [0.02, 0.10, 0.35, 0.58, 0.66])]
    lo, hi = usable_band(rows)
    assert lo < hi
    assert hi <= 0.5
    assert all(r.abs_m < SATURATION for r in rows if lo <= r.beta_j <= hi)


def test_a_provisional_row_is_excluded_from_the_band():
    """A row whose chains disagree must not silently set the band's edge."""
    cs = [0.1, 0.2, 0.3]
    rows = [RegimeRow(beta_j=b, size=16, abs_m=m, abs_m_err=0.01, chi=1.0,
                      binder=u, tau=1.0, n_eff=100.0, r_hat=rh,
                      ess_reason="ok", provisional=rh > 1.01)
            for b, m, u, rh in zip(cs, [0.05, 0.40, 0.80],
                                   [0.02, 0.35, 0.58], [1.0, 1.9, 1.0])]
    band = usable_band(rows)
    # Asserting this only `if band is not None` would let a usable_band that
    # always returns None pass unconditionally -- the exact vacuous shape this
    # project has now caught 18 times. The band must exist AND exclude the
    # provisional row.
    assert band is not None, "a provisional row must be skipped, not abort the band"
    assert 0.2 not in band


@pytest.mark.slow
def test_sweep_runs_end_to_end_and_carries_the_ess_refusal_through():
    """A real (tiny) sampling run through sweep() itself -- not just the
    synthetic RegimeRow objects the tests above construct by hand -- to
    confirm analyse -> build_program -> sample_chains -> tsu.ess/r_hat are
    actually wired together correctly and produce well-typed rows.

    Makes NO claim about where a transition sits on this toy: a 3x3 lattice
    sampled for 40 draws/chain is far too small/short for that, and asserting
    a location here would be exactly the fabrication this module exists to
    refuse. What IS checked is the refusal itself: n_chains*n_samples = 160 is
    far below tsu.ess's own reliability floor (RELIABILITY_MIN_N_OVER_TAU =
    5000 -- see tsu/ess.py), so effective_sample_size MUST return
    ess=None/iat=None regardless of how well the chain mixes, and sweep() must
    carry that refusal through as None fields and provisional=True rather than
    substituting a raw standard error (trap #2 in this task's brief).
    """
    from world.fields import compile_layer

    def model_fn(size, beta_j):
        _, _, ising, _ = compile_layer(size, beta_j)
        return ising

    rows = sweep(model_fn, sizes=[3], couplings=[0.2, 0.6], seed=0,
                n_chains=4, n_samples=40, n_warmup=100, steps=2)

    assert len(rows) == 2
    for r in rows:
        assert isinstance(r, RegimeRow)
        assert r.size == 3
        assert 0.0 <= r.abs_m <= 1.0
        assert -1e-9 <= r.binder <= 2.0 / 3.0 + 1e-9
        assert r.r_hat > 0.0
        # 4*40 = 160 total draws is nowhere near tsu.ess's reliability floor
        # of 5000, at ANY tau -- this is not a statement about mixing quality.
        assert r.n_eff is None
        assert r.tau is None
        assert r.abs_m_err is None
        assert r.provisional is True
        assert r.ess_reason != ""
