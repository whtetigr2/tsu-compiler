"""Task 3: the estimators every reported number depends on.

Each is checked against a value known ANALYTICALLY rather than against our own
output. An estimator tested against itself proves only self-consistency, which is
exactly what a reviewer discounts."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu.preflight.diagnostics import (autocorr_time, n_eff, r_hat,
                                       stderr_corrected, RHAT_THRESHOLD)


def ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    """A first-order autoregressive series, whose integrated autocorrelation
    time is known in closed form: tau = (1 + phi) / (2 * (1 - phi))."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_tau_of_white_noise_is_one_half():
    """For independent draws the autocorrelation sum vanishes and tau -> 1/2 by
    the standard convention tau = 1/2 + sum_k rho_k."""
    x = np.random.default_rng(1).standard_normal(200_000)
    assert autocorr_time(x) == pytest.approx(0.5, abs=0.08)


@pytest.mark.parametrize("phi", [0.5, 0.8, 0.9])
def test_tau_of_an_ar1_series_matches_its_closed_form(phi):
    """tau = (1 + phi) / (2(1 - phi)): 1.5 at phi=0.5, 4.5 at 0.8, 9.5 at 0.9.
    This is the test that says the estimator is right rather than merely
    plausible."""
    expected = (1 + phi) / (2 * (1 - phi))
    got = autocorr_time(ar1(phi, 400_000))
    assert got == pytest.approx(expected, rel=0.15)


def test_tau_grows_with_correlation():
    assert autocorr_time(ar1(0.9, 200_000)) > autocorr_time(ar1(0.5, 200_000))


def test_n_eff_halves_the_count_for_independent_draws():
    """With tau = 1/2, N_eff = N / (2 * 1/2) = N."""
    assert n_eff(1000, 0.5) == pytest.approx(1000.0)
    assert n_eff(1000, 5.0) == pytest.approx(100.0)


def test_n_eff_never_drops_below_one():
    """A tau larger than the run length means the chain never decorrelated. The
    floor keeps downstream error bars finite, and the caller is expected to flag
    the row rather than trust it."""
    assert n_eff(10, 10_000.0) == pytest.approx(1.0)


def test_corrected_stderr_is_wider_than_the_naive_one():
    """THE POINT OF THE WHOLE MODULE. Correlated samples carry less information
    than their count suggests, so the naive standard error is too small -- which
    is how every spread this project reported before now was too narrow."""
    x = ar1(0.9, 50_000)
    naive = float(np.std(x, ddof=1) / np.sqrt(x.size))
    corrected = stderr_corrected(x, autocorr_time(x))
    assert corrected > naive * 3.0


def test_r_hat_of_identical_chains_is_the_bda3_closed_form():
    """Gelman-Rubin uses var_plus = ((n-1)/n)W + B/n, so chains with zero
    between-chain disagreement give R-hat = sqrt((n-1)/n) EXACTLY, a value
    slightly below 1. That is the published estimator's behaviour and not a
    defect: asserting 1.0 here (as this project's brief originally did) would
    force the discount term out and yield a statistic that is not R-hat, while
    RHAT_THRESHOLD is calibrated for the one that is."""
    n = 500
    chains = np.tile(np.random.default_rng(3).standard_normal(n), (4, 1))
    assert r_hat(chains) == pytest.approx(np.sqrt((n - 1) / n), abs=1e-12)
    assert r_hat(chains) < 1.0


def test_rhat_of_independent_chains_from_one_distribution_is_near_one():
    c = np.random.default_rng(3).standard_normal((8, 4000))
    assert r_hat(c) < RHAT_THRESHOLD


def test_rhat_detects_chains_stuck_in_different_places():
    """The failure R-hat exists to catch: each chain is internally converged and
    they disagree with each other. A per-chain average would hide it entirely."""
    rng = np.random.default_rng(4)
    c = np.stack([rng.standard_normal(2000) + off
                  for off in (-5.0, -2.0, 2.0, 5.0)])
    assert r_hat(c) > 1.5


def test_rhat_flags_a_small_but_genuine_offset_above_threshold():
    """The BDA3 discount that keeps identical chains near (not exactly) 1 must
    not also mask a real, if modest, disagreement between chains -- an offset
    small enough to be easy to miss by eye still has to clear the threshold.
    Offsets of +-0.1/+-0.3 (a fraction of the chains' own unit spread) were
    checked to clear RHAT_THRESHOLD for every seed in range(200), not just
    this one -- a margin this thin from a single lucky draw would test noise,
    not sensitivity."""
    rng = np.random.default_rng(5)
    n = 500
    c = np.stack([rng.standard_normal(n) + off
                  for off in (-0.3, -0.1, 0.1, 0.3)])
    assert r_hat(c) > RHAT_THRESHOLD


def test_rhat_needs_at_least_two_chains():
    with pytest.raises(ValueError, match="two chains"):
        r_hat(np.zeros((1, 100)))
