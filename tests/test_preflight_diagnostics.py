"""Task 3 rework: this module now keeps only R-hat and the ESS-based standard
error. Integrated autocorrelation time and effective sample size come from
tsu_compiler.ess (see src/tsu_compiler/preflight/diagnostics.py's module docstring for why --
that module has a stronger, cross-validated estimator with a reliability
floor this one never had). Each remaining estimator here is still checked
against a value known ANALYTICALLY rather than against our own output, plus
one test that pins the tau convention tsu_compiler.ess publishes, since a silent
switch there would corrupt every error bar this module computes without ever
looking wrong.
"""
import numpy as np
import pytest

from tsu_compiler.ess import effective_sample_size, integrated_autocorrelation_time
from tsu_compiler.preflight.diagnostics import RHAT_THRESHOLD, r_hat, stderr_from_ess


def ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    """A first-order autoregressive series. Under tsu_compiler.ess's tau_A convention
    (tau_A = 1 + 2*sum_k rho_k), its integrated autocorrelation time is known
    in closed form: tau_A = (1 + phi) / (1 - phi)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------------------
# The convention this module's stderr_from_ess relies on tsu_compiler.ess to publish.
# tsu_compiler.ess and this module's old, now-deleted tau disagreed by exactly 2x, so
# this is pinned rather than assumed.
# ---------------------------------------------------------------------------

def test_the_ess_module_uses_the_tau_A_convention_this_module_relies_on():
    """tsu_compiler.ess reports tau_A = 1 + 2*sum(rho), NOT tau_A/2. An AR(1) series has
    tau_A = (1+phi)/(1-phi) in closed form. If this ever silently switched to the
    half convention, every error bar derived from ESS would be wrong by a factor
    of two while still looking entirely plausible."""
    phi, n = 0.8, 200_000
    rng = np.random.default_rng(11)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    tau = integrated_autocorrelation_time(x[None, :]).tau
    assert tau == pytest.approx((1 + phi) / (1 - phi), rel=0.15)
    assert tau > 5.0   # the half convention would give 4.5 and fail here


# ---------------------------------------------------------------------------
# stderr_from_ess: this module's one remaining tau/ESS-adjacent estimator,
# now taking ESS directly so no convention can be misread at a call site.
# ---------------------------------------------------------------------------

def test_stderr_from_ess_matches_the_closed_form():
    x = np.random.default_rng(2).standard_normal(1000)
    ess = 250.0
    expected = np.std(x, ddof=1) / np.sqrt(ess)
    assert stderr_from_ess(x, ess) == pytest.approx(expected)


def test_stderr_from_ess_shrinks_as_ess_grows():
    """Less autocorrelation -> higher ESS -> a tighter (not wider) error bar
    for the same underlying spread of x."""
    x = np.random.default_rng(3).standard_normal(2000)
    wide = stderr_from_ess(x, 50.0)
    narrow = stderr_from_ess(x, 2000.0)
    assert narrow < wide


# ---------------------------------------------------------------------------
# The refusal this whole design depends on: effective_sample_size must never
# hand a plausible-looking number to stderr_from_ess for a chain too short to
# support one. That refusal is a spec requirement; pin it here since this
# module's caller-facing contract depends on it even though it never calls
# effective_sample_size itself.
# ---------------------------------------------------------------------------

def test_effective_sample_size_refuses_a_chain_too_short_to_support_an_estimate():
    phi = 0.9   # tau_analytic = 19 under the tau_A convention
    x = ar1(phi, 500, seed=5)[None, :]     # N/tau ~ 26, far below the floor
    r = effective_sample_size(x)
    assert r.ess is None
    assert r.reason  # non-empty: a caller must have something to report
                      # instead of a number before ever reaching stderr_from_ess


# ---------------------------------------------------------------------------
# r_hat -- unchanged by this rework; tsu_compiler.ess has no Gelman-Rubin analogue.
# ---------------------------------------------------------------------------

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
