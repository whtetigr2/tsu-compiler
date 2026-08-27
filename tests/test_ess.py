"""Validation of tsu.ess: an estimator is a measurement instrument, and this
file is its known-answer control, its cross-implementation control, and its
must-fail potency control -- see src/tsu/ess.py's module docstring and
sdd/2026-08-26-tsu-compiler-vertical-slice/ess-report.md.

`arviz` and `statsmodels` are TEST-ONLY dependencies (per the task brief);
this is the only place in the repo that should import them.
"""
import numpy as np
import pytest

from tsu.ess import (
    DEFAULT_C,
    RELIABILITY_MIN_N_OVER_TAU,
    EssEstimate,
    autocorrelation,
    effective_sample_size,
    geyer_initial_positive_sequence_tau,
    integrated_autocorrelation_time,
)


def ar1_chains(n, phi, seed, n_chains=1):
    """n_chains independent AR(1) series x_t = phi*x_{t-1} + noise_t, noise_t
    ~ N(0,1). The exact integrated autocorrelation time of an AR(1) process is
    tau = (1+phi)/(1-phi) -- this is the external, analytically-known
    reference every other check in this file is measured against."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_chains, n))
    noise = rng.normal(size=(n_chains, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + noise[:, t]
    return x


# ---------------------------------------------------------------------------
# 1. Known-answer, analytic: AR(1) has an EXACT tau = (1+phi)/(1-phi).
# ---------------------------------------------------------------------------

AR1_CASES = [0.0, 0.5, 0.8, 0.9, 0.95]


@pytest.mark.parametrize("phi", AR1_CASES)
def test_ar1_known_answer_recovers_analytic_tau(phi):
    """Chain length chosen so N/tau ~= 20000, comfortably above this module's
    own RELIABILITY_MIN_N_OVER_TAU=5000 threshold -- this is the reliable
    regime, so the estimate must land close to the exact value. Tolerance is
    20% relative error: the AR(1) sweep behind RELIABILITY_MIN_N_OVER_TAU
    shows worst-of-50-seeds error of ~17% even AT N/tau=5000, and this test
    runs at N/tau~20000 (a single fixed seed, not a 50-seed sweep), so 20%
    leaves real margin without being so loose it stops being a control."""
    tau_analytic = (1 + phi) / (1 - phi) if phi < 1 else float("inf")
    n = max(int(round(20000 * max(tau_analytic, 1.0))), 4000)
    x = ar1_chains(n, phi, seed=42)
    iat = integrated_autocorrelation_time(x)
    rel_err = abs(iat.tau - tau_analytic) / tau_analytic
    assert rel_err < 0.20, (
        f"phi={phi}: analytic tau={tau_analytic:.3f}, estimated={iat.tau:.3f}, "
        f"relative error={rel_err:.3f} exceeds the 20% tolerance")


def test_ar1_known_answer_table_is_stable_across_phi():
    """A single consolidated table (used verbatim in ess-report.md): the
    estimator must not merely pass its per-phi tolerance but show ERROR THAT
    DOES NOT SYSTEMATICALLY GROW with phi at a fixed N/tau ratio -- if it did,
    that would mean the estimator's failure mode is hidden by only ever
    testing at fixed N rather than fixed N/tau."""
    rows = []
    for phi in AR1_CASES:
        tau_analytic = (1 + phi) / (1 - phi) if phi < 1 else float("inf")
        n = max(int(round(20000 * max(tau_analytic, 1.0))), 4000)
        x = ar1_chains(n, phi, seed=42)
        iat = integrated_autocorrelation_time(x)
        rel_err = abs(iat.tau - tau_analytic) / tau_analytic
        rows.append((phi, tau_analytic, iat.tau, rel_err))
    errs = [r[3] for r in rows]
    assert max(errs) < 0.20
    # not a monotonically exploding error as phi -> 1: the spread across phi
    # at a FIXED N/tau ratio should stay small in absolute terms (an additive
    # margin, not a ratio -- the smallest error here is close enough to zero
    # that a pure ratio bound is dominated by estimator noise, not signal).
    assert max(errs) - min(errs) < 0.10


def test_ar1_reports_the_failure_honestly_if_it_ever_occurs():
    """Per the task brief: 'If your estimator does not reproduce the AR(1)
    analytic value, report that rather than tuning tolerances until it
    passes.' This test exists to make that an executable norm, not just a
    process note: it re-runs the exact same check as
    test_ar1_known_answer_recovers_analytic_tau and fails LOUDLY (with the
    full comparison) rather than silently, so a future change that breaks the
    estimator cannot slip through as a quiet off-by-a-bit regression."""
    for phi in AR1_CASES:
        tau_analytic = (1 + phi) / (1 - phi) if phi < 1 else float("inf")
        n = max(int(round(20000 * max(tau_analytic, 1.0))), 4000)
        x = ar1_chains(n, phi, seed=42)
        iat = integrated_autocorrelation_time(x)
        rel_err = abs(iat.tau - tau_analytic) / tau_analytic
        assert rel_err < 0.20, (
            f"AR(1) validation FAILED at phi={phi}: analytic tau="
            f"{tau_analytic:.4f}, estimated tau={iat.tau:.4f}, error="
            f"{rel_err:.1%}. This is a real result: do not raise this "
            f"tolerance to make it pass without noting the failure in "
            f"ess-report.md.")


# ---------------------------------------------------------------------------
# 2. Cross-implementation: agreement with independently-written code.
# ---------------------------------------------------------------------------

def test_autocorrelation_matches_statsmodels_acf():
    import statsmodels.tsa.stattools as sm_tsa

    x = ar1_chains(5000, 0.8, seed=3)[0]
    rho_mine = autocorrelation(x[None, :])
    rho_sm = sm_tsa.acf(x, nlags=50, fft=True, adjusted=False)
    # statsmodels' `adjusted=False` is the same biased (1/n) normalisation
    # this module uses -- these should agree near machine precision, not just
    # approximately, since they are computing the identical estimator.
    assert np.allclose(rho_mine[:51], rho_sm, atol=1e-8), (
        "autocorrelation() disagrees with statsmodels.tsa.stattools.acf "
        "(adjusted=False) -- these should be the SAME estimator")


def test_ess_agrees_with_arviz_ess_on_multi_chain_ar1():
    import arviz as az

    phi = 0.8
    tau_analytic = (1 + phi) / (1 - phi)
    n_chains, n = 4, 20000   # N/tau ~ 8888, reliable regime
    x = ar1_chains(n, phi, seed=7, n_chains=n_chains)

    mine = effective_sample_size(x)
    assert mine.reliable, f"expected a reliable estimate, got: {mine.reason}"

    arviz_ess = float(az.ess(x, method="bulk"))

    # Different methods (Sokal windowing here; arviz's default 'bulk' is a
    # rank-normalised, split-chain Geyer/Gelman-Rubin combination) -- exact
    # agreement is not expected, but both must be within 30% of each other,
    # and both must be within 30% of the analytic N/tau_analytic reference,
    # or the "independent implementation agrees" claim is not actually true.
    n_total = x.size
    analytic_ess = n_total / tau_analytic
    rel_diff = abs(mine.ess - arviz_ess) / arviz_ess
    assert rel_diff < 0.30, (
        f"mine={mine.ess:.1f} vs arviz(bulk)={arviz_ess:.1f}, "
        f"relative difference {rel_diff:.2f} too large")
    assert abs(mine.ess - analytic_ess) / analytic_ess < 0.30
    assert abs(arviz_ess - analytic_ess) / analytic_ess < 0.30


# ---------------------------------------------------------------------------
# 3. Potency / must-fail: an estimator that cannot discriminate is useless.
# ---------------------------------------------------------------------------

def test_iid_noise_gives_ess_close_to_n():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 20000))
    r = effective_sample_size(x)
    assert r.reliable, r.reason
    assert r.iat == pytest.approx(1.0, abs=0.3)
    assert r.ess / r.n_total > 0.7, \
        "i.i.d. noise must give ESS close to N (IAT close to 1)"


def test_strongly_correlated_chain_gives_ess_much_less_than_n():
    phi = 0.98
    tau_analytic = (1 + phi) / (1 - phi)   # = 99
    n = int(round(6000 * tau_analytic))    # N/tau ~ 6000, reliable
    x = ar1_chains(n, phi, seed=9)
    r = effective_sample_size(x)
    assert r.reliable, r.reason
    assert r.ess / r.n_total < 0.05, \
        "a strongly autocorrelated chain must give ESS << N"


def test_estimator_discriminates_iid_from_strongly_correlated():
    """The potency control itself: if i.i.d. noise and a strongly correlated
    chain came back with SIMILAR ess, the estimator would not be measuring
    anything, and this test (not just the two tests above individually) is
    what catches that failure mode directly."""
    rng = np.random.default_rng(2)
    iid = rng.normal(size=(1, 20000))
    phi = 0.98
    n = int(round(6000 * (1 + phi) / (1 - phi)))
    corr = ar1_chains(n, phi, seed=11)

    r_iid = effective_sample_size(iid)
    r_corr = effective_sample_size(corr)
    assert r_iid.reliable and r_corr.reliable
    assert r_iid.ess / r_iid.n_total > 50 * (r_corr.ess / r_corr.n_total), (
        f"iid ess-fraction={r_iid.ess / r_iid.n_total:.4f} is not far enough "
        f"above correlated ess-fraction={r_corr.ess / r_corr.n_total:.4f}; "
        f"an estimator that cannot separate these two regimes by orders of "
        f"magnitude measures nothing")


# ---------------------------------------------------------------------------
# 4. Validity domain: too short must return None with a reason, never a guess.
# ---------------------------------------------------------------------------

def test_short_chain_returns_none_with_a_reason_not_a_number():
    phi = 0.9   # tau_analytic = 19
    x = ar1_chains(500, phi, seed=5)     # N/tau ~ 26, far below threshold
    r = effective_sample_size(x)
    assert r.ess is None
    assert r.iat is None
    assert r.reliable is False
    assert r.reason.startswith("unavailable")
    assert "5000" in r.reason or str(int(RELIABILITY_MIN_N_OVER_TAU)) in r.reason


def test_too_few_samples_per_chain_returns_none_with_a_reason():
    x = np.array([[0.1, 0.2, 0.3]])   # 3 samples, below the hard minimum
    r = effective_sample_size(x)
    assert r.ess is None and r.iat is None
    assert r.reliable is False
    assert r.reason.startswith("unavailable")


def test_constant_series_returns_none_with_a_reason():
    x = np.ones((2, 500))
    r = effective_sample_size(x)
    assert r.ess is None and r.iat is None
    assert "constant" in r.reason


def test_reliability_gate_flips_at_the_threshold():
    """Same generative process (phi=0.5, tau_analytic=3), only N changed: a
    chain far below RELIABILITY_MIN_N_OVER_TAU must read unreliable, and the
    SAME process run long enough to clear it must read reliable -- the gate
    must be a real function of the data, not a constant."""
    phi = 0.5
    short = ar1_chains(3000, phi, seed=21)     # N/tau ~ 1000
    long_ = ar1_chains(30000, phi, seed=21)    # N/tau ~ 10000
    r_short = effective_sample_size(short)
    r_long = effective_sample_size(long_)
    assert r_short.reliable is False
    assert r_long.reliable is True


def test_to_dict_never_leaks_none_when_unreliable():
    x = ar1_chains(500, 0.9, seed=5)
    r = effective_sample_size(x)
    d = r.to_dict()
    assert d["ess"] != None  # noqa: E711 -- must be the reason STRING, not null
    assert isinstance(d["ess"], str) and d["ess"].startswith("unavailable")
    assert isinstance(d["iat"], str) and d["iat"].startswith("unavailable")


# ---------------------------------------------------------------------------
# Multi-chain combination and the secondary (Geyer) cross-check.
# ---------------------------------------------------------------------------

def test_multichain_combination_agrees_with_one_long_chain():
    """4 chains of 5000 pooled the way this module combines chains (average
    per-chain autocovariance) should recover close to the same tau as 1 chain
    of 20000 from the same process -- the combination must not systematically
    bias the estimate relative to just having drawn one long chain."""
    phi = 0.8
    multi = ar1_chains(5000, phi, seed=99, n_chains=4)
    single = ar1_chains(20000, phi, seed=100, n_chains=1)
    tau_multi = integrated_autocorrelation_time(multi).tau
    tau_single = integrated_autocorrelation_time(single).tau
    assert abs(tau_multi - tau_single) / tau_single < 0.25


def test_geyer_initial_positive_sequence_cross_checks_sokal():
    """Two independently-constructed estimators (Sokal automatic windowing
    and Geyer's initial positive sequence) on the SAME data must agree with
    each other, not just with the analytic answer -- this is the
    self-consistency check that catches a bug shared between 'the estimator'
    and 'the known answer' (there isn't one here, since the known answer is
    external) but more importantly catches a bug in the windowing/truncation
    logic specifically, since Geyer's truncation rule is different from
    Sokal's."""
    for phi in (0.0, 0.5, 0.8, 0.9):
        tau_analytic = (1 + phi) / (1 - phi) if phi < 1 else float("inf")
        n = max(int(round(20000 * max(tau_analytic, 1.0))), 4000)
        x = ar1_chains(n, phi, seed=42)
        tau_sokal = integrated_autocorrelation_time(x).tau
        tau_geyer = geyer_initial_positive_sequence_tau(x)
        rel_diff = abs(tau_sokal - tau_geyer) / max(tau_analytic, 1.0)
        assert rel_diff < 0.15, (
            f"phi={phi}: Sokal tau={tau_sokal:.3f} vs Geyer tau={tau_geyer:.3f} "
            f"disagree by more than 15% of the analytic reference "
            f"({tau_analytic:.3f})")
