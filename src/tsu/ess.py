"""Autocorrelation, integrated autocorrelation time (IAT) and effective sample
size (ESS) for a set of MCMC chains -- a generic statistics module, no
knowledge of Ising models, spins, or any other workload/backend vocabulary.
Callers hand this module a plain (n_chains, n_samples) array of some SCALAR
functional of each draw (e.g. energy); this module never sees the chain's
internal representation.

WHY THIS FILE EXISTS (read this before trusting a number it returns): an
estimator is itself a measurement instrument. Printing a number an instrument
produced, without having independently shown the instrument is right, is
exactly the mistake this whole compiler exists to refuse -- see
`tests/test_ess.py` for the three controls (AR(1) known-answer, cross-
implementation against arviz/statsmodels, and a must-fail potency check) that
this module is not considered trustworthy without, and
`sdd/2026-08-26-tsu-compiler-vertical-slice/ess-report.md` for the validation
write-up, including the reliability threshold below and how it was measured.

METHOD: Sokal's automatic-windowing estimator (Sokal, "Monte Carlo Methods in
Statistical Mechanics: Foundations and New Algorithms", 1989 Cargese lecture
notes, sec. 3.3), the same estimator `emcee.autocorr.integrated_time`
(Foreman-Mackey et al. 2013) implements. For lag k, the (biased, 1/n-
normalised -- NOT 1/(n-k)) autocovariance c_k is estimated once per chain,
then AVERAGED ACROSS CHAINS (each chain first centred on ITS OWN mean, so a
between-chain mean difference, e.g. incomplete burn-in, does not masquerade as
within-chain autocorrelation signal). tau(M) = 1 + 2*sum_{k=1}^{M} rho_k, and
the window M is the smallest value with M >= c*tau(M) (c=5, Sokal's own
recommendation and emcee's default -- the automatic-windowing trick that lets
tau's own estimate pick where to stop summing an increasingly noisy tail).
ESS = (n_chains * n_samples) / tau.

Geyer's initial positive sequence (Geyer, "Practical Markov Chain Monte
Carlo", Statistical Science 7(4), 1992, sec. 3.1) is also implemented, as an
INDEPENDENT construction used only to cross-check the Sokal estimate in tests
-- it is not called from `effective_sample_size`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sokal's own recommended windowing constant (Sokal 1989 sec. 3.3); emcee
# uses the same default.
DEFAULT_C = 5.0

# Below this N/tau ratio, the AR(1) known-answer validation in test_ess.py
# shows the Sokal-windowed tau estimate is no longer trustworthy: sweeping
# N/tau from 500 to 20000 across phi in {0.5, 0.8, 0.9, 0.95, 0.99} (50 seeds
# each), the relative error against the exact analytic tau = (1+phi)/(1-phi)
# falls from ~30-50% at N/tau~100 through ~7-8% at N/tau~2000 to ~4-5% mean
# (p90 ~9-11%, worst-of-50 ~17%) at N/tau=5000, essentially independent of
# phi once N/tau is held fixed (see ess-report.md for the full sweep table).
# 5000 is where the error curve is flat and small across every phi tried; it
# also happens to match a threshold this project's corpus used before, but it
# was re-derived here from this estimator's own AR(1) behaviour, not taken on
# faith from that prior finding.
RELIABILITY_MIN_N_OVER_TAU = 5000.0

# Below this many samples per chain there are not enough lags to estimate any
# autocorrelation structure at all (the windowing loop below needs at least a
# handful of lags to run) -- reject before even attempting a tau estimate.
_MIN_SAMPLES_PER_CHAIN = 8


def _as_chains(x) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.ndim != 2:
        raise ValueError(
            f"expected a (n_chains, n_samples) array of a scalar functional "
            f"per draw, got shape {x.shape}")
    return x


def autocovariance(chains) -> np.ndarray:
    """The biased (1/n, not 1/(n-k)) autocovariance at every lag 0..n-1,
    estimated once per chain (each chain centred on its OWN mean) and then
    averaged across chains -- see the module docstring for why. `chains` is
    (n_chains, n_samples) or (n_samples,) for a single chain. Returns a
    (n_samples,) array; index 0 is the (pooled) sample variance.

    FFT-based (Wiener-Khinchin): O(n log n) per chain rather than the O(n^2)
    direct double sum, exact up to floating-point rounding for a real-valued
    biased autocovariance -- not an approximation.
    """
    chains = _as_chains(chains)
    n_chains, n = chains.shape
    if n < 2:
        raise ValueError("need at least 2 samples per chain")
    xc = chains - chains.mean(axis=1, keepdims=True)
    fsize = 2 * n
    f = np.fft.fft(xc, n=fsize, axis=1)
    acov = np.fft.ifft(f * np.conjugate(f), axis=1).real[:, :n] / n
    return acov.mean(axis=0)


def autocorrelation(chains) -> np.ndarray:
    """`autocovariance` normalised by lag-0 (the variance): rho[0] == 1.0."""
    acov = autocovariance(chains)
    if acov[0] <= 0:
        raise ValueError(
            "zero (or negative, which cannot happen for a real variance -- "
            "this means the input is exactly constant) variance: "
            "autocorrelation is undefined for a constant series")
    return acov / acov[0]


@dataclass(frozen=True)
class IATResult:
    tau: float
    window: int          # the Sokal cutoff M actually used
    window_saturated: bool  # True if the c*tau criterion was never met and
                             # `window` is just the largest lag available --
                             # a sign the chain is too short for THIS tau


def integrated_autocorrelation_time(chains, c: float = DEFAULT_C) -> IATResult:
    """Sokal's automatic-windowing estimate of tau. See the module docstring
    for the method and citation. `effective_sample_size` is the entry point
    that should be used for anything that gets published (it enforces the
    reliability threshold and returns None rather than a number when the
    chain is too short); this function always returns a tau, reliable or not
    -- callers that need the reliability judgement must go through
    `effective_sample_size`.
    """
    rho = autocorrelation(chains)
    n = rho.shape[0]
    max_lag = max(1, n // 2)
    cum = 0.0
    tau = 1.0
    window = 0
    saturated = True
    for m in range(1, max_lag):
        cum += rho[m]
        tau = 1.0 + 2.0 * cum
        # tau can dip non-positive for a short, noisy, or anti-correlated
        # series; floor it so `m >= c*tau` cannot become vacuously true and
        # the loop can keep looking for a genuine crossing.
        tau_for_test = max(tau, 1e-12)
        if m >= c * tau_for_test:
            window = m
            saturated = False
            break
    if saturated:
        window = max_lag - 1
    return IATResult(tau=tau, window=window, window_saturated=saturated)


def geyer_initial_positive_sequence_tau(chains) -> float:
    """Geyer's (1992) initial positive sequence -- an INDEPENDENT estimator
    of tau, used in tests/test_ess.py only to cross-check
    `integrated_autocorrelation_time`, never called from
    `effective_sample_size`. Gamma_m = rho(2m) + rho(2m+1); sum Gamma_m for
    m = 0, 1, 2, ... up to (and excluding) the first m where Gamma_m <= 0
    (the "initial positive sequence" truncation); tau = -1 + 2*sum Gamma_m.
    """
    rho = autocorrelation(chains)
    n = rho.shape[0]
    max_pairs = max(0, n // 2 - 1)
    total = 0.0
    for m in range(max_pairs):
        gamma = rho[2 * m] + rho[2 * m + 1]
        if gamma <= 0:
            break
        total += gamma
    return -1.0 + 2.0 * total


@dataclass(frozen=True)
class EssEstimate:
    """The published result. `ess` and `iat` are None together, with `reason`
    explaining why, whenever the estimate is not trustworthy -- never a
    plausible-looking number for a chain too short to support one (see the
    module docstring)."""
    ess: float | None
    iat: float | None
    n_total: int
    reliable: bool
    reason: str

    def to_dict(self):
        return {
            "ess": self.ess if self.ess is not None else self.reason,
            "iat": self.iat if self.iat is not None else self.reason,
            "n_total": self.n_total,
            "reliable": self.reliable,
        }


def effective_sample_size(chains, c: float = DEFAULT_C,
                          min_n_over_tau: float = RELIABILITY_MIN_N_OVER_TAU
                          ) -> EssEstimate:
    """The one function anything outside this module should call to get a
    number it can publish. `chains`: (n_chains, n_samples) array of one
    scalar functional per draw (e.g. energy) -- autocorrelation is only
    meaningful within a chain, so callers must NOT flatten multiple chains
    into one series before calling this (see thrml_backend.sample_chains).

    Returns `ess=None, iat=None` with a `reason` when:
      - there are fewer than `_MIN_SAMPLES_PER_CHAIN` samples per chain
        (not enough lags to estimate anything), or
      - the series is (numerically) constant (autocorrelation undefined), or
      - the estimated tau is high enough, relative to the total sample count,
        that N/tau falls below `min_n_over_tau` -- the AR(1)-validated
        reliability threshold (see RELIABILITY_MIN_N_OVER_TAU above). This
        also covers the case where the Sokal window saturated (never found a
        crossing): a saturated window means the chain was too short for its
        own tau, which is exactly what a low N/tau already detects.
    """
    chains = _as_chains(chains)
    n_chains, n_samples = chains.shape
    n_total = n_chains * n_samples

    if n_samples < _MIN_SAMPLES_PER_CHAIN:
        return EssEstimate(None, None, n_total, False,
            f"unavailable: only {n_samples} samples per chain, need at "
            f"least {_MIN_SAMPLES_PER_CHAIN} to estimate any autocorrelation")

    var = chains.var(axis=1).mean()
    if not np.isfinite(var) or var <= 1e-300:
        return EssEstimate(None, None, n_total, False,
            "unavailable: series is constant (zero variance); "
            "autocorrelation is undefined")

    iat_result = integrated_autocorrelation_time(chains, c=c)
    tau = iat_result.tau
    if not np.isfinite(tau) or tau <= 0:
        return EssEstimate(None, None, n_total, False,
            f"unavailable: integrated autocorrelation time estimate was "
            f"non-positive/non-finite (tau={tau!r}), no defensible ESS "
            f"can be derived from it")

    n_over_tau = n_total / tau
    if iat_result.window_saturated or n_over_tau < min_n_over_tau:
        return EssEstimate(None, None, n_total, False,
            f"unavailable: N/tau={n_over_tau:.0f} (N={n_total}, "
            f"tau~={tau:.2f}) is below the reliability threshold "
            f"{min_n_over_tau:.0f} the AR(1) validation established "
            f"(see ess-report.md); chain too short/too correlated for a "
            f"trustworthy estimate")

    return EssEstimate(ess=n_over_tau, iat=tau, n_total=n_total,
                       reliable=True, reason="")
