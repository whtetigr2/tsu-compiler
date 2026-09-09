"""The estimators every reported number depends on.

WHY THIS MODULE DECIDES WHETHER THE TOOL IS BELIEVABLE. Consecutive Gibbs
samples are correlated, so `std / sqrt(N)` overstates precision by a factor that
depends on the chain and the coupling -- and it is worst exactly at a phase
transition, where autocorrelation grows and where the interesting numbers are.
Every error bar this tool prints is therefore computed from N_eff = N / (2*tau),
not from N.

Pure statistics over arrays: no sampling, no I/O, no compiler. That is what lets
each estimator be checked against a value known in closed form rather than
against our own output.
"""
from __future__ import annotations

import numpy as np

RHAT_THRESHOLD = 1.01
"""Above this, chains disagree enough that their pooled numbers are reported as
provisional. The conventional threshold; stated as a constant so a reader can
see which value was used rather than infer it."""


def autocorr_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time, tau = 1/2 + sum_k rho_k.

    Uses Sokal's automatic windowing: truncate the sum at the smallest M with
    M >= c * tau(M). Summing every lag instead would add pure noise from the
    long-lag tail, where rho is estimated from almost no independent data.

    Returns 1/2 for independent draws, and (1+phi)/(2(1-phi)) for an AR(1)
    series -- both checked against their closed forms in the tests.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 0.5
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0.0:
        return 0.5  # a constant series has no correlation structure to measure

    # autocovariance by FFT, which is O(n log n) rather than O(n^2)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n].real
    rho = acf / acf[0]

    # `taus` here is Sokal/emcee's convention tau_A(M) = 1 + 2*sum_{k=1}^{M} rho_k;
    # the windowing criterion M >= c*tau_A(M) is evaluated in that convention.
    # This module's convention is tau_B = 1/2 + sum_{k=1}^{M} rho_k = tau_A/2
    # (chosen so that n_eff = n/(2*tau) matches the classic N/(1+2*sum) ESS
    # formula and the AR(1) closed form (1+phi)/(2(1-phi))) -- hence the final
    # halving, with no additive offset.
    taus = 2.0 * np.cumsum(rho) - 1.0
    window = np.arange(n)
    ok = window >= c * taus
    m = int(np.argmax(ok)) if ok.any() else n - 1
    return float(max(0.5, 0.5 * taus[m]))


def n_eff(n_samples: int, tau: float) -> float:
    """Effective independent sample count, floored at 1.

    A tau exceeding the run length means the chain never decorrelated. The floor
    keeps downstream error bars finite; the caller is expected to FLAG such a row
    rather than treat its number as measured.
    """
    return float(max(1.0, n_samples / (2.0 * max(tau, 0.5))))


def stderr_corrected(x: np.ndarray, tau: float) -> float:
    """Standard error using N_eff rather than N."""
    x = np.asarray(x, dtype=float).ravel()
    return float(np.std(x, ddof=1) / np.sqrt(n_eff(x.size, tau)))


def r_hat(chains: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction on shape (n_chains, n_samples).

    Compares between-chain variance to within-chain variance. Equals 1 when the
    chains are indistinguishable and grows when they are each internally settled
    but disagree with one another -- the failure a per-chain average hides
    completely.
    """
    c = np.asarray(chains, dtype=float)
    if c.ndim != 2 or c.shape[0] < 2:
        raise ValueError("r_hat needs at least two chains, shaped "
                         "(n_chains, n_samples)")
    m, n = c.shape
    if n < 2:
        raise ValueError("r_hat needs at least two samples per chain")
    means = c.mean(axis=1)
    w = float(c.var(axis=1, ddof=1).mean())
    b = float(n * means.var(ddof=1))
    if w <= 0.0:
        return 1.0
    # No (n-1)/n discount on w: the textbook BDA3 var_plus = (n-1)/n*w + b/n
    # never reaches exactly 1 for identical chains at finite n (it gives
    # sqrt((n-1)/n) < 1 when b=0), which is wrong -- identical chains carry
    # zero between-chain disagreement and R-hat must equal 1 exactly.
    var_plus = w + b / n
    return float(np.sqrt(var_plus / w))
