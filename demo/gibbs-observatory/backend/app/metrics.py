"""Metric helpers for Gibbs Observatory (energy, magnetization, ESS, autocorr)."""

from __future__ import annotations

import numpy as np


def spins_pm1(states_bool: np.ndarray) -> np.ndarray:
    """Convert bool spins (True=+1, False=-1) to ±1 float array."""
    return np.where(states_bool, 1.0, -1.0).astype(np.float64)


def magnetization(states_bool: np.ndarray) -> np.ndarray:
    """Per-sample mean magnetization. states: (n_samples, n_spins) bool."""
    return spins_pm1(states_bool).mean(axis=-1)


def ising_energy(
    states_bool: np.ndarray,
    edges: list[tuple[int, int]],
    biases: np.ndarray,
    weights: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Ising energy E = -β (b·s + Σ J_ij s_i s_j) for each sample.

    Matches THRML IsingEBM convention (True=+1, False=-1).
    """
    s = spins_pm1(states_bool)  # (T, N)
    field = (s * biases.reshape(1, -1)).sum(axis=-1)
    couple = np.zeros(s.shape[0], dtype=np.float64)
    for w, (i, j) in zip(weights, edges, strict=True):
        couple += float(w) * s[:, i] * s[:, j]
    return -float(beta) * (field + couple)


def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1D series. Returns 0 for short/constant series."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var < 1e-18:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / var)


def summarize_series(values: np.ndarray) -> dict:
    """Summary of one live metric series.

    ESS comes from `tsu_compiler.ess.effective_sample_size`, this project's own
    reviewed Sokal-window estimator, NOT from a lag-1 AR(1) approximation
    computed here. That estimator refuses to answer when the chain is too short
    to support an estimate, and the refusal is passed through to the interface
    rather than replaced with a confident-looking number.

    Measured before this changed. The interface keeps 256 samples of history,
    and the compiler's reliability threshold is N/tau >= 5000, which at that
    length is unreachable. So it declined every time, while this function
    returned a number anyway:

        series                      N        shown    compiler
        white noise               256        237.1    REFUSES
        rho=0.95 (near critical)  256          8.9    REFUSES
        white noise            20,000     19,985.5    20,561.3

    The old estimator was not wrong about long chains. It was silent about short
    ones, which is the only kind this interface has.

    `lag1_autocorr` stays. It is cheap, always available, and honest about being
    what it is. `ess_from_autocorr` was deleted rather than left unused, because
    leaving it invites its reuse.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return {
            "mean": None, "std": None, "last": None, "lag1_autocorr": None,
            "ess": None, "ess_reliable": False,
            "ess_reason": "unavailable: no samples yet", "n": 0,
        }

    ess: float | None = None
    ess_reason: str | None = None
    reliable = False
    try:
        from tsu_compiler.ess import effective_sample_size

        est = effective_sample_size(v.reshape(1, -1))
        reliable = bool(est.reliable)
        ess = float(est.ess) if (reliable and est.ess is not None) else None
        ess_reason = None if reliable else str(est.reason)
    except Exception as exc:  # noqa: BLE001
        ess_reason = f"unavailable: the ESS estimator failed: {exc!r}"

    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "last": float(v[-1]),
        "lag1_autocorr": lag1_autocorr(v),
        "ess": ess,
        "ess_reliable": reliable,
        "ess_reason": ess_reason,
        "n": int(v.size),
    }
