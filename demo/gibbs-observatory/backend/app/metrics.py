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


def ess_from_autocorr(n: int, rho1: float) -> float:
    """Rough ESS from lag-1 AR(1) approximation: ESS ≈ n * (1-ρ)/(1+ρ)."""
    if n <= 0:
        return 0.0
    rho = float(np.clip(rho1, -0.999, 0.999))
    return float(n) * (1.0 - rho) / (1.0 + rho)


def summarize_series(values: np.ndarray) -> dict:
    """Summary stats + lag-1 autocorr + ESS for a metric series."""
    v = np.asarray(values, dtype=np.float64).ravel()
    rho = lag1_autocorr(v)
    return {
        "mean": float(v.mean()) if v.size else 0.0,
        "std": float(v.std()) if v.size else 0.0,
        "last": float(v[-1]) if v.size else 0.0,
        "lag1_autocorr": rho,
        "ess": ess_from_autocorr(v.size, rho),
        "n": int(v.size),
    }
