"""Minimal tests for metrics helpers."""

import numpy as np

from backend.app.metrics import (
    ess_from_autocorr,
    ising_energy,
    lag1_autocorr,
    magnetization,
    spins_pm1,
    summarize_series,
)


def test_spins_pm1():
    x = np.array([[True, False], [False, True]])
    s = spins_pm1(x)
    assert np.allclose(s, [[1, -1], [-1, 1]])


def test_magnetization():
    x = np.array([[True, True, False, False]])
    m = magnetization(x)
    assert np.allclose(m, [0.0])


def test_ising_energy_ferro_aligned():
    # Two spins, J=1, beta=1, aligned => E=-1
    states = np.array([[True, True], [False, False], [True, False]])
    edges = [(0, 1)]
    biases = np.zeros(2)
    weights = np.array([1.0])
    e = ising_energy(states, edges, biases, weights, beta=1.0)
    assert np.allclose(e, [-1.0, -1.0, 1.0])


def test_lag1_autocorr_constant():
    assert lag1_autocorr(np.ones(20)) == 0.0


def test_lag1_autocorr_alternating():
    x = np.array([1.0, -1.0] * 50)
    rho = lag1_autocorr(x)
    assert rho < -0.9


def test_ess_and_summarize():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    rho = lag1_autocorr(x)
    ess = ess_from_autocorr(len(x), rho)
    assert 0 < ess <= len(x) * 1.5
    s = summarize_series(x)
    assert s["n"] == 200
    assert "lag1_autocorr" in s
