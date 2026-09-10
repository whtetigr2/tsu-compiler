"""Where should this model be sampled?

A model below its ordering transition produces noise no matter how good its
constraints are. This project lost months generating structureless worlds at
beta*J = 0.2 against a transition at 0.44, so the question is not academic.

THE TRANSITION IS MEASURED, NEVER ASSUMED. Onsager's Kc = 0.4407 is exact for a
uniform square lattice in zero field, and this project treated it as a property
of the chip until measuring the same ferromagnet order at beta*J ~ 0.075 on
degree-16 connectivity -- six times lower. So the location comes from the data.

IT IS LOCATED BY FINITE-SIZE SCALING. The Binder cumulant
U = 1 - <m^4>/(3<m^2>^2) tends to 0 deep in the disordered phase (where m is
Gaussian and <m^4> = 3<m^2>^2) and to 2/3 deep in the ordered phase (where m sits
at +-m0). Those limits hold for ANY model, which is what lets curves for two
sizes cross at the critical coupling without either curve knowing what model it
came from. A susceptibility peak is reported alongside, but it drifts with size
and so cannot locate the transition on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tsu.ess import effective_sample_size
from tsu.preflight.diagnostics import RHAT_THRESHOLD, r_hat, stderr_from_ess
from tsu.passes.analyse import analyse
from tsu.passes.program import build_program
from tsu.backends.thrml_backend import sample_chains

SATURATION = 0.9
"""|m| above which one state has effectively swallowed the system. The upper
edge of the usable band: past here the model is ordered and produces a single
configuration, which is as useless as noise."""


@dataclass(frozen=True)
class RegimeRow:
    """One (size, coupling) measurement.

    `abs_m_err`, `tau` and `n_eff` are None together whenever tsu.ess judged the
    run too short to support a trustworthy estimate; `ess_reason` then says why.
    They are optional rather than zero-filled on purpose -- a zero error bar
    reads as an exact measurement, which is the opposite of what happened."""
    beta_j: float
    size: int
    abs_m: float
    abs_m_err: float | None
    chi: float
    binder: float
    tau: float | None
    n_eff: float | None
    r_hat: float
    ess_reason: str
    provisional: bool


def binder(m: np.ndarray) -> float:
    """U = 1 - <m^4> / (3 <m^2>^2). Zero for a Gaussian, 2/3 for two deltas."""
    m = np.asarray(m, dtype=float).ravel()
    m2 = float(np.mean(m ** 2))
    if m2 <= 0.0:
        return 0.0  # degenerate sample: report the disordered limit, not NaN
    return float(1.0 - np.mean(m ** 4) / (3.0 * m2 * m2))


def susceptibility(m: np.ndarray, n_spins: int) -> float:
    """chi = N (<m^2> - <|m|>^2)."""
    m = np.asarray(m, dtype=float).ravel()
    return float(n_spins * (np.mean(m ** 2) - np.mean(np.abs(m)) ** 2))


def sweep(model_fn, sizes, couplings, *, seed: int = 0, n_chains: int = 8,
          n_samples: int = 400, n_warmup: int = 4000,
          steps: int = 8) -> list[RegimeRow]:
    """Sample `model_fn(size, beta_j) -> IsingModel` over every (size, coupling).

    Diagnostics are computed PER CHAIN before pooling: tau from the concatenated
    per-chain series of the order parameter, R-hat across chains. Pooling first
    would destroy exactly the structure both statistics exist to detect.
    """
    rows: list[RegimeRow] = []
    for size in sizes:
        for bj in couplings:
            ising = model_fn(size, bj)
            rep = analyse(ising)
            prog = build_program(ising, rep)
            # sample_chains, NOT sample: sample() flattens the chain boundary
            # on purpose, and its own docstring warns that autocorrelation and
            # ESS are meaningless across it. Reshaping sample()'s output by hand
            # happens to recover the right order today, but only by coincidence.
            draws = np.asarray(sample_chains(
                prog, n_chains=n_chains, n_samples=n_samples,
                n_warmup=n_warmup, steps_per_sample=steps, seed=seed))
            spins = 2 * draws.astype(int) - 1
            per_chain = spins.mean(axis=2)          # (n_chains, n_samples)

            m_all = per_chain.ravel()
            est = effective_sample_size(np.abs(per_chain))
            rh = r_hat(np.abs(per_chain))
            # est.ess is None when the run cannot support a trustworthy
            # estimate; the row is then provisional and carries est.reason
            # rather than a plausible-looking error bar.
            rows.append(RegimeRow(
                beta_j=float(bj), size=int(size),
                abs_m=float(np.mean(np.abs(m_all))),
                abs_m_err=(stderr_from_ess(np.abs(m_all), est.ess)
                           if est.ess is not None else None),
                chi=susceptibility(m_all, rep.n_nodes),
                binder=binder(m_all), tau=est.iat, n_eff=est.ess, r_hat=rh,
                ess_reason=est.reason,
                provisional=bool(rh > RHAT_THRESHOLD or not est.reliable)))
    return rows


def crossing(rows_a, rows_b):
    """The coupling where two sizes' Binder curves meet, or None.

    Below the transition a larger system is MORE disordered (lower U); above it,
    more ordered. The sign of the difference therefore flips exactly at the
    critical coupling, and the crossing is found by linear interpolation on that
    difference. Returns None when the sweep never brackets a flip -- reporting a
    transition that was not observed would be worse than reporting none.
    """
    a = {r.beta_j: r.binder for r in rows_a}
    b = {r.beta_j: r.binder for r in rows_b}
    common = sorted(set(a) & set(b))
    for lo, hi in zip(common, common[1:]):
        d0, d1 = a[lo] - b[lo], a[hi] - b[hi]
        if d0 == 0.0:
            return float(lo)
        if d0 * d1 < 0.0:
            return float(lo + (hi - lo) * abs(d0) / (abs(d0) + abs(d1)))
    return None


def usable_band(rows, saturate: float = SATURATION):
    """(low, high) coupling where the model orders but has not saturated.

    Provisional rows are excluded: a coupling whose chains disagree must not set
    an edge of the band a user will trust.
    """
    good = sorted((r for r in rows if not r.provisional),
                  key=lambda r: r.beta_j)
    if not good:
        return None
    started = [r for r in good if r.binder > 0.1]
    if not started:
        return None
    lo = started[0].beta_j
    below = [r for r in good if r.beta_j >= lo and r.abs_m < saturate]
    if not below:
        return None
    return (float(lo), float(below[-1].beta_j))
