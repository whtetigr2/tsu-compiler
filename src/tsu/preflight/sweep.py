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
    reads as an exact measurement, which is the opposite of what happened.

    `ess_unavailable` and `provisional` are DELIBERATELY SEPARATE fields, not
    one merged flag, because they mean different things:
      - `provisional` (R-hat > RHAT_THRESHOLD): the chains disagree with each
        other, so the MEAN ITSELF (abs_m, binder, chi) is suspect. A row like
        this must never set the edge of a band a user will trust.
      - `ess_unavailable` (tsu.ess judged N/tau too low): only the tau
        ESTIMATE is too noisy to quantify an ERROR BAR. This does not make the
        mean wrong -- abs_m/binder/chi are still the best estimates the draws
        support -- it makes the uncertainty on that mean unquantified.
    Conflating the two would be actively harmful: tau genuinely diverges near
    an ordering transition (critical slowing down), which is exactly the
    region `usable_band` exists to locate. A row there is disproportionately
    likely to be ESS-unavailable precisely because the physics is interesting,
    so treating ESS-unavailability as disqualifying would make this tool
    refuse to find the band where it actually is.

    `n_samples_used` is the `n_samples` the escalation loop in `sweep`
    actually settled on for THIS row -- equal to the `n_samples` argument
    when the first attempt already cleared tsu.ess's reliability floor, and
    larger than it when the row needed one or more doublings to get there (or
    to exhaust `max_samples` and still refuse). It is itself a physics
    readout, not just bookkeeping: tau grows near an ordering transition
    (critical slowing down), so the rows that needed the most draws to
    certify are the ones nearest the transition."""
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
    ess_unavailable: bool
    provisional: bool
    n_samples_used: int


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


def sweep(model_fn, sizes, couplings, *, seed: int = 0, n_chains: int = 16,
          n_samples: int = 2000, n_warmup: int = 4000, steps: int = 8,
          max_samples: int = 32_000) -> list[RegimeRow]:
    """Sample `model_fn(size, beta_j) -> IsingModel` over every (size, coupling).

    Diagnostics are computed PER CHAIN before pooling: tau from the concatenated
    per-chain series of the order parameter, R-hat across chains. Pooling first
    would destroy exactly the structure both statistics exist to detect.

    DEFAULTS: `n_chains=16, n_samples=2000` gives `n_total=32,000`, clearing
    `tsu.ess.RELIABILITY_MIN_N_OVER_TAU=5000` even once tau exceeds 1 by a
    healthy margin. The original defaults here (`n_chains=8, n_samples=400`,
    `n_total=3,200`) could never clear that floor -- even literally i.i.d.
    draws (tau~1, the best case) give N/tau~3,362 < 5,000, so every row would
    be ESS-unavailable regardless of how well the chain mixed. Measured cost:
    raising to these defaults is free, not a tradeoff -- JAX vmaps the chains
    in parallel and warmup dominates wall time, so 10x the draws (16x2000 vs
    8x400) cost about the same wall-clock per (size, coupling) point.

    ESCALATION: tau grows near an ordering transition (critical slowing
    down), so a FIXED draw count is guaranteed to fail to certify a row
    exactly where the measurement matters most -- refusing outright there
    would mean this tool only ever answers the easy half of its own question.
    When `effective_sample_size` refuses a row, `sweep` resamples THAT
    (size, coupling) point with `n_samples` doubled and tries again, doubling
    repeatedly until either the estimate clears tsu.ess's reliability floor
    or the NEXT doubling would exceed `max_samples`. `max_samples` is a
    STATED COMPUTE BUDGET, not a judgement that the quantity is unmeasurable:
    a row that still refuses once the budget is exhausted reports
    `n_samples_used` as the largest attempt actually made, and `ess_reason`
    names that attempt, rather than being reported as if tau there were
    inherently inaccessible.
    """
    rows: list[RegimeRow] = []
    for size in sizes:
        for bj in couplings:
            ising = model_fn(size, bj)
            rep = analyse(ising)
            prog = build_program(ising, rep)

            cur_n_samples = n_samples
            while True:
                # sample_chains, NOT sample: sample() flattens the chain
                # boundary on purpose, and its own docstring warns that
                # autocorrelation and ESS are meaningless across it.
                # Reshaping sample()'s output by hand happens to recover the
                # right order today, but only by coincidence.
                draws = np.asarray(sample_chains(
                    prog, n_chains=n_chains, n_samples=cur_n_samples,
                    n_warmup=n_warmup, steps_per_sample=steps, seed=seed))
                spins = 2 * draws.astype(int) - 1
                per_chain = spins.mean(axis=2)      # (n_chains, n_samples)

                m_all = per_chain.ravel()
                est = effective_sample_size(np.abs(per_chain))
                rh = r_hat(np.abs(per_chain))
                if est.reliable:
                    break
                next_n_samples = cur_n_samples * 2
                if next_n_samples > max_samples:
                    # Budget exhausted: report the largest attempt actually
                    # made, not a plausible-looking number from a run that
                    # never happened.
                    break
                cur_n_samples = next_n_samples

            # est.ess is None when the run cannot support a trustworthy tau/
            # ESS estimate even after escalation; abs_m_err then carries None
            # too rather than a plausible-looking error bar, and
            # ess_unavailable=True records WHY without disqualifying the
            # row's mean-based fields (abs_m, binder, chi) -- see RegimeRow's
            # docstring for why the two are kept separate. `provisional`
            # reflects R-hat ALONE: only chain disagreement makes the mean
            # itself untrustworthy.
            reason = est.reason if est.reliable else (
                f"{est.reason} (largest attempt: n_samples={cur_n_samples}, "
                f"budget max_samples={max_samples})")
            rows.append(RegimeRow(
                beta_j=float(bj), size=int(size),
                abs_m=float(np.mean(np.abs(m_all))),
                abs_m_err=(stderr_from_ess(np.abs(m_all), est.ess)
                           if est.ess is not None else None),
                chi=susceptibility(m_all, rep.n_nodes),
                binder=binder(m_all), tau=est.iat, n_eff=est.ess, r_hat=rh,
                ess_reason=reason,
                ess_unavailable=not est.reliable,
                provisional=bool(rh > RHAT_THRESHOLD),
                n_samples_used=cur_n_samples))
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


def usable_band(rows, crossing, saturate: float = SATURATION):
    """(low, high) coupling where the model orders but has not saturated, or
    None when no such band can be measured.

    THE LOWER EDGE IS THE MEASURED BINDER CROSSING -- nothing else. This
    function used to derive its own lower edge from a hardcoded
    `binder > 0.1` literal, invented while writing this module's plan and
    never justified against the design (which already says the lower edge
    IS the crossing). That literal was a real, shipped defect: run against a
    16-node 1D Ising ring (uniform ferromagnetic J, zero bias) -- which has
    NO finite-temperature phase transition at any coupling (exact result,
    Ising 1925) -- it fired on correlated-but-disordered fluctuation (binder
    reached ~0.13, comfortably past the 0.1 literal, with no genuine
    ordering anywhere in range) and reported a confident-looking band on a
    model with a known-correct answer of "no band exists". `crossing` is
    accepted as a required parameter specifically so this function cannot
    reconstruct that mistake: when `crossing` is None (no size family was
    swept, or the sweep never bracketed one -- see `crossing()` above),
    there is no principled lower edge, and this returns None outright
    rather than inventing one from `binder` alone.

    THE UPPER EDGE MUST BE A REAL SATURATION OBSERVATION, never "the last
    coupling the sweep happened to try": if no row at or above the crossing
    actually reaches `saturate`, the band is open above (`high = None`) --
    "the sweep didn't reach saturation within its own range" is a fact
    about the swept range, not a measurement of where the model itself
    would saturate, and printing the sweep's own upper bound as if it were
    that measurement is the same category of mistake as the deleted lower
    edge. `SATURATION` itself stays a named, stated operational definition
    (spec: |m| above which one state has effectively swallowed the system)
    -- unlike the deleted 0.1, it was never presented as a measurement, so
    it is not the same defect.

    Filters on `provisional` ONLY -- R-hat above threshold, meaning the
    chains disagree and the mean itself is suspect -- never on
    `ess_unavailable`. An ESS-unavailable row's mean (abs_m) is still the
    best estimate the draws support, only its error bar is unquantified;
    tau genuinely diverges near an ordering transition (critical slowing
    down), so ESS-unavailable rows cluster exactly where the band's edge
    is, and dropping them would make this function refuse to locate the
    band precisely where it exists.
    """
    if crossing is None:
        return None
    good = sorted((r for r in rows if not r.provisional),
                  key=lambda r: r.beta_j)
    if not good:
        return None
    lo = float(crossing)
    above = [r for r in good if r.beta_j >= lo]
    if not above:
        return None
    saturated = [r for r in above if r.abs_m >= saturate]
    if not saturated:
        # Open above: nothing at or above the crossing saturated within
        # this sweep's own range. `above[-1].beta_j` (the largest coupling
        # actually tried) is exactly the fake edge this function used to
        # report -- reporting None here instead is the fix.
        return (lo, None)
    below = [r for r in above if r.abs_m < saturate]
    if not below:
        return None
    return (lo, float(below[-1].beta_j))
