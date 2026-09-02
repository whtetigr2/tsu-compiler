"""Task 5 & 6: pure, headlessly-tested statistics behind the LATTICE demo's
REGIME (temperature) and SCOPE panels. No Tk here -- see
tests/test_scope.py for the headless suite; demo/lattice_app.py imports
these functions and does the Tk drawing.

TASK 5 -- beta_to_temperature: beta is inverse temperature. The
convention exists because exp(-beta*E) is tidier to carry through the
compiler than exp(-E/T), not because temperature is unusable -- both
belong on screen (see lattice_app.py's REGIME panel). beta<=0 is refused:
it is not a colder or hotter model, it is a meaningless one -- exp(-beta*E)
would invert (beta<0, low-energy states become LEAST likely) or flatten
(beta==0, every state equally likely, T undefined/infinite) the
distribution the rest of this app treats as physical.

TASK 6 -- the SCOPE panel's four readouts:

  autocorrelation(series, max_lag): reuses `tsu.ess.autocorrelation` (the
  compiler's own Sokal/FFT-based estimator, already validated in
  tests/test_ess.py against an AR(1) known-answer sweep -- see that
  module's docstring) rather than reimplementing a second, unvalidated
  copy. A single series is passed through `tsu.ess`'s (n_chains,
  n_samples) contract as one chain (shape (1, n)); this file only adds
  the max_lag truncation and the list[float] return shape the demo needs.

  magnetization(draws): the standard Ising order parameter, mean spin per
  draw. Draws are 0/1 OCCUPANCY (thrml's own encoding, see the thrml
  skill / tsu.passes.lower's own "spins s = 2*occupancy - 1" convention,
  the SAME mapping demo/lattice_app.py's energy_of_draw already uses) --
  so a draw is mapped s = 2*n - 1 before averaging. [1,0,1,0] -> spins
  [1,-1,1,-1] -> mean 0.0, not 0.5 (which would be the raw occupancy mean,
  a different and less informative number: the order parameter is defined
  on spins, not counts).

  energy_histogram(series, bins): a thin, honest wrapper over
  np.histogram -- the distribution the energy TRACE (a time series) only
  samples one point of at a time.

  local_field_response(draws, ising, bins): the MEASURED analogue of
  Extropic's DTM paper (arXiv 2510.23972) Figure 4a, which plots P(x=1)
  against bias voltage as an S-curve. Ours is measurable rather than
  drawn: THRML's chromatic block Gibbs conditional sampler for a spin site
  i is exactly P(s_i=1 | neighbours) = sigmoid(2*gamma_i), where gamma_i
  is the LOCAL FIELD (see the thrml skill's "How the API composes" table
  and `tsu.passes.lower`'s own sign convention, restated here since this
  file computes gamma independently rather than importing a private
  thrml/tsu symbol):

      IsingModel's (b, J) are defined so that thrml's energy is
      E_thrml(s) = -beta * (sum_i b_i*s_i + sum_(i,j) J_ij*s_i*s_j).
      Fixing every spin except s_i, the log-odds of s_i=+1 vs s_i=-1 is
      2*beta*(b_i + sum_{j~i} J_ij*s_j) -- call that 2*gamma_i -- so
      P(s_i=1 | neighbours) = sigmoid(2*gamma_i), gamma_i = beta*(b_i +
      sum_{j~i} J_ij*s_j).

  This function computes gamma_i FROM THE SAMPLER'S OWN DRAWS (the actual
  neighbour spins that draw actually had, not an assumed or averaged
  neighbourhood), pools (gamma, s) pairs across every spin site and every
  draw handed in, bins by gamma, and reports the EMPIRICAL P(s=1) per bin
  alongside its count. The caller overlays the analytic sigmoid(2*gamma)
  on the SAME axes -- if the measured points do not sit on that curve,
  that is a real finding about the sampler (mixing, thinning, a sign
  error) and must be reported as such, never smoothed away by picking
  more flattering bins.

  A bin with fewer than MIN_LOCAL_FIELD_BIN_COUNT pooled samples reports
  its count but NOT a probability (its probability entry is NaN, never a
  plausible-looking float) -- a bin of, say, 3 samples can only read
  0/3, 1/3, 2/3, or 3/3, all of which are far too coarse to plot as if
  they were a measured point on a curve. 30 is the conventional rule-of-
  thumb minimum for a binomial proportion's normal approximation to be
  reasonable (n*p and n*(1-p) both comfortably above 5 across the
  P~0.2-0.8 range this model's fields typically produce); it is a
  labelled convention, not a derived optimum for this specific model.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tsu.ess import autocorrelation as _chain_autocorrelation

MIN_LOCAL_FIELD_BIN_COUNT = 30


def beta_to_temperature(beta: float) -> float:
    """T = 1/beta. beta<=0 raises -- see module docstring."""
    if beta <= 0:
        raise ValueError(
            f"beta must be positive (got {beta!r}); beta<=0 is not a colder "
            f"or hotter model, it is a meaningless one -- exp(-beta*E) would "
            f"invert (beta<0) or flatten (beta==0) the distribution")
    return 1.0 / beta


def autocorrelation(series: Sequence[float], max_lag: int) -> list[float]:
    """Normalised autocorrelation of ONE series, lags 0..max_lag inclusive
    (acf[0] == 1.0 always). Delegates to `tsu.ess.autocorrelation` -- see
    module docstring for why this is a reuse, not a reimplementation."""
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    rho = _chain_autocorrelation(series)
    if max_lag >= len(rho):
        raise ValueError(
            f"max_lag={max_lag} exceeds the available lags ({len(rho) - 1}) "
            f"for a series of length {len(series)}")
    return [float(v) for v in rho[:max_lag + 1]]


def magnetization(draws: Sequence[Sequence[int]]) -> list[float]:
    """Mean spin per draw, s = 2*occupancy - 1 mapped before averaging
    (see module docstring). One float per draw, in [-1, 1]."""
    out = []
    for row in draws:
        arr = np.asarray(row, dtype=float)
        if arr.size == 0:
            raise ValueError("a draw with zero spins has no magnetization")
        s = 2.0 * arr - 1.0
        out.append(float(s.mean()))
    return out


def energy_histogram(series: Sequence[float], bins: int) -> tuple[list[float], list[int]]:
    """Bin edges and counts for `series` -- the distribution the energy
    TRACE only samples one point of at a time. len(edges) == len(counts)+1
    always (np.histogram's own convention, passed straight through)."""
    counts, edges = np.histogram(np.asarray(series, dtype=float), bins=bins)
    return [float(e) for e in edges], [int(c) for c in counts]


def sigmoid(x):
    """1/(1+exp(-x)), the analytic curve `local_field_response`'s
    empirical points are checked against. A plain function (not folded
    into local_field_response) so the demo's panel-drawing code can
    overlay the SAME closed form it is comparing measurements to,
    rather than a second hand-copied expression that could silently
    drift from this one."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def local_field_response(draws: Sequence[Sequence[int]], ising, bins: int
                         ) -> tuple[list[float], list[float], list[int]]:
    """Bin centres, empirical P(s=1), and per-bin counts -- see module
    docstring for the gamma = beta*(b_i + sum J_ij*s_j) convention and the
    MIN_LOCAL_FIELD_BIN_COUNT honesty rule. `ising` needs only `.biases`,
    `.weights`, `.edges`, `.beta` (duck-typed against
    tsu.passes.lower.IsingModel -- see tests/test_scope.py's _FakeIsing
    for the minimal shape). `draws` is (n_draws, n_spins) of 0/1
    occupancy, one row per raw physical sample (valid or not -- the field
    is a property of the raw chain, the same convention
    demo/lattice_app.py's energy trace already uses).

    Returns ([], [], []) for zero draws -- no data is not zero bins of
    data, but there is nothing to bin either; an empty result is the
    honest one, not a fabricated bin layout."""
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.size == 0:
        return [], [], []
    if draws_arr.ndim != 2:
        raise ValueError(
            f"expected draws as a (n_draws, n_spins) array, got shape "
            f"{draws_arr.shape}")
    n_draws, n_spins = draws_arr.shape
    biases = np.asarray(ising.biases, dtype=float)
    weights = np.asarray(ising.weights, dtype=float)
    if len(biases) != n_spins:
        raise ValueError(
            f"ising carries {len(biases)} biases but draws have {n_spins} "
            f"spins per row -- draws must be over the SAME node set ising "
            f"was built from")
    beta = float(ising.beta)

    spins = 2.0 * draws_arr - 1.0  # (n_draws, n_spins), s in {-1, +1}
    field = np.tile(biases, (n_draws, 1))  # (n_draws, n_spins)
    for k, (u, v) in enumerate(ising.edges):
        w = weights[k]
        field[:, u] += w * spins[:, v]
        field[:, v] += w * spins[:, u]
    gamma = beta * field  # (n_draws, n_spins), the local field per (draw, spin)

    gamma_flat = gamma.ravel()
    occ_flat = draws_arr.ravel()  # 0/1 occupancy; s=1 iff occupancy==1

    gmin, gmax = float(gamma_flat.min()), float(gamma_flat.max())
    if gmin == gmax:
        # A perfectly constant field (e.g. one edgeless spin, every draw
        # the same) -- widen the range so a single bin still has a real
        # (non-zero-width) extent to report, rather than dividing by zero.
        gmin, gmax = gmin - 0.5, gmax + 0.5
    bin_edges = np.linspace(gmin, gmax, bins + 1)
    bin_idx = np.clip(np.digitize(gamma_flat, bin_edges[1:-1]), 0, bins - 1)

    centers: list[float] = []
    probs: list[float] = []
    counts: list[int] = []
    for b in range(bins):
        mask = bin_idx == b
        n = int(mask.sum())
        centers.append(float(0.5 * (bin_edges[b] + bin_edges[b + 1])))
        counts.append(n)
        if n >= MIN_LOCAL_FIELD_BIN_COUNT:
            probs.append(float(occ_flat[mask].mean()))
        else:
            probs.append(float("nan"))
    return centers, probs, counts
