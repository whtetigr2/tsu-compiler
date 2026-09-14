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

  autocorrelation(series, max_lag): reuses `tsu_compiler.ess.autocorrelation` (the
  compiler's own Sokal/FFT-based estimator, already validated in
  tests/test_ess.py against an AR(1) known-answer sweep -- see that
  module's docstring) rather than reimplementing a second, unvalidated
  copy. A single series is passed through `tsu_compiler.ess`'s (n_chains,
  n_samples) contract as one chain (shape (1, n)); this file only adds
  the max_lag truncation and the list[float] return shape the demo needs.

  magnetization(draws): the standard Ising order parameter, mean spin per
  draw. Draws are 0/1 OCCUPANCY (thrml's own encoding, see the thrml
  skill / tsu_compiler.passes.lower's own "spins s = 2*occupancy - 1" convention,
  the SAME mapping demo/lattice_app.py's energy_of_draw already uses) --
  so a draw is mapped s = 2*n - 1 before averaging. [1,0,1,0] -> spins
  [1,-1,1,-1] -> mean 0.0, not 0.5 (which would be the raw occupancy mean,
  a different and less informative number: the order parameter is defined
  on spins, not counts).

  energy_histogram(series, bins): a thin, honest wrapper over
  np.histogram -- the distribution the energy TRACE (a time series) only
  samples one point of at a time.

  local_field_response(draws, ising, bins): the measurable analogue of
  Extropic's DTM paper (arXiv 2510.23972) Figure 4a, which plots P(x=1)
  against bias voltage as an S-curve. THRML's chromatic block Gibbs
  conditional sampler for a spin site i is exactly P(s_i=1 | neighbours)
  = sigmoid(2*gamma_i), where gamma_i is the LOCAL FIELD (see the thrml
  skill's "How the API composes" table and `tsu_compiler.passes.lower`'s own sign
  convention, restated here since this file computes gamma independently
  rather than importing a private thrml/tsu symbol):

      IsingModel's (b, J) are defined so that thrml's energy is
      E_thrml(s) = -beta * (sum_i b_i*s_i + sum_(i,j) J_ij*s_i*s_j).
      Fixing every spin except s_i, the log-odds of s_i=+1 vs s_i=-1 is
      2*beta*(b_i + sum_{j~i} J_ij*s_j) -- call that 2*gamma_i -- so
      P(s_i=1 | neighbours) = sigmoid(2*gamma_i), gamma_i = beta*(b_i +
      sum_{j~i} J_ij*s_j).

  IMPORTANT DISTINCTION (added after fix-round 1 -- see
  task-5-6-report.md's "Fix round 1" section for the full investigation):
  this function does NOT measure that conditional directly. THRML exposes
  no public API for the local field at the moment a spin was actually
  flipped, so gamma_i here is RECONSTRUCTED POST HOC from the final
  recorded joint draw -- every OTHER spin's value in that same draw is
  read as if it were "the neighbourhood s_i was conditioned on." Because
  the reconstructed gamma is itself correlated with the very spin being
  measured, E[s_i | gamma_reconstructed] is NOT guaranteed to equal
  sigmoid(2*gamma_reconstructed) even for a perfectly correct sampler --
  the gap is expected to be largest exactly where the sigmoid is
  steepest (near gamma=0) and smallest in the saturated tails, which is
  what was actually measured against demo/receipts/small.

  A block-split check (see task-5-6-report.md) ruled OUT one specific
  alternative explanation -- per-substep staleness (the hypothesis that
  the LAST-updated colour block's snapshot would match the reconstructed
  field while the FIRST-updated block's would not): both blocks showed
  the SAME-sign, similarly-sized deviation in the SAME bins, which
  staleness alone would not produce. It did NOT rule out a genuine
  sampler defect -- the reconstruction artifact and a real defect are
  currently INDISTINGUISHABLE from this function's output alone, because
  thrml does not expose the local field at flip time. This function
  pools (gamma, s) pairs across every spin site and every draw handed
  in, bins by the RECONSTRUCTED gamma, and reports the EMPIRICAL P(s=1)
  per bin alongside its count. The caller overlays the analytic
  sigmoid(2*gamma) on the SAME axes FOR REFERENCE, not as ground truth
  the measured points are expected to match -- see render_sigmoid_plot
  in demo/lattice_app.py for the on-screen caption carrying this same
  distinction. Never smooth a measured deviation away by picking more
  flattering bins; report it, with this caveat attached.

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

from tsu_compiler.ess import autocorrelation as _chain_autocorrelation

MIN_LOCAL_FIELD_BIN_COUNT = 30


def beta_to_temperature(beta: float) -> float:
    """T = 1/beta. beta<=0 raises -- see module docstring."""
    if beta <= 0:
        raise ValueError(
            f"beta must be positive (got {beta!r}); beta<=0 is not a colder "
            f"or hotter model, it is a meaningless one -- exp(-beta*E) would "
            f"invert (beta<0) or flatten (beta==0) the distribution")
    return 1.0 / beta


def temperature_control_state(ising) -> tuple[str, str | None]:
    """Task 8: the temperature control's two explicit states --
    ("adjustable", None) or ("fixed", reason) -- keyed on EXACTLY the fact
    `tsu_compiler.passes.route.assert_beta_consistent` gates sampling on: whether
    `ising.mediator_nodes` is empty. That function is a no-op (accepts ANY
    requested beta) iff mediator_nodes is empty; this returns "adjustable"
    under precisely that condition, so the two can never silently drift
    apart -- see tests/test_scope.py's own
    test_temperature_control_state_keys_on_the_same_fact_... for a direct
    cross-check against assert_beta_consistent.

    The reason names the mediator count and the beta they were coupled at
    -- NEVER a layer name. A prior review flagged the locked-state message
    hardcoding the word "base" in demo/lattice_app.py: that would become a
    lie the moment any OTHER layer ever compiled with mediator spins, since
    "locked" is a property of THIS model, not of being named "base". The
    caller (whichever panel is showing this) already displays which layer
    is selected; this function has no opinion on that and takes no layer
    argument at all, so there is nothing left to hardcode."""
    if not ising.mediator_nodes:
        return ("adjustable", None)
    reason = (
        f"fixed at beta={ising.beta:.4g} -- {len(ising.mediator_nodes)} "
        f"mediator spin(s) were coupled at this temperature; any other "
        f"beta would silently reproduce the wrong couplings, so "
        f"assert_beta_consistent refuses it (BetaMismatchError, spec 5.3.5)")
    return ("fixed", reason)


def autocorrelation(series: Sequence[float], max_lag: int) -> list[float]:
    """Normalised autocorrelation of ONE series, lags 0..max_lag inclusive
    (acf[0] == 1.0 always). Delegates to `tsu_compiler.ess.autocorrelation` -- see
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


def per_cell_occupancy(draws: Sequence[Sequence[int]], n_world_spins: int,
                        spins_per_cell: int, grid_w: int) -> np.ndarray:
    """Task 10: per-cell mean WORLD-spin occupancy across `draws` -- the
    per-cell heatmap's own data source (demo/lattice_app.py's
    render_heatmap_image). Only the first `n_world_spins` columns of each
    draw are read (mediator spins, which follow the world spins in every
    draw row -- see demo/lattice_app.py's spin_cell_position -- belong to
    no cell and are excluded, never averaged in). Cell index -> (x, y) is
    the SAME (cell % grid_w, cell // grid_w) convention spin_cell_position
    already uses, so result[y, x] lines up directly with
    cell_block_bounds(x, y, ...) with no re-derivation needed by the
    caller. A draw contributes each cell's OWN spins_per_cell sub-spins'
    raw 0/1 occupancy, averaged first within that one draw's cell, THEN
    across draws -- RAW physical occupancy, valid or not (the same
    "every draw, valid or not" convention magnetization/energy_of_draw
    already use elsewhere in this app, not the conditional-valid subset
    DECODED WORLD shows).

    Zero draws returns an all-NaN (grid_h, grid_w) array -- honestly
    "no data yet", never a fabricated 0.0 (mirrors local_field_response's
    own NaN-for-insufficient-data convention). A grid_w/spins_per_cell
    that does not evenly divide n_world_spins is a genuine shape mismatch,
    not a "no data" case, and raises ValueError rather than silently
    guessing a layout."""
    if n_world_spins <= 0 or spins_per_cell <= 0 or grid_w <= 0:
        raise ValueError(
            f"n_world_spins ({n_world_spins}), spins_per_cell "
            f"({spins_per_cell}) and grid_w ({grid_w}) must all be positive")
    if n_world_spins % spins_per_cell != 0:
        raise ValueError(
            f"n_world_spins ({n_world_spins}) is not evenly divisible by "
            f"spins_per_cell ({spins_per_cell}) -- cannot derive a cell "
            f"count without guessing")
    n_cells = n_world_spins // spins_per_cell
    if n_cells % grid_w != 0:
        raise ValueError(
            f"cell count ({n_cells}) is not evenly divisible by grid_w "
            f"({grid_w}) -- cannot derive a grid height without guessing")
    grid_h = n_cells // grid_w
    if not draws:
        return np.full((grid_h, grid_w), np.nan)
    arr = np.asarray(draws, dtype=float)[:, :n_world_spins]
    per_draw_cell = arr.reshape(arr.shape[0], n_cells, spins_per_cell).mean(axis=2)
    cell_mean = per_draw_cell.mean(axis=0)
    return cell_mean.reshape(grid_h, grid_w)


def sigmoid(x):
    """1/(1+exp(-x)), the analytic curve `local_field_response`'s
    empirical points are plotted alongside FOR REFERENCE (not as ground
    truth they are expected to coincide with -- see that function's own
    docstring). A plain function (not folded into local_field_response)
    so the demo's panel-drawing code can overlay the SAME closed form
    it is comparing measurements to, rather than a second hand-copied
    expression that could silently drift from this one."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def local_field_response(draws: Sequence[Sequence[int]], ising, bins: int
                         ) -> tuple[list[float], list[float], list[int]]:
    """Bin centres, empirical P(s=1), and per-bin counts -- see module
    docstring for the gamma = beta*(b_i + sum J_ij*s_j) convention and the
    MIN_LOCAL_FIELD_BIN_COUNT honesty rule. `ising` needs only `.biases`,
    `.weights`, `.edges`, `.beta` (duck-typed against
    tsu_compiler.passes.lower.IsingModel -- see tests/test_scope.py's _FakeIsing
    for the minimal shape). `draws` is (n_draws, n_spins) of 0/1
    occupancy, one row per raw physical sample (valid or not -- the field
    is a property of the raw chain, the same convention
    demo/lattice_app.py's energy trace already uses).

    NOT the sampler's live conditional: gamma is RECONSTRUCTED POST HOC
    from each draw's own final joint state (thrml exposes no local field
    at flip time), so it is correlated with the spin it's paired with.
    The empirical P(s=1) this function returns is therefore not expected
    to equal sigmoid(2*gamma) near the transition even for a defect-free
    sampler -- see the module docstring's "IMPORTANT DISTINCTION" section
    for the full reasoning and the block-split evidence that ruled out
    per-substep staleness specifically (without ruling out a genuine
    sampler defect: the two remain indistinguishable from this output
    alone).

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
