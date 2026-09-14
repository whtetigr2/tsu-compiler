"""R4 -- sampling semantics (plan Wave 2). Independent, EMPIRICAL
re-verification of R-Wave 1's Critical finding C-1 (truth_table.md: the
live SCOPE panel's tau/ACF plot runs tsu_compiler.ess's Sokal estimator over a flat
series that is not a Markov-chain trajectory), plus direct checks of the
other named items: n_chains/n_warmup/steps_per_sample, chain persistence
between calls, and whether successive draws are independent.

This script does not re-read lattice_app.py's source a third time to
reconfirm C-1 -- that was already done directly (see R4.md's own
narrative) -- it instead REPRODUCES THE DEFECT NUMERICALLY, on real
sampler output from the real code path (`tsu_compiler.backends.thrml_backend.
sample`/`sample_chains`, the exact functions `demo/lattice_app.py` calls
as `thrml_sample`), so the finding rests on a live number, not just a
reading of two docstrings.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from tsu_compiler.ir import Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef  # noqa: E402
from tsu_compiler.passes.lower import lower  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.program import build_program  # noqa: E402
from tsu_compiler.backends.thrml_backend import sample as thrml_sample, sample_chains  # noqa: E402
from tsu_compiler import ess  # noqa: E402

# A slightly bigger instance than R3/R7/R15's 3-spin one, so there is
# enough per-chain autocorrelation structure for tau to be a meaningful
# number at all (a 3-spin, near-independent-across-sweeps chain has almost
# no lag structure to measure). 6 spins, a ring of couplings (each spin
# coupled to its two ring-neighbours) -- large enough to mix slowly at a
# high-ish beta, small enough to build without going through place/route.
n = 6
names = [f"x{i}" for i in range(n)]
variables = tuple(Var(nm, Binary()) for nm in names)
terms = []
for i in range(n):
    terms.append(Product(LinearForm({VarRef(names[i]): 1.0}),
                         LinearForm({VarRef(names[(i + 1) % n]): 1.0}), 1.4))
model = EnergyModel(variables=variables, terms=tuple(terms), beta=2.0)
im = lower(model)
report = analyse(im)
prog = build_program(im, report)

N_CHAINS = 8
N_SAMPLES = 400
N_WARMUP = 300           # demo/lattice_app.py's own N_WARMUP constant
STEPS_PER_SAMPLE = 4     # demo/lattice_app.py's own STEPS_PER_SAMPLE constant
SEED = 20260903

print("=" * 78)
print("1. CHAIN PERSISTENCE BETWEEN CALLS -- is state carried over, or is")
print("   every call a fresh, from-scratch draw given only (prog, seed)?")
print("=" * 78)
chains_a = sample_chains(prog, n_chains=N_CHAINS, n_samples=N_SAMPLES,
                         n_warmup=N_WARMUP, steps_per_sample=STEPS_PER_SAMPLE, seed=SEED)
chains_b = sample_chains(prog, n_chains=N_CHAINS, n_samples=N_SAMPLES,
                         n_warmup=N_WARMUP, steps_per_sample=STEPS_PER_SAMPLE, seed=SEED)
identical = np.array_equal(chains_a, chains_b)
print(f"Two independent calls to sample_chains() with the SAME (prog, seed, "
      f"n_chains, n_samples, n_warmup, steps_per_sample) produce IDENTICAL "
      f"output: {identical}")
print("  -> " + ("This is a PURE function of its arguments: no hidden module-level "
                 "chain state persists between calls (confirms thrml_backend.py's "
                 "own docstring claim empirically, not just by reading it)."
                 if identical else
                 "UNEXPECTED: two calls with identical arguments produced DIFFERENT "
                 "output -- would indicate hidden global/persisted state. FINDING."))

print("\n" + "=" * 78)
print("2. ARE SUCCESSIVE DRAWS WITHIN ONE UNCLAMPED TICK (n_samples=1,")
print("   n_chains>1) MUTUALLY INDEPENDENT ACROSS CHAINS?")
print("=" * 78)
# Simulate exactly what _run_unclamped_tick does at Full speed: n_chains
# parallel chains, N_SAMPLES_PER_CALL=1 each, fresh seed, fresh warmup.
tick_chains = sample_chains(prog, n_chains=6, n_samples=1, n_warmup=N_WARMUP,
                            steps_per_sample=STEPS_PER_SAMPLE, seed=SEED + 1)
flat_tick = tick_chains.reshape(-1, tick_chains.shape[-1])
print(f"One simulated unclamped tick at n_chains=6: shape {tick_chains.shape} "
      f"-> flattened {flat_tick.shape} (this IS what thrml_sample/`sample` returns "
      f"and what lattice_app.py pushes row-by-row into energy_trace).")
print("Each of the 6 rows came from an INDEPENDENT parallel chain (separate "
      "jax.random.split key, separate fresh warmup) with exactly ONE sample "
      "drawn from each -- there is no 'later in the same trajectory' relationship "
      "between any two of these 6 rows, by construction of sample_chains' own "
      "vmap-over-independent-keys implementation (thrml_backend.py:221-228).")

print("\n" + "=" * 78)
print("3. REPRODUCING C-1 NUMERICALLY: Sokal tau on a CHAIN-MAJOR-FLATTENED")
print("   multi-chain series (what render_acf_plot actually computes) vs.")
print("   the SAME data correctly shaped and gated (what tsu_compiler.ess demands)")
print("=" * 78)
energy_biases = np.asarray(im.biases)
energy_weights = np.asarray(im.weights)


def energy_of_row(row):
    s = 2.0 * row.astype(float) - 1.0
    total = im.offset
    total -= energy_biases @ s
    for k, (u, v) in enumerate(im.edges):
        total -= energy_weights[k] * s[u] * s[v]
    return total


chains = sample_chains(prog, n_chains=N_CHAINS, n_samples=N_SAMPLES, n_warmup=N_WARMUP,
                       steps_per_sample=STEPS_PER_SAMPLE, seed=SEED + 2)
energy_per_chain = np.array([[energy_of_row(chains[c, t]) for t in range(N_SAMPLES)]
                             for c in range(N_CHAINS)])   # shape (n_chains, n_samples)

# (A) CORRECT shape, exactly as tsu_compiler.ess's own contract demands and
#     demo/ess_run.py already does elsewhere in this codebase.
correct_result = ess.effective_sample_size(energy_per_chain)
print(f"(A) CORRECT (n_chains={N_CHAINS}, n_samples={N_SAMPLES}) shape, "
      f"tsu_compiler.ess.effective_sample_size: reliable={correct_result.reliable} "
      f"ess={correct_result.ess} iat={correct_result.iat} reason={correct_result.reason!r}")

# (B) render_acf_plot's OWN actual computation: chain-major-flattened into
#     ONE series (exactly `simulate.py:151`'s `chains.reshape(-1, ...)`
#     pattern, replayed here on the ENERGY series instead of the spin
#     rows -- the flattening step is identical), fed to
#     integrated_autocorrelation_time as a SINGLE chain of shape (1, N).
flattened_series = energy_per_chain.reshape(-1)   # chain 0's N_SAMPLES, then chain 1's, ...
iat_flat = ess.integrated_autocorrelation_time(flattened_series)
print(f"(B) FLATTENED (chain-major, as render_acf_plot/energy_trace.ys actually "
      f"receives it) shape (1, {flattened_series.shape[0]}), "
      f"tsu_compiler.ess.integrated_autocorrelation_time (render_acf_plot's own entry "
      f"point -- NOT effective_sample_size, no reliability gate at all): "
      f"tau={iat_flat.tau:.4f} window={iat_flat.window} "
      f"window_saturated={iat_flat.window_saturated}")
print(f"    render_acf_plot would print exactly this on screen: 'tau~{iat_flat.tau:.1f}' "
      f"-- a specific, confident-looking number, computed over data "
      f"tsu_compiler.ess's own module docstring says autocorrelation is undefined for "
      f"(N_CHAINS={N_CHAINS} mutually independent chains concatenated end to end).")

# (C) Sanity: what tau does (A)'s CORRECTLY-shaped data actually have,
#     bypassing the reliability gate (integrated_autocorrelation_time
#     directly, not effective_sample_size), for a side-by-side number.
iat_correct = ess.integrated_autocorrelation_time(energy_per_chain)
print(f"(C) Same draws, CORRECTLY shaped (n_chains, n_samples), tau estimate "
      f"(bypassing the reliability gate, for comparison only): "
      f"tau={iat_correct.tau:.4f} window={iat_correct.window} "
      f"window_saturated={iat_correct.window_saturated}")

print(f"\n  (B)'s flattened tau ({iat_flat.tau:.4f}) vs (C)'s correctly-shaped tau "
      f"({iat_correct.tau:.4f}): {'THE SAME NUMBER BY COINCIDENCE' if abs(iat_flat.tau-iat_correct.tau)<1e-9 else 'DIFFERENT NUMBERS'} "
      f"-- (B) is what a viewer of the live SCOPE panel actually sees; (C) is "
      f"never displayed anywhere in the live app (only demo/ess_run.py, an "
      f"unused-by-the-UI script, computes anything shaped like (C)).")

print("\n" + "=" * 78)
print("4. OTHER LIVE STATISTICS: are they order/chain-boundary dependent too?")
print("   (Direct re-check of C-1's own 'what was tried' scope, on THIS")
print("   script's real sampler output, not just by reading scope.py.)")
print("=" * 78)
import importlib
sys.path.insert(0, str(REPO_ROOT / "demo"))
import scope as demo_scope  # noqa: E402

flat_draws = chains.reshape(-1, chains.shape[-1])           # chain-major flattened, like the app
shuffled_draws = flat_draws[np.random.default_rng(0).permutation(flat_draws.shape[0])]

mag_in_order = demo_scope.magnetization(list(flat_draws))
mag_shuffled = demo_scope.magnetization(list(shuffled_draws))
# magnetization is per-draw; shuffling the ORDER of draws must leave the
# MULTISET of per-draw values unchanged if (and only if) it truly has no
# order-dependence -- compare the SORTED value lists, not position-by-position.
mag_order_independent = np.allclose(sorted(mag_in_order), sorted(mag_shuffled))
print(f"magnetization(): sorted values identical whether draws are chain-major "
      f"or randomly shuffled: {mag_order_independent} "
      f"(confirms this statistic carries no chain-trajectory assumption -- "
      f"it cannot, because it is a single-draw functional averaged with no "
      f"memory of position, unlike autocorrelation which is DEFINED by position).")

hist_edges_a, hist_counts_a = demo_scope.energy_histogram(
    [energy_of_row(r) for r in flat_draws], bins=10)
hist_edges_b, hist_counts_b = demo_scope.energy_histogram(
    [energy_of_row(r) for r in shuffled_draws], bins=10)
hist_order_independent = hist_counts_a == hist_counts_b and hist_edges_a == hist_edges_b
print(f"energy_histogram(): identical histogram whether draws are chain-major "
      f"or shuffled: {hist_order_independent} (same reasoning -- a histogram is "
      f"order-invariant by construction).")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  sample_chains() is a pure function of its arguments (no persisted chain "
      f"state between calls): {identical}")
print(f"  C-1 reproduced numerically: chain-major-flattened series yields a "
      f"specific displayed tau={iat_flat.tau:.4f} (window_saturated="
      f"{iat_flat.window_saturated}) with NO reliability gate, on data "
      f"tsu_compiler.ess's own effective_sample_size (the gated, correct entry point) "
      f"says: reliable={correct_result.reliable}, reason={correct_result.reason!r}")
print(f"  magnetization()/energy_histogram() independently confirmed order-invariant "
      f"(no C-1-style defect): {mag_order_independent and hist_order_independent}")
