"""R15 -- independent oracle test: the app's own sampled distribution vs
`audit.oracles.exact.exact_boltzmann`, on an exactly-enumerable instance,
with a statistical criterion STATED BELOW BEFORE the sampler is ever run
(plan Wave 2's own explicit requirement -- "do not tune sampling
parameters until it passes").

===========================================================================
PRE-REGISTERED CRITERION (written before this script was ever executed;
left completely unchanged after -- see the git history of this file if
that is ever in doubt):

  A Pearson chi-square goodness-of-fit test, H0: the sampler's stationary
  distribution over the model's 2**n states equals exact_boltzmann's
  distribution. Expected counts = N * exact_probs (every expected count
  will be checked to be >= 5, the standard rule-of-thumb minimum for the
  chi-square approximation to be valid -- if it is not, N is increased,
  decided here, before running, not after seeing a failing result).
  Significance level alpha = 0.01 (one-sided: reject only for an
  excessively large statistic -- a p-value has no lower-tail direction
  here, it already discounts on both sides). PASS iff p >= alpha, i.e. the
  observed sample counts are not so different from the exact distribution
  that they would occur by chance less than 1% of the time under a
  perfectly correct sampler.

  Sampler configuration: `tsu_compiler.backends.thrml_backend.sample` -- the SAME
  function `demo/lattice_app.py` imports as `thrml_sample` and calls from
  its own unclamped-tick worker (see truth_table.md's `sample` row) -- at
  the app's own FIXED constants for that path: n_warmup=300,
  steps_per_sample=4 (`demo/lattice_app.py:118-119`, `N_WARMUP`/
  `STEPS_PER_SAMPLE`), n_chains=200 and n_samples=50 (10,000 total draws;
  scaled UP from the live app's own tiny per-tick n_chains so this single
  batch has enough draws for the chi-square approximation to be valid on
  an 8-state model -- the live app instead accumulates draws across many
  ticks to build up its own energy_trace; this script draws them all in
  one call instead, but through the identical sampler entry point,
  n_warmup and steps_per_sample). Seed=20260903 (today's date, chosen
  arbitrarily and fixed before running, not cherry-picked after).

  If this fails, the plan is explicit: that IS the finding. This script
  will not be re-run with different parameters to make it pass.
===========================================================================
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "audit"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from tsu_compiler.ir import Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef  # noqa: E402
from tsu_compiler.passes.lower import lower  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.program import build_program  # noqa: E402
from tsu_compiler.backends.thrml_backend import sample as thrml_sample  # noqa: E402

from oracles.exact import exact_boltzmann  # noqa: E402

ALPHA = 0.01
N_CHAINS = 200
N_SAMPLES = 50
N_WARMUP = 300          # demo/lattice_app.py's own N_WARMUP constant
STEPS_PER_SAMPLE = 4    # demo/lattice_app.py's own STEPS_PER_SAMPLE constant
SEED = 20260903

# The SAME asymmetric 3-spin instance R3 used (deliberately asymmetric --
# no relabelling/sign symmetry to hide a bug behind), reused here rather
# than inventing a new one, per the "don't hand-roll a second copy of
# something already built and checked this wave" principle.
x0, x1, x2 = Var("x0", Binary()), Var("x1", Binary()), Var("x2", Binary())
terms = (
    Linear(LinearForm({VarRef("x0"): 1.0}), 0.7),
    Linear(LinearForm({VarRef("x1"): 1.0}), -0.3),
    Linear(LinearForm({VarRef("x2"): 1.0}), 1.1),
    Product(LinearForm({VarRef("x0"): 1.0}), LinearForm({VarRef("x1"): 1.0}), 0.9),
    Product(LinearForm({VarRef("x1"): 1.0}), LinearForm({VarRef("x2"): 1.0}), -0.5),
    Product(LinearForm({VarRef("x0"): 1.0}), LinearForm({VarRef("x2"): 1.0}), 0.2),
)
model = EnergyModel(variables=(x0, x1, x2), terms=terms, beta=1.0)
im = lower(model)
report = analyse(im)
prog = build_program(im, report)

print("=" * 78)
print("Instance: same asymmetric 3-spin model as R3.")
print(f"nodes={im.nodes} edges={im.edges} weights={im.weights.tolist()} "
      f"biases={im.biases.tolist()} offset={im.offset} beta={im.beta}")
print(f"Graph colouring: colour_blocks={report.colour_blocks} bipartite={report.bipartite} "
      f"(chromatic block Gibbs is exact for ANY proper colouring, bipartite or not)")
print("=" * 78)

# Exact reference (independent oracle).
J = {(u, v): float(w) for (u, v), w in zip(im.edges, im.weights)}
b = [float(x) for x in im.biases]
exact_states, exact_probs = exact_boltzmann(J, b, im.beta)
exact_probs = np.asarray(exact_probs)
n_states = len(exact_states)

expected_min = N_CHAINS * N_SAMPLES * exact_probs.min()
print(f"\nExpected count of the RAREST state at N={N_CHAINS*N_SAMPLES}: "
      f"{expected_min:.2f} (chi-square rule-of-thumb wants >= 5)")
if expected_min < 5:
    raise SystemExit(
        f"PRE-REGISTRATION VIOLATION GUARD: expected_min={expected_min:.2f} < 5 -- "
        f"this script refuses to silently proceed with an invalid chi-square "
        f"approximation. (Not expected to fire for this instance/N; if it does, "
        f"this is reported as-is, not patched after the fact.)")

print(f"\nRunning tsu_compiler.backends.thrml_backend.sample (== demo/lattice_app.py's "
      f"thrml_sample) with n_chains={N_CHAINS}, n_samples={N_SAMPLES}, "
      f"n_warmup={N_WARMUP}, steps_per_sample={STEPS_PER_SAMPLE}, seed={SEED} ...")
draws = thrml_sample(prog, n_chains=N_CHAINS, n_samples=N_SAMPLES,
                     n_warmup=N_WARMUP, steps_per_sample=STEPS_PER_SAMPLE, seed=SEED)
print(f"Got {draws.shape[0]} draws of {draws.shape[1]} spins each.")

state_index = {s: i for i, s in enumerate(exact_states)}
observed = np.zeros(n_states, dtype=int)
for row in draws:
    observed[state_index[tuple(int(x) for x in row)]] += 1
N = int(observed.sum())
expected = exact_probs * N

print(f"\n{'state':>10} | {'observed':>9} | {'expected (exact*N)':>19} | {'observed/N':>11} | {'exact_prob':>11}")
for s, o, e, p in zip(exact_states, observed, expected, exact_probs):
    print(f"{str(s):>10} | {o:9d} | {e:19.3f} | {o/N:11.6f} | {p:11.6f}")

chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
tv = 0.5 * np.abs(observed / N - exact_probs).sum()

print(f"\nPearson chi-square statistic = {chi2_stat:.6f}  (df={n_states-1})")
print(f"p-value = {p_value:.6f}")
print(f"Total variation distance (observed vs exact) = {tv:.6f}")
print(f"Total draws N = {N}")

verdict = "PASS" if p_value >= ALPHA else "FAIL"
print(f"\nPRE-REGISTERED CRITERION: PASS iff p >= alpha={ALPHA}")
print(f"RESULT: p={p_value:.6f} -> {verdict}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  chi2={chi2_stat:.4f}  df={n_states-1}  p={p_value:.6f}  alpha={ALPHA}  "
      f"TV={tv:.6f}  N={N}  -> {verdict}")
