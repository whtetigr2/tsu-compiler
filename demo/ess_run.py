"""Task B3: a real ESS number for demo/receipts/small, or an honest
impracticality statement -- whichever the data actually gives.

The receipt's own verification.json reads:

    ess: unavailable: N/tau=1147 (N=6400, tau~=5.58) is below the
    reliability threshold 5000 ...

That was measured at the compiler's own fixed verify parameters
(n_chains=32, n_samples=200, n_warmup=400, steps_per_sample=2 -- see
`tsu_compiler.passes.search._VERIFY_SAMPLE_PARAMS`). This script asks a narrower
question: starting from that exact same program (the receipt's own
reconstructed sampling program, never a fresh compile -- `tsuc compile` is
NEVER invoked here), how much LONGER a chain does it actually take, in
real wall-clock time measured on THIS machine, to clear
`tsu_compiler.ess.RELIABILITY_MIN_N_OVER_TAU` (5000)?

It does this the only honest way available: run progressively longer
chains (n_chains fixed at the receipt's own 32, n_samples increasing),
computing each draw's energy with the SAME formula
`tsu_compiler.passes.search._from_ising` uses (reimplemented locally below, in
`_energy_of`, to avoid importing a leading-underscore name from another
module -- the identical formula `demo/lattice_app.py`'s `energy_of_draw`
also carries, for the same reason), and feeding the unflattened
(n_chains, n_samples) energy array to `tsu_compiler.ess.effective_sample_size` --
the SAME function, SAME reliability threshold, the compiler itself uses.
NOTHING here lowers the threshold or tunes sampling to manufacture a pass;
every row below is what that exact function returned for that exact run.

Both outcomes named in the brief are results: a configuration that clears
5000 is reported with its own N, tau, N/tau and wall time; if none is
reached within this script's own time/step budget, the LAST (largest,
still-unreliable) row's own extrapolation is reported instead, plainly
labelled as an extrapolation, never as a measurement.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsu_compiler.simulate import reconstruct_program  # noqa: E402
from tsu_compiler.backends.thrml_backend import sample_chains  # noqa: E402
from tsu_compiler.ess import (RELIABILITY_MIN_N_OVER_TAU, effective_sample_size,
                     integrated_autocorrelation_time)  # noqa: E402

RECEIPT_DIR = REPO_ROOT / "demo" / "receipts" / "small"

# Held fixed at the receipt's own compile-time verify values (spec: "started
# from that exact same program") -- only n_samples (the chain LENGTH) grows
# from step to step, which is what "run progressively longer chains" means.
N_CHAINS = 32
N_WARMUP = 400
STEPS_PER_SAMPLE = 2

# n_samples schedule: doubling from the receipt's own baseline (200, which
# reproduces N=6400/tau~5.58/N-over-tau~1147 -- the receipt's own recorded
# numbers) until either the reliability threshold clears or the schedule
# runs out. Chosen empirically during this task (see b-report.md): tau
# itself grows as the chain lengthens (a short-chain Sokal-window estimate
# is biased low), so N/tau does not grow linearly with n_samples the way a
# constant-tau assumption would predict -- doubling is a robust, cheap way
# to bracket where it actually crosses, rather than guessing a single big
# jump that might overshoot the time budget for no reason.
N_SAMPLES_SCHEDULE = (200, 400, 800, 1600, 2000, 2400, 3200, 4000, 6400)

# Stop early if the cumulative wall time of every run so far would already
# exceed this many seconds -- "bound your total runtime to a few minutes"
# (the brief). Every step so far has been well under a second per run on
# this machine, so this is a generous ceiling, not an expected outcome.
TOTAL_WALL_BUDGET_S = 150.0


def _energy_of(im, row: np.ndarray) -> float:
    """E(x) for one raw {0,1} draw, from IsingModel's own documented sign
    convention (lower.py: 'sum b s + sum J s s == -E(x)', s = 2*occ-1) --
    the SAME formula `tsu_compiler.passes.search._from_ising` and
    `demo/lattice_app.py`'s `energy_of_draw` both use."""
    s = 2.0 * row.astype(float) - 1.0
    total = im.offset
    total -= float((im.biases * s).sum())
    for k, (u, v) in enumerate(im.edges):
        total -= im.weights[k] * s[u] * s[v]
    return total


@dataclass(frozen=True)
class EssRow:
    n_chains: int
    n_samples: int
    n_warmup: int
    steps_per_sample: int
    seed: int
    wall_s: float
    n_total: int
    tau_point: float          # the Sokal point estimate, ALWAYS available
                               # (integrated_autocorrelation_time never
                               # refuses) -- shown for the diagnostic "how
                               # did tau move as the chain grew" table the
                               # brief asks for, even on an unreliable row.
    n_over_tau_point: float   # n_total / tau_point, the SAME diagnostic
                               # quantity -- NOT the reliability-gated `ess`.
    reliable: bool
    reason: str
    ess: float | None         # only set (by tsu_compiler.ess itself) once reliable;
                               # this is the number that would actually be
                               # PUBLISHED, never tau_point/n_over_tau_point.


def run_one(prog, im, n_chains: int, n_samples: int, n_warmup: int,
           steps_per_sample: int, seed: int) -> EssRow:
    t0 = time.time()
    chains = sample_chains(prog, n_chains=n_chains, n_samples=n_samples,
                           n_warmup=n_warmup, steps_per_sample=steps_per_sample,
                           seed=seed)
    wall_s = time.time() - t0
    energy = np.array([[_energy_of(im, chains[ci, ti]) for ti in range(n_samples)]
                       for ci in range(n_chains)])
    est = effective_sample_size(energy)
    tau_point = integrated_autocorrelation_time(energy).tau
    n_total = n_chains * n_samples
    return EssRow(n_chains, n_samples, n_warmup, steps_per_sample, seed,
                 wall_s, n_total, tau_point, n_total / tau_point,
                 est.reliable, est.reason, est.ess)


def render_row(r: EssRow) -> str:
    verdict = "RELIABLE" if r.reliable else "below threshold"
    return (f"  n_chains={r.n_chains:<3} n_samples={r.n_samples:<5} "
           f"N={r.n_total:<7} wall={r.wall_s:6.2f}s  tau~={r.tau_point:<7.2f} "
           f"N/tau={r.n_over_tau_point:<8.0f} [{verdict}]")


def main() -> None:
    prog = reconstruct_program(str(RECEIPT_DIR))
    im = prog.ising

    print(f"ESS run -- demo/receipts/small, program reconstructed verbatim "
         f"(never recompiled). n_chains={N_CHAINS}, n_warmup={N_WARMUP}, "
         f"steps_per_sample={STEPS_PER_SAMPLE} held fixed; n_samples grows.")
    print(f"reliability threshold (tsu_compiler.ess.RELIABILITY_MIN_N_OVER_TAU): "
         f"{RELIABILITY_MIN_N_OVER_TAU:.0f}")
    print()

    rows: list[EssRow] = []
    cumulative = 0.0
    first_reliable: EssRow | None = None
    for i, n_samples in enumerate(N_SAMPLES_SCHEDULE):
        if cumulative > TOTAL_WALL_BUDGET_S:
            print(f"STOPPING: cumulative wall time {cumulative:.1f}s would "
                 f"exceed the {TOTAL_WALL_BUDGET_S:.0f}s budget -- reporting "
                 f"what was measured so far, not continuing.")
            break
        row = run_one(prog, im, N_CHAINS, n_samples, N_WARMUP,
                     STEPS_PER_SAMPLE, seed=i)
        cumulative += row.wall_s
        rows.append(row)
        print(render_row(row))
        if row.reliable and first_reliable is None:
            first_reliable = row
            break  # the brief asks for the FIRST configuration that clears it

    print()
    print(f"total wall time this run: {cumulative:.2f}s")
    print()

    if first_reliable is not None:
        r = first_reliable
        print("RESULT: a real, reliable ESS was reached.")
        print(f"  first configuration to clear N/tau>={RELIABILITY_MIN_N_OVER_TAU:.0f}: "
             f"n_chains={r.n_chains}, n_samples={r.n_samples}, "
             f"n_warmup={r.n_warmup}, steps_per_sample={r.steps_per_sample}, "
             f"seed={r.seed}")
        print(f"  N={r.n_total}, tau~={r.tau_point:.2f}, "
             f"N/tau={r.n_over_tau_point:.0f}, ESS={r.ess:.0f}, wall time "
             f"for THIS run={r.wall_s:.2f}s")
    else:
        last = rows[-1]
        print("RESULT: no configuration in this schedule cleared the "
             "threshold within the time budget.")
        if last.tau_point:
            # Extrapolation, labelled as such: N/tau grew roughly linearly
            # in N across the last few steps of THIS run's own data (see
            # b-report.md for the actual fitted rows) -- solving
            # N/tau(N) >= threshold for N gives the extrapolated chain
            # length, reported as an estimate, never dressed up as measured.
            needed_n = RELIABILITY_MIN_N_OVER_TAU * last.tau_point
            needed_samples = needed_n / N_CHAINS
            per_sample_wall = last.wall_s / last.n_samples
            projected_wall = per_sample_wall * needed_samples
            print(f"  EXTRAPOLATION from the last measured row (tau~={last.tau_point:.2f}, "
                 f"NOT assumed constant -- see b-report.md for how tau moved "
                 f"across this run's own steps): reaching N/tau>="
                 f"{RELIABILITY_MIN_N_OVER_TAU:.0f} at this tau would need "
                 f"roughly n_samples~={needed_samples:.0f} per chain "
                 f"(N~={needed_n:.0f}), projected wall time ~"
                 f"{projected_wall:.1f}s at this machine's measured "
                 f"per-sample cost -- NOT run, reported as an estimate only.")


if __name__ == "__main__":
    main()
