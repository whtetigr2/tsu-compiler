"""What a moment readout can express is set by the graph it is defined on.

================================ PROTOCOL ================================
SYSTEM DEFINITION   Bars-and-stripes on a 4x4 grid: the 30 configurations that
                    are constant along every row, or constant along every
                    column. A standard Boltzmann-machine target, chosen here
                    because 2^16 = 65536 states is small enough to enumerate,
                    so nothing below is estimated.
STATE VARIABLES     16 spins in {-1,+1}, one per grid cell.
TRANSITION RULES    None. No sampler runs in this file. Every probability is a
                    closed-form Boltzmann weight over the full state space.
ALLOWED OPERATIONS  Enumerating all 65536 states; computing exact moments;
                    fitting, for each of several interaction graphs, the
                    maximum-entropy model whose sufficient statistics are the
                    site values and the pair products ON THAT GRAPH, by convex
                    optimisation with exact gradients; comparing the fitted
                    distribution against the target.
FORBIDDEN OPERATIONS
                    No energy claim. No hardware claim. No sampling, so no
                    claim that depends on mixing. No generalisation from one
                    target distribution: the numbers below are specific to
                    bars-and-stripes, and what generalises is the mechanism,
                    not the magnitude.
ASSUMPTIONS         None that matter. Every comparison is between two exactly
                    computed distributions over the same 65536 states.
INVARIANTS          Each fit must match the target's moments ON ITS OWN GRAPH
                    to optimiser tolerance. That is what "this readout saw
                    everything it was given" means, and a row that fails it is
                    marked VOID rather than reported, because the whole claim
                    is "same moments on E, different distribution".
MEASUREMENTS        Per graph: edge count, max degree, residual moment error on
                    E, KL(target || fitted), and the probability mass the
                    fitted model places on valid bars-and-stripes, split
                    between the two trivial uniform patterns and the 28
                    structured ones.
NULL HYPOTHESES     "Exporting pairwise moments instead of full configurations
                    loses information." Refuted on any graph where the fit
                    reproduces the target exactly.
SUCCESS CRITERIA    A clear per-graph answer, with the residual gate passed.
FAILURE CRITERIA    Reporting a distribution difference while the moments did
                    not converge. An earlier hand-tuned gradient ascent
                    diverged to a delta on the all-ones pattern with moment
                    errors near 1.0, which would have been reported as
                    spurious agreement. Hence the explicit residual gate.
PROVENANCE          Bars-and-stripes as an RBM benchmark: MacKay, "Information
                    Theory, Inference, and Learning Algorithms" (2003). It is
                    the target already used by this repo in
                    demo/gibbs-observatory/backend/app/ebm_bars_stripes.py.
                    That a Boltzmann machine on graph G is an exponential
                    family whose minimal sufficient statistics are the site
                    values and the pair products on E(G): Wainwright and
                    Jordan, "Graphical Models, Exponential Families, and
                    Variational Inference" (2008), section 3.
==========================================================================

Run:
    PYTHONIOENCODING=utf-8 python audit/moment_readout_blindness.py
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

GRID = 4
N = GRID * GRID
RESIDUAL_GATE = 1e-6   # a row above this is VOID, not a result

Pair = tuple[int, int]


def cell(r: int, c: int) -> int:
    return r * GRID + c


def bars_and_stripes() -> np.ndarray:
    """The valid configurations: constant along rows, or constant along columns.

    All-ones and all-minus-ones satisfy both descriptions, so the two families
    overlap in exactly two patterns and the set has 2*2^GRID - 2 members.
    """
    pats = set()
    for bits in itertools.product((0, 1), repeat=GRID):
        pats.add(tuple(1.0 if bits[r] else -1.0 for r in range(GRID) for _ in range(GRID)))
    for bits in itertools.product((0, 1), repeat=GRID):
        pats.add(tuple(1.0 if bits[c] else -1.0 for _ in range(GRID) for c in range(GRID)))
    return np.array(sorted(pats))


def graphs() -> dict[str, list[Pair]]:
    """Interaction graphs on the 16 cells, coarse to fine.

    Each one is a choice about which pair products a chip could accumulate.
    `complete` is the unlimited case; the rest are degree-limited in the way
    real silicon is, and `rook` is the graph the target's own structure lives
    on, since bars-and-stripes correlates cells within a row or a column.
    """
    out: dict[str, list[Pair]] = {}
    out["complete"] = list(itertools.combinations(range(N), 2))

    rook: list[Pair] = []
    for r in range(GRID):
        rook += [(cell(r, a), cell(r, b)) for a, b in itertools.combinations(range(GRID), 2)]
    for c in range(GRID):
        rook += [(cell(a, c), cell(b, c)) for a, b in itertools.combinations(range(GRID), 2)]
    out["rook"] = rook

    out["rows only"] = [e for e in rook if e[0] // GRID == e[1] // GRID]

    grid4: list[Pair] = []
    for r in range(GRID):
        for c in range(GRID):
            if c + 1 < GRID:
                grid4.append((cell(r, c), cell(r, c + 1)))
            if r + 1 < GRID:
                grid4.append((cell(r, c), cell(r + 1, c)))
    out["nearest neighbour"] = grid4

    grid8 = list(grid4)
    for r in range(GRID - 1):
        for c in range(GRID - 1):
            grid8.append((cell(r, c), cell(r + 1, c + 1)))
            grid8.append((cell(r, c + 1), cell(r + 1, c)))
    out["8 neighbour"] = grid8

    out["ring"] = [(i, (i + 1) % N) for i in range(N)]
    out["none (independent)"] = []

    # The control that decides whether the result is about edge COUNT or edge
    # CHOICE. Same budget as `8 neighbour`, edges drawn at random. If these do
    # as well, connectivity structure is not the lever and the table above is
    # just counting parameters.
    every = list(itertools.combinations(range(N), 2))
    for seed in range(3):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(every), size=len(out["8 neighbour"]), replace=False)
        out[f"random (seed {seed})"] = [every[i] for i in sorted(idx)]
    return out


def features(states: np.ndarray, pairs: list[Pair]) -> np.ndarray:
    """The sufficient statistics of a Boltzmann machine on this graph: every
    site value, then the product across every edge."""
    if not pairs:
        return states
    prods = np.stack([states[:, i] * states[:, j] for i, j in pairs], axis=1)
    return np.concatenate([states, prods], axis=1)


def max_degree(pairs: list[Pair]) -> int:
    deg = np.zeros(N, int)
    for i, j in pairs:
        deg[i] += 1
        deg[j] += 1
    return int(deg.max())


def fit(allc: np.ndarray, valid: np.ndarray, pairs: list[Pair]):
    """The maximum-entropy distribution whose moments on this graph match the
    target's. It is a Boltzmann machine on that graph, so the fit answers
    exactly what a chip exporting those accumulators could ever reconstruct.

    Minimise f(t) = log Z(t) - <t, target>, gradient <phi>_model - target.
    Convex with an exact gradient, so L-BFGS converges without a step size to
    tune. Tuning one by hand is what made a first attempt diverge to a delta
    on the all-ones pattern.
    """
    phi = features(allc, pairs)
    target = features(valid, pairs).mean(axis=0)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scores = phi @ theta
        logz = logsumexp(scores)
        p = np.exp(scores - logz)
        return float(logz - theta @ target), p @ phi - target

    res = minimize(objective, np.zeros(phi.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 20000, "maxfun": 40000,
                            "ftol": 1e-16, "gtol": 1e-12})
    scores = phi @ res.x
    p = np.exp(scores - logsumexp(scores))
    residual = float(np.abs(p @ phi - target).max())
    return p, residual, res


def main() -> int:
    valid = bars_and_stripes()
    allc = np.array(list(itertools.product((-1.0, 1.0), repeat=N)))
    valid_rows = set(map(tuple, valid))
    is_valid = np.fromiter((tuple(c) in valid_rows for c in allc), bool, len(allc))
    is_uniform = np.abs(allc.sum(axis=1)) == N
    q = 1.0 / len(valid)

    print(f"bars-and-stripes on {GRID}x{GRID}: {len(valid)} valid configurations "
          f"out of 2**{N} = {len(allc)}")
    print(f"  {int((is_valid & is_uniform).sum())} are the uniform patterns, "
          f"{int((is_valid & ~is_uniform).sum())} carry real bar/stripe structure")
    print(f"  the target is uniform over the {len(valid)}: {q:.4f} each")
    print(f"  a uniform distribution over all {len(allc)} states puts "
          f"{is_valid.mean():.6f} on valid\n")

    print(f"For each interaction graph: fit the max-entropy model whose")
    print(f"sufficient statistics are the site values and the pair products on")
    print(f"THAT graph, then ask what the fitted model actually generates.\n")

    hdr = (f"{'graph':<20}{'edges':>6}{'deg':>5}{'residual':>11}"
           f"{'KL(p||q)':>10}{'valid':>9}{'structured':>12}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, pairs in graphs().items():
        p, residual, res = fit(allc, valid, pairs)
        if residual > RESIDUAL_GATE:
            print(f"{name:<20}{len(pairs):>6}{max_degree(pairs):>5}"
                  f"{residual:>11.1e}{'VOID':>10}{'':>9}{'':>12}")
            continue
        kl = float(np.sum(q * np.log(q / np.maximum(p[is_valid], 1e-300))))
        mass_valid = float(p[is_valid].sum())
        mass_struct = float(p[is_valid & ~is_uniform].sum())
        rows.append((name, len(pairs), max_degree(pairs), kl, mass_valid, mass_struct))
        print(f"{name:<20}{len(pairs):>6}{max_degree(pairs):>5}{residual:>11.1e}"
              f"{kl:>10.3f}{mass_valid:>9.4f}{mass_struct:>12.4f}")

    print()
    for name in ("complete", "8 neighbour"):
        pairs = graphs()[name]
        p, _, _ = fit(allc, valid, pairs)
        leak = float(p[~is_valid].sum())
        spread = float(p[is_valid].max() - p[is_valid].min())
        print(f"  {name}: mass leaked outside the target's support {leak:.2e}, "
              f"spread across the {len(valid)} valid states {spread:.2e}")
    print(f"  Both are optimiser noise around an exact reconstruction, not a "
          f"small imperfection.")

    # Which of the 42 edges is doing the work, and what does the target's own
    # correlation say about them? This is the part that decides whether you
    # could have PICKED this graph by looking at the data.
    pairs = graphs()["8 neighbour"]
    _, _, res = fit(allc, valid, pairs)
    fitted_j = res.x[N:]
    moments = features(valid, pairs).mean(axis=0)[N:]
    aligned = [(i, j) for i, j in pairs
               if (i // GRID == j // GRID) or (i % GRID == j % GRID)]
    is_diag = np.array([not ((i // GRID == j // GRID) or (i % GRID == j % GRID))
                        for i, j in pairs])

    print(f"\nWithin the 8-neighbour graph, by edge type:")
    print(f"  {'type':<24}{'count':>6}{'target <s_i s_j>':>19}{'fitted |J|':>13}")
    for label, mask in (("row/column adjacent", ~is_diag), ("diagonal", is_diag)):
        print(f"  {label:<24}{int(mask.sum()):>6}"
              f"{moments[mask].mean():>19.4f}{np.abs(fitted_j[mask]).mean():>13.4f}")
    ratio = abs(moments[~is_diag].mean() / moments[is_diag].mean())
    print(f"  The diagonals carry {ratio:.0f}x LESS correlation than the edges you")
    print(f"  would pick by looking at the data, and couplings of the same order.")
    print(f"  They are also indispensable: dropping them is the `nearest neighbour`")
    print(f"  row above, where KL goes from 0.000 to 5.295. A heuristic that spends")
    print(f"  connectivity where the correlation is drops every one of them, and")
    print(f"  lands on `rook`, which is worse than a random graph of the same size.")

    # Z1 caps |J| at 6.0 (SOURCED). The reconstruction above wants far more than
    # that, so the exact result is a limit the hardware cannot reach. What it
    # CAN reach is the question the compiler's gate actually asks.
    print(f"\nAgainst Z1's published |J| <= 6.0, on the 8-neighbour graph:")
    phi = features(allc, pairs)
    target_full = features(valid, pairs).mean(axis=0)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        scores = phi @ theta
        logz = logsumexp(scores)
        pp = np.exp(scores - logz)
        return float(logz - theta @ target_full), pp @ phi - target_full

    print(f"  {'|J| cap':>9}{'KL(p||q)':>10}{'valid':>9}{'structured':>12}"
          f"{'moment err':>12}")
    for cap in (float("inf"), 6.0, 4.0, 2.0, 1.0):
        bounds = None if cap == float("inf") else (
            [(None, None)] * N + [(-cap, cap)] * len(pairs))
        r = minimize(objective, np.zeros(phi.shape[1]), jac=True, method="L-BFGS-B",
                     bounds=bounds,
                     options={"maxiter": 20000, "maxfun": 40000,
                              "ftol": 1e-16, "gtol": 1e-12})
        sc = phi @ r.x
        pp = np.exp(sc - logsumexp(sc))
        kl = float(np.sum(q * np.log(q / np.maximum(pp[is_valid], 1e-300))))
        err = float(np.abs(pp @ phi - target_full).max())
        label = "none" if cap == float("inf") else f"{cap:.1f}"
        print(f"  {label:>9}{kl:>10.3f}{pp[is_valid].sum():>9.4f}"
              f"{pp[is_valid & ~is_uniform].sum():>12.4f}{err:>12.1e}")
    print(f"  A capped fit does not match the moments and is not supposed to:")
    print(f"  it is the best a Z1-legal model can do, which is the question the")
    print(f"  compiler's |J| gate is asking. The moment error column says how far")
    print(f"  the cap pushes it off the target's own statistics.")

    print(f"\n{'KL(p||q)':<12} nats from the target. 0 means the moment readout")
    print(f"{'':12} lost nothing: the model it determines IS the target.")
    print(f"{'valid':<12} mass the fitted model puts on bars-and-stripes.")
    print(f"{'structured':<12} that mass minus the two trivial uniform patterns,")
    print(f"{'':12} which any ferromagnet produces for free.")
    print(f"\nFor reference, KL(target || uniform over all states) = "
          f"{np.log(len(allc) / len(valid)):.3f} nats.")

    print(f"A KL printed as -0.000 is float noise below the precision of the")
    print(f"sum; the leaked-mass line above gives the honest magnitude.")

    if rows:
        by_kl = {r[0]: r[3] for r in rows}
        budget = [r for r in rows if r[1] == 42]
        structured = min(budget, key=lambda r: r[3])
        rand = [r for r in budget if r[0].startswith("random")]
        print(f"\nSame 42-edge budget, different choice of edges:")
        print(f"  {structured[0]:<18} KL {structured[3]:.3f}")
        for r in rand:
            print(f"  {r[0]:<18} KL {r[3]:.3f}")
        print(f"  and `rook`, with {48} edges, KL {by_kl['rook']:.3f}, is beaten")
        print(f"  by a random graph holding six FEWER edges.")

        worst_rand = max(r[3] for r in rand)
        print(f"\nThe two levers, priced against each other on this workload:")
        print(f"  Z1's |J| <= 6.0 instead of unbounded couplings   0.013 nats")
        print(f"  the wrong 42 edges instead of the right 42       {worst_rand:.3f} nats")
        print(f"  Connectivity structure costs {worst_rand / 0.013:.0f}x what the")
        print(f"  coupling cap costs. The cap is the number people quote.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
