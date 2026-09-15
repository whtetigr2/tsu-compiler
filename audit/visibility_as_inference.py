"""Visibility without a raycaster: which wall you see, as a sampled quantity.

================================ PROTOCOL ================================
SYSTEM DEFINITION   First-hit visibility for a 2D level: for each screen column,
                    which surface the viewer sees. Posed as an Ising model on a
                    (column x depth) lattice and solved by sampling, with no
                    raycaster anywhere in the solve path.
STATE VARIABLES     v[c][k] in {0,1}: "the ray for column c has reached depth k
                    without hitting anything yet." Depth of first hit is the
                    number of 1s in the column's chain.
TRANSITION RULES    Chromatic block Gibbs, the same two-colour sweep Z1 performs.
ALLOWED OPERATIONS  Building the energy from the level's occupancy grid;
                    sampling it; decoding a depth per column; comparing that
                    decode against an exact raycast used ONLY as ground truth.
FORBIDDEN OPERATIONS
                    The raycaster must never appear in the solve path. It scores
                    the answer; it does not produce it, seed it, or bias it. No
                    hardware claim -- this is simulation.
ASSUMPTIONS         That occupancy along each column's ray is precomputed from
                    level geometry. That is the INPUT, the way a prompt is input
                    to a sampler; the claim is about visibility being inferred,
                    not about geometry being conjured.
INVARIANTS          The lattice must be BIPARTITE -- it is a grid with couplings
                    only between orthogonal neighbours, so a (c+k) parity
                    colouring must 2-colour it. If it does not, the builder is
                    wrong. And the monotone chain must never be violated in a
                    decoded sample at high beta.
MEASUREMENTS        Per beta: agreement between the sampled depth profile and
                    the exact raycast, count of monotonicity violations, and the
                    lattice's degree, parity and Z1 fit.
NULL HYPOTHESES     "Sampling cannot recover the exact visibility solution."
                    Refuted if agreement reaches 100% as beta grows. The control
                    against fooling ourselves is the LOW-beta row: if agreement
                    were high everywhere, the energy would not be doing the work
                    and something else would be.
SUCCESS CRITERIA    Agreement -> 100% as beta rises, and visibly worse at low
                    beta. Lattice bipartite, within Z1's degree.
FAILURE CRITERIA    Agreement high at low beta (the answer is coming from
                    somewhere other than the energy); monotonicity violated at
                    high beta; the lattice non-bipartite. NOT a failure: cells
                    reading 1 behind the first hit. That region is unconstrained
                    by construction and cannot reach the decode.
PROVENANCE          The encoding is the standard monotone-chain formulation of
                    first-hit visibility. Z1 limits from target.py.
SCOPE OF VALIDITY   2D, first-hit depth only. It does not texture, shade, or
                    render anything, and it says nothing about whether a full
                    scene looks right. It establishes one thing: that WHICH
                    SURFACE IS VISIBLE can be a sample rather than a computation.
==========================================================================

WHY THIS EXISTS.

A demo of DOOM on a sampler currently raycasts the scene conventionally, writes
the finished frame into `target[]`, and relaxes a Potts lattice toward it. That
is a lattice denoising a rendered picture. The interesting claim -- that which
wall you see is itself a sample -- is not what the code does.

This is the smallest honest version of that claim. No raycaster in the solve.

THE ENCODING, and why it fits a pairwise bipartite fabric.

Depth cannot be a categorical variable here: a one-hot over D depths needs an
exclusion clique, which is non-bipartite, and whose bias requirement grows like
lambda*(D-2)/2 -- the obstruction measured in audit/softmax_as_boltzmann.py and
again in audit/doom_lattice_preflight.py. Twice was enough.

So depth is encoded as a MONOTONE CHAIN instead. v[c][k] = 1 means the ray has
got as far as depth k still travelling. Three energy terms, all pairwise or
local:

  MONOTONICITY   penalise v[c][k]=0 with v[c][k+1]=1. Once the ray stops it
                 stays stopped. This is a coupling between vertically adjacent
                 cells, and it is what makes occlusion a CONSTRAINT rather than
                 a computation: nothing behind a hit can be seen, because the
                 chain cannot turn back on.

  OCCUPANCY      a per-cell bias from the level: keep going through empty space,
                 stop at solid. This is the level geometry entering as input.

  COHERENCE      a coupling between horizontally adjacent cells, so neighbouring
                 columns prefer similar depth. Walls are continuous. This is the
                 term that makes the problem a LATTICE rather than D independent
                 one-dimensional puzzles, and it is why the result is a sample
                 from a joint distribution rather than a per-column argmin.

Every coupling is between orthogonal neighbours on a grid, so colouring by
(c+k) mod 2 two-colours it exactly. Bipartite, degree 4, which is a subgraph of
Z1's degree-16 chessboard.

WHAT THE RAYCASTER IS FOR HERE. Scoring, and nothing else. It never touches the
state, the biases or the initial configuration. If it did, this file would prove
nothing at all.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.target import PROFILES

OUT = Path("out/visibility")
Z1 = PROFILES["z1"]

N_COLS, N_DEPTH = 64, 48
W_MONO = 6.0          # strength of the "once stopped, stay stopped" constraint
W_OCC = 3.0           # how strongly geometry says keep-going / stop
W_COH = 0.6           # neighbouring columns prefer similar depth
SWEEPS, BURN = 900, 300


def level_occupancy(n_cols=N_COLS, n_depth=N_DEPTH, seed=1):
    """solid[c][k] = is there wall at depth k along column c.

    Stands in for a level: a back wall, a near pillar over some columns, and a
    mid-depth ledge over others. The SHAPE does not matter; what matters is that
    it is fixed input and the solver never sees the answer it implies."""
    solid = np.zeros((n_cols, n_depth), dtype=bool)
    solid[:, n_depth - 1] = True                       # far wall, always there
    solid[10:22, 14] = True                            # a pillar, close in
    solid[34:52, 27] = True                            # a ledge, further out
    solid[52:60, 8] = True                             # something very near
    return solid


def exact_raycast(solid):
    """GROUND TRUTH ONLY. Never used to build, seed or bias the model."""
    n_cols, n_depth = solid.shape
    out = np.empty(n_cols, dtype=int)
    for c in range(n_cols):
        hit = np.flatnonzero(solid[c])
        out[c] = hit[0] if hit.size else n_depth - 1
    return out


def biases(solid):
    """Per-cell field: empty space says keep going, solid says stop."""
    return np.where(solid, -W_OCC, +W_OCC)


def build_ising(solid) -> IsingModel:
    """The same energy, expressed as an IsingModel so preflight can weigh it."""
    n_cols, n_depth = solid.shape
    idx = lambda c, k: c * n_depth + k
    edges, weights = [], []
    for c in range(n_cols):
        for k in range(n_depth):
            if k + 1 < n_depth:                        # monotonicity, vertical
                edges.append((idx(c, k), idx(c, k + 1))); weights.append(-W_MONO)
            if c + 1 < n_cols:                         # coherence, horizontal
                edges.append((idx(c, k), idx(c + 1, k))); weights.append(-W_COH)
    b = biases(solid).reshape(-1)
    n = n_cols * n_depth
    order = np.argsort([e[0] * n + e[1] for e in edges])
    return IsingModel(nodes=tuple(f"v{i}" for i in range(n)),
                      edges=tuple(edges[i] for i in order),
                      weights=np.array([weights[i] for i in order]),
                      biases=b, beta=1.0, offset=0.0)


def sample(solid, beta, sweeps=SWEEPS, burn=BURN, seed=0):
    """Chromatic block Gibbs on the (column x depth) lattice.

    Starts from a uniform random state -- NOT from the raycast, not from
    anything derived from it."""
    rng = np.random.default_rng(seed)
    n_cols, n_depth = solid.shape
    v = rng.integers(0, 2, size=(n_cols, n_depth)).astype(np.int8)
    # A WALL IS A CONSTRAINT, NOT A PREFERENCE. Solid cells are CLAMPED to 0:
    # the ray cannot be still travelling where there is rock. Written first as a
    # soft bias, which failed -- monotonicity pulls a cell toward 1 whenever the
    # cell below it is 1, so at W_MONO > W_OCC the geometry could never stop the
    # ray and every column reported the far wall. Clamping is both correct and
    # what the hardware offers: `prog.clamped` is exactly this.
    v[solid] = 0
    bias = biases(solid)
    cc, kk = np.meshgrid(np.arange(n_cols), np.arange(n_depth), indexing="ij")
    parity = (cc + kk) % 2
    acc = np.zeros((n_cols, n_depth))
    kept = 0
    for s in range(sweeps):
        for colour in (0, 1):
            up = np.zeros_like(bias)
            # MONOTONICITY IS ASYMMETRIC, and that is the whole point.
            # v_k = 0 forces v_{k+1} = 0; v_k = 1 forces nothing. A symmetric
            # ferromagnetic coupling gets this wrong -- it penalises both
            # transitions equally, so the chain just wants to be uniform, and at
            # any reasonable strength uniformity beats the occupancy field and
            # every column reports the far wall. The control caught exactly that.
            # Each clause fires in one direction only:
            up[:, :-1] += W_MONO * v[:, 1:]              # something below is still
                                                         # travelling -> I must be too
            up[:, 1:] -= W_MONO * (1 - v[:, :-1])        # the cell above has stopped
                                                         # -> I must be stopped
            # coherence with the neighbouring columns
            up[:-1, :] += W_COH * (2 * v[1:, :] - 1)
            up[1:, :] += W_COH * (2 * v[:-1, :] - 1)
            field = bias + up
            p = 1.0 / (1.0 + np.exp(-2.0 * beta * field))
            draw = (rng.random((n_cols, n_depth)) < p).astype(np.int8)
            v = np.where(parity == colour, draw, v)
            v[solid] = 0                       # clamp re-applied every sweep
        if s >= burn:
            acc += v; kept += 1
    return v, acc / max(kept, 1)


def decode(v):
    """Depth of first hit = length of the leading run of 1s."""
    n_cols, n_depth = v.shape
    out = np.empty(n_cols, dtype=int)
    for c in range(n_cols):
        zeros = np.flatnonzero(v[c] == 0)
        out[c] = zeros[0] if zeros.size else n_depth - 1
    return out


def early_stops(v, truth):
    """Columns that stop in FREE SPACE, before the wall. This is the invariant
    that matters, and the one a wrong answer would break."""
    return int(sum(1 for c in range(v.shape[0]) if np.any(v[c, :truth[c]] == 0)))


def tail_ones(v):
    """Cells still reading 1 behind the first hit.

    NOT an error, and measuring it as one was a mistake in the first version of
    this file. Behind a wall the ray never passed, "has the ray got here" is a
    question with no answer, so the region is unconstrained -- and it is
    unconstrained in the ENERGY too, on purpose: the monotonicity penalty is paid
    once at the boundary while the free-space bias pays per cell, so a run of 1s
    behind the wall is the genuine ground state, not a failure to converge.
    The decode reads the FIRST zero, so none of it can reach the answer.
    Reported for transparency, not scored."""
    return int(np.sum((v[:, :-1] == 0) & (v[:, 1:] == 1)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    solid = level_occupancy()
    truth = exact_raycast(solid)

    model = build_ising(solid)
    rep = analyse(model)
    lattice = {"pbits": rep.n_nodes, "edges": rep.n_edges,
               "max_degree": rep.max_degree, "bipartite": bool(rep.bipartite),
               "degree_within_z1": rep.max_degree <= Z1.degree.value,
               "fits_node_budget": rep.n_nodes <= Z1.node_budget.value,
               "z1_degree": Z1.degree.value,
               "z1_node_budget": Z1.node_budget.value}

    rows, failures = [], []
    for beta in (0.05, 0.2, 0.5, 1.0, 2.0, 4.0):
        v, _ = sample(solid, beta)
        got = decode(v)
        agree = float(np.mean(got == truth))
        rows.append({"beta": beta, "agreement_with_raycast": round(agree, 4),
                     "exact_columns": int(np.sum(got == truth)),
                     "n_columns": len(truth),
                     "mean_abs_depth_error": round(float(np.mean(np.abs(got - truth))), 3),
                     "early_stops_in_free_space": early_stops(v, truth),
                     "dont_care_tail_cells": tail_ones(v)})

    if not rep.bipartite:
        failures.append("the visibility lattice came back NON-bipartite; it is a "
                        "grid with orthogonal couplings only, so the builder is wrong")
    hi = rows[-1]
    if hi["agreement_with_raycast"] < 0.99:
        failures.append(f"at beta={hi['beta']} agreement is only "
                        f"{hi['agreement_with_raycast']:.2%}; sampling is not "
                        f"recovering the exact solution")
    if hi["early_stops_in_free_space"] > 0:
        failures.append(f"{hi['early_stops_in_free_space']} columns stop in free "
                        f"space at beta={hi['beta']} -- the ray is halting where "
                        f"there is nothing to halt it")
    lo = rows[0]
    if lo["agreement_with_raycast"] > 0.5:
        failures.append(f"at beta={lo['beta']} agreement is already "
                        f"{lo['agreement_with_raycast']:.2%}. The answer is coming "
                        f"from somewhere other than the energy, and this file "
                        f"proves nothing.")

    (OUT / "visibility_as_inference.json").write_text(json.dumps(
        {"lattice": lattice, "beta_sweep": rows, "control_failures": failures,
         "encoding": "monotone depth chain; occlusion is a pairwise constraint, "
                     "not a computation",
         "raycaster_role": "ground truth for scoring only -- never used to "
                           "build, seed or bias the model",
         "scope": "2D first-hit depth. No texturing, no shading, no scene.",
         "hardware": "none -- simulation"},
        indent=2), encoding="utf-8")

    print("WHICH WALL YOU SEE, AS A SAMPLE\n")
    print(f"  lattice: {lattice['pbits']:,} pbits, {lattice['edges']:,} couplings, "
          f"degree {lattice['max_degree']}, bipartite {lattice['bipartite']}")
    print(f"  Z1: degree {Z1.degree.value} -> {'within' if lattice['degree_within_z1'] else 'OVER'}; "
          f"budget {Z1.node_budget.value:,} -> "
          f"{'fits' if lattice['fits_node_budget'] else 'OVER'}\n")
    print(f"  {'beta':>6}{'agrees with raycast':>22}{'exact cols':>12}"
          f"{'mean |err|':>12}{'early stops':>13}{'tail':>7}")
    for r in rows:
        print(f"  {r['beta']:>6.2f}{r['agreement_with_raycast']:>21.1%}"
              f"{r['exact_columns']:>8}/{r['n_columns']:<3}"
              f"{r['mean_abs_depth_error']:>12.2f}"
              f"{r['early_stops_in_free_space']:>13}{r['dont_care_tail_cells']:>7}")
    print()
    print("  The raycaster scored these. It did not produce them: the chains")
    print("  start uniformly random and are moved only by the energy.")
    print("  Low beta is the control -- if agreement were high there, the answer")
    print("  would be coming from somewhere other than the sampling.")
    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'visibility_as_inference.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
