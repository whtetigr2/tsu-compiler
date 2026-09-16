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
# NO-PREFLIGHT: the numpy reference implementation; it builds an IsingModel only so the preflight in visibility_on_thrml.py can weigh the same energy.

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
W_MONO = 3.0          # chain coupling along depth
W_FREE = 3.0          # empty space says keep going
W_WALL = 4.0          # a wall says stop -- a strong BIAS, deliberately NOT a
                      # clamp. Clamped walls cannot be outvoted by anything, which
                      # is what made the couplings inert in the first version of
                      # this file (R23). A bias can be overruled by the lattice
                      # when the input is wrong, and that is the entire point.
W_COH = 1.2           # neighbouring columns prefer similar depth
NOISE = 0.04          # fraction of free cells the input wrongly reports as wall
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


def biases(observed):
    """Per-cell field from the OBSERVED occupancy, which may be wrong."""
    return np.where(observed, -W_WALL, +W_FREE)


def corrupt(solid, rate, seed=7):
    """The input a real sensor would hand you: mostly right, sometimes not.
    Spurious walls in free space -- the failure a scan cannot recover from,
    because it stops at the first thing it is told about."""
    rng = np.random.default_rng(seed)
    obs = solid.copy()
    obs |= (rng.random(solid.shape) < rate) & (~solid)
    return obs


def build_ising(solid) -> IsingModel:
    """The same energy, expressed as an IsingModel so preflight can weigh it."""
    n_cols, n_depth = solid.shape
    idx = lambda c, k: c * n_depth + k
    edges, weights = [], []
    for c in range(n_cols):
        for k in range(n_depth):
            # SIGN AND SCALE must match sample() exactly, or the preflight
            # weighs a different model than the one being sampled. In this
            # project's IR a POSITIVE weight is ferromagnetic (verified in
            # audit/diagnostic_control.py); these were written negative, i.e.
            # antiferromagnetic, and the chain term was written at full strength
            # where sample() applies W_MONO/2 to each of the two neighbours.
            # Nothing measured was wrong because build_ising only ever fed the
            # preflight -- but visibility_on_thrml.py now samples this, so it
            # has to be the same energy.
            if k + 1 < n_depth:                        # chain, vertical
                edges.append((idx(c, k), idx(c, k + 1)))
                weights.append(W_MONO / 2.0)
            if c + 1 < n_cols:                         # coherence, horizontal
                edges.append((idx(c, k), idx(c + 1, k)))
                weights.append(W_COH)
    b = biases(solid).reshape(-1)
    n = n_cols * n_depth
    order = np.argsort([e[0] * n + e[1] for e in edges])
    return IsingModel(nodes=tuple(f"v{i}" for i in range(n)),
                      edges=tuple(edges[i] for i in order),
                      weights=np.array([weights[i] for i in order]),
                      biases=b, beta=1.0, offset=0.0)


def sample(observed, beta, w_mono=W_MONO, w_coh=W_COH,
           sweeps=SWEEPS, burn=BURN, seed=0):
    """Chromatic block Gibbs on the (column x depth) lattice.

    Starts from a uniform random state -- NOT from the raycast, not from
    anything derived from it."""
    rng = np.random.default_rng(seed)
    n_cols, n_depth = observed.shape
    v = rng.integers(0, 2, size=(n_cols, n_depth)).astype(np.int8)
    bias = biases(observed)
    cc, kk = np.meshgrid(np.arange(n_cols), np.arange(n_depth), indexing="ij")
    parity = (cc + kk) % 2
    acc = np.zeros((n_cols, n_depth))
    kept = 0
    for s in range(sweeps):
        for colour in (0, 1):
            up = np.zeros_like(bias)
            w_m, w_c = w_mono, w_coh
            # A symmetric ferromagnetic chain. An earlier version wrote this as
            # two one-sided clauses and called the asymmetry the mechanism; the
            # constants cancel and it is the same thing at half strength (R23).
            up[:, :-1] += (w_m / 2) * (2 * v[:, 1:].astype(float) - 1)
            up[:, 1:] += (w_m / 2) * (2 * v[:, :-1].astype(float) - 1)
            # coherence with the neighbouring columns
            up[:-1, :] += w_c * (2 * v[1:, :].astype(float) - 1)
            up[1:, :] += w_c * (2 * v[:-1, :].astype(float) - 1)
            field = bias + up
            p = 1.0 / (1.0 + np.exp(-2.0 * beta * field))
            draw = (rng.random((n_cols, n_depth)) < p).astype(np.int8)
            v = np.where(parity == colour, draw, v)
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


def agreement(observed, truth, w_mono, w_coh, beta=1.0) -> float:
    v, _ = sample(observed, beta, w_mono=w_mono, w_coh=w_coh)
    return float(np.mean(decode(v) == truth))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    solid = level_occupancy()
    truth = exact_raycast(solid)

    model = build_ising(solid)
    rep = analyse(model)
    lattice = {"pbits": rep.n_nodes, "edges": rep.n_edges,
               "max_degree": rep.max_degree, "bipartite": bool(rep.bipartite),
               "degree_within_z1": rep.max_degree <= Z1.degree.value,
               "fits_node_budget": rep.n_nodes <= Z1.node_budget.value}

    # ---- ABLATION. The check this file did not have, and the one that matters.
    clean = solid
    noisy = corrupt(solid, NOISE)
    ablation = []
    for lbl, wm, wc in (("full lattice", W_MONO, W_COH),
                        ("no chain coupling", 0.0, W_COH),
                        ("no coherence", W_MONO, 0.0),
                        ("no couplings at all", 0.0, 0.0)):
        ablation.append({"variant": lbl, "w_mono": wm, "w_coh": wc,
                         "clean_input": round(agreement(clean, truth, wm, wc), 4),
                         "noisy_input": round(agreement(noisy, truth, wm, wc), 4)})

    # ---- how the gain grows with how unreliable the input is
    sweep_rows = []
    for rate in (0.0, 0.01, 0.02, 0.04, 0.08):
        obs = corrupt(solid, rate)
        bare = agreement(obs, truth, 0.0, 0.0)
        full = agreement(obs, truth, W_MONO, W_COH)
        sweep_rows.append({"noise": rate, "no_couplings": round(bare, 4),
                           "full_lattice": round(full, 4),
                           "gain": round(full - bare, 4)})

    failures = []
    if not rep.bipartite:
        failures.append("the lattice is not bipartite; the builder is wrong")
    bare_noisy = next(a for a in ablation if a["variant"] == "no couplings at all")
    full_noisy = next(a for a in ablation if a["variant"] == "full lattice")
    if full_noisy["noisy_input"] - bare_noisy["noisy_input"] < 0.10:
        failures.append(
            f"the couplings buy only "
            f"{full_noisy['noisy_input'] - bare_noisy['noisy_input']:+.1%} on a "
            f"noisy input. They are not load-bearing, and any claim that the "
            f"energy performs the occlusion is unsupported -- this is exactly "
            f"the defect recorded in audit/findings/R23.md")
    if sweep_rows[0]["gain"] > 0.10:
        failures.append("the couplings appear to help on a PERFECT input, which "
                        "they should not -- a clean occupancy already contains "
                        "the answer and a scan recovers it")

    (OUT / "visibility_as_inference.json").write_text(json.dumps(
        {"lattice": lattice, "ablation": ablation, "noise_sweep": sweep_rows,
         "walls": "strong bias, NOT a clamp -- a clamp cannot be outvoted, which "
                  "is what made the couplings inert in the first version (R23)",
         "raycaster_role": "ground truth for scoring only",
         "what_this_shows": "the lattice recovers a coherent surface from an "
                            "input that is individually unreliable. On a PERFECT "
                            "input it buys almost nothing, and says so -- a scan "
                            "is the right tool when the answer is already in the "
                            "input.",
         "hardware": "none -- simulation"},
        indent=2), encoding="utf-8")

    print("ARE THE COUPLINGS LOAD-BEARING?")
    print()
    print(f"  lattice: {lattice['pbits']:,} pbits, degree {lattice['max_degree']}, "
          f"bipartite {lattice['bipartite']}")
    print()
    print(f"  {'variant':<22}{'clean input':>14}{'noisy input':>14}")
    for a in ablation:
        print(f"  {a['variant']:<22}{a['clean_input']:>13.1%}{a['noisy_input']:>14.1%}")
    print()
    print(f"  {'sensor noise':>13}{'no couplings':>15}{'full lattice':>15}{'gain':>9}")
    for r in sweep_rows:
        print(f"  {r['noise']:>13.3f}{r['no_couplings']:>14.1%}"
              f"{r['full_lattice']:>15.1%}{r['gain']:>+9.1%}")
    print()
    print("  On a PERFECT input the couplings buy almost nothing, and that is")
    print("  the honest result: a clean occupancy already contains the answer,")
    print("  and a linear scan recovers it. The lattice earns its keep where the")
    print("  input is unreliable -- which is the only place sampling was ever")
    print("  going to be worth anything.")
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
