"""The same visibility, sampled by THRML rather than by a loop I wrote.

================================ PROTOCOL ================================
SYSTEM DEFINITION   First-hit visibility on a (column x depth) lattice, written
                    as couplings and biases, compiled through this project's own
                    passes and sampled by `thrml`'s block Gibbs -- Extropic's
                    library, not a hand-written Gibbs loop.
STATE VARIABLES     One spin per (column, depth). A solid cell is a strong
                    BIAS toward "stopped". It is NOT clamped, and that is the
                    correction this file carries -- see below.
TRANSITION RULES    `thrml.block_sampling.sample_states`, reached through
                    `tsu_compiler.backends.thrml_backend.sample_chains`.
ALLOWED OPERATIONS  Deriving the couplings from the energy; compiling; sampling;
                    decoding; comparing against an exact raycast and against the
                    numpy reference implementation.
FORBIDDEN OPERATIONS
                    No hand-written Gibbs in the solve. No raycaster in the solve
                    path -- it scores the answer and nothing else. No hardware
                    claim: thrml simulates on CPU/JAX.
ASSUMPTIONS         Occupancy along each ray is input, precomputed from level
                    geometry, the way a prompt is input to a sampler. At
                    NOISE > 0 that input is WRONG about some cells, on purpose.
INVARIANTS          The lattice must be bipartite (grid, orthogonal couplings
                    only) or `build_program` cannot colour it. The energy handed
                    to thrml must be the same energy `visibility_as_inference.py`
                    samples in numpy -- same constants, same signs, same scale.
MEASUREMENTS        Agreement with the exact raycast, on a clean input and on a
                    corrupted one, with the couplings ON and OFF.
NULL HYPOTHESES     (1) "The answer only appears because of how I wrote the
                    sampler." Refuted if thrml reproduces what numpy gets from
                    the same couplings. That is the original reason for this file.
                    (2) "The couplings are inert and the decode is doing the
                    work." This is not hypothetical -- it was TRUE of the first
                    version of this model, and R23 is the retraction. The check
                    is an ablation: zero the couplings and see if the answer
                    moves.
SUCCESS CRITERIA    thrml within a few points of numpy on every cell of the
                    ablation table; a LARGE coupling gain on the corrupted input;
                    bipartite lattice.
FAILURE CRITERIA    thrml disagreeing with numpy at the same beta; the couplings
                    making no difference on the corrupted input, which would mean
                    they are still inert; a non-bipartite lattice.
                    NOT a failure: a small or negative coupling gain on the CLEAN
                    input. That is the expected, honest result, and it is
                    reported rather than hidden.
PROVENANCE          Transcribed directly from the field computed in
                    `visibility_as_inference.sample`:
                        field = bias  +  (W_MONO/2) * chain neighbours
                                      +   W_COH     * column neighbours
                    so the IR carries biases of +-W and weights of W_MONO/2 and
                    W_COH. Constants are IMPORTED from that module rather than
                    retyped, so the two cannot drift apart.
SCOPE OF VALIDITY   2D first-hit depth. thrml on CPU. No silicon exists to run
                    this on, and no timing or energy claim is made.
==========================================================================

WHY THIS FILE EXISTS, AND IT IS TWO CORRECTIONS.

The first: `audit/visibility_as_inference.py` sampled this model with a Gibbs
loop written for that file, in numpy. That leaves an obvious objection open --
maybe the result depends on how the sampler was written rather than on the
energy -- and it fails a standard this project holds everywhere else. The whole
point of compiling to an Ising model is that somebody else's sampler can run it.

The second is worse, and writing this file is what exposed it. This model used
to CLAMP solid cells off. Porting it to thrml meant writing the energy out as
real couplings, and the algebra then said what the prose had not: with the walls
clamped, every cell is independent. A free cell takes 1 with probability
sigmoid(2*beta*W_FREE) = 0.999994, a wall is pinned to 0, and "first 0" is
simply "first wall". The couplings could be deleted without changing a single
decoded column, and the decode -- a linear scan for the nearest wall -- was
doing the entire job. That is a raycast in a different notation.

Every control in the old version passed: agreement 100%, zero free-space stops,
0% at the cold control, bipartite, reproduced by thrml. Every one of those is
consistent with inert couplings, because none of them asked whether the
couplings mattered. The missing check is an ablation -- turn a term off and see
if the answer moves -- and this file now runs one.

A wall is a strong bias here, not a clamp. A clamp cannot be outvoted by
anything; a bias can. That is the difference between a lattice that decorates a
scan and one that can repair an input the scan would have to trust.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "audit")

import numpy as np

from tsu_compiler.backends.thrml_backend import sample_chains
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.program import build_program
from tsu_compiler.preflight.check import preflight
from tsu_compiler.target import PROFILES

# The constants are IMPORTED, not retyped. If they drift, the two files stop
# describing the same model and the cross-check below stops meaning anything.
from visibility_as_inference import (  # noqa: E402
    W_COH,
    W_FREE,
    W_MONO,
    W_WALL,
    biases,
    corrupt,
    decode,
    exact_raycast,
    level_occupancy,
)
from visibility_as_inference import sample as numpy_sample  # noqa: E402

OUT = Path("out/visibility")

N_COLS, N_DEPTH = 48, 32
NOISE = 0.08
BETA = 1.0
N_CHAINS, N_WARMUP, STEPS = 8, 4000, 4


def build(observed, beta: float, couplings: bool = True) -> IsingModel:
    """The energy as a real IsingModel.

    A wall is a bias of -W_WALL, free space +W_FREE. Nothing is clamped, so the
    couplings are able to outvote the input -- which is the entire point, and
    was impossible in the clamped version.
    """
    n_cols, n_depth = observed.shape
    idx = lambda c, k: c * n_depth + k
    edges, weights = [], []

    for c in range(n_cols):
        for k in range(n_depth):
            if k + 1 < n_depth:                      # chain, along depth
                edges.append((idx(c, k), idx(c, k + 1)))
                # SIGN: this project's IR weight IS the oracle's J, NOT its
                # negation -- verified in audit/diagnostic_control.py, where IR
                # weight -1.0 measured as ANTIferromagnetic. Writing a negative
                # here produced a perfectly alternating 1010101... chain, which
                # is what antiferromagnetic looks like, and is a trap this
                # project has walked into three times.
                # SCALE: W_MONO/2 per neighbour, because the reference applies
                # half the chain strength to each of the two neighbours.
                weights.append(W_MONO / 2.0 if couplings else 0.0)
            if c + 1 < n_cols:                       # coherence, across columns
                edges.append((idx(c, k), idx(c + 1, k)))
                weights.append(W_COH if couplings else 0.0)

    order = sorted(range(len(edges)), key=lambda i: edges[i])
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(n_cols * n_depth)),
        edges=tuple(edges[i] for i in order),
        weights=np.array([weights[i] for i in order]),
        biases=biases(observed).reshape(-1).astype(float),
        beta=beta, offset=0.0)


def run_thrml(observed, truth, beta: float, couplings: bool,
              n_samples: int = 40) -> dict:
    """Sample the energy with thrml and score the decode."""
    model = build(observed, beta, couplings=couplings)
    rep = analyse(model)
    prog = build_program(model, rep)
    draws = np.asarray(sample_chains(prog, n_chains=N_CHAINS,
                                     n_samples=n_samples, n_warmup=N_WARMUP,
                                     steps_per_sample=STEPS, seed=0))
    n_cols, n_depth = observed.shape
    last = draws[:, -1, :].reshape(N_CHAINS, n_cols, n_depth)
    agree = [float(np.mean(decode(last[ch]) == truth)) for ch in range(N_CHAINS)]
    return {"agreement_mean": float(np.mean(agree)),
            "agreement_best": float(np.max(agree)),
            "bipartite": bool(rep.bipartite),
            "colour_blocks": rep.colour_blocks,
            "pbits": rep.n_nodes,
            "degree": rep.max_degree}


def run_numpy(observed, truth, beta: float, couplings: bool) -> float:
    """The same energy under the reference sampler, for cross-checking."""
    v, _ = numpy_sample(observed, beta,
                        w_mono=W_MONO if couplings else 0.0,
                        w_coh=W_COH if couplings else 0.0)
    return float(np.mean(decode(v) == truth))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    solid = level_occupancy(N_COLS, N_DEPTH)
    truth = exact_raycast(solid)
    clean = solid
    noisy = corrupt(solid, NOISE)

    # The front door for the hardware question, rather than a hand-rolled
    # degree/cap check. Carries gate provenance, mediators, placement, node
    # budget and the distinct-coupling count in one call.
    pf = preflight(build(clean, BETA), PROFILES["z1"])
    print(f"preflight vs z1: {pf.verdict} (mediators {pf.mediators}, "
          f"placed {pf.placed}, {pf.distinct_couplings} distinct |J|)")
    for g in pf.gates:
        if g.status != "ok":
            print(f"  GATE {g.name}: {g.status} -- {g.note}")
    print()

    rows = []
    for label, observed in (("clean", clean), (f"{NOISE:.0%} noise", noisy)):
        for couplings in (False, True):
            t = run_thrml(observed, truth, BETA, couplings)
            n = run_numpy(observed, truth, BETA, couplings)
            rows.append({"input": label, "couplings": couplings,
                         "thrml": round(t["agreement_mean"], 4),
                         "thrml_best": round(t["agreement_best"], 4),
                         "numpy": round(n, 4),
                         "gap": round(abs(t["agreement_mean"] - n), 4),
                         "bipartite": t["bipartite"], "degree": t["degree"],
                         "pbits": t["pbits"]})

    def get(inp, cpl):
        return next(r for r in rows if r["input"] == inp and r["couplings"] is cpl)

    clean_gain = get("clean", True)["thrml"] - get("clean", False)["thrml"]
    noisy_gain = (get(f"{NOISE:.0%} noise", True)["thrml"]
                  - get(f"{NOISE:.0%} noise", False)["thrml"])

    failures = []
    worst = max(r["gap"] for r in rows)
    if worst > 0.12:
        failures.append(
            f"thrml and numpy disagree by up to {worst:.1%} on the same energy; "
            f"the answer depends on the sampler, which is the thing this file "
            f"exists to rule out")
    if noisy_gain < 0.20:
        failures.append(
            f"the couplings buy only {noisy_gain:+.1%} on a corrupted input; "
            f"they are not load-bearing and this is R23 all over again")
    if not all(r["bipartite"] for r in rows):
        failures.append("the lattice is not bipartite; block Gibbs cannot apply")

    (OUT / "visibility_on_thrml.json").write_text(json.dumps(
        {"rows": rows, "control_failures": failures,
         "clean_coupling_gain": round(clean_gain, 4),
         "noisy_coupling_gain": round(noisy_gain, 4),
         "beta": BETA, "noise": NOISE,
         "sampler": "thrml.block_sampling.sample_states via "
                    "tsu_compiler.backends.thrml_backend.sample_chains",
         "walls": "strong bias (-W_WALL), NOT clamped -- see audit/findings/R23",
         "constants": {"W_MONO": W_MONO, "W_FREE": W_FREE,
                       "W_WALL": W_WALL, "W_COH": W_COH},
         "raycaster_role": "ground truth for scoring only",
         "hardware": "none -- thrml on CPU/JAX"},
        indent=2), encoding="utf-8")

    print("VISIBILITY, SAMPLED BY THRML\n")
    r0 = rows[0]
    print(f"  {r0['pbits']:,} pbits, degree {r0['degree']}, "
          f"bipartite {r0['bipartite']}, nothing clamped\n")
    print(f"  {'input':>12}{'couplings':>12}{'thrml':>10}{'numpy':>10}{'gap':>8}")
    for r in rows:
        print(f"  {r['input']:>12}{('on' if r['couplings'] else 'off'):>12}"
              f"{r['thrml']:>10.1%}{r['numpy']:>10.1%}{r['gap']:>8.1%}")
    print()
    print(f"  coupling gain, clean input:     {clean_gain:+.1%}")
    print(f"  coupling gain, {NOISE:.0%} corrupted:   {noisy_gain:+.1%}")
    print()
    print("  Two things are being checked here, not one.")
    print("  thrml agreeing with numpy says the answer is in the ENERGY and")
    print("  not in my sampler. The ablation says the COUPLINGS are what")
    print("  produce it -- the check the clamped version never ran, and would")
    print("  have failed. A near-zero gain on the clean input is expected: a")
    print("  clean occupancy already contains the answer and a scan recovers")
    print("  it. The corrupted input is where the lattice earns its keep.")
    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'visibility_on_thrml.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
