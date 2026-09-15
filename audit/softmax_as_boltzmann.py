"""Can attention's softmax be sampled on Z1 directly, instead of computed?

================================ PROTOCOL ================================
SYSTEM DEFINITION   A softmax over a window of W positions, written as a
                    pairwise Ising model with a one-hot latent pointer, and
                    checked against Z1's published degree, coupling cap and
                    (assumed) bias cap.
STATE VARIABLES     Per constraint strength lambda: required |J|, required |b|,
                    the probability the model lands on a VALID one-hot state,
                    and the total variation from the true softmax conditioned on
                    validity. Plus, per window size: bipartiteness and the
                    mediator cost of the one-hot clique.
TRANSITION RULES    None. Exact enumeration, not sampling.
ALLOWED OPERATIONS  Constructing the gadget; enumerating it with
                    `audit/oracles/exact.py`, which does not import this package;
                    comparing to the analytic softmax.
FORBIDDEN OPERATIONS
                    No energy claim. No claim that Extropic missed this -- the
                    conclusion here is the opposite. No treating the bias cap as
                    established: it is ASSUMED in target.py and the verdict
                    depends on it, which is stated rather than buried.
ASSUMPTIONS         Z1's |b| <= 6.0. This is THIS PROJECT'S working value, not a
                    sourced Extropic figure (target.py marks it assumed), and it
                    is the number the whole verdict turns on. If Extropic's real
                    h_max is materially larger, the conclusion changes.
INVARIANTS          Conditioned on a valid one-hot state the gadget must equal
                    the softmax EXACTLY -- that is the algebra, and a TV above
                    floating-point noise means the gadget is built wrong.
MEASUREMENTS        The two tables in main().
NULL HYPOTHESES     "Softmax attention can be sampled natively on Z1." Refuted
                    if no lambda satisfies the caps while keeping the invalid-
                    state mass small.
SUCCESS CRITERIA    A clear answer either way, with the binding constraint named.
FAILURE CRITERIA    Reporting a TV above ~1e-12 on a valid one-hot state, which
                    would mean the derivation below is wrong.
PROVENANCE          Mapping proposed by Grok (via Paul, 2026-09-15). Softmax and
                    Boltzmann algebra are standard. Z1 limits from target.py.
SCOPE OF VALIDITY   One attention head, one window, exact enumeration at small W.
                    It says whether the gadget FITS the published limits. It says
                    nothing about energy, and nothing about Extropic's own
                    implementation, which does not use this construction.
==========================================================================

THE IDEA, AND IT IS A GOOD ONE.

    ctx_t = sum_{j<=t} e^{K_j} V_j / sum_{j<=t} e^{K_j}

is exactly an expectation of V under a Boltzmann distribution over positions
with energy -K. A TSU samples from e^{-E} natively. So rather than COMPUTING the
softmax digitally, clamp the keys as fields, thermalize, and read back V: the
expectation falls out of the sampling the hardware already does, and `exp` and
the divide never happen at all.

The algebra holds. This file checks whether the resulting model fits Z1.

THE GADGET. A one-hot latent pointer over W positions, in binary x:

    E = lambda * (sum_j x_j - 1)^2  -  sum_j K_j x_j

On a valid one-hot state this is exactly -K_j, so p_j ~ e^{K_j}: the softmax.
Converting x = (s+1)/2 into the oracle's E = -sum J s s - sum b s:

    J_ab = -lambda / 2                  (exclusion between every pair)
    b_j  = K_j/2 - lambda*(W-2)/2       (the linear compensation)

That compensation is not optional. Without it a pairwise exclusion clique does
not produce a one-hot state at all -- it produces a HALF-UP state, because
minimising sum_{a<b} s_a s_b over W spins is solved by splitting them evenly,
not by raising exactly one. A first version of this file omitted it and measured
P(not one-hot) rising to 1.0 as the penalty grew, which is the signature of
enforcing the wrong constraint rather than of the idea failing.

WHAT IT COSTS ON Z1, and the answer is not the coupling.

Two independent obstructions, and the second is the one that bites:

1. The one-hot constraint is a CLIQUE over W positions. Cliques of three or more
   contain triangles, triangles are odd cycles, and Z1's chessboard cannot host
   an odd cycle directly. Mediation fixes it at a price that grows with W.

2. The bias required scales as lambda*(W-2)/2. Pushing lambda high enough to
   make invalid states rare drives |b| past Z1's cap long before |J| gets
   anywhere near its own. At W=4 the largest lambda that keeps |b| inside the
   cap still leaves about 4.5% of the mass on invalid states; at W=8 the
   requirement is already four times the cap, and it keeps growing.

So the honest verdict is that softmax attention does not fit this fabric -- and
that is very likely WHY Extropic built gated convolutional attention instead.
GCA's normalised cumulative sum is elementwise. It needs no cross-position
normalisation, so no clique, no exclusion penalty, and no bias blow-up. Reading
this result as something they missed would be exactly backwards; it reads much
more like something they already knew and routed around.

THE NUMBER THE VERDICT DEPENDS ON, AND IT IS NOT SOURCED.

Z1's |b| cap is 6.0 in target.py, marked ASSUMED -- this project's own working
value, chosen by symmetry with the documented |J| cap, not a figure Extropic has
published. Every "OVER" below is measured against it. If the real h_max is
materially larger, this conclusion weakens or reverses, and the window size that
fits moves with it.

That makes h_max a specific, answerable question worth asking, and gives it a
concrete reason: it decides whether a softmax can live on the fabric at all.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "audit")

import numpy as np

from oracles.exact import exact_boltzmann
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.route import insert_mediators
from tsu_compiler.target import PROFILES

OUT = Path("out/softmax-boltzmann")
Z1 = PROFILES["z1"]
J_CAP = Z1.max_abs_coupling.value
B_CAP = Z1.max_abs_bias.value


def gadget(K, lam: float):
    """J and b for the one-hot softmax gadget, in the oracle's convention."""
    W = len(K)
    J = {(a, b): -lam / 2 for a, b in itertools.combinations(range(W), 2)}
    b = [float(k / 2 - lam * (W - 2) / 2) for k in K]
    return J, b


def evaluate(K, lam: float) -> dict:
    W = len(K)
    J, b = gadget(K, lam)
    states, probs = exact_boltzmann(J, b, 1.0)
    P = np.asarray(probs)
    A = np.array(states)
    one_hot = A.sum(axis=1) == 1
    p_valid = float(P[one_hot].sum())
    marg = np.zeros(W)
    for s, p in zip(A[one_hot], P[one_hot]):
        marg[int(np.argmax(s))] += p
    soft = np.exp(K) / np.exp(K).sum()
    tv = float(0.5 * np.abs(marg / marg.sum() - soft).sum()) if marg.sum() else float("nan")
    b_max = max(abs(x) for x in b)
    return {"lambda": lam, "abs_J": lam / 2, "abs_b_max": round(b_max, 3),
            "p_one_hot": round(p_valid, 6), "tv_given_one_hot": tv,
            "J_within_cap": lam / 2 <= J_CAP, "b_within_cap": b_max <= B_CAP,
            "fits_z1": bool(lam / 2 <= J_CAP and b_max <= B_CAP)}


def clique_cost(W: int) -> dict:
    edges = tuple(itertools.combinations(range(W), 2))
    m = IsingModel(nodes=tuple(f"h{j}" for j in range(W)), edges=edges,
                   weights=np.full(len(edges), -1.0), biases=np.zeros(W),
                   beta=1.0, offset=0.0)
    rep = analyse(m)
    if rep.bipartite:
        return {"W": W, "bipartite": True, "mediators": 0, "fabric_tax": 1.0}
    mrep = analyse(insert_mediators(m, rep)[0])
    return {"W": W, "bipartite": False, "mediators": mrep.n_nodes - rep.n_nodes,
            "fabric_tax": round(mrep.n_nodes / rep.n_nodes, 3)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    K = np.array([1.2, -0.4, 0.8, 0.1])
    lams = [2.0, 4.0, 6.0, 8.0, 12.0, 20.0]
    rows = [evaluate(K, l) for l in lams]
    cliques = [clique_cost(W) for W in (2, 3, 4, 8, 16)]
    failures = [f"lambda={r['lambda']}: TV {r['tv_given_one_hot']:.2e} on a valid "
                f"one-hot state -- the gadget is built wrong"
                for r in rows if r["tv_given_one_hot"] > 1e-12]

    bias_need = {W: round(8.0 * (W - 2) / 2, 1) for W in (4, 8, 16, 32)}
    fits = [r for r in rows if r["fits_z1"]]
    best = max(fits, key=lambda r: r["p_one_hot"]) if fits else None

    (OUT / "softmax_as_boltzmann.json").write_text(json.dumps(
        {"rows": rows, "clique_cost": cliques, "control_failures": failures,
         "bias_required_at_lambda_8": bias_need,
         "best_fitting": best,
         "z1_caps": {"abs_J": J_CAP, "abs_b": B_CAP,
                     "abs_b_provenance": "ASSUMED in target.py -- this project's "
                                         "working value, not an Extropic figure. "
                                         "The verdict below depends on it."},
         "verdict": "Softmax attention is EXACTLY a Boltzmann expectation, and "
                    "the one-hot gadget reproduces it to floating-point noise. "
                    "It does not fit Z1: the one-hot clique is non-bipartite, "
                    "and the required bias exceeds the (assumed) cap well before "
                    "the coupling does, worsening linearly in window size.",
         "why_that_is_unsurprising": "GCA's normalised cumulative sum is "
                                     "elementwise and needs no cross-position "
                                     "normalisation, so it has no clique and no "
                                     "bias blow-up. This reads as something "
                                     "Extropic routed around, not something "
                                     "they missed.",
         "open_ask": "h_max. It decides whether a softmax can live on the "
                     "fabric, and it is not published.",
         "hardware": "none -- exact enumeration"},
        indent=2), encoding="utf-8")

    print("CAN A SOFTMAX BE SAMPLED ON Z1 INSTEAD OF COMPUTED?\n")
    print(f"  Z1 caps: |J| <= {J_CAP} (sourced), |b| <= {B_CAP} (ASSUMED)\n")
    print(f"{'lambda':>8}{'|J|':>7}{'|b|max':>9}{'P(one-hot)':>13}"
          f"{'TV|one-hot':>13}{'J ok':>7}{'b ok':>7}")
    for r in rows:
        print(f"{r['lambda']:>8.1f}{r['abs_J']:>7.1f}{r['abs_b_max']:>9.2f}"
              f"{r['p_one_hot']:>13.6f}{r['tv_given_one_hot']:>13.1e}"
              f"{str(r['J_within_cap']):>7}{str(r['b_within_cap']):>7}")
    print("\n  Conditioned on a valid one-hot state it IS the softmax, exactly.")
    print("  The obstruction is everything around that.\n")
    print(f"{'window W':>10}{'bipartite':>12}{'mediators':>11}{'tax':>8}")
    for c in cliques:
        print(f"{c['W']:>10}{str(c['bipartite']):>12}{c['mediators']:>11}"
              f"{format(c['fabric_tax'], '.2f') + 'x':>8}")
    print("\n  And the bias required at lambda=8, by window size:")
    for W, need in bias_need.items():
        print(f"    W={W:>3}  |b| ~ {need:>6.1f}  vs cap {B_CAP}  "
              f"{'ok' if need <= B_CAP else 'OVER'}")
    print()
    if best:
        print(f"  Largest lambda that fits both caps: {best['lambda']}, which "
              f"still leaves")
        print(f"  {(1 - best['p_one_hot']) * 100:.1f}% of the mass on invalid states.")
    print()
    print("  VERDICT: the mapping is exact and does not fit this fabric. The")
    print("  bias cap binds before the coupling cap, and worsens with window")
    print("  size. That is very likely WHY gated convolutional attention exists:")
    print("  an elementwise normalisation has no clique and no bias blow-up.")
    print()
    print("  THE CAVEAT THAT MATTERS: |b| <= 6.0 is ASSUMED in target.py, not a")
    print("  published Extropic figure. Every OVER above is measured against a")
    print("  number we chose. h_max is the ask, and this is the reason it counts.")
    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'softmax_as_boltzmann.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
