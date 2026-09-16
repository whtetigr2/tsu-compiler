"""Would DOOM's sampler lattice fit on Z1? Not as written. Yes with one change.

================================ PROTOCOL ================================
SYSTEM DEFINITION   The Potts lattice behind the "DOOM on the sampler" demo at
                    whtetigr2.github.io/doom.html, read out of its own source:
                    a 4-neighbour grid of about 25,000 cells, K=64 states,
                    energy H = sum_i bias(x_i) + J * sum_edges [x_i != x_j],
                    relaxed by chromatic block Gibbs on a checkerboard.
STATE VARIABLES     Per encoding and per K: pbits, coupling edges, max degree,
                    bipartiteness, mediator cost, required bias, and whether the
                    whole lattice clears Z1's node budget.
TRANSITION RULES    None. Structural, answered before sampling.
ALLOWED OPERATIONS  Building the factor graph each encoding implies; analyse();
                    insert_mediators(); comparing against target.py's Z1 limits.
FORBIDDEN OPERATIONS
                    No claim that the demo runs, or could run, on hardware. No
                    TSU silicon exists to test on. No energy or speed claim. The
                    demo is and remains browser simulation.
ASSUMPTIONS         Z1's |b| <= 6.0, which target.py marks ASSUMED -- our own
                    working value, not an Extropic figure. It decides the Potts
                    verdict below, exactly as it decided the softmax one in
                    audit/softmax_as_boltzmann.py.
INVARIANTS          A 4-neighbour grid is bipartite by chessboard parity. Any
                    encoding that adds no within-cell coupling must therefore
                    come back bipartite; if it does not, the builder is wrong.
MEASUREMENTS        The two tables in main().
NULL HYPOTHESES     "A 64-state Potts lattice of this size cannot fit Z1."
                    True as written, and FALSE under a different encoding of the
                    same picture -- which is the finding.
SUCCESS CRITERIA    A fitting configuration at the shipped resolution and state
                    count, or a clear statement of what has to give.
FAILURE CRITERIA    Reporting a fit while degree, budget or bias is exceeded.
PROVENANCE          Lattice parameters read from doom.html's own halfSweep() and
                    energy comment. Z1 limits from target.py.
SCOPE OF VALIDITY   TOPOLOGY AND CAPACITY ONLY. It says the graph fits. It does
                    NOT say the picture looks the same -- the Hamming encoding
                    changes the energy, and whether that still renders DOOM is an
                    empirical question this cannot answer.
==========================================================================

WHAT THE DEMO ACTUALLY DOES, from its own source rather than its description:

    function halfSweep(color){ ...
      for(let cc=(color^(r&1)); cc<RW; cc+=2){ ...
        if(m&1)tmp[s[top+cc]]++;  if(m&2)tmp[s[bot+cc]]++;
        if(m&4)tmp[s[i-1]]++;     if(m&8)tmp[s[i+1]]++;

Four neighbours, swept in checkerboard parity, Gumbel-max over K states, with a
bitmask that CUTS edges across geometry boundaries. The spatial graph is already
a chessboard -- the same shape Z1 is -- so the parity question that cost Z1T's
attention layer 1.19x to 1.56x costs this nothing.

The obstruction is the STATE SPACE, not the geometry.

Z1's pbits are binary. A K=64 Potts variable has to be encoded, and the encoding
decides whether the coupling stays pairwise:

  ONE-HOT: [x != y] = 1 - sum_k x_k y_k, which IS pairwise in the one-hot bits.
  But one-hot needs an exclusion constraint inside every cell, and that
  constraint is a CLIQUE over K bits. Cliques of three or more are non-bipartite,
  and -- the part that actually kills it -- the constraint's linear compensation
  term makes the required bias grow like lambda*(K-2)/2. At K=4 it is already
  over the cap. At K=64 it is about 248 against a cap of 6.

  That is the same obstruction, in the same week, as
  audit/softmax_as_boltzmann.py found for attention. Both times the one-hot
  gadget's BIAS requirement binds long before its coupling does.

  HAMMING: keep a plain binary code and charge for disagreement per bit,
  H = sum bias + J * Hamming(x_i, x_j). Hamming distance is a sum of independent
  per-bit disagreements, so it is pairwise with NO within-cell constraint at all:
  no clique, no exclusion, no bias blow-up. Six bits carry the full 64 states at
  degree 4 of 16, bipartite, 1.00x, using 150,000 of 269,568 pbits.

WHAT THAT COSTS, AND IT IS NOT NOTHING. Hamming is not Potts. Potts charges the
same J whether two cells differ slightly or completely; Hamming charges by code
distance. For texture indices that is a genuinely different model, and whether
it still looks like DOOM is an empirical question -- this file cannot answer it.

It may also be an improvement. Potts treats "brick beside slightly different
brick" exactly like "brick beside sky". Assign codes so visually similar
textures sit at small Hamming distance and the energy starts expressing
something Potts cannot: that some disagreements matter more than others. That is
a design question, and it is the interesting one.
"""
# NO-PREFLIGHT: checks degree, caps and parity inline against target.PROFILES; its committed results predate preflight() and rewriting would invalidate them.

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.route import insert_mediators
from tsu_compiler.target import PROFILES

OUT = Path("out/doom-preflight")
Z1 = PROFILES["z1"]
CELLS = 25_000          # as the demo states
STATES = 64             # K=64, as the demo states
PROBE_SIDE = 6          # small grid; degree and parity are local properties


def _mk(n, edges):
    e = tuple(sorted(edges))
    return IsingModel(nodes=tuple(f"p{i}" for i in range(n)), edges=e,
                      weights=np.full(len(e), 1.0), biases=np.zeros(n),
                      beta=1.0, offset=0.0)


def potts_onehot(side: int, K: int) -> IsingModel:
    idx = lambda r, c, k: (r * side + c) * K + k
    edges = set()
    for r in range(side):
        for c in range(side):
            for a, b in itertools.combinations(range(K), 2):
                edges.add((idx(r, c, a), idx(r, c, b)))
            for dr, dc in ((1, 0), (0, 1)):
                r2, c2 = r + dr, c + dc
                if r2 < side and c2 < side:
                    for k in range(K):
                        u, v = idx(r, c, k), idx(r2, c2, k)
                        edges.add((min(u, v), max(u, v)))
    return _mk(side * side * K, edges)


def hamming(side: int, bits: int) -> IsingModel:
    idx = lambda r, c, b: (r * side + c) * bits + b
    edges = set()
    for r in range(side):
        for c in range(side):
            for dr, dc in ((1, 0), (0, 1)):
                r2, c2 = r + dr, c + dc
                if r2 < side and c2 < side:
                    for b in range(bits):
                        u, v = idx(r, c, b), idx(r2, c2, b)
                        edges.add((min(u, v), max(u, v)))
    return _mk(side * side * bits, edges)


def measure(model, per_cell: int, label: str, bias_need: float | None) -> dict:
    rep = analyse(model)
    if rep.bipartite:
        tax, med = 1.0, 0
    else:
        mrep = analyse(insert_mediators(model, rep)[0])
        med = mrep.n_nodes - rep.n_nodes
        tax = round(mrep.n_nodes / rep.n_nodes, 2)
    total = CELLS * per_cell * (tax if tax > 1 else 1)
    row = {"encoding": label, "pbits_per_cell": per_cell,
           "max_degree": rep.max_degree,
           "degree_ok": rep.max_degree <= Z1.degree.value,
           "bipartite": bool(rep.bipartite), "mediators_on_probe": med,
           "fabric_tax": tax,
           "pbits_at_full_size": int(total),
           "budget_ok": int(total) <= Z1.node_budget.value}
    if bias_need is not None:
        row["bias_required"] = round(bias_need, 1)
        row["bias_ok"] = bias_need <= Z1.max_abs_bias.value
    row["fits_z1"] = bool(row["degree_ok"] and row["budget_ok"]
                          and row.get("bias_ok", True))
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    onehot = [measure(potts_onehot(PROBE_SIDE, K), K, f"Potts one-hot, K={K}",
                      8.0 * (K - 2) / 2) for K in (2, 4, 8, 16)]
    ham = [measure(hamming(PROBE_SIDE, b), b,
                   f"Hamming, {b} bits = {2 ** b} states", None)
           for b in (1, 2, 4, 6, 8)]

    failures = [f"{r['encoding']}: reported fits_z1 while a limit is exceeded"
                for r in onehot + ham
                if r["fits_z1"] and not (r["degree_ok"] and r["budget_ok"])]
    grid = hamming(PROBE_SIDE, 4)
    if not analyse(grid).bipartite:
        failures.append("a 4-neighbour grid with no within-cell coupling came "
                        "back non-bipartite; the builder is wrong")

    best = [r for r in ham if r["fits_z1"] and 2 ** r["pbits_per_cell"] >= STATES]
    headroom = None
    if best:
        b = min(best, key=lambda r: r["pbits_per_cell"])
        headroom = {
            "encoding": b["encoding"],
            "pbits_used": b["pbits_at_full_size"],
            "pbits_spare": Z1.node_budget.value - b["pbits_at_full_size"],
            "die_spare_pct": round(100 * (1 - b["pbits_at_full_size"]
                                          / Z1.node_budget.value), 1),
            "ports_used_per_node": b["max_degree"],
            "ports_spare_per_node": Z1.degree.value - b["max_degree"],
            "max_cells_at_this_encoding":
                Z1.node_budget.value // b["pbits_per_cell"],
            "approx_square_side":
                int((Z1.node_budget.value // b["pbits_per_cell"]) ** 0.5)}

    (OUT / "doom_lattice_preflight.json").write_text(json.dumps(
        {"demo": {"cells": CELLS, "states": STATES,
                  "neighbourhood": "4-neighbour grid, checkerboard parity",
                  "energy": "sum bias(x_i) + J*sum_edges [x_i != x_j]",
                  "source": "doom.html halfSweep() and its energy comment"},
         "potts_one_hot": onehot, "hamming": ham,
         "headroom_at_64_states": headroom, "control_failures": failures,
         "bias_cap_provenance": "|b| <= 6.0 is ASSUMED in target.py, not an "
                                "Extropic figure. It decides the Potts verdict.",
         "scope": "topology and capacity only. Hamming is NOT Potts -- it "
                  "changes the energy, and whether the result still looks like "
                  "DOOM is empirical and not answered here.",
         "hardware": "none. The demo is browser simulation and remains so."},
        indent=2), encoding="utf-8")

    print("WOULD DOOM'S LATTICE FIT Z1?\n")
    print(f"  demo: {CELLS:,} cells, K={STATES}, 4-neighbour grid, checkerboard")
    print(f"  Z1:   {Z1.node_budget.value:,} pbits, degree {Z1.degree.value}, "
          f"|b| <= {Z1.max_abs_bias.value} (ASSUMED)\n")
    print("  AS WRITTEN -- Potts, one-hot encoded:")
    print(f"  {'encoding':<26}{'deg':>5}{'bip':>6}{'tax':>7}{'pbits':>12}"
          f"{'|b|':>7}{'fits':>6}")
    for r in onehot:
        print(f"  {r['encoding']:<26}{r['max_degree']:>5}"
              f"{str(r['bipartite'])[:1]:>6}{format(r['fabric_tax'], '.2f'):>7}"
              f"{r['pbits_at_full_size']:>12,}{r['bias_required']:>7.0f}"
              f"{str(r['fits_z1']):>6}")
    print(f"\n  At K={STATES} the one-hot constraint would need |b| ~ "
          f"{8.0 * (STATES - 2) / 2:.0f} against a cap of "
          f"{Z1.max_abs_bias.value}.\n")
    print("  RE-ENCODED -- Hamming coupling, no one-hot constraint:")
    print(f"  {'encoding':<26}{'deg':>5}{'bip':>6}{'tax':>7}{'pbits':>12}{'fits':>6}")
    for r in ham:
        print(f"  {r['encoding']:<26}{r['max_degree']:>5}"
              f"{str(r['bipartite'])[:1]:>6}{format(r['fabric_tax'], '.2f'):>7}"
              f"{r['pbits_at_full_size']:>12,}{str(r['fits_z1']):>6}")
    if headroom:
        h = headroom
        print(f"\n  {h['encoding']} carries all {STATES} states and fits:")
        print(f"    {h['pbits_used']:,} of {Z1.node_budget.value:,} pbits "
              f"({h['die_spare_pct']}% of the die spare)")
        print(f"    {h['ports_used_per_node']} of {Z1.degree.value} ports used "
              f"({h['ports_spare_per_node']} spare per node)")
        print(f"    room for {h['max_cells_at_this_encoding']:,} cells, about "
              f"{h['approx_square_side']}x{h['approx_square_side']}")
    print("\n  WHAT THIS DOES NOT SAY: Hamming is not Potts. It charges by code")
    print("  distance rather than a flat J for any disagreement, so it is a")
    print("  different energy and may render differently. That is empirical.")
    print("  It may also be better: assign codes so similar textures sit close")
    print("  and the energy can say some disagreements matter more than others,")
    print("  which flat Potts cannot.")
    if failures:
        print("\nCONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"\n  -> {OUT / 'doom_lattice_preflight.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
