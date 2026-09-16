"""What does Z1's topology cost the published DTM architectures?

================================ PROTOCOL ================================
SYSTEM DEFINITION   The grid topologies used by Denoising Thermodynamic Models,
                    as published in the DTM reference implementation
                    (github.com/pschilliOrange/dtm-replication, linked from
                    Extropic's GitHub; its pyproject names the upstream as
                    extropic-ai/thrmlDenoising, which is not public). Each
                    architecture is a side_len x side_len grid with a set of
                    "jumps" (offsets), optionally toroidal.
STATE VARIABLES     Per architecture: side length, jump offsets, spins, couplings,
                    max degree, bipartiteness, and per jump whether Z1 can realise
                    it DIRECTLY or only by routing it across several hops.
TRANSITION RULES    None. This is a structural question, answered before sampling.
ALLOWED OPERATIONS  Constructing the graph from the PUBLISHED topology parameters;
                    running this project's analyse() on it; BFS over Z1's own 16
                    offsets to find the shortest legal route for a long jump.
FORBIDDEN OPERATIONS
                    No vendoring of their code -- that repository carries no
                    licence, so only our own measurements may be published. No
                    training, no sampling, and NO CLAIM THAT ANYTHING THEY
                    PUBLISHED IS WRONG. Nothing here is an error report; their
                    presets are internally consistent and satisfy their own
                    stated assertion.
ASSUMPTIONS         That the published (side_len, jumps) presets describe the
                    topology faithfully. Node counts are the GRID only; a real DTM
                    adds label nodes, so these are a floor.
INVARIANTS          Every jump satisfies (dx + dy) odd -- the reference
                    implementation asserts this itself. Our analyse() must agree
                    that the resulting graph is bipartite.
MEASUREMENTS        Spins, couplings, measured degree, bipartiteness, and the
                    chain-embedding cost in intermediate spins for any jump Z1
                    cannot realise in one step.
NULL HYPOTHESES     "Bipartiteness decides whether a DTM graph suits Z1."
                    CONTROL: every architecture here is bipartite and passes the
                    parity assertion, yet their costs differ by seventeen-fold.
                    If parity decided the question, every row would cost the same.
SUCCESS CRITERIA    Every architecture gets a COST, not a verdict, and any jump
                    needing routing names its hop count.
FAILURE CRITERIA    Reporting a jump as directly realisable when it is not in Z1's
                    offset set; reporting a cost without saying it is an estimate.
PROVENANCE          Topology from the public reference implementation. Z1 offsets
                    and node budget from target.py, sourced to F-14 and Fig. 05.
                    No hardware, no sampling, nothing of theirs redistributed.
SCOPE OF VALIDITY   TOPOLOGY ONLY, and the chain costs are a LOWER BOUND.
                    This compiler does not implement chain embedding -- its
                    mediator insertion fixes bipartite PARITY conflicts, which is
                    a different technique (see audit/findings/R2.md). The hop
                    counts come from BFS over Z1's offsets and are therefore the
                    best case: a real embedding pays more for routing congestion
                    and cannot reuse spins freely. Nothing here says a trained
                    DTM's coupling magnitudes clear the |J| cap.
==========================================================================

WHAT THIS FOUND, AND WHAT IT DID NOT.

It did NOT find an error. Every published preset is internally consistent, and
every jump satisfies the assertion the reference implementation makes about its
own offsets:

    assert (dx + dy) % 2 == 1, "To ensure bipartitness for parallel sampling ..."

That is exactly the constraint Z1's lattice imposes, and it holds everywhere.

What it found is a PRICE. Z1's longest offset is (4,1), a reach of 4.12. Presets
whose jumps sit within that reach place directly and cost nothing extra. Presets
using jumps that reach 13 to 24 cells are still realisable -- one long edge can
be routed across several legal Z1 hops -- but each such edge spends intermediate
spins, and the cost runs to an order of magnitude.

So parity is necessary, and reach sets the price rather than deciding
possibility. An earlier version of this script reported those presets as "cannot
be placed", which was wrong and is corrected here.
"""
# NO-PREFLIGHT: checks published DTM topologies inline against target.PROFILES; its committed results predate preflight() and rewriting would invalidate them.

import json
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.target import PROFILES

OUT = Path("out/dtm-preflight")

# Published presets, by their key in the reference implementation. The key's
# trailing digits are NOT asserted to mean anything: an obvious reading is
# {side_len}{degree}, but preset 88 measures degree 4, not 8. Only side_len and
# jumps are taken from the source; degree is MEASURED.
PRESETS = {
    "88   (8x8)":     (8,  [(0, 1), (4, 1)]),
    "448  (44x44)":   (44, [(0, 1), (4, 1)]),
    "608  (60x60)":   (60, [(0, 1), (4, 1)]),
    "908  (90x90)":   (90, [(0, 1), (4, 1)]),
    "4412 (44x44)":   (44, [(0, 1), (4, 1), (10, 9)]),
    "6016 (60x60)":   (60, [(0, 1), (4, 1), (10, 9), (11, 14)]),
    "6020 (60x60)":   (60, [(0, 1), (4, 1), (10, 9), (11, 14), (23, 6)]),
}


def min_hops(target, offsets, cap: int = 8):
    """Fewest legal Z1 offsets that sum to `target`, by BFS. 1 means Z1 realises
    the jump directly. None means no route was found within `cap` hops.

    This is a LOWER BOUND on what an embedding costs: it routes one edge in
    isolation, with no congestion and no competition for intermediate cells."""
    seen = {(0, 0)}
    q = deque([((0, 0), 0)])
    while q:
        (x, y), d = q.popleft()
        if (x, y) == target:
            return d
        if d >= cap:
            continue
        for dx, dy in offsets:
            n = (x + dx, y + dy)
            if n not in seen and abs(n[0]) <= 40 and abs(n[1]) <= 40:
                seen.add(n)
                q.append((n, d + 1))
    return None


def build(side: int, jumps, torus: bool = True) -> IsingModel:
    """The published construction: a side x side grid, an edge at every jump
    offset from every node, wrapped if toroidal, deduplicated."""
    idx = lambda i, j: i * side + j
    edges = set()
    for i in range(side):
        for j in range(side):
            for di, dj in jumps:
                if torus:
                    a, b = (i + di) % side, (j + dj) % side
                else:
                    a, b = i + di, j + dj
                    if not (0 <= a < side and 0 <= b < side):
                        continue
                u, v = idx(i, j), idx(a, b)
                if u != v:
                    edges.add((min(u, v), max(u, v)))
    e = tuple(sorted(edges))
    n = side * side
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=e,
                      weights=np.full(len(e), 1.0), biases=np.zeros(n),
                      beta=1.0, offset=0.0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    offsets = list(PROFILES["z1"].offsets.value)
    budget = PROFILES["z1"].node_budget.value
    reach = max((dx * dx + dy * dy) ** 0.5 for dx, dy in offsets)
    rows, failures = [], []

    for name, (side, jumps) in PRESETS.items():
        model = build(side, jumps)
        rep = analyse(model)
        n = rep.n_nodes

        per_jump, extra = [], 0
        for j in jumps:
            h = min_hops(j, offsets)
            direct = h == 1
            # one edge per node per jump on a torus; each needs h-1 intermediates
            cost = 0 if (h is None or direct) else n * (h - 1)
            extra += cost
            per_jump.append({
                "jump": list(j), "reach": round((j[0] ** 2 + j[1] ** 2) ** 0.5, 2),
                "z1_hops": h, "direct": direct,
                "intermediate_spins_per_edge": 0 if (direct or h is None) else h - 1,
                "chain_spins_for_this_jump": cost,
            })
            if h is None:
                failures.append(f"{name}: jump {j} has no route within 8 hops")

        total = n + extra
        rows.append({
            "preset": name, "side_len": side, "jumps": [list(j) for j in jumps],
            "spins": n, "couplings": rep.n_edges, "max_degree": rep.max_degree,
            "bipartite": bool(rep.bipartite),
            "all_jumps_odd_parity": all((a + b) % 2 == 1 for a, b in jumps),
            "per_jump": per_jump,
            "all_jumps_direct": extra == 0,
            "chain_spins_estimate": extra,
            "physical_spins_estimate": total,
            "fabric_tax_estimate": round(total / n, 2),
            "within_node_budget": bool(total <= budget),
        })

    (OUT / "dtm_preflight.json").write_text(json.dumps(
        {"rows": rows, "control_failures": failures,
         "z1_max_reach": round(reach, 2), "z1_node_budget": budget,
         "estimate_note": "chain costs are a LOWER BOUND from BFS over Z1's "
                          "offsets; this compiler does not implement chain "
                          "embedding, and a real embedding pays routing "
                          "congestion on top",
         "not_an_error_report": "every preset is internally consistent and "
                                "satisfies the reference implementation's own "
                                "parity assertion; nothing here claims their "
                                "published work is wrong",
         "source": "topology parameters from the public DTM reference "
                   "implementation; no vendor code redistributed",
         "hardware": "none -- structural analysis, no sampling"},
        indent=2), encoding="utf-8")

    print("preset".ljust(16) + "spins".rjust(8) + "couplings".rjust(11)
          + "deg".rjust(5) + "+chain".rjust(9) + "total".rjust(9)
          + "tax".rjust(7) + "  budget")
    for r in rows:
        print(r["preset"].ljust(16)
              + format(r["spins"], ",").rjust(8)
              + format(r["couplings"], ",").rjust(11)
              + str(r["max_degree"]).rjust(5)
              + format(r["chain_spins_estimate"], ",").rjust(9)
              + format(r["physical_spins_estimate"], ",").rjust(9)
              + (format(r["fabric_tax_estimate"], ".2f") + "x").rjust(7)
              + ("  fits" if r["within_node_budget"] else "  OVER BUDGET"))
    print()
    print("  Every preset is bipartite and every jump has odd dx+dy -- the")
    print("  reference implementation asserts that itself, and it holds. Nothing")
    print("  here reports an error in their work.")
    print(f"  Z1's longest offset reaches {reach:.2f}. Jumps inside that place")
    print("  directly at 1.00x; longer jumps are ROUTED across several hops, so")
    print("  reach sets the PRICE rather than deciding possibility.")
    print("  Chain costs are a LOWER BOUND -- see SCOPE OF VALIDITY above.")
    if failures:
        print("\nCONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"  -> {OUT / 'dtm_preflight.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
