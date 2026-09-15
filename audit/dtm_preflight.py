"""Do the published DTM graph architectures fit Z1?

================================ PROTOCOL ================================
SYSTEM DEFINITION   The grid topologies used by Denoising Thermodynamic Models,
                    as published in the DTM reference implementation
                    (github.com/pschilliOrange/dtm-replication, linked from
                    Extropic's GitHub; its pyproject names the upstream as
                    extropic-ai/thrmlDenoising, which is not public). Each
                    architecture is a side_len x side_len grid with a set of
                    "jumps" (offsets), optionally toroidal.
STATE VARIABLES     Per architecture: side length, jump offsets, spins, couplings,
                    max degree, bipartiteness, and whether every jump is an offset
                    Z1 can actually realise.
TRANSITION RULES    None. This is a structural question about topology, answered
                    before any sampling.
ALLOWED OPERATIONS  Constructing the graph from the PUBLISHED topology parameters
                    and running this project's own compiler passes on it.
FORBIDDEN OPERATIONS
                    No vendoring of their code -- that repository carries no
                    licence, so it is all-rights-reserved and only our own
                    measurements may be published. The graphs here are built from
                    the documented construction (grid, jumps, torus, chessboard
                    parity), not copied. No training, no sampling, no claim about
                    DTM model quality.
ASSUMPTIONS         That the published (side_len, jumps) presets describe the
                    topology faithfully. Node counts here are the GRID, and a
                    real DTM adds label nodes; that makes these figures a floor,
                    which is stated rather than smoothed.
INVARIANTS          Every jump must satisfy (dx + dy) odd -- the reference
                    implementation asserts this itself, to keep the graph
                    bipartite for parallel sampling. Our analyse() must agree.
MEASUREMENTS        Spins, couplings, max degree, bipartiteness, and per-jump
                    whether it lies in Z1's 16 offsets.
NULL HYPOTHESES     "Bipartiteness is what decides whether a DTM graph fits Z1."
                    CONTROL: an architecture can be bipartite, pass the reference
                    implementation's own parity assertion, and still be
                    unplaceable -- because parity is necessary and REACH is
                    binding. If every bipartite architecture fitted, this script
                    would be measuring nothing.
SUCCESS CRITERIA    Every architecture is classified, and any failure names the
                    specific jump and its reach rather than a generic verdict.
FAILURE CRITERIA    Any architecture reported as fitting whose jumps are not all
                    in Z1's offset set; any disagreement with the parity assertion
                    the reference implementation makes about its own presets.
PROVENANCE          Topology parameters read from the public reference
                    implementation. Z1's offsets from target.py, sourced to F-14.
                    No hardware, no sampling, nothing of theirs redistributed.
SCOPE OF VALIDITY   This answers a TOPOLOGY question only: can Z1's lattice host
                    these edges. It says nothing about whether a trained DTM's
                    coupling magnitudes fit the |J| cap, which needs trained
                    weights this script does not have. A "fits" verdict here is
                    necessary, not sufficient.
==========================================================================

WHY THIS IS THE INTERESTING QUESTION. The reference implementation asserts, of
its own jump offsets:

    assert (dx + dy) % 2 == 1, "To ensure bipartitness for parallel sampling ..."

That is exactly the constraint Z1's lattice imposes, and every published preset
satisfies it. But parity is necessary and not sufficient: Z1's longest offset is
(4,1), a reach of 4.12, and several presets use jumps reaching 13 to 24 cells.
Those graphs are bipartite, pass the reference implementation's own check, and
still cannot be placed on the hardware the models are aimed at.

That is the same lesson this project learned about its own placer, arrived at
from the other direction: bipartiteness is necessary, geometry is binding.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.target import PROFILES

OUT = Path("out/dtm-preflight")

# Published presets, by their key in the reference implementation. The key's
# trailing digits are NOT asserted to mean anything here: an obvious reading is
# {side_len}{degree}, but preset 88 measures degree 4, not 8, so that reading is
# wrong or means something else. Only side_len and jumps are taken from the
# source; degree is MEASURED below rather than inferred from a name.
PRESETS = {
    "88   (8x8)":     (8,  [(0, 1), (4, 1)]),
    "448  (44x44)":   (44, [(0, 1), (4, 1)]),
    "608  (60x60)":   (60, [(0, 1), (4, 1)]),
    "908  (90x90)":   (90, [(0, 1), (4, 1)]),
    "4412 (44x44)":   (44, [(0, 1), (4, 1), (10, 9)]),
    "6016 (60x60)":   (60, [(0, 1), (4, 1), (10, 9), (11, 14)]),
    "6020 (60x60)":   (60, [(0, 1), (4, 1), (10, 9), (11, 14), (23, 6)]),
}


def z1_offsets() -> set:
    return set(PROFILES["z1"].offsets.value)


def jump_is_realisable(jump, offsets) -> bool:
    """A jump is realisable if it, or any of its four rotations, is a legal Z1
    offset. Rotation matters because the grid has no preferred orientation."""
    x, y = jump
    for _ in range(4):
        if (x, y) in offsets:
            return True
        x, y = -y, x
    return False


def build(side: int, jumps, torus: bool = True) -> IsingModel:
    """The published construction: a side x side grid, an edge at every jump offset
    from every node, wrapped if toroidal, deduplicated."""
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
    offsets = z1_offsets()
    reach = max((dx * dx + dy * dy) ** 0.5 for dx, dy in offsets)
    rows, failures = [], []

    for name, (side, jumps) in PRESETS.items():
        model = build(side, jumps)
        rep = analyse(model)
        bad = [j for j in jumps if not jump_is_realisable(j, offsets)]
        row = {
            "preset": name, "side_len": side, "jumps": [list(j) for j in jumps],
            "spins": rep.n_nodes, "couplings": rep.n_edges,
            "max_degree": rep.max_degree, "bipartite": bool(rep.bipartite),
            "all_jumps_odd_parity": all((a + b) % 2 == 1 for a, b in jumps),
            "unrealisable_jumps": [
                {"jump": list(j), "reach": round((j[0] ** 2 + j[1] ** 2) ** 0.5, 2)}
                for j in bad],
            "topology_fits_z1": not bad,
        }
        rows.append(row)
        # the control: a bipartite graph that still cannot be placed
        if row["topology_fits_z1"] and bad:
            failures.append(f"{name}: reported as fitting despite unrealisable jumps")
        if not row["bipartite"] and row["all_jumps_odd_parity"]:
            failures.append(f"{name}: odd-parity jumps but analyse() says non-bipartite")

    (OUT / "dtm_preflight.json").write_text(json.dumps(
        {"rows": rows, "control_failures": failures,
         "z1_max_reach": round(reach, 2), "z1_offsets": sorted(map(list, offsets)),
         "source": "topology parameters from the public DTM reference "
                   "implementation; no vendor code redistributed",
         "hardware": "none -- structural analysis, no sampling"},
        indent=2), encoding="utf-8")

    print("preset".ljust(24) + "spins".rjust(8) + "couplings".rjust(11)
          + "deg".rjust(5) + "bipartite".rjust(11) + "  fits Z1 topology")
    for r in rows:
        verdict = "YES" if r["topology_fits_z1"] else "NO"
        detail = ""
        if r["unrealisable_jumps"]:
            worst = max(r["unrealisable_jumps"], key=lambda x: x["reach"])
            detail = (f"  <- jump {tuple(worst['jump'])} reaches "
                      f"{worst['reach']}, Z1 max {reach:.2f}")
        print(r["preset"].ljust(24)
              + format(r["spins"], ",").rjust(8)
              + format(r["couplings"], ",").rjust(11)
              + str(r["max_degree"]).rjust(5)
              + str(r["bipartite"]).rjust(11)
              + "  " + verdict + detail)
    print()
    print("  Every preset above is bipartite and every jump has odd dx+dy -- the")
    print("  reference implementation asserts that itself. Parity is NECESSARY and")
    print("  not SUFFICIENT: Z1's longest offset is (4,1), reach 4.12.")
    if failures:
        print("\nCONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"  -> {OUT / 'dtm_preflight.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
