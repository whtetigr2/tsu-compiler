"""Is the land at criticality actually WALKABLE? (design exploration)

audit/critical_sweep.py found where structure lives (beta*J in [0.38, 0.45]).
Structure is not the same as traversability: a correlation length of 2 cells and
visible coastlines say nothing about whether a connected path exists across the
map. Random site percolation on a square lattice needs p_c ~ 0.5927 occupancy,
and the critical worlds render at ~48% fill -- but Ising clusters are CORRELATED,
not random, so that threshold does not transfer and the question has to be
measured directly.

This is exploration to ground a design decision, not a feature. It answers three
things per coupling, over independent worlds:

  span      fraction of worlds where ONE connected component touches both the
            left and right edges (or top and bottom) -- the operational
            definition of "you can walk across this map"
  lcc       largest connected component as a fraction of all walkable cells --
            how much of the land is reachable from one starting point
  parts     how many separate landmasses there are

Measured on BOTH phases (cells==1 and cells==0), because at zero field which
value wins is a symmetry-broken accident of the seed, so "land" has to mean
"the phase you chose to walk on", not a fixed label.

Independence: only the FINAL sample of each chain is used, across several seeds.
Consecutive samples within one chain are correlated and would inflate the count
of "independent worlds" without adding information.
"""
import sys, json, math
sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np

from critical_sweep import compile_layer, grid_of, onsager_kc
from tsu_compiler.backends.thrml_backend import sample as thrml_sample

N = 64
SEEDS = [0, 1, 2, 3, 4]
COUPLINGS = [-0.20, -0.34, -0.38, -0.40, -0.42, -0.44, -0.46, -0.50]
SAMPLE = dict(n_chains=6, n_samples=8, n_warmup=4000, steps_per_sample=25)


def components(mask: np.ndarray):
    """4-connected components of a boolean mask, by iterative flood fill.
    Hand-rolled rather than pulled from scipy so the audit keeps its habit of
    not adding a dependency for twenty lines of well-understood code."""
    n, m = mask.shape
    lab = np.full((n, m), -1, dtype=int)
    sizes = []
    for sy in range(n):
        for sx in range(m):
            if not mask[sy, sx] or lab[sy, sx] >= 0:
                continue
            cid = len(sizes)
            stack = [(sy, sx)]
            lab[sy, sx] = cid
            count = 0
            while stack:
                y, x = stack.pop()
                count += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < n and 0 <= nx < m and mask[ny, nx] and lab[ny, nx] < 0:
                        lab[ny, nx] = cid
                        stack.append((ny, nx))
            sizes.append(count)
    return lab, sizes


def spans(lab: np.ndarray, sizes) -> bool:
    """True when a SINGLE component touches both opposite edges on either axis."""
    for cid in range(len(sizes)):
        m = lab == cid
        if (m[:, 0].any() and m[:, -1].any()) or (m[0, :].any() and m[-1, :].any()):
            return True
    return False


def worlds(w, seeds):
    """One world per chain per seed, taking only the FINAL sample of each chain."""
    spec, enc, ising, prog, j, b = compile_layer(N, w)
    out = []
    for sd in seeds:
        draws = thrml_sample(prog, seed=sd, **SAMPLE)
        rows = np.asarray(draws)
        per_chain = rows.reshape(SAMPLE["n_chains"], SAMPLE["n_samples"], -1)
        for c in range(SAMPLE["n_chains"]):
            bits = dict(zip(ising.nodes, per_chain[c, -1].tolist()))
            if enc.is_codeword(bits):
                out.append(grid_of(enc.decode(bits), N))
    return out, float(ising.beta) * j


def main():
    kc = onsager_kc()
    print("Traversability at %dx%d  (Kc = %.4f, %d seeds x %d chains)\n"
          % (N, N, kc, len(SEEDS), SAMPLE["n_chains"]))
    print("  beta*J  vs Kc   phase  fill   span    lcc    parts")
    rows = []
    for w in COUPLINGS:
        gs, bj = worlds(w, SEEDS)
        if not gs:
            print("  %.3f  -- NO valid codewords drawn" % bj)
            continue
        for phase in (1, 0):
            fills, spanned, lccs, parts = [], [], [], []
            for g in gs:
                mask = (g == phase)
                tot = int(mask.sum())
                fills.append(tot / mask.size)
                if tot == 0:
                    spanned.append(False); lccs.append(0.0); parts.append(0)
                    continue
                lab, sizes = components(mask)
                spanned.append(spans(lab, sizes))
                lccs.append(max(sizes) / tot)
                parts.append(len(sizes))
            row = dict(weight=w, beta_j=bj, ratio=bj / kc, phase=phase,
                       fill=float(np.mean(fills)),
                       span=float(np.mean(spanned)),
                       lcc=float(np.mean(lccs)),
                       parts=float(np.mean(parts)), n=len(gs))
            rows.append(row)
            print("  %.3f  %.2fx    %d     %.3f  %.2f   %.3f   %5.1f"
                  % (bj, bj / kc, phase, row["fill"], row["span"],
                     row["lcc"], row["parts"]))
        print()
        with open("audit/traversability.json", "w") as fh:
            json.dump(dict(grid=N, kc=kc, seeds=SEEDS, sample=SAMPLE,
                           rows=rows), fh, indent=2)
    print("-> audit/traversability.json")


if __name__ == "__main__":
    main()
