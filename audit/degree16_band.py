"""Does using Z1's FULL degree-16 connectivity change the usable band?

PRE-REGISTERED. Committed before the run.

WHY. Every world this project generates uses a 4-neighbour grid: the `(1,0)`
offset and its rotations, which is 4 of the 16 edges Z1 actually offers. That was
never a considered choice about the hardware -- it is what `generate: {kind:
grid}` produces. An external reviewer (Grok) flagged the gap, and the
consequences are larger than "unused capacity":

  1. Onsager's Kc = 0.4407 is exact for the SQUARE LATTICE, z = 4. It is not
     Z1's critical coupling. Our measured usable band of 0.38-0.45 is therefore
     a square-lattice band, and this project has been quietly treating it as if
     it were a property of the chip.

  2. The spec states, as a conclusion, that "correlation length cannot carry
     scale separation" -- tunable only over 0.75 to 2.66 cells, a 3.5x range --
     and builds the whole grid-size-plus-upsampling architecture on it. That
     measurement was taken at z = 4. Z1's offsets reach FOUR CELLS, so a
     degree-16 lattice propagates structure much further per sweep. If the
     tunable range is materially wider at z = 16, the architecture was designed
     around an artifact of an unexamined default.

Z1's offsets are (1,0), (2,1), (2,3), (4,1) and their four rotations. Every one
has dx+dy odd, so the full degree-16 graph is still BIPARTITE and still places as
a direct subgraph -- using all 16 costs no mediators and no placement search.

PREDICTION, recorded before running:
  - Kc(z=16) is substantially BELOW Kc(z=4) = 0.4407. More neighbours order at
    weaker coupling; a mean-field scaling of 1/z would put it near 0.11, and the
    true value should sit above that.
  - Peak correlation length at z=16 EXCEEDS the 2.11 cells measured at z=4,
    because the (4,1) and (2,3) offsets move influence four cells per hop rather
    than one.

FALSIFIER: if the peak correlation length at z=16 is not materially longer than
2.11 cells, then degree was not the limitation, the spec's conclusion stands as
written, and the grid-size architecture is vindicated rather than merely
adequate.

Either outcome is reported. A confirmation means a documented conclusion of this
project was an artifact of a default nobody chose deliberately.
"""
# NO-PREFLIGHT: pre-registered and committed before its run; it reads target.PROFILES and analyse() directly, and rewriting it to route through preflight() would alter a protocol that was fixed in advance.

import sys
import json
import math

sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np

from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.program import build_program
from tsu_compiler.backends.thrml_backend import sample as thrml_sample
from tsu_compiler.target import PROFILES

N = 32
BETA = 1.0
SAMPLE = dict(n_chains=6, n_samples=30, n_warmup=4000, steps_per_sample=8)
COUPLINGS_4 = [0.15, 0.25, 0.35, 0.40, 0.44, 0.50, 0.60]
COUPLINGS_16 = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
NEAREST_ONLY = {(1, 0), (-1, 0), (0, 1), (0, -1)}


def build(n: int, offsets, j: float) -> IsingModel:
    """A uniform ferromagnet on `offsets`, open boundaries, zero field."""
    idx = {(x, y): y * n + x for y in range(n) for x in range(n)}
    edges, seen = [], set()
    for (x, y), i in idx.items():
        for dx, dy in offsets:
            nb = (x + dx, y + dy)
            if nb in idx:
                k = tuple(sorted((i, idx[nb])))
                if k not in seen:
                    seen.add(k)
                    edges.append(k)
    return IsingModel(
        nodes=tuple(f"g{x}_{y}" for y in range(n) for x in range(n)),
        edges=tuple(edges),
        weights=np.full(len(edges), float(j)),
        biases=np.zeros(n * n),
        beta=BETA, offset=0.0)


def connected_corr(s: np.ndarray, max_r: int):
    m = float(s.mean())
    den = 1.0 - m * m
    out = [1.0]
    for r in range(1, max_r + 1):
        if r >= s.shape[0] or den <= 1e-12:
            out.append(float("nan")); continue
        acc = [float((s[:, :-r] * s[:, r:]).mean()),
               float((s[:-r, :] * s[r:, :]).mean())]
        out.append((float(np.mean(acc)) - m * m) / den)
    return out


def corr_length(c):
    thr = 1.0 / math.e
    for r in range(1, len(c)):
        if not math.isfinite(c[r]):
            return None
        if c[r] < thr:
            prev = c[r - 1]
            return (r - 1) + (prev - thr) / (prev - c[r]) if prev != c[r] else float(r)
    return None


def run(label, offsets, couplings):
    print(f"\n  {label}  (degree {len(offsets)})")
    print("     beta*J    |m|      xi (cells)   bipartite  edges")
    rows = []
    for j in couplings:
        im = build(N, offsets, j)
        rep = analyse(im)
        prog = build_program(im, rep)
        draws = np.asarray(thrml_sample(prog, seed=0, **SAMPLE))
        S = 2 * draws.astype(int) - 1
        grids = [row.reshape(N, N) for row in S]
        mags = [abs(float(g.mean())) for g in grids]
        cbar = np.nanmean(np.array([connected_corr(g, 12) for g in grids],
                                   dtype=float), axis=0).tolist()
        xi = corr_length(cbar)
        rows.append(dict(beta_j=j, abs_m=float(np.mean(mags)), xi=xi,
                         bipartite=bool(rep.bipartite), edges=len(im.edges)))
        xs = "exceeds window" if xi is None else f"{xi:5.2f}"
        print(f"     {j:.3f}   {np.mean(mags):.3f}    {xs}        "
              f"{str(rep.bipartite):<9} {len(im.edges)}")
    return rows


def main():
    print(f"Z1 degree-16 connectivity vs the 4-neighbour default, {N}x{N}, beta=1")
    z1 = set(PROFILES["z1"].offsets.value)
    print(f"  Z1 offsets: {len(z1)}, max reach "
          f"{max(max(abs(a), abs(b)) for a, b in z1)} cells, "
          f"all dx+dy odd: {all((a + b) % 2 == 1 for a, b in z1)}")
    out = dict(
        deg4=run("4-neighbour (what every world uses today)",
                 NEAREST_ONLY, COUPLINGS_4),
        deg16=run("Z1 full connectivity", z1, COUPLINGS_16))
    for k in out:
        best = max((r for r in out[k] if r["xi"] is not None),
                   key=lambda r: r["xi"], default=None)
        if best:
            print(f"\n  {k}: peak xi = {best['xi']:.2f} cells at "
                  f"beta*J = {best['beta_j']}")
    with open("audit/degree16_band.json", "w") as fh:
        json.dump(dict(size=N, beta=BETA, sample=SAMPLE, arms=out), fh, indent=2)
    print("\n-> audit/degree16_band.json")


if __name__ == "__main__":
    main()
