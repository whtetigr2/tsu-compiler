"""Look at the worlds the critical sweep found, and locate the usable band.

audit/critical_sweep.py established WHERE the phase transition is. This asks
the only question that matters for LATTICE: does the critical band actually
produce something that reads as geography, or just a different flavour of
noise? A correlation length is a number; a world is a thing you look at.

Two outputs:
  1. A FINE sweep across the transition (the coarse sweep's steps were too
     wide to see the usable band's edges).
  2. ASCII renders of one drawn world per coupling, so the answer is visible
     rather than inferred from a statistic.

The renders draw from the SAME sampler settings and seed as the sweep, and are
a single realization each -- one draw, honestly labelled, not a best-of.
"""
import sys, math, json
sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np

from critical_sweep import (compile_layer, grid_of, connected_corr,
                            corr_length, onsager_kc, N, MAX_R, SAMPLE)
from tsu.backends.thrml_backend import sample as thrml_sample

FINE = [-0.30, -0.34, -0.38, -0.41, -0.43, -0.45, -0.47, -0.50, -0.55]
RENDER_AT = [-0.20, -0.40, -0.44, -0.48, -0.60]
GLYPH = {0: ".", 1: "#"}


def draw(w, seed=0):
    spec, enc, ising, prog, j, b = compile_layer(N, w)
    draws = thrml_sample(prog, seed=seed, **SAMPLE)
    grids = []
    for row in draws:
        bits = dict(zip(ising.nodes, row.tolist()))
        if enc.is_codeword(bits):
            grids.append(grid_of(enc.decode(bits), N))
    return grids, float(ising.beta) * j


def main():
    kc = onsager_kc()

    print("FINE SWEEP across the transition (grid %dx%d, Kc = %.4f)\n" % (N, N, kc))
    rows = []
    for w in FINE:
        grids, bj = draw(w)
        spins = [2 * g - 1 for g in grids]
        m = float(np.mean([abs(float(s.mean())) for s in spins]))
        cbar = np.nanmean(np.array([connected_corr(s, MAX_R) for s in spins],
                                   dtype=float), axis=0).tolist()
        xi = corr_length(cbar)
        rows.append(dict(weight=w, beta_j=bj, ratio=bj / kc, abs_m=m, xi=xi))
        xs = "exceeds window" if xi is None else "%.2f" % xi
        print("  beta*J=%.3f (%.2f x Kc)   |m|=%.3f   xi=%s" % (bj, bj / kc, m, xs))

    print("\n\nONE DRAWN WORLD PER COUPLING (single realization, not best-of)")
    for w in RENDER_AT:
        grids, bj = draw(w)
        g = grids[-1]
        frac = float(g.mean())
        print("\n  beta*J = %.3f  (%.2f x Kc)   fraction '#' = %.3f" %
              (bj, bj / kc, frac))
        for r in range(N):
            print("    " + "".join(GLYPH[int(v)] for v in g[r]))

    with open("audit/critical_fine.json", "w") as fh:
        json.dump(dict(grid=N, kc=kc, sample=SAMPLE, rows=rows), fh, indent=2)
    print("\n-> audit/critical_fine.json")


if __name__ == "__main__":
    main()
