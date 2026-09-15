"""Drawing the blindness: a free energy surface in coordinates that can see it.

================================ PROTOCOL ================================
SYSTEM DEFINITION   A 16x16 periodic square-lattice Ising model in ZERO FIELD,
                    at both coupling signs, swept across beta*J. Onsager gives
                    the exact critical point for this lattice,
                    Kc = ln(1+sqrt(2))/2 = 0.440687, so "ordered" is known in
                    advance rather than inferred from the picture.
STATE VARIABLES     Per (sign, beta*J): the joint histogram of the two SUBLATTICE
                    magnetisations, the free energy estimated from it, the scalar
                    and per-spin R-hat on the same draws, and the count of
                    visited cells.
TRANSITION RULES    thrml chromatic block Gibbs via `sample_chains`, 16 chains
                    from independent random starts.
ALLOWED OPERATIONS  Sampling; histogramming; finite differences on the estimated
                    surface.
FORBIDDEN OPERATIONS
                    No interpolation across unvisited cells -- F = -ln 0 is
                    infinite and painting those regions would invent landscape
                    the sampler never entered. No smoothing that crosses a mask
                    boundary. No hardware. The coupling SIGN is confirmed against
                    `audit/oracles/exact.py`, never assumed from our IR.
ASSUMPTIONS         None beyond Onsager's exact result for this lattice.
INVARIANTS          The two sublattices are interchangeable by symmetry, so the
                    surface must be symmetric under swapping the axes, and under
                    global negation. A visibly asymmetric surface means too few
                    draws, not broken physics.
MEASUREMENTS        F = -ln P over (m_A, m_B) up to an additive constant; the
                    drift -grad F; and, on the SAME draws, scalar R-hat and
                    max per-spin R-hat.
NULL HYPOTHESES     "A scalar order parameter is enough to see whether the
                    chains mixed." The antiferromagnetic panels refute it: both
                    basins sit at total magnetisation zero, so the scalar sees
                    ONE point where the surface shows TWO basins.
SUCCESS CRITERIA    The ordered panels show two separated basins; the disordered
                    panels show one; the antiferromagnetic basins lie on the
                    anti-diagonal where total m = 0.
FAILURE CRITERIA    Any painted region the sampler never visited. Any panel whose
                    basin structure contradicts the R-hat printed on it.
PROVENANCE          Onsager 1944 for Kc. Sign from the independent oracle.
                    Diagnostics from `tsu_compiler`. No Z1 hardware anywhere --
                    thrml simulates on CPU.
SCOPE OF VALIDITY   This is a picture of OUR sampler on a model chosen because
                    its answer is known. It is not a claim about any workload,
                    any hardware, or any performance.
==========================================================================

WHY THESE COORDINATES AND NOT THE OBVIOUS ONES.

`audit/free_energy_surface.py` already draws a free energy surface over the mean
spin of the lattice's left and right halves, and for a FERROMAGNET that works:
the halves order together, so below Kc two basins appear on the diagonal.

Those coordinates cannot see an ANTIFERROMAGNET at all. Its ordered states are
the two checkerboards, and the left half of a checkerboard averages to zero just
as the right half does -- so both ordered states land on the same spot at the
origin, and the surface shows a single basin for a model that has two.

Splitting by the chessboard parity instead -- sublattice A is the cells with
(i+j) even, B the cells with (i+j) odd -- separates them:

    ferromagnetic order   -> m_A and m_B agree   -> basins on the DIAGONAL
    antiferromagnetic     -> m_A and m_B oppose  -> basins on the ANTI-DIAGONAL
    disordered            -> one basin at the origin

The total magnetisation is (m_A + m_B)/2, which is the projection onto the
diagonal. On the anti-diagonal that projection is ZERO for both basins. So the
figure shows, geometrically, why the finding in `audit/findings/R20.md` happened:
a scalar order parameter collapses the two antiferromagnetic basins onto one
number, reports R-hat = 1.0000, and certifies as mixed a set of chains that are
each frozen in a different configuration. The per-spin diagnostic, printed on the
same panel from the same draws, reads 11.39.

The point of the picture is not that the landscape is pretty. It is that the
information the scalar throws away is visible once you choose an axis that keeps
it -- and that nothing about the chains changed between the two readings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "audit")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsu_compiler.backends.thrml_backend import sample_chains
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.program import build_program
from tsu_compiler.preflight.diagnostics import RHAT_THRESHOLD, r_hat

OUT_DIR = Path("audit")
RECEIPT = Path("out/free-energy")
L = 16
N = L * L
KC = float(np.log(1 + np.sqrt(2)) / 2)
BINS = 34
MIN_COUNT = 3          # a cell needs this many visits before F is drawn at all
BETAS = (0.20, 0.44, 0.60, 0.80)
N_CHAINS, N_SAMPLES, N_WARMUP, STEPS = 16, 6000, 3000, 4

BG, FG, DIM = "#0d1117", "#e8eaed", "#9aa2ad"


def _torus_edges(side):
    idx = lambda i, j: (i % side) * side + (j % side)
    return sorted({(min(idx(i, j), idx(i + di, j + dj)),
                    max(idx(i, j), idx(i + di, j + dj)))
                   for i in range(side) for j in range(side)
                   for di, dj in ((1, 0), (0, 1))
                   if idx(i, j) != idx(i + di, j + dj)})


def lattice(beta_j, weight):
    e = _torus_edges(L)
    return IsingModel(nodes=tuple(f"n{i}" for i in range(N)), edges=tuple(e),
                      weights=np.full(len(e), weight), biases=np.zeros(N),
                      beta=beta_j, offset=0.0)


def confirm_sign(weight):
    """Name the sign from the INDEPENDENT oracle. This project has shipped a
    wrong-signed coupling before, and a sign error here would invert the whole
    reading of the figure."""
    from oracles.exact import exact_boltzmann
    side = 4
    e = _torus_edges(side)
    states, probs = exact_boltzmann({ab: weight for ab in e},
                                    [0.0] * (side * side), 0.6)
    P = np.asarray(probs)
    S = np.array([[2 * v - 1 for v in s] for s in states])
    nn = float(np.mean([P @ (S[:, a] * S[:, b]) for a, b in e]))
    return "ferromagnetic" if nn > 0 else "antiferromagnetic"


# chessboard parity: the only split that separates the two AFM ground states
PARITY = np.fromfunction(lambda i, j: (i + j) % 2 == 0, (L, L)).ravel()


def measure(beta_j, weight):
    model = lattice(beta_j, weight)
    draws = np.asarray(sample_chains(
        build_program(model, analyse(model)), n_chains=N_CHAINS,
        n_samples=N_SAMPLES, n_warmup=N_WARMUP, steps_per_sample=STEPS, seed=0))
    S = 2 * draws.astype(np.int8) - 1                  # (chains, samples, spins)

    mA = S[:, :, PARITY].mean(axis=2)
    mB = S[:, :, ~PARITY].mean(axis=2)

    edges = np.linspace(-1.0, 1.0, BINS + 1)
    counts, _, _ = np.histogram2d(mA.ravel(), mB.ravel(), bins=[edges, edges])
    # A cell below MIN_COUNT is NOT drawn. F = -ln 0 is infinite, and a single
    # stray visit is not evidence of a basin -- painting either would invent
    # landscape. The mask is the honesty of this figure.
    p = np.where(counts >= MIN_COUNT, counts, np.nan)
    p = p / np.nansum(p)
    F = -np.log(p)
    F = F - np.nanmin(F)

    per_spin = [float(r_hat((2.0 * draws[:, :, j] - 1.0).astype(np.float64)))
                for j in range(N)
                if (2.0 * draws[:, :, j] - 1.0).std() > 0]
    total_m = 0.5 * (mA + mB)
    return {
        "F": F, "edges": edges,
        "mA_last": mA[:, -1], "mB_last": mB[:, -1],
        "visited": int(np.isfinite(F).sum()),
        "r_hat_total_m": float(r_hat(total_m)),
        "r_hat_abs_total_m": float(r_hat(np.abs(total_m))),
        "max_r_hat_per_spin": max(per_spin) if per_spin else float("nan"),
    }


def draw_panel(ax, m, beta_j, sign, show_y):
    c = 0.5 * (m["edges"][:-1] + m["edges"][1:])
    F = m["F"]
    im = ax.pcolormesh(c, c, F.T, cmap="coolwarm", shading="auto",
                       vmin=0, vmax=np.nanpercentile(F, 96))
    ax.contour(c, c, F.T, levels=6, colors="#0d1117",
               linewidths=0.35, alpha=0.5)

    # drift = -grad F, computed only where F is defined and never across the
    # mask boundary: a gradient that straddles a NaN is not a measurement.
    filled = np.isfinite(F)
    Fq = np.where(filled, F, np.nan)
    gy, gx = np.gradient(np.nan_to_num(Fq, nan=np.nanmax(Fq)))
    # Only inside the well-sampled region. At the mask boundary the gradient is
    # a cliff between "measured" and "never visited", which records where
    # sampling stopped rather than a drift the sampler feels -- drawing those
    # arrows would be the vector equivalent of painting unvisited cells.
    interior = filled & (F <= np.nanpercentile(F, 70))
    step = 2
    sl = (slice(None, None, step), slice(None, None, step))
    ok = interior[sl]
    X, Y = np.meshgrid(c[::step], c[::step], indexing="ij")
    ax.quiver(X[ok], Y[ok], -gx[sl][ok], -gy[sl][ok],
              color="#5eb8ff", scale=34, width=0.005, alpha=0.95)

    # where the 16 chains actually ended up
    ax.scatter(m["mA_last"], m["mB_last"], s=34, c="#ffffff",
               edgecolors="#0d1117", linewidths=0.8, zorder=5)

    ordered = beta_j > KC
    ax.set_title(f"beta*J = {beta_j}   ({beta_j / KC:.2f} x Kc)   "
                 f"{'ORDERED' if ordered else 'disordered'}",
                 color=FG, fontsize=10.5, pad=40)
    blind = m["r_hat_total_m"] <= RHAT_THRESHOLD
    ax.text(0.5, 1.005,
            f"R-hat on total m = {m['r_hat_total_m']:.4f}"
            f"{'   <-- BLIND' if (blind and ordered) else ''}\n"
            f"max R-hat per spin = {m['max_r_hat_per_spin']:.2f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2,
            family="monospace",
            color="#ff7b72" if (blind and ordered) else DIM)
    # A nearly-blank panel is a RESULT, not a failed plot: the chains were so
    # stuck they never entered the rest of the plane. Saying so on the panel
    # stops a reader reading emptiness as a rendering bug.
    ax.text(0.5, -0.155, f"{m['visited']} of {BINS * BINS} cells visited",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            family="monospace", color=DIM)
    ax.set_xlabel("m$_A$  (sublattice A)", color=DIM, fontsize=9)
    if show_y:
        ax.set_ylabel(f"{sign}\n\nm$_B$  (sublattice B)", color=FG, fontsize=9)
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.set_aspect("equal")
    return im


def main() -> int:
    RECEIPT.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []

    fig, axes = plt.subplots(2, len(BETAS), figsize=(19.5, 10.6),
                             facecolor=BG)
    for r, weight in enumerate((1.0, -1.0)):
        sign = confirm_sign(weight)
        for c_i, bj in enumerate(BETAS):
            m = measure(bj, weight)
            im = draw_panel(axes[r][c_i], m, bj, sign, show_y=(c_i == 0))
            rows.append({
                "sign_per_exact_oracle": sign, "ir_weight": weight,
                "beta_j": bj, "ordered": bool(bj > KC),
                "cells_visited": m["visited"], "cells_total": BINS * BINS,
                "r_hat_total_m": round(m["r_hat_total_m"], 4),
                "r_hat_abs_total_m": round(m["r_hat_abs_total_m"], 4),
                "max_r_hat_per_spin": round(m["max_r_hat_per_spin"], 4),
                "scalar_reports_failure":
                    bool(m["r_hat_total_m"] > RHAT_THRESHOLD),
                "per_spin_reports_failure":
                    bool(m["max_r_hat_per_spin"] > RHAT_THRESHOLD),
            })

    # CONTROL: the antiferromagnet at beta*J = 0.80 is the whole point. If the
    # scalar ever starts catching it, the claim in R20 must be rewritten rather
    # than left standing.
    afm = [r for r in rows
           if r["sign_per_exact_oracle"] == "antiferromagnetic"
           and r["beta_j"] == 0.80]
    for r in afm:
        if r["scalar_reports_failure"]:
            failures.append("the scalar order parameter DID flag the ordered "
                            "antiferromagnet; R20's claim needs rewriting")
        if not r["per_spin_reports_failure"]:
            failures.append("the per-spin diagnostic did NOT flag the ordered "
                            "antiferromagnet; the figure contradicts itself")

    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label("free energy  F = -ln P   (arbitrary offset; blue = visited "
                 "often = low F)", color=DIM, fontsize=9)
    cb.ax.tick_params(colors=DIM, labelsize=8)
    fig.suptitle(
        "The same chains, read two ways  |  "
        f"{L}x{L} zero-field Ising, {N_CHAINS * N_SAMPLES:,} draws per panel, "
        "thrml on CPU (no Z1 hardware)\n"
        "White dots are where the 16 chains ended. Arrows are -grad F. "
        "Blank cells were never visited and are not painted.",
        color=FG, fontsize=12.5, y=0.975)
    path = OUT_DIR / "free_energy_sublattice.png"
    fig.savefig(path, dpi=125, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    (RECEIPT / "free_energy_sublattice.json").write_text(json.dumps(
        {"lattice": f"{L}x{L} periodic, zero field", "n_spins": N,
         "onsager_kc": round(KC, 6), "bins": BINS, "min_count_per_cell": MIN_COUNT,
         "draws_per_panel": N_CHAINS * N_SAMPLES, "rows": rows,
         "control_failures": failures,
         "coordinates": "sublattice magnetisations by chessboard parity; the "
                        "only split that separates the two antiferromagnetic "
                        "ground states, both of which have total m = 0",
         "hardware": "none -- thrml on CPU"},
        indent=2), encoding="utf-8")

    print(f"Onsager Kc = {KC:.6f}\n")
    print(f"{'sign':>18} {'beta*J':>7} {'regime':>11} {'R-hat(m)':>10}"
          f" {'maxR-hat/spin':>14} {'cells':>12}")
    for r in rows:
        print(f"{r['sign_per_exact_oracle']:>18} {r['beta_j']:>7.2f}"
              f" {'ordered' if r['ordered'] else 'disordered':>11}"
              f" {r['r_hat_total_m']:>10.4f} {r['max_r_hat_per_spin']:>14.2f}"
              f" {r['cells_visited']:>6}/{r['cells_total']:<5}")
    print()
    if failures:
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("  The antiferromagnetic panels are the point: two basins on the")
    print("  ANTI-diagonal, both at total m = 0. The scalar collapses them onto")
    print("  one number and reports R-hat ~ 1.000; the per-spin diagnostic, on")
    print("  the same draws, reports a maximum in the double digits.")
    print(f"  -> {path}")
    print(f"  -> {RECEIPT / 'free_energy_sublattice.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
