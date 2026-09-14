"""The energy landscape our sampler actually explores, from thrml draws.

WHY NOT A DIRECT ENERGY PLOT. A two-variable constraint can be drawn as E(x, y)
straight from its energy function. Our model is 4,096 binary spins, so its state
space has 2**4096 points and there is no axis to put them on. Plotting E over
two of the spins would show almost nothing, because the interesting structure is
collective rather than per-site.

WHAT IS PLOTTED INSTEAD. A FREE energy surface over two collective coordinates:
the mean spin of the lattice's left half and of its right half. For each cell of
that plane we count how often the sampler visits it and take F = -ln P, up to an
additive constant. This is estimated FROM THE DRAWS -- it is not the model's
energy function evaluated on a grid, and the difference matters: free energy
includes entropy, so a broad shallow basin here means "many configurations",
not "one low-energy configuration".

The arrows are -grad F, the direction the sampler is pushed. They are computed by
finite differences on the smoothed surface, so they are a readout of the measured
histogram rather than an independent quantity.

WHAT IT SHOWS. Below Kc the distribution is one basin at the origin: no net
magnetisation, left and right agree only by chance. Above Kc the up-down symmetry
breaks and TWO basins appear on the diagonal, because the two halves order
together. Near Kc the surface flattens and the basins merge, which is the
critical region that made worlds look like geography.

Unsampled cells are left blank rather than filled: F = -ln 0 is infinite, and
painting those regions would invent landscape the sampler never visited.

No Z1 hardware is involved. thrml simulates the sampling on CPU.
"""
import sys
import math

sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world.fields import compile_layer
from tsu_compiler.backends.thrml_backend import sample as thrml_sample
from world.studio import KC

N = 24
BINS = 26
COUPLINGS = [0.20, 0.44, 0.70]
SAMPLE = dict(n_chains=96, n_samples=220, n_warmup=3000, steps_per_sample=6)


def draws(beta_j: float) -> np.ndarray:
    """Spins as (samples, N, N) in {-1, +1}."""
    _spec, _enc, ising, prog = compile_layer(N, beta_j)
    rows = np.asarray(thrml_sample(prog, seed=0, **SAMPLE))
    return (2 * rows.astype(int) - 1).reshape(-1, N, N)


def surface(spins: np.ndarray):
    """F = -ln P over (mean spin of left half, mean spin of right half)."""
    half = N // 2
    ml = spins[:, :, :half].mean(axis=(1, 2))
    mr = spins[:, :, half:].mean(axis=(1, 2))
    edges = np.linspace(-1.0, 1.0, BINS + 1)
    counts, _, _ = np.histogram2d(ml, mr, bins=[edges, edges])
    p = counts / counts.sum()
    with np.errstate(divide="ignore"):
        f = -np.log(p)
    f = np.where(np.isfinite(f), f, np.nan)
    return f - np.nanmin(f), edges, int((counts > 0).sum())


def main():
    fig, axes = plt.subplots(1, len(COUPLINGS), figsize=(16.5, 5.6),
                             facecolor="#0d0d0d")
    for ax, bj in zip(axes, COUPLINGS):
        f, edges, filled = surface(draws(bj))
        c = 0.5 * (edges[:-1] + edges[1:])
        im = ax.pcolormesh(c, c, f.T, cmap="afmhot_r", vmin=0, vmax=8,
                           shading="auto")
        ax.contour(c, c, f.T, levels=np.arange(0.5, 8, 0.75),
                   colors="black", linewidths=0.4, alpha=0.55)
        # arrows: -grad F on the smoothed surface, only where it is defined
        g = np.where(np.isfinite(f), f, np.nanmax(f))
        gy, gx = np.gradient(g)
        step = 2
        s = (slice(None, None, step), slice(None, None, step))
        mask = np.isfinite(f)[s]
        X, Y = np.meshgrid(c[::step], c[::step], indexing="ij")
        ax.quiver(X[mask], Y[mask], -gx[s][mask], -gy[s][mask],
                  color="#39ff5a", scale=26, width=0.004, alpha=0.9)
        ax.set_title(f"beta*J = {bj}   ({bj / KC:.2f} x Kc)   "
                     f"{filled} of {BINS * BINS} cells visited",
                     color="#e8e8e8", fontsize=11)
        ax.set_xlabel("mean spin, left half", color="#bdbdbd")
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#8a8a8a")
        for sp in ax.spines.values():
            sp.set_color("#3a3a3a")
    axes[0].set_ylabel("mean spin, right half", color="#bdbdbd")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("free energy  F = -ln P   (arbitrary offset)", color="#bdbdbd")
    cb.ax.tick_params(colors="#8a8a8a")
    fig.suptitle(
        f"What the sampler actually explores  |  {N}x{N} binary lattice, "
        f"{SAMPLE['n_chains'] * SAMPLE['n_samples']:,} draws per panel, "
        f"thrml on CPU (no Z1 hardware)",
        color="#f0f0f0", fontsize=13)
    fig.savefig("audit/free_energy_surface.png", dpi=125,
                facecolor="#0d0d0d", bbox_inches="tight")
    print("  -> audit/free_energy_surface.png")


if __name__ == "__main__":
    main()
