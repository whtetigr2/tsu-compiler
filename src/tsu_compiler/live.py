"""Watch a model being sampled, live, and watch the diagnostics disagree.

WHAT THIS IS FOR. `audit/free_energy_sublattice.py` draws a finished free energy
surface and prints, on the same draws, a scalar R-hat next to the maximum R-hat
over individual spins. On an ordered antiferromagnet the first reads 1.0000 and
the second reads 11.11. That is the finding in `audit/findings/R20.md`, and as a
static figure it asks the reader to take the numbers on trust.

Live, it stops being a claim. You watch the surface fill in, watch the basins
separate, and watch the two numbers come apart in the same window while the
chains are visibly failing to cross between them.

THE COORDINATES ARE THE COMPILER'S OWN. Any graph this compiler can sample has
been 2-coloured by `route`/`program` -- that is what makes block Gibbs apply at
all -- so every model arrives with a natural pair of collective coordinates: the
mean spin of colour block 0 and of colour block 1. On a square lattice those are
exactly the two chessboard sublattices, which is why this reproduces the audit
figure; on any other model they are still the two halves the sampler alternates
between. Nothing here is chosen per-model by hand.

WHAT "REAL TIME" HONESTLY MEANS HERE, measured rather than hoped:

    batch     ms   ms/sample   updates/s
       10  533.5       53.35         1.9
       40  371.6        9.29         2.7
      100  389.1        3.89         2.6
      400  485.6        1.21         2.1

Wall time is dominated by per-call dispatch into thrml, not by sampling: 400
samples cost about 30% more than 40. So this takes LARGE batches at a LOW update
rate -- roughly 2.5 updates per second is the ceiling no matter how little work
each one does, and asking for smaller batches makes the view slower AND less
informative. This is a live instrument, not a 30fps animation, and it does not
pretend otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends.thrml_backend import stream_chains
from .preflight.diagnostics import RHAT_THRESHOLD, r_hat

# Per-spin R-hat is computed over a fixed subset when a model is large, because
# doing it for every spin at every frame would cost more than the sampling. The
# subset is drawn ONCE with a fixed seed and never resampled, so the trace is a
# consistent estimate over frames rather than a different quantity each time.
MAX_TRACKED_SPINS = 384
# Retained history per spin, in samples. R-hat over an unbounded history would
# grow without limit in a long session; this is a WINDOW and is labelled as one.
MAX_HISTORY = 4000


@dataclass
class LiveAccumulator:
    """Running free energy histogram and diagnostics over continuing chains."""
    blocks: tuple
    n_spins: int
    bins: int = 34
    min_count: int = 3
    seed: int = 0
    counts: np.ndarray = field(init=False)
    edges: np.ndarray = field(init=False)
    tracked: np.ndarray = field(init=False)
    hist_x: list = field(default_factory=list)   # per-batch (chains,) means
    hist_y: list = field(default_factory=list)
    hist_spin: list = field(default_factory=list)
    trace: list = field(default_factory=list)    # (draws, scalar, per_spin)
    draws: int = 0

    def __post_init__(self):
        self.edges = np.linspace(-1.0, 1.0, self.bins + 1)
        self.counts = np.zeros((self.bins, self.bins), dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        self.tracked = (np.arange(self.n_spins) if self.n_spins <= MAX_TRACKED_SPINS
                        else np.sort(rng.choice(self.n_spins, MAX_TRACKED_SPINS,
                                                replace=False)))

    def update(self, batch: np.ndarray) -> None:
        S = 2 * batch.astype(np.int8) - 1                # (chains, samples, spins)
        a = S[:, :, list(self.blocks[0])].mean(axis=2)
        b = (S[:, :, list(self.blocks[1])].mean(axis=2)
             if len(self.blocks) > 1 else a)
        c, _, _ = np.histogram2d(a.ravel(), b.ravel(),
                                 bins=[self.edges, self.edges])
        self.counts += c.astype(np.int64)
        self.draws += batch.shape[0] * batch.shape[1]

        self.hist_x.append(a)
        self.hist_y.append(b)
        self.hist_spin.append(S[:, :, self.tracked].astype(np.int8))
        while sum(h.shape[1] for h in self.hist_x) > MAX_HISTORY:
            self.hist_x.pop(0)
            self.hist_y.pop(0)
            self.hist_spin.pop(0)

        total = 0.5 * (np.concatenate(self.hist_x, axis=1)
                       + np.concatenate(self.hist_y, axis=1))
        spins = np.concatenate(self.hist_spin, axis=1)
        scalar = float(r_hat(total)) if total.shape[1] >= 2 else float("nan")
        per = [float(r_hat(spins[:, :, j].astype(np.float64)))
               for j in range(spins.shape[2])
               if spins[:, :, j].std() > 0]
        self.trace.append((self.draws, scalar, max(per) if per else float("nan")))

    def free_energy(self) -> np.ndarray:
        """F = -ln P, with cells below `min_count` left as NaN.

        Not painting them is the point: F = -ln 0 is infinite, and one stray
        visit is not evidence of a basin. A live view that smoothed over them
        would invent landscape the sampler never entered -- and would do it
        frame by frame, where it is hardest to notice."""
        p = np.where(self.counts >= self.min_count, self.counts, np.nan)
        with np.errstate(invalid="ignore"):
            p = p / np.nansum(p)
            F = -np.log(p)
        return F - np.nanmin(F)

    @property
    def visited(self) -> int:
        return int((self.counts >= self.min_count).sum())


def run(prog, n_chains: int, batch: int, warmup: int, steps: int, seed: int,
        frames: int | None = None):
    """Yield `(accumulator, batch_index)` after each batch of continuing chains."""
    acc = LiveAccumulator(blocks=tuple(prog.blocks),
                          n_spins=len(prog.ising.nodes), seed=seed)
    gen = stream_chains(prog, n_chains=n_chains, batch_samples=batch,
                        n_warmup=warmup, steps_per_sample=steps, seed=seed)
    i = 0
    while frames is None or i < frames:
        acc.update(next(gen))
        i += 1
        yield acc, i


def watch(prog, n_chains=16, batch=150, warmup=2000, steps=4, seed=0,
          frames=None, save=None, title="") -> int:
    """Live matplotlib view. Returns 0 when the window closes or frames run out.

    matplotlib is imported here rather than at module scope so that `tsuc`'s
    other commands do not require it."""
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, FG, DIM = "#0d1117", "#e8eaed", "#9aa2ad"
    fig, (axF, axR) = plt.subplots(1, 2, figsize=(14.5, 6.4), facecolor=BG,
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    fig.suptitle(title or "tsuc watch", color=FG, fontsize=12)
    centres = None
    snapshots = []

    for acc, i in run(prog, n_chains, batch, warmup, steps, seed, frames):
        F = acc.free_energy()
        if centres is None:
            centres = 0.5 * (acc.edges[:-1] + acc.edges[1:])

        axF.clear()
        axF.pcolormesh(centres, centres, F.T, cmap="coolwarm", shading="auto",
                       vmin=0, vmax=max(1e-9, np.nanpercentile(F, 96)))
        axF.scatter(acc.hist_x[-1][:, -1], acc.hist_y[-1][:, -1], s=30,
                    c="#ffffff", edgecolors=BG, linewidths=0.8, zorder=5)
        axF.set_xlabel("mean spin, colour block 0", color=DIM, fontsize=9)
        axF.set_ylabel("mean spin, colour block 1", color=DIM, fontsize=9)
        axF.set_title(f"free energy  F = -ln P     {acc.draws:,} draws     "
                      f"{acc.visited} of {acc.bins ** 2} cells visited",
                      color=FG, fontsize=10)
        axF.set_facecolor(BG)
        axF.set_aspect("equal")

        t = np.array(acc.trace)
        axR.clear()
        axR.plot(t[:, 0], t[:, 1], color="#58a6ff", lw=2,
                 label="R-hat, scalar order parameter")
        axR.plot(t[:, 0], t[:, 2], color="#ff7b72", lw=2,
                 label="max R-hat over individual spins")
        axR.axhline(RHAT_THRESHOLD, color=DIM, ls="--", lw=1,
                    label=f"threshold {RHAT_THRESHOLD}")
        axR.set_yscale("log")
        axR.set_xlabel("draws", color=DIM, fontsize=9)
        axR.set_title("the same chains, read two ways", color=FG, fontsize=10)
        axR.set_facecolor(BG)
        axR.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor=FG,
                   fontsize=8.5, loc="upper right")
        n_tracked = len(acc.tracked)
        axR.text(0.02, 0.03,
                 f"per-spin max over {n_tracked} spins"
                 + ("" if n_tracked == acc.n_spins else
                    f" (fixed subset of {acc.n_spins})")
                 + f"; R-hat over a window of the last {MAX_HISTORY:,} samples",
                 transform=axR.transAxes, fontsize=7.5, color=DIM)

        for ax in (axF, axR):
            ax.tick_params(colors=DIM, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#30363d")

        if save:
            fig.canvas.draw()
            snapshots.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        else:
            plt.pause(0.001)
            if not plt.fignum_exists(fig.number):
                break

    if save:
        # Written frame by frame with pillow rather than through FuncAnimation:
        # the frames are already rendered, so replaying them through an
        # animation object only adds a blitting contract to get wrong.
        from PIL import Image
        imgs = [Image.fromarray(s, mode="RGBA").convert("P", palette=Image.ADAPTIVE)
                for s in snapshots]
        imgs[0].save(save, save_all=True, append_images=imgs[1:],
                     duration=400, loop=0)
        print(f"  -> {save}  ({len(imgs)} frames, {acc.draws:,} draws)")
    else:
        plt.show()
    return 0
