"""Task 6 (plan 2026-09-04-lattice-rule-taxonomy): sample the ALREADY-
COMPILED `emergence_8x8` receipt and measure whether the four rules built
into `specs/emergence_8x8.yaml` produce recognisable structure or noise.

This does NOT call `tsu compile` -- the task's own budget allows exactly
one compile, already spent (`audit/receipts/emergence_8x8`). It calls
`tsu.simulate.simulate` (a cheap re-sample of the already-compiled physical
program -- see that module's own docstring on why this never re-derives
the energy model), then decodes EVERY draw itself (not just `simulate`'s
own single `decoded_example`) so this script can measure density, spatial
clustering, and the two hand-written rules' (PAIRWISE, GAMEPLAY) own
specific claims -- none of which a single decoded example can support.

Every number this script prints is either read straight from the
`simulate()` output or computed here, over the actual decoded draws --
nothing is asserted without being computed against real samples first,
per this project's own verification-never-fabricates rule.

Run with the project's pinned interpreter, from the repo root:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
        audit/emergence_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsu.passes.encode import encode
from tsu.simulate import _selected_encoding, simulate
from tsu.spec import load_spec

RECEIPT_DIR = REPO_ROOT / "audit" / "receipts" / "emergence_8x8"
RENDER_PATH = REPO_ROOT / "audit" / "receipts" / "emergence_8x8" / "worlds.png"
HEATMAP_PATH = REPO_ROOT / "audit" / "receipts" / "emergence_8x8" / "per_cell_probability.png"
WIDTH, HEIGHT = 8, 8


def _grid(decoded: dict[str, int]) -> np.ndarray:
    g = np.zeros((HEIGHT, WIDTH), dtype=int)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            g[y, x] = decoded[f"g{x}_{y}"]
    return g


def _largest_component(grid: np.ndarray) -> tuple[int, int]:
    """4-connected flood fill over cells == 1. Returns (largest component
    size, number of components) -- 0, 0 for an all-zero grid."""
    seen = np.zeros_like(grid, dtype=bool)
    sizes = []
    h, w = grid.shape
    for y0 in range(h):
        for x0 in range(w):
            if grid[y0, x0] != 1 or seen[y0, x0]:
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 1 and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(size)
    if not sizes:
        return 0, 0
    return max(sizes), len(sizes)


def _boundary_length(grid: np.ndarray) -> int:
    """Count of 4-adjacent cell pairs with DIFFERING values -- the direct
    complement of what GRADIENT's (1,1)-only `product_over_edges` term
    rewards (a (1,1) edge lowers energy; a (0,1)/(1,0) edge is neutral). A
    genuinely clumped configuration has FEWER such boundary edges than an
    i.i.d. field at the same density; this is a sharper test than largest-
    component size when density is high enough that i.i.d. noise already
    percolates (see the module docstring's P1 discussion)."""
    h, w = grid.shape
    count = 0
    for y in range(h):
        for x in range(w):
            if x + 1 < w and grid[y, x] != grid[y, x + 1]:
                count += 1
            if y + 1 < h and grid[y, x] != grid[y + 1, x]:
                count += 1
    return count


def _iid_baseline_boundary_length(density: float, n_trials: int, rng: np.random.Generator) -> float:
    lens = []
    for _ in range(n_trials):
        g = (rng.random((HEIGHT, WIDTH)) < density).astype(int)
        lens.append(_boundary_length(g))
    return float(np.mean(lens))


def _iid_baseline_largest_component(density: float, n_trials: int, rng: np.random.Generator) -> float:
    """Mean largest-4-connected-component size over `n_trials` INDEPENDENT
    Bernoulli(density) 8x8 grids -- the null hypothesis this experiment's
    P1 must beat: if GRADIENT's clumping has no effect, sampled worlds
    should look like this, not more clustered than this."""
    sizes = []
    for _ in range(n_trials):
        g = (rng.random((HEIGHT, WIDTH)) < density).astype(int)
        largest, _ = _largest_component(g)
        sizes.append(largest)
    return float(np.mean(sizes))


def main() -> None:
    spec = load_spec(str(RECEIPT_DIR / "spec.yaml"))
    encoding = _selected_encoding(RECEIPT_DIR)
    enc = encode(spec, encoding)

    n_chains, n_samples, n_warmup, seed = 32, 200, 400, 1
    path, got, im = simulate(RECEIPT_DIR, n_chains=n_chains, n_samples=n_samples,
                             n_warmup=n_warmup, seed=seed,
                             output_dir=RECEIPT_DIR)
    print(f"simulate() wrote: {path}")
    print(f"total draws: {len(got)} (n_chains={n_chains} x n_samples={n_samples})")

    decoded_all = []
    valid_all = []
    codeword_violations = 0
    for row in got:
        bits = dict(zip(im.nodes, row.tolist()))
        if not enc.is_codeword(bits):
            codeword_violations += 1
            continue
        decoded = enc.decode(bits)
        result = spec.contract.validate(decoded)
        decoded_all.append(decoded)
        valid_all.append(result.ok)

    n_draws = len(got)
    n_codeword = len(decoded_all)
    n_valid = sum(valid_all)
    print(f"codeword_violations: {codeword_violations} ({codeword_violations / n_draws:.4f})")
    print(f"task_validity (this run): {n_valid / n_draws:.4f}  ({n_valid}/{n_draws})")

    valid_decoded = [d for d, ok in zip(decoded_all, valid_all) if ok]
    grids = [_grid(d) for d in valid_decoded]

    # --- P2: density -------------------------------------------------
    densities = [g.sum() / g.size for g in grids]
    print(f"\n--- P2: density (raised-cell fraction, VALID samples only) ---")
    print(f"n valid grids: {len(grids)}")
    print(f"mean density: {np.mean(densities):.4f}  std: {np.std(densities):.4f}")
    print(f"min density: {np.min(densities):.4f}  max density: {np.max(densities):.4f}")
    n_distinct_grids = len({g.tobytes() for g in grids})
    print(f"distinct grids among valid samples: {n_distinct_grids} / {len(grids)}")

    # --- P1: spatial clustering vs i.i.d. baseline at the SAME density ---
    print(f"\n--- P1: spatial clustering (largest 4-connected component) ---")
    largest_sizes = []
    n_components = []
    for g in grids:
        largest, ncomp = _largest_component(g)
        largest_sizes.append(largest)
        n_components.append(ncomp)
    mean_largest = float(np.mean(largest_sizes))
    mean_ncomp = float(np.mean(n_components))
    print(f"mean largest raised component: {mean_largest:.2f} cells "
          f"(out of {WIDTH*HEIGHT})")
    print(f"mean number of raised components: {mean_ncomp:.2f}")
    rng = np.random.default_rng(0)
    mean_density = float(np.mean(densities))
    baseline_largest = _iid_baseline_largest_component(mean_density, 2000, rng)
    print(f"i.i.d. Bernoulli({mean_density:.3f}) baseline, mean largest "
          f"component over 2000 trials: {baseline_largest:.2f} cells")
    print(f"ratio (sampled / i.i.d. baseline): "
          f"{mean_largest / baseline_largest:.3f}  (both saturate near the "
          f"grid size once density is deep in the percolating regime -- "
          f"see the boundary-length statistic below for a sharper test)")

    mean_boundary = float(np.mean([_boundary_length(g) for g in grids]))
    baseline_boundary = _iid_baseline_boundary_length(mean_density, 2000, rng)
    print(f"mean boundary length (adjacent DIFFERING-value pairs): "
          f"{mean_boundary:.2f} out of 112 possible edges")
    print(f"i.i.d. Bernoulli({mean_density:.3f}) baseline boundary length, "
          f"2000 trials: {baseline_boundary:.2f}")
    print(f"ratio (sampled / i.i.d. baseline): "
          f"{mean_boundary / baseline_boundary:.3f}  (< 1.0 means MORE "
          f"clumped than chance at the same density; this is the test "
          f"GRADIENT's own reward shape should move)")

    # --- P3: PAIRWISE landmark correlation ----------------------------
    print(f"\n--- P3: PAIRWISE (g0_0, g7_7) co-occurrence ---")
    p_00 = np.mean([g[0, 0] for g in grids])
    p_77 = np.mean([g[7, 7] for g in grids])
    p_both = np.mean([g[0, 0] * g[7, 7] for g in grids])
    print(f"P(g0_0=1)={p_00:.4f}  P(g7_7=1)={p_77:.4f}  "
          f"P(both=1)={p_both:.4f}  independent-product={p_00*p_77:.4f}")
    # baseline pair: the OTHER diagonal (g0_7, g7_0) -- same Chebyshev
    # distance, opposite corners, but named by NO rule in this spec.
    p_07 = np.mean([g[7, 0] for g in grids])   # g0_7 -> row y=7, col x=0
    p_70 = np.mean([g[0, 7] for g in grids])   # g7_0 -> row y=0, col x=7
    p_other_both = np.mean([g[7, 0] * g[0, 7] for g in grids])
    print(f"baseline pair (g0_7, g7_0), named by no rule: "
          f"P(g0_7=1)={p_07:.4f}  P(g7_0=1)={p_70:.4f}  "
          f"P(both=1)={p_other_both:.4f}  independent-product={p_07*p_70:.4f}")

    # --- P4: GAMEPLAY exactly-one --------------------------------------
    print(f"\n--- P4: GAMEPLAY (g3_3, g3_4, g4_3) settlement ---")
    counts = [g[3, 3] + g[4, 3] + g[3, 4] for g in grids]  # y,x order: g3_3->[3,3], g3_4->[4,3], g4_3->[3,4]
    # NOTE: grid[y, x]; g3_3 -> x=3,y=3 -> grid[3,3]; g3_4 -> x=3,y=4 -> grid[4,3];
    # g4_3 -> x=4,y=3 -> grid[3,4]. counts already matches this.
    n_zero = sum(1 for c in counts if c == 0)
    n_one = sum(1 for c in counts if c == 1)
    n_two_plus = sum(1 for c in counts if c >= 2)
    print(f"among {len(grids)} valid samples: 0 raised={n_zero} "
          f"({n_zero/len(grids):.4f}), exactly 1 raised={n_one} "
          f"({n_one/len(grids):.4f}), 2+ raised={n_two_plus} "
          f"({n_two_plus/len(grids):.4f}) [2+ should be 0: contract-enforced]")
    naive_p_at_most_one = 3 * mean_density * (1 - mean_density) ** 2 + (1 - mean_density) ** 3
    print(f"naive independent-Bernoulli({mean_density:.3f}) P(at most 1 of "
          f"3 iid): {naive_p_at_most_one:.4f} vs measured task_validity "
          f"{n_valid/n_draws:.4f}")

    # --- per-cell raised-probability, aggregated over ALL valid samples --
    # not one of the "several worlds" the brief asks to render -- an
    # AGGREGATE statistic, kept separate and clearly labelled as such. Built
    # because the six rendered worlds below show what looked, by eye, like
    # a recurring light patch near the GAMEPLAY cells; this checks that
    # impression against every valid sample rather than trusting six.
    stacked = np.stack(grids)
    per_cell = stacked.mean(axis=0)
    print(f"\n--- per-cell P(raised), all {len(grids)} valid samples ---")
    for y in range(HEIGHT):
        print("  " + " ".join(f"{per_cell[y, x]:.2f}" for x in range(WIDTH)))
    print(f"g3_3 (gameplay): {per_cell[3, 3]:.4f}  g3_4: {per_cell[4, 3]:.4f}  "
          f"g4_3: {per_cell[3, 4]:.4f}  -- vs grid-wide mean {per_cell.mean():.4f}")

    fig2, ax2 = plt.subplots(figsize=(5, 4.5))
    im2 = ax2.imshow(per_cell, cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax2.set_xticks(range(WIDTH))
    ax2.set_yticks(range(HEIGHT))
    ax2.set_title(f"P(raised) per cell, {len(grids)} valid samples")
    fig2.colorbar(im2, ax=ax2, label="P(raised)")
    fig2.tight_layout()
    fig2.savefig(HEATMAP_PATH, dpi=150)
    print(f"rendered per-cell probability heatmap to {HEATMAP_PATH}")

    # A first look at the six rendered worlds (below) suggested a recurring
    # light patch near the grid's edges -- tested here as a real hypothesis
    # (boundary cells have lower grid-graph degree, hence weaker GRADIENT
    # coupling, hence maybe more likely to be the "hole") against ALL valid
    # samples, not just six. Reported honestly whichever way it comes out.
    def cell_degree(x: int, y: int) -> int:
        d = 0
        if x > 0: d += 1
        if x < WIDTH - 1: d += 1
        if y > 0: d += 1
        if y < HEIGHT - 1: d += 1
        return d

    by_degree: dict[int, list[float]] = {2: [], 3: [], 4: []}
    for y in range(HEIGHT):
        for x in range(WIDTH):
            by_degree[cell_degree(x, y)].append(per_cell[y, x])
    print(f"\n--- boundary-position hypothesis check (all valid samples) ---")
    print(f"mean P(raised): corner (deg 2, n=4)={np.mean(by_degree[2]):.4f}  "
          f"edge (deg 3, n=24)={np.mean(by_degree[3]):.4f}  "
          f"interior (deg 4, n=36)={np.mean(by_degree[4]):.4f}")

    # --- render several worlds ------------------------------------------
    n_render = min(6, len(grids))
    idx = np.linspace(0, len(grids) - 1, n_render).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    cmap = matplotlib.colors.ListedColormap(["#dbe9f4", "#5b7f4f"])
    for ax, i in zip(axes.flat, idx):
        g = grids[i]
        ax.imshow(g, cmap=cmap, vmin=0, vmax=1, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"sample #{i} (density={g.sum()/g.size:.2f})", fontsize=9)
    fig.suptitle("emergence_8x8 -- sampled worlds (light=0, green=raised)")
    fig.tight_layout()
    fig.savefig(RENDER_PATH, dpi=150)
    print(f"\nrendered {n_render} worlds to {RENDER_PATH}")


if __name__ == "__main__":
    main()
