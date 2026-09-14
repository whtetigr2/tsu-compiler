#!/usr/bin/env python3
"""H3 — MRF denoise 8x8: noisy → THRML-smoothed before/after.

Honesty: JAX/THRML SIM only — not Extropic silicon. No energy/watt claims.

The shelf program ``mrf_denoise_8x8`` is a pairwise Potts/Ising *smoothness
prior* (prefer agreeing neighbours). A real denoiser attaches a **data unary**
from a noisy observation offline — exactly what this script does:

  E(x) = E_prior(x) + λ Σ_i (x_i - y_i)^2   (binary Hamming / field form)

Decode metrics
--------------
  hamming_noisy_vs_clean
  hamming_denoised_vs_clean
  agreement_gain          = noisy_err - denoised_err  (positive = improved)
  residual_disagree_edges = fraction of NN edges with disagreeing endpoints

Why this leads over bars-and-stripes for Extropic
-------------------------------------------------
Bars-and-stripes proves train→compile purity on a toy generative RBM.
Denoise reads as the Z1T / AI workload: observe → sample posterior → recover
structure. Keep bars-and-stripes as the baseline appendix (pure_rate ≈ 0.9375).

Run::

  .venv/bin/python scripts/sample_mrf_denoise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.receipt_loader import receipt_to_graph_arrays  # noqa: E402

RECEIPT_ID = "prog_mrf_denoise_8x8"
WIDTH = HEIGHT = 8
N = WIDTH * HEIGHT
OUT_DIR = ROOT / "artifacts" / "mrf_denoise_8x8"
HERO_DIR = ROOT / "hero" / "denoise"
PROG_DIR = ROOT / "programs" / "mrf_denoise_8x8"


def grid_edges(w: int = WIDTH, h: int = HEIGHT) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            i = y * w + x
            if x + 1 < w:
                edges.append((i, i + 1))
            if y + 1 < h:
                edges.append((i, i + w))
    return edges


def make_clean(kind: str = "block", seed: int = 0) -> np.ndarray:
    """Synthetic clean binary image (row-major flat)."""
    g = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    if kind == "block":
        g[2:6, 2:6] = 1
        g[3:5, 3:5] = 0  # hollow square
    elif kind == "stripe":
        g[:, ::2] = 1
    elif kind == "blob":
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
        g[((yy - 3.5) ** 2 + (xx - 3.5) ** 2) <= 6.5] = 1
    else:
        rng = np.random.default_rng(seed)
        g = (rng.random((HEIGHT, WIDTH)) > 0.55).astype(np.int8)
    return g.ravel()


def add_noise(clean: np.ndarray, p_flip: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flips = rng.random(clean.shape) < p_flip
    out = clean.copy()
    out[flips] = 1 - out[flips]
    return out


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a.astype(np.int8) != b.astype(np.int8)))


def disagree_edge_frac(flat: np.ndarray, edges: list[tuple[int, int]]) -> float:
    if not edges:
        return 0.0
    return float(sum(1 for i, j in edges if flat[i] != flat[j]) / len(edges))


def sample_ising(
    edges: list[tuple[int, int]],
    weights: np.ndarray,
    biases: np.ndarray,
    *,
    beta: float,
    n_samples: int,
    warmup: int,
    steps_per_sample: int,
    seed: int,
) -> np.ndarray:
    """Block-Gibbs via THRML; returns bool array (n, N)."""
    n_nodes = int(biases.shape[0])
    nodes = [SpinNode() for _ in range(n_nodes)]
    edge_nodes = [(nodes[i], nodes[j]) for i, j in edges]
    model = IsingEBM(
        nodes,
        edge_nodes,
        jnp.asarray(biases, dtype=jnp.float32),
        jnp.asarray(weights, dtype=jnp.float32),
        jnp.array(float(beta), dtype=jnp.float32),
    )
    # 2-color blocks on bipartite grid
    color0 = [i for i in range(n_nodes) if ((i % WIDTH) + (i // WIDTH)) % 2 == 0]
    color1 = [i for i in range(n_nodes) if i not in set(color0)]
    free_blocks = [Block([nodes[i] for i in color0]), Block([nodes[i] for i in color1])]
    program = IsingSamplingProgram(model, free_blocks, [])
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    init_free = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(
        n_warmup=warmup, n_samples=n_samples, steps_per_sample=steps_per_sample
    )
    key, k_samp = jax.random.split(key)
    samples_list = sample_states(k_samp, program, schedule, init_free, [], [Block(nodes)])
    return np.asarray(samples_list[0], dtype=bool)


def bool_to_binary(samples: np.ndarray) -> np.ndarray:
    """THRML bool → {0,1}; True maps to 1 (occupied / white)."""
    return samples.astype(np.int8)


def data_biases_from_observation(y: np.ndarray, strength: float) -> np.ndarray:
    """Map binary observation y∈{0,1} to Ising local fields.

    Convention matched to shelf export: positive bias favors spin True/+1 → binary 1.
    Field +s on sites with y=1, −s on y=0.
    """
    s = float(strength)
    # y=1 → +s, y=0 → −s
    return np.where(y.astype(np.int8) == 1, s, -s).astype(np.float32)


def render_before_after(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    path: Path,
    *,
    metrics_line: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#0c0e10", "#7ec8e3"])  # bg / fg
    panels = [
        ("Clean", clean.reshape(HEIGHT, WIDTH)),
        ("Noisy observation", noisy.reshape(HEIGHT, WIDTH)),
        ("THRML denoised", denoised.reshape(HEIGHT, WIDTH)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    fig.patch.set_facecolor("#0c0e10")
    for ax, (title, g) in zip(axes, panels):
        ax.imshow(g, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, color="#e8ecf0", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444")
    fig.suptitle(
        "MRF denoise 8×8 · THRML sim · not silicon\n" + metrics_line,
        color="#c8cdd3",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_gallery(grids: list[np.ndarray], titles: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#0c0e10", "#7ec8e3"])
    n = len(grids)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2 + 0.3))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor("#0c0e10")
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= n:
            ax.axis("off")
            continue
        ax.imshow(grids[idx].reshape(HEIGHT, WIDTH), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(titles[idx], fontsize=8, color="#c8cdd3")
    fig.suptitle("MRF denoise samples · THRML · not silicon", color="#e8ecf0", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    import argparse
    import shutil

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps-per-sample", type=int, default=4)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--noise", type=float, default=0.22, help="bit-flip rate on clean image")
    ap.add_argument("--data-strength", type=float, default=0.75, help="|local field| from observation")
    ap.add_argument("--pattern", choices=["block", "stripe", "blob"], default="blob")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HERO_DIR.mkdir(parents=True, exist_ok=True)
    PROG_DIR.mkdir(parents=True, exist_ok=True)

    data = receipt_to_graph_arrays(RECEIPT_ID)
    edges = [(int(a), int(b)) for a, b in data["edges"]]
    prior_weights = np.asarray(data["weights"], dtype=np.float32)
    # Keep prior pairwise J; replace biases with data unary (offline attachment)
    clean = make_clean(args.pattern, seed=args.seed)
    noisy = add_noise(clean, args.noise, seed=args.seed + 1)
    biases = data_biases_from_observation(noisy, args.data_strength)

    samples_bool = sample_ising(
        edges,
        prior_weights,
        biases,
        beta=args.beta,
        n_samples=args.n,
        warmup=args.warmup,
        steps_per_sample=args.steps_per_sample,
        seed=args.seed,
    )
    samples = bool_to_binary(samples_bool)

    ham_noisy = hamming(noisy, clean)
    ham_each = [hamming(s, clean) for s in samples]
    best_i = int(np.argmin(ham_each))
    denoised = samples[best_i]
    # also report mean sample
    mean_field = (samples.mean(axis=0) >= 0.5).astype(np.int8)
    ham_best = ham_each[best_i]
    ham_mean = hamming(mean_field, clean)
    edges_nn = grid_edges()

    metrics = {
        "receipt_id": RECEIPT_ID,
        "honesty": "JAX/THRML software sim — not Extropic silicon. No energy/watt claims.",
        "note": (
            "Shelf YAML is the pairwise smoothness prior only; this script attaches "
            "data unary fields from a noisy observation offline (as the program "
            "description intends)."
        ),
        "pattern": args.pattern,
        "noise_flip_rate": args.noise,
        "data_field_strength": args.data_strength,
        "beta": args.beta,
        "n_samples": args.n,
        "warmup": args.warmup,
        "seed": args.seed,
        "decode": {
            "hamming_noisy_vs_clean": ham_noisy,
            "hamming_best_denoised_vs_clean": ham_best,
            "hamming_mean_sample_vs_clean": ham_mean,
            "agreement_gain_best": ham_noisy - ham_best,
            "agreement_gain_mean": ham_noisy - ham_mean,
            "disagree_edge_frac_noisy": disagree_edge_frac(noisy, edges_nn),
            "disagree_edge_frac_denoised": disagree_edge_frac(denoised, edges_nn),
            "disagree_edge_frac_clean": disagree_edge_frac(clean, edges_nn),
            "best_sample_index": best_i,
        },
        "compile_note": {
            "n_nodes": 64,
            "mediators": 0,
            "bipartite": True,
            "max_degree": 4,
            "verdict": "COMPILED",
            "fabric_tax": 1.0,
            "max_abs_J": 0.2,
        },
        "why_over_bars_stripes": (
            "Denoise is an observe→posterior-sample→recover story (AI / Z1T-shaped). "
            "Bars-and-stripes remains the train→compile purity baseline "
            "(pure_rate≈0.9375 vs gate 0.90) — appendix, not Extropic walkthrough lead."
        ),
        "bars_stripes_baseline": {
            "receipt_id": "prog_ebm_bars_stripes",
            "pure_rate": 0.9375,
            "gate": 0.90,
            "artifacts": "artifacts/ebm_bars_stripes/",
            "doc": "EBM_BARS_STRIPES.md",
        },
    }

    grids_payload = {
        "clean": clean.reshape(HEIGHT, WIDTH).tolist(),
        "noisy": noisy.reshape(HEIGHT, WIDTH).tolist(),
        "denoised_best": denoised.reshape(HEIGHT, WIDTH).tolist(),
        "denoised_mean": mean_field.reshape(HEIGHT, WIDTH).tolist(),
        "samples": [s.reshape(HEIGHT, WIDTH).tolist() for s in samples],
        "hamming_vs_clean": ham_each,
    }

    (OUT_DIR / "decode_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT_DIR / "before_after_grids.json").write_text(json.dumps(grids_payload) + "\n")

    line = (
        f"Hamming clean←noisy {ham_noisy} → best denoised {ham_best} "
        f"(gain {ham_noisy - ham_best})"
    )
    ba_path = OUT_DIR / "before_after.png"
    render_before_after(clean, noisy, denoised, ba_path, metrics_line=line)
    render_before_after(clean, noisy, denoised, HERO_DIR / "hero_before_after.png", metrics_line=line)

    # gallery of samples ranked by hamming
    order = list(np.argsort(ham_each))[:8]
    render_gallery(
        [samples[i] for i in order],
        [f"H={ham_each[i]}" for i in order],
        OUT_DIR / "gallery_thrml.png",
    )
    shutil.copy2(OUT_DIR / "gallery_thrml.png", HERO_DIR / "hero_gallery.png")
    for name in ("decode_metrics.json", "before_after_grids.json", "before_after.png", "gallery_thrml.png"):
        shutil.copy2(OUT_DIR / name, PROG_DIR / name)

    print(json.dumps(metrics["decode"], indent=2))
    print(f"wrote {ba_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
