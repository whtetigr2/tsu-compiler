#!/usr/bin/env python3
"""H2 — hard-core placement 8x8: THRML sample gallery + exclusion decode.

Honesty: JAX/THRML SIM only — not Extropic silicon. No energy/watt claims.

Decode metric
-------------
A layout is *valid* iff no two occupied (binary 1) sites share a lattice edge
(classical hard-core / packing keep-out). Metrics:

  valid_rate          = (# valid samples) / N
  exclusion_violation_rate = 1 - valid_rate
  mean_violations     = mean count of occupied-occupied NN edges
  mean_occupancy      = mean number of occupied sites

Run (Observatory venv)::

  cd demo/gibbs-observatory   # or /workspace/gibbs-observatory
  .venv/bin/python scripts/sample_placement_exclusion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.sampler_engine import SamplerConfig, SamplerEngine  # noqa: E402

RECEIPT_ID = "prog_placement_8x8_exclusion"
WIDTH = HEIGHT = 8
N_SITES = WIDTH * HEIGHT
OUT_DIR = ROOT / "artifacts" / "placement_8x8_exclusion"
HERO_DIR = ROOT / "hero" / "placement"
PROG_DIR = ROOT / "programs" / "placement_8x8_exclusion"


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


def to_binary(state) -> np.ndarray:
    arr = np.asarray(state).ravel()
    if arr.dtype == bool or np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.int8)
    if np.any(arr < 0):
        return (arr > 0).astype(np.int8)
    return (arr.astype(np.float64) > 0.5).astype(np.int8)


def exclusion_stats(flat: np.ndarray, edges: list[tuple[int, int]]) -> dict:
    flat = flat[:N_SITES]
    viol = sum(1 for i, j in edges if flat[i] == 1 and flat[j] == 1)
    occ = int(flat.sum())
    return {
        "valid": viol == 0,
        "n_violations": int(viol),
        "occupancy": occ,
        "density": float(occ / N_SITES),
        "grid": flat.reshape(HEIGHT, WIDTH).tolist(),
    }


def render_gallery(
    grids: list[list[list[int]]],
    titles: list[str],
    path: Path,
    *,
    cols: int = 4,
    cell: int = 14,
    title: str = "Hard-core placement 8×8 · THRML sim · not silicon",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n = len(grids)
    cols = min(cols, max(1, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4 + 0.4))
    axes = np.atleast_2d(axes)
    cmap = ListedColormap(["#1a1d21", "#e8b84a"])  # empty / occupied
    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= n:
            ax.axis("off")
            continue
        g = np.asarray(grids[idx], dtype=float)
        ax.imshow(g, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(titles[idx], fontsize=8, color="#c8cdd3")
        for spine in ax.spines.values():
            spine.set_color("#333")
    fig.patch.set_facecolor("#0c0e10")
    fig.suptitle(title, color="#e8ecf0", fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64, help="number of THRML samples")
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--steps-per-sample", type=int, default=4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HERO_DIR.mkdir(parents=True, exist_ok=True)
    PROG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = SamplerConfig(
        receipt_id=RECEIPT_ID,
        size=WIDTH,
        beta=args.beta,
        warmup=args.warmup,
        steps_per_sample=args.steps_per_sample,
        batch_size=min(args.n, 64),
        seed=args.seed,
    )
    engine = SamplerEngine(cfg)
    if engine.config.sampling_fallback:
        print("ERROR: receipt not THRML-ready; engine fell back to lattice2d", file=sys.stderr)
        return 2

    # sample in chunks of ≤64
    remaining = args.n
    all_states: list[np.ndarray] = []
    while remaining > 0:
        take = min(64, remaining)
        batch = engine.sample_batch(n_samples=take, warmup=args.warmup if not all_states else 0)
        states = batch.get("states") or []
        for s in states:
            all_states.append(to_binary(s))
        remaining -= len(states)
        if not states:
            break

    edges = grid_edges()
    decoded = [exclusion_stats(s, edges) for s in all_states]
    n = len(decoded)
    n_valid = sum(1 for d in decoded if d["valid"])
    valid_rate = float(n_valid / n) if n else 0.0
    viol_rate = 1.0 - valid_rate
    mean_viol = float(np.mean([d["n_violations"] for d in decoded])) if n else 0.0
    mean_occ = float(np.mean([d["occupancy"] for d in decoded])) if n else 0.0

    # receipt single-sample decode (already on shelf) for cross-check
    receipt_sample = json.loads((ROOT / "receipts" / RECEIPT_ID / "sample.json").read_text())
    rs_flat = np.zeros(N_SITES, dtype=np.int8)
    for k, v in receipt_sample.get("decoded", {}).items():
        # g{x}_{y}
        parts = k[1:].split("_")
        x, y = int(parts[0]), int(parts[1])
        rs_flat[y * WIDTH + x] = int(v)
    receipt_decode = exclusion_stats(rs_flat, edges)

    metrics = {
        "receipt_id": RECEIPT_ID,
        "honesty": "JAX/THRML software sim — not Extropic silicon. No energy/watt claims.",
        "n_samples": n,
        "beta": args.beta,
        "warmup": args.warmup,
        "steps_per_sample": args.steps_per_sample,
        "seed": args.seed,
        "decode": {
            "valid_hardcore_rate": valid_rate,
            "exclusion_violation_rate": viol_rate,
            "mean_edge_violations": mean_viol,
            "mean_occupancy": mean_occ,
            "n_valid": n_valid,
            "n_invalid": n - n_valid,
            "definition": (
                "valid iff no occupied-occupied nearest-neighbour edge "
                "(hard-core exclusion / keep-out)"
            ),
        },
        "receipt_sample_decode": receipt_decode,
        "compile_note": {
            "n_nodes": 64,
            "mediators": 0,
            "bipartite": True,
            "max_degree": 4,
            "verdict": "COMPILED",
            "fabric_tax": 1.0,
        },
        "verification_shelf": {
            "task_validity": 0.25296875,
            "note": "shelf verification.json task_validity ≈ fraction of valid hard-core layouts under longer chain",
        },
    }

    (OUT_DIR / "decode_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    sample_payload = {
        "n": n,
        "grids": [d["grid"] for d in decoded],
        "valid": [d["valid"] for d in decoded],
        "n_violations": [d["n_violations"] for d in decoded],
        "occupancy": [d["occupancy"] for d in decoded],
    }
    (OUT_DIR / "sample_grids.json").write_text(json.dumps(sample_payload) + "\n")

    # gallery: prefer valid layouts first
    ranked = sorted(range(n), key=lambda i: (not decoded[i]["valid"], decoded[i]["n_violations"], -decoded[i]["occupancy"]))
    show = ranked[:8]
    grids = [decoded[i]["grid"] for i in show]
    titles = [
        f"{'OK' if decoded[i]['valid'] else 'VIOL×'+str(decoded[i]['n_violations'])} · occ={decoded[i]['occupancy']}"
        for i in show
    ]
    gallery_path = OUT_DIR / "gallery_thrml.png"
    render_gallery(grids, titles, gallery_path)

    # hero: best valid + one invalid contrast if any
    best_valid = next((decoded[i] for i in ranked if decoded[i]["valid"]), decoded[0])
    render_gallery(
        [best_valid["grid"]],
        [f"valid hard-core · occ={best_valid['occupancy']}"],
        HERO_DIR / "hero_valid_layout.png",
        cols=1,
        title="Placement Lab · valid hard-core · THRML · not silicon",
    )
    # copy gallery to hero + programs mirror
    import shutil

    shutil.copy2(gallery_path, HERO_DIR / "hero_gallery.png")
    shutil.copy2(gallery_path, PROG_DIR / "gallery_thrml.png")
    shutil.copy2(OUT_DIR / "decode_metrics.json", PROG_DIR / "decode_metrics.json")
    shutil.copy2(OUT_DIR / "sample_grids.json", PROG_DIR / "sample_grids.json")

    print(json.dumps(metrics["decode"], indent=2))
    print(f"wrote {gallery_path}")
    print(f"valid_rate={valid_rate:.4f} exclusion_violation_rate={viol_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
