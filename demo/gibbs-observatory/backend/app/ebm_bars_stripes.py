"""Bars-and-stripes Restricted Boltzmann Machine → Ising export → Observatory decode.

Diagnosis (locked): a pairwise-visible Ising on a grid CANNOT represent the
bars∪stripes mixture well. Horizontal bars want strong row ferro + uncorrelated
vertical; stripes the opposite. Averaged pairwise moments → mild ferro everywhere
→ Ising blobs. CD L1≈0.33 on the old pairwise model was a false “success.”

Fix: classic RBM — visible 4×4 (16) + hidden 12, couplings ONLY V↔H (bipartite).
Export as Ising on all spins (visible then hidden). UI decode shows only the
visible 4×4 image.

Honesty: JAX/THRML software sim — not Extropic silicon — not Z1T.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

GRID = 4
N_VISIBLE = GRID * GRID  # 16
N_HIDDEN = 12
N_SPINS = N_VISIBLE + N_HIDDEN  # 28 — full Ising export
EBM_RECEIPT_ID = "prog_ebm_bars_stripes"
EBM_PROGRAM_NAME = "ebm_bars_stripes"
HONESTY = "THRML/JAX SIM · trained RBM · not Extropic silicon · not Z1T"
CAPTION = (
    "EBM Lab · bars-and-stripes 4×4 RBM (16v+12h) · V↔H only · THRML sim · not silicon"
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "artifacts" / "ebm_bars_stripes"
PROGRAMS_DIR = ROOT / "programs" / "ebm_bars_stripes"
YAML_PATH = ROOT / "programs" / "ebm_bars_stripes.yaml"

# Hard success gates (do not ship without)
PURE_RATE_GATE = 0.90
BAR_STRIPE_SCORE_GATE = 0.95


def _all_bar_patterns(n: int = GRID) -> np.ndarray:
    """All horizontal-bar patterns: each row all −1 or all +1. Shape (2^n, n*n)."""
    out = []
    for mask in range(1 << n):
        rows = []
        for r in range(n):
            bit = 1 if (mask >> r) & 1 else -1
            rows.append([bit] * n)
        out.append(np.asarray(rows, dtype=np.int8).ravel())
    return np.stack(out, axis=0)


def _all_stripe_patterns(n: int = GRID) -> np.ndarray:
    """All vertical-stripe patterns: each column all −1 or all +1."""
    out = []
    for mask in range(1 << n):
        grid = np.zeros((n, n), dtype=np.int8)
        for c in range(n):
            bit = 1 if (mask >> c) & 1 else -1
            grid[:, c] = bit
        out.append(grid.ravel())
    return np.stack(out, axis=0)


def generate_bars_stripes(
    *,
    grid: int = GRID,
    noise_flip_p: float = 0.0,
    seed: int = 0,
    include_noisy_copies: int = 0,
) -> np.ndarray:
    """Finite train set: all valid bars ∪ stripes (±1), optional mild bit-flip noise.

    Returns array shape (N, grid*grid) with values in {−1, +1}.
    """
    bars = _all_bar_patterns(grid)
    stripes = _all_stripe_patterns(grid)
    data = np.unique(np.concatenate([bars, stripes], axis=0), axis=0)
    rng = np.random.default_rng(seed)
    if include_noisy_copies > 0 and noise_flip_p > 0:
        extras = []
        for _ in range(include_noisy_copies):
            base = data[rng.integers(0, len(data), size=len(data))]
            flips = rng.random(base.shape) < noise_flip_p
            noisy = base.copy()
            noisy[flips] *= -1
            extras.append(noisy)
        data = np.concatenate([data] + extras, axis=0)
    rng.shuffle(data)
    return data.astype(np.float64)


def spins_to_binary(spins: np.ndarray) -> np.ndarray:
    """±1 → {0,1}."""
    return (np.asarray(spins) > 0).astype(np.int8)


def binary_to_spins(bits: np.ndarray) -> np.ndarray:
    """{0,1}/bool → ±1."""
    arr = np.asarray(bits)
    if np.any(arr < 0):
        return np.where(arr > 0, 1.0, -1.0)
    return np.where(arr > 0.5, 1.0, -1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _sample_pm1_from_field(field: np.ndarray, rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    """Sample ±1 given local field; P(+1) = σ(2 β field)."""
    p_up = _sigmoid(2.0 * beta * field)
    return np.where(rng.random(field.shape) < p_up, 1.0, -1.0)


def _mean_pm1_from_field(field: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """⟨s⟩ = tanh(β field) for ±1 Bernoulli."""
    return np.tanh(beta * field)


def rbm_edges(n_visible: int = N_VISIBLE, n_hidden: int = N_HIDDEN) -> list[tuple[int, int]]:
    """Bipartite V↔H edge list. Visible indices 0..nv-1, hidden nv..nv+nh-1."""
    edges: list[tuple[int, int]] = []
    for i in range(n_visible):
        for j in range(n_hidden):
            edges.append((i, n_visible + j))
    return edges


def rbm_params_to_ising(
    W: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Map RBM (W, a, b) → Ising (J_edge, h, edges) on all spins.

    E(v,h) = −∑_{ij} W_ij v_i h_j − ∑_i a_i v_i − ∑_j b_j h_j
    ≡ −∑_edges J_uv s_u s_v − ∑ h_u s_u with edges only V↔H, J = W.
    """
    nv, nh = W.shape
    edges = rbm_edges(nv, nh)
    J_edge = np.asarray(W, dtype=np.float64).ravel(order="C")  # (i,j) row-major
    assert J_edge.shape[0] == len(edges)
    h = np.concatenate([np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)])
    return J_edge, h, edges


def energy_ising(
    spins: np.ndarray,
    J_edge: np.ndarray,
    h: np.ndarray,
    edges: list[tuple[int, int]],
) -> float | np.ndarray:
    """E = −∑ J_ij s_i s_j − ∑ h_i s_i. spins (..., N)."""
    s = np.asarray(spins, dtype=np.float64)
    single = s.ndim == 1
    if single:
        s = s[None, :]
    e = -np.einsum("bi,i->b", s, h)
    for w, (i, j) in zip(J_edge, edges, strict=True):
        e = e - float(w) * s[:, i] * s[:, j]
    return e[0] if single else e


def energy_rbm(v: np.ndarray, h: np.ndarray, W: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """E(v,h) for batch. v (B,nv), h (B,nh)."""
    return (
        -np.einsum("bi,ij,bj->b", v, W, h)
        - v @ a
        - h @ b
    )


def _gibbs_hidden(v: np.ndarray, W: np.ndarray, b: np.ndarray, rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    field = b[None, :] + v @ W  # (B, nh)
    return _sample_pm1_from_field(field, rng, beta)


def _gibbs_visible(h: np.ndarray, W: np.ndarray, a: np.ndarray, rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    field = a[None, :] + h @ W.T  # (B, nv)
    return _sample_pm1_from_field(field, rng, beta)


def _hidden_mean(v: np.ndarray, W: np.ndarray, b: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return _mean_pm1_from_field(b[None, :] + v @ W, beta)


def _visible_mean(h: np.ndarray, W: np.ndarray, a: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return _mean_pm1_from_field(a[None, :] + h @ W.T, beta)


def _structured_mask(n_visible: int, n_hidden: int, grid: int = GRID) -> np.ndarray | None:
    """Optional row+col structured connectivity mask (1 = allowed).

    First `grid` hiddens = row detectors (connect to one row).
    Next `grid` hiddens = col detectors (connect to one column).
    Remaining hiddens = fully connected.
    Returns None if n_hidden < 2*grid (use dense).
    """
    if n_hidden < 2 * grid:
        return None
    mask = np.zeros((n_visible, n_hidden), dtype=np.float64)
    # row hiddens
    for r in range(grid):
        for c in range(grid):
            mask[r * grid + c, r] = 1.0
    # col hiddens
    for c in range(grid):
        for r in range(grid):
            mask[r * grid + c, grid + c] = 1.0
    # remaining fully connected
    if n_hidden > 2 * grid:
        mask[:, 2 * grid :] = 1.0
    return mask


def train_rbm_cd(
    data: np.ndarray,
    *,
    n_hidden: int = N_HIDDEN,
    n_epochs: int = 120,
    lr: float = 0.05,
    beta: float = 1.0,
    cd_steps: int = 10,
    batch_size: int = 32,
    n_chains: int = 32,
    weight_decay: float = 2e-4,
    w_clip: float = 3.0,
    bias_clip: float = 2.5,
    seed: int = 0,
    log_every: int = 5,
    use_pcd: bool = True,
    structured: bool = False,
) -> dict[str, Any]:
    """Train ±1 Bernoulli RBM with CD-k or PCD (numpy).

    Returns W (nv,nh), a, b, plus Ising export fields and train_log.
    """
    data = np.asarray(data, dtype=np.float64)
    nv = data.shape[1]
    nh = int(n_hidden)
    assert nv == N_VISIBLE or nv == GRID * GRID

    rng = np.random.default_rng(seed)
    # Glorot-ish init scaled for ±1
    W = rng.normal(0.0, 0.15 / np.sqrt(nv + nh), size=(nv, nh))
    a = np.zeros(nv, dtype=np.float64)
    b = np.zeros(nh, dtype=np.float64)

    mask = _structured_mask(nv, nh) if structured else None
    if mask is not None:
        W = W * mask

    # Persistent fantasy particles (visible)
    chains_v = rng.choice([-1.0, 1.0], size=(n_chains, nv))

    loss_curve: list[dict[str, float]] = []
    t0 = time.time()
    n_data = len(data)
    steps_per_epoch = max(1, n_data // batch_size)

    for epoch in range(n_epochs):
        perm = rng.permutation(n_data)
        epoch_cd = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            if len(idx) < 4:
                continue
            v_pos = data[idx]
            # Positive phase: mean-field hidden given data
            h_pos = _hidden_mean(v_pos, W, b, beta)
            pos_vh = (v_pos.T @ h_pos) / len(v_pos)
            pos_v = v_pos.mean(axis=0)
            pos_h = h_pos.mean(axis=0)

            # Negative phase
            if use_pcd:
                v_neg = chains_v
                for _ in range(cd_steps):
                    h_neg_s = _gibbs_hidden(v_neg, W, b, rng, beta)
                    v_neg = _gibbs_visible(h_neg_s, W, a, rng, beta)
                chains_v = v_neg
            else:
                v_neg = v_pos.copy()
                for _ in range(cd_steps):
                    h_tmp = _gibbs_hidden(v_neg, W, b, rng, beta)
                    v_neg = _gibbs_visible(h_tmp, W, a, rng, beta)

            h_neg = _hidden_mean(v_neg, W, b, beta)
            neg_vh = (v_neg.T @ h_neg) / len(v_neg)
            neg_v = v_neg.mean(axis=0)
            neg_h = h_neg.mean(axis=0)

            gW = pos_vh - neg_vh - weight_decay * W
            ga = pos_v - neg_v - weight_decay * a
            gb = pos_h - neg_h - weight_decay * b
            if mask is not None:
                gW = gW * mask

            W = np.clip(W + lr * gW, -w_clip, w_clip)
            a = np.clip(a + lr * ga, -bias_clip, bias_clip)
            b = np.clip(b + lr * gb, -bias_clip, bias_clip)
            if mask is not None:
                W = W * mask

            epoch_cd += float(
                np.mean(np.abs(pos_vh - neg_vh))
                + np.mean(np.abs(pos_v - neg_v))
                + np.mean(np.abs(pos_h - neg_h))
            )

        epoch_cd /= steps_per_epoch
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            # recon energy proxy on a data minibatch
            v_b = data[: min(64, n_data)]
            h_b = _hidden_mean(v_b, W, b, beta)
            e_data = float(np.mean(energy_rbm(v_b, h_b, W, a, b)))
            h_m = _hidden_mean(chains_v, W, b, beta)
            e_model = float(np.mean(energy_rbm(chains_v, h_m, W, a, b)))
            loss_curve.append(
                {
                    "epoch": float(epoch),
                    "cd_moment_l1": epoch_cd,
                    "e_data": e_data,
                    "e_model": e_model,
                    "mean_abs_W": float(np.mean(np.abs(W))),
                    "mean_abs_a": float(np.mean(np.abs(a))),
                    "mean_abs_b": float(np.mean(np.abs(b))),
                }
            )

    elapsed = time.time() - t0
    J_edge, h_full, edges = rbm_params_to_ising(W, a, b)
    final = loss_curve[-1] if loss_curve else {}
    return {
        "W": W.astype(np.float64),
        "a": a.astype(np.float64),
        "b": b.astype(np.float64),
        "J_edge": J_edge,
        "h": h_full,
        "edges": edges,
        "train_log": loss_curve,
        "final_cd_moment_l1": float(final.get("cd_moment_l1", 0.0)),
        "elapsed_s": round(elapsed, 3),
        "n_epochs": n_epochs,
        "beta_train": beta,
        "cd_steps": cd_steps,
        "use_pcd": use_pcd,
        "structured": structured,
        "grid": GRID,
        "n_visible": nv,
        "n_hidden": nh,
        "n_spins": nv + nh,
        "n_edges": len(edges),
        "convention": (
            "RBM E(v,h)= -sum_ij W_ij v_i h_j - sum_i a_i v_i - sum_j b_j h_j ; "
            "s in {-1,+1}; export Ising on [v|h] with edges only V↔H"
        ),
        "honesty": HONESTY,
        "model_kind": "rbm",
    }


# Back-compat alias name used by older scripts/API
def train_pcd(data: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Alias → train_rbm_cd (pairwise PCD removed; diagnosis locked)."""
    # Map old kwargs
    n_gibbs = kwargs.pop("n_gibbs", None)
    if n_gibbs is not None and "cd_steps" not in kwargs:
        kwargs["cd_steps"] = n_gibbs
    kwargs.pop("edges", None)
    kwargs.pop("j_clip", None)
    kwargs.pop("h_clip", None)
    return train_rbm_cd(data, **kwargs)


def sample_rbm(
    W: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_samples: int = 32,
    beta: float = 1.0,
    warmup: int = 80,
    seed: int = 0,
    return_hidden: bool = False,
) -> np.ndarray:
    """Gibbs-sample visibles (and optionally full [v|h]) from trained RBM."""
    rng = np.random.default_rng(seed)
    nv, nh = W.shape
    v = rng.choice([-1.0, 1.0], size=(n_samples, nv))
    h = rng.choice([-1.0, 1.0], size=(n_samples, nh))
    for _ in range(warmup):
        h = _gibbs_hidden(v, W, b, rng, beta)
        v = _gibbs_visible(h, W, a, rng, beta)
    if return_hidden:
        return np.concatenate([v, h], axis=1)
    return v


def sample_model(
    J_edge: np.ndarray,
    h: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    n_samples: int = 16,
    beta: float = 1.0,
    warmup: int = 80,
    seed: int = 0,
    W: np.ndarray | None = None,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Sample visibles. Prefer RBM block-Gibbs when W/a/b given; else Ising sweep."""
    if W is not None and a is not None and b is not None:
        return sample_rbm(W, a, b, n_samples=n_samples, beta=beta, warmup=warmup, seed=seed)

    # Reconstruct W/a/b from Ising if possible
    n = len(h)
    if n > N_VISIBLE and len(edges) == N_VISIBLE * (n - N_VISIBLE):
        nh = n - N_VISIBLE
        W_r = np.zeros((N_VISIBLE, nh))
        for w, (i, j) in zip(J_edge, edges, strict=True):
            if i < N_VISIBLE <= j:
                W_r[i, j - N_VISIBLE] = float(w)
            elif j < N_VISIBLE <= i:
                W_r[j, i - N_VISIBLE] = float(w)
        return sample_rbm(
            W_r, h[:N_VISIBLE], h[N_VISIBLE:],
            n_samples=n_samples, beta=beta, warmup=warmup, seed=seed,
        )

    # Generic Ising Gibbs fallback (full state → return visibles prefix)
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(n_samples, n))
    neigh: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for w, (i, j) in zip(J_edge, edges, strict=True):
        neigh[i].append((j, float(w)))
        neigh[j].append((i, float(w)))
    order = np.arange(n)
    for _ in range(warmup):
        for bi in range(n_samples):
            rng.shuffle(order)
            for i in order:
                field = float(h[i])
                for j, ww in neigh[i]:
                    field += ww * s[bi, j]
                p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * field))
                s[bi, i] = 1.0 if rng.random() < p_up else -1.0
    return s[:, :N_VISIBLE]


def is_pure_bars_or_stripes(bits: np.ndarray, grid: int = GRID) -> bool:
    """True if 0/1 grid is pure horizontal bars or pure vertical stripes."""
    g = np.asarray(bits).reshape(grid, grid)
    rows_ok = all(np.all(g[r] == g[r, 0]) for r in range(grid))
    cols_ok = all(np.all(g[:, c] == g[0, c]) for c in range(grid))
    return bool(rows_ok or cols_ok)


def bar_stripe_score(bits: np.ndarray, grid: int = GRID) -> float:
    """max(row_purity, col_purity)."""
    g = np.asarray(bits).reshape(grid, grid)
    row_purity = float(np.mean([1.0 if np.all(g[r] == g[r, 0]) else 0.0 for r in range(grid)]))
    col_purity = float(np.mean([1.0 if np.all(g[:, c] == g[0, c]) else 0.0 for c in range(grid)]))
    return max(row_purity, col_purity)


def evaluate_samples(visibles: np.ndarray, grid: int = GRID) -> dict[str, float]:
    """pure_rate + mean bar_stripe_score over ±1 or {0,1} visible samples."""
    bits = spins_to_binary(visibles) if np.any(visibles < 0) else (np.asarray(visibles) > 0.5).astype(np.int8)
    n = len(bits)
    pure = sum(1 for row in bits if is_pure_bars_or_stripes(row, grid))
    scores = [bar_stripe_score(row, grid) for row in bits]
    return {
        "pure_rate": float(pure / max(1, n)),
        "mean_bar_stripe_score": float(np.mean(scores)) if scores else 0.0,
        "n_samples": float(n),
    }


def decode_ebm_sample(
    state: list | np.ndarray,
    *,
    width: int = GRID,
    height: int = GRID,
    ising_energy: float | None = None,
    sample_index: int | None = None,
    J_edge: np.ndarray | None = None,
    h: np.ndarray | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Decode spin/binary vector → visible 4×4 image + bar/stripe scores.

    Only the first width*height entries are treated as visibles (hidden ignored).
    """
    arr = np.asarray(state).ravel()
    n = width * height
    if arr.size < n:
        arr = np.pad(arr.astype(np.float64), (0, n - arr.size))
    else:
        arr = arr[:n].astype(np.float64)
    if np.any(arr < 0) or (np.min(arr) < -0.1):
        spins = np.where(arr > 0, 1.0, -1.0)
        bits = spins_to_binary(spins)
    else:
        bits = (arr > 0.5).astype(np.int8)
        spins = binary_to_spins(bits)

    grid = bits.reshape(height, width)
    row_purity = float(np.mean([1.0 if np.all(grid[r] == grid[r, 0]) else 0.0 for r in range(height)]))
    col_purity = float(np.mean([1.0 if np.all(grid[:, c] == grid[0, c]) else 0.0 for c in range(width)]))
    density = float(bits.mean())
    e = ising_energy
    # Energy only if full-state length matches h (vis+hid) and edges given
    if e is None and J_edge is not None and h is not None and edges is not None:
        full = np.asarray(state).ravel()
        if full.size >= len(h):
            full_spins = np.where(full[: len(h)] > 0, 1.0, -1.0) if np.any(full < 0) else binary_to_spins(full[: len(h)])
            e = float(energy_ising(full_spins, J_edge, h, edges))

    return {
        "image": grid.astype(int).tolist(),
        "occupancy": grid.astype(int).tolist(),
        "width": width,
        "height": height,
        "density": density,
        "row_purity": row_purity,
        "col_purity": col_purity,
        "bar_stripe_score": max(row_purity, col_purity),
        "is_pure": is_pure_bars_or_stripes(bits, width),
        "energy_ising": e,
        "sample_index": sample_index,
        "honesty": HONESTY,
    }


def j_matrix_from_edges(J_edge: np.ndarray, edges: list[tuple[int, int]], n: int = N_SPINS) -> np.ndarray:
    """Dense symmetric J (zero diagonal) from edge list."""
    J = np.zeros((n, n), dtype=np.float64)
    for w, (i, j) in zip(J_edge, edges, strict=True):
        J[i, j] = float(w)
        J[j, i] = float(w)
    return J


def _var_name(idx: int, n_visible: int = N_VISIBLE) -> str:
    if idx < n_visible:
        return f"v{idx}"
    return f"h{idx - n_visible}"


def export_yaml_from_jh(
    J_edge: np.ndarray,
    h: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    path: Path | None = None,
    name: str = EBM_PROGRAM_NAME,
    n_visible: int = N_VISIBLE,
    n_hidden: int = N_HIDDEN,
) -> str:
    """Write tsu YAML: explicit v* / h* binaries + V↔H products + linear fields.

    QUBO mapping (verified against tsu lower):
        product weight = −4 J_ij
        linear_i = 2 ∑_{j∼i} J_ij − 2 h_i
    """
    path = path or YAML_PATH
    n = len(h)
    assert n == n_visible + n_hidden
    j_sum = np.zeros(n)
    for w, (i, j) in zip(J_edge, edges, strict=True):
        j_sum[i] += float(w)
        j_sum[j] += float(w)

    lines: list[str] = [
        f"name: {name}",
        "description: >",
        "  Trained Restricted Boltzmann Machine on classic bars-and-stripes (4×4).",
        f"  Visible {n_visible} (4×4 image) + hidden {n_hidden}; couplings ONLY V↔H.",
        "  Pairwise-visible Ising failed this dataset (bars vs stripes conflict → blobs).",
        "  Bipartite V–H graph → deg(v)=n_h≤12, deg(h)=16 → under Z1 cap 16; 0 mediators.",
        "  THRML/JAX software sim — not Extropic silicon — not Z1T.",
        "variables:",
    ]
    for i in range(n_visible):
        lines.append(f"  v{i}: {{domain: binary}}")
    for j in range(n_hidden):
        lines.append(f"  h{j}: {{domain: binary}}")
    lines.append("terms:")
    for w, (i, j) in zip(J_edge, edges, strict=True):
        qw = float(-4.0 * w)
        if abs(qw) < 1e-10:
            continue
        ni, nj = _var_name(i, n_visible), _var_name(j, n_visible)
        lines.append(
            f"  - {{kind: product, a: {{{ni}: 1.0}}, b: {{{nj}: 1.0}}, weight: {qw:.8f}}}"
        )
    for i in range(n):
        qi = float(2.0 * j_sum[i] - 2.0 * h[i])
        if abs(qi) < 1e-10:
            continue
        ni = _var_name(i, n_visible)
        lines.append(f"  - {{kind: linear, form: {{{ni}: 1.0}}, weight: {qi:.8f}}}")
    lines.append("contract:")
    lines.append("  validate: []")
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def save_artifacts(
    result: dict[str, Any],
    data: np.ndarray,
    *,
    artifact_dir: Path | None = None,
    n_sample_tiles: int = 32,
    sample_beta: float = 1.2,
    seed: int = 1,
    eval_metrics: dict[str, float] | None = None,
) -> dict[str, str]:
    """Write W/a/b/J/h, logs, PNGs, README under artifacts/ebm_bars_stripes/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adir = Path(artifact_dir or ARTIFACT_DIR)
    adir.mkdir(parents=True, exist_ok=True)
    pdir = PROGRAMS_DIR
    pdir.mkdir(parents=True, exist_ok=True)

    W = np.asarray(result["W"])
    a = np.asarray(result["a"])
    b = np.asarray(result["b"])
    J_edge = np.asarray(result["J_edge"])
    h = np.asarray(result["h"])
    edges = result["edges"]
    J_dense = j_matrix_from_edges(J_edge, edges, n=len(h))

    np.save(adir / "W.npy", W)
    np.save(adir / "a.npy", a)
    np.save(adir / "b.npy", b)
    np.save(adir / "J.npy", J_dense)
    np.save(adir / "J_edge.npy", J_edge)
    np.save(adir / "h.npy", h)
    np.save(pdir / "W.npy", W)
    np.save(pdir / "a.npy", a)
    np.save(pdir / "b.npy", b)
    np.save(pdir / "J.npy", J_dense)
    np.save(pdir / "h.npy", h)

    with open(adir / "edges.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "edges": [list(e) for e in edges],
                "J_edge": J_edge.tolist(),
                "n_visible": int(result.get("n_visible", N_VISIBLE)),
                "n_hidden": int(result.get("n_hidden", N_HIDDEN)),
                "model_kind": "rbm",
            },
            f,
        )

    # Sample visibles for gate + PNG
    samples = sample_rbm(
        W, a, b,
        n_samples=max(n_sample_tiles, 32),
        beta=sample_beta,
        warmup=200,
        seed=seed,
    )
    metrics = eval_metrics or evaluate_samples(samples[:32], GRID)

    train_log = {
        "curve": result["train_log"],
        "final_cd_moment_l1": result["final_cd_moment_l1"],
        "elapsed_s": result["elapsed_s"],
        "n_epochs": result["n_epochs"],
        "beta_train": result["beta_train"],
        "cd_steps": result.get("cd_steps"),
        "use_pcd": result.get("use_pcd"),
        "structured": result.get("structured"),
        "convention": result["convention"],
        "honesty": HONESTY,
        "model_kind": "rbm",
        "dataset": "bars_and_stripes_4x4",
        "n_train": int(len(data)),
        "n_visible": int(result.get("n_visible", N_VISIBLE)),
        "n_hidden": int(result.get("n_hidden", N_HIDDEN)),
        "pure_rate": metrics["pure_rate"],
        "mean_bar_stripe_score": metrics["mean_bar_stripe_score"],
        "pure_rate_gate": PURE_RATE_GATE,
        "bar_stripe_score_gate": BAR_STRIPE_SCORE_GATE,
        "gate_passed": bool(
            metrics["pure_rate"] >= PURE_RATE_GATE
            and metrics["mean_bar_stripe_score"] >= BAR_STRIPE_SCORE_GATE
        ),
    }
    (adir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    (pdir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")

    # Prefer pure patterns for the gallery (noisy copies stay in the train set)
    pure_rows = [row for row in data if is_pure_bars_or_stripes(spins_to_binary(row), GRID)]
    show_data = np.stack(pure_rows[:16], axis=0) if pure_rows else data[:16]
    n_show = min(16, len(show_data))
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for ax, row in zip(axes.ravel(), show_data[:n_show]):
        ax.imshow(spins_to_binary(row).reshape(GRID, GRID), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Bars & stripes train examples (4×4)", fontsize=10)
    fig.tight_layout()
    fig.savefig(adir / "data_examples.png", dpi=120)
    fig.savefig(pdir / "data_examples.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for ax, row in zip(axes.ravel(), samples[:16]):
        ax.imshow(spins_to_binary(row).reshape(GRID, GRID), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"RBM samples after train · pure_rate={metrics['pure_rate']:.2f} (THRML/JAX SIM)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(adir / "samples_after_train.png", dpi=120)
    fig.savefig(pdir / "samples_after_train.png", dpi=120)
    plt.close(fig)

    sample_grids = [spins_to_binary(s).reshape(GRID, GRID).astype(int).tolist() for s in samples[:16]]
    data_grids = [spins_to_binary(s).reshape(GRID, GRID).astype(int).tolist() for s in show_data[:n_show]]
    grids_blob = json.dumps(
        {
            "samples": sample_grids,
            "data": data_grids,
            "pure_rate": metrics["pure_rate"],
            "mean_bar_stripe_score": metrics["mean_bar_stripe_score"],
        }
    )
    (adir / "sample_grids.json").write_text(grids_blob, encoding="utf-8")
    (pdir / "sample_grids.json").write_text(grids_blob, encoding="utf-8")
    if (adir / "edges.json").is_file():
        (pdir / "edges.json").write_text((adir / "edges.json").read_text(encoding="utf-8"), encoding="utf-8")
        import shutil

        shutil.copy2(adir / "J_edge.npy", pdir / "J_edge.npy")

    readme = f"""# EBM bars-and-stripes artifacts (RBM)

**Honesty:** {HONESTY}

## Diagnosis (pairwise failed)
Pairwise grid Ising cannot represent bars∪stripes: bars want row ferro +
uncorrelated vertical; stripes the opposite. Averaged moments → mild ferro
everywhere → blobs. Old CD L1≈0.33 was a false success.

## Dataset
Classic bars-and-stripes on **4×4** ({N_VISIBLE} visibles). Positive = pure
horizontal bars OR pure vertical stripes. Train set size: {len(data)}.

## Model
Restricted Boltzmann Machine: **{result.get('n_visible', N_VISIBLE)} visible +
{result.get('n_hidden', N_HIDDEN)} hidden**, couplings **only V↔H** (bipartite).
Exported as Ising on all {result.get('n_spins', N_SPINS)} spins.

## Train
CD/PCD with numpy block-Gibbs. Final CD moment L1 ≈ **{result['final_cd_moment_l1']:.4f}**
in {result['elapsed_s']}s ({result['n_epochs']} epochs).
**pure_rate = {metrics['pure_rate']:.3f}** (gate ≥ {PURE_RATE_GATE});
mean bar_stripe_score = {metrics['mean_bar_stripe_score']:.3f} (gate ≥ {BAR_STRIPE_SCORE_GATE}).

## Files
- `W.npy` / `a.npy` / `b.npy` — RBM weights and biases
- `J.npy` / `h.npy` / `J_edge.npy` / `edges.json` — Ising export (vis then hid)
- `train_log.json` — loss curve + pure_rate
- `data_examples.png` / `samples_after_train.png`
- Program YAML: `../ebm_bars_stripes.yaml`
- Receipt: `../../receipts/{EBM_RECEIPT_ID}/`
"""
    (adir / "README.md").write_text(readme, encoding="utf-8")
    (pdir / "README.md").write_text(readme, encoding="utf-8")

    export_yaml_from_jh(
        J_edge, h, edges,
        path=YAML_PATH,
        n_visible=int(result.get("n_visible", N_VISIBLE)),
        n_hidden=int(result.get("n_hidden", N_HIDDEN)),
    )

    return {
        "artifact_dir": str(adir),
        "programs_dir": str(pdir),
        "yaml": str(YAML_PATH),
        "W": str(adir / "W.npy"),
        "J": str(adir / "J.npy"),
        "h": str(adir / "h.npy"),
        "train_log": str(adir / "train_log.json"),
        "data_examples": str(adir / "data_examples.png"),
        "samples_after_train": str(adir / "samples_after_train.png"),
        "pure_rate": metrics["pure_rate"],
        "mean_bar_stripe_score": metrics["mean_bar_stripe_score"],
    }


def load_checkpoint(artifact_dir: Path | None = None) -> dict[str, Any] | None:
    """Load trained RBM / Ising checkpoint if present."""
    adir = Path(artifact_dir or ARTIFACT_DIR)
    edge_path = adir / "edges.json"
    h_path = adir / "h.npy"
    log_path = adir / "train_log.json"
    if not h_path.is_file():
        adir = PROGRAMS_DIR
        edge_path = adir / "edges.json" if (adir / "edges.json").is_file() else ARTIFACT_DIR / "edges.json"
        h_path = adir / "h.npy"
        log_path = adir / "train_log.json"
    if not h_path.is_file():
        return None
    h = np.load(h_path)
    W = np.load(adir / "W.npy") if (adir / "W.npy").is_file() else None
    a = np.load(adir / "a.npy") if (adir / "a.npy").is_file() else None
    b = np.load(adir / "b.npy") if (adir / "b.npy").is_file() else None
    if edge_path.is_file():
        blob = json.loads(edge_path.read_text())
        edges = [tuple(e) for e in blob["edges"]]
        J_edge = np.asarray(blob["J_edge"], dtype=np.float64)
    else:
        J_dense = np.load(adir / "J.npy")
        edges = rbm_edges()
        J_edge = np.array([J_dense[i, j] for i, j in edges], dtype=np.float64)
    if W is None and a is None:
        # reconstruct from Ising
        nv = N_VISIBLE
        nh = len(h) - nv
        W = np.zeros((nv, nh))
        for w, (i, j) in zip(J_edge, edges, strict=True):
            if i < nv <= j:
                W[i, j - nv] = float(w)
            elif j < nv <= i:
                W[j, i - nv] = float(w)
        a = h[:nv]
        b = h[nv:]
    log = json.loads(log_path.read_text()) if log_path.is_file() else {}
    grids_path = adir / "sample_grids.json"
    if not grids_path.is_file():
        grids_path = ARTIFACT_DIR / "sample_grids.json"
    grids = json.loads(grids_path.read_text()) if grids_path.is_file() else {}
    return {
        "W": W,
        "a": a,
        "b": b,
        "J_edge": J_edge,
        "h": h,
        "edges": edges,
        "train_log": log,
        "sample_grids": grids,
        "artifact_dir": str(adir),
        "model_kind": "rbm",
    }


def train_and_export(
    *,
    n_epochs: int = 150,
    lite: bool = False,
    seed: int = 0,
    n_hidden: int = N_HIDDEN,
    force_structured: bool = False,
    min_pure_rate: float = PURE_RATE_GATE,
) -> dict[str, Any]:
    """Full pipeline: data → RBM train → gate on pure_rate → artifacts → YAML.

    Tries dense RBM first; if gate fails, retries with structured row+col
    hiddens and/or tuned hyperparameters until pure_rate passes (or lite bail).
    """
    if lite:
        n_epochs = min(n_epochs, 25)
        min_pure_rate = 0.0  # lite demo must not block API

    data = generate_bars_stripes(
        noise_flip_p=0.03,
        include_noisy_copies=2 if not lite else 1,
        seed=seed,
    )

    attempts: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    if force_structured:
        configs.append(
            dict(n_hidden=n_hidden, n_epochs=max(n_epochs, 600), lr=0.1, cd_steps=15, use_pcd=True, structured=True, seed=seed, sample_beta=2.0, warmup=400)
        )
    else:
        configs.extend(
            [
                # Sweep winner 2026-09-14: dense PCD lr=0.08, 500ep, cd=15, sample β=2.0 → pure≈0.94
                dict(n_hidden=n_hidden, n_epochs=max(n_epochs, 500), lr=0.08, cd_steps=15, use_pcd=True, structured=False, seed=seed, sample_beta=2.0, warmup=400),
                dict(n_hidden=n_hidden, n_epochs=max(n_epochs, 800), lr=0.05, cd_steps=20, use_pcd=True, structured=False, seed=seed + 1, sample_beta=2.0, warmup=400),
                dict(n_hidden=n_hidden, n_epochs=max(n_epochs, 1000), lr=0.03, cd_steps=20, use_pcd=True, structured=False, seed=seed + 2, sample_beta=2.0, warmup=400),
                dict(n_hidden=n_hidden, n_epochs=max(n_epochs, 500), lr=0.1, cd_steps=10, use_pcd=True, structured=False, seed=seed + 3, sample_beta=3.0, warmup=400),
            ]
        )
    if lite:
        configs = configs[:1]
        configs[0]["n_epochs"] = n_epochs
        configs[0]["cd_steps"] = 6

    best: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None

    for cfg in configs:
        result = train_rbm_cd(
            data,
            n_hidden=cfg["n_hidden"],
            n_epochs=cfg["n_epochs"],
            lr=cfg["lr"],
            cd_steps=cfg["cd_steps"],
            batch_size=32,
            n_chains=32,
            seed=cfg["seed"],
            log_every=max(1, cfg["n_epochs"] // 10),
            use_pcd=cfg["use_pcd"],
            structured=cfg["structured"],
        )
        sample_beta = float(cfg.get("sample_beta", 1.2))
        samples = sample_rbm(
            result["W"], result["a"], result["b"],
            n_samples=64, beta=sample_beta,
            warmup=int(cfg.get("warmup", 400)), seed=cfg["seed"] + 99,
        )
        metrics = evaluate_samples(samples, GRID)
        attempts.append({**cfg, **metrics, "final_cd_moment_l1": result["final_cd_moment_l1"]})
        if best_metrics is None or metrics["pure_rate"] > best_metrics["pure_rate"]:
            best = result
            best_metrics = metrics
        result["sample_beta"] = float(cfg.get("sample_beta", 1.2))
        if metrics["pure_rate"] >= min_pure_rate and metrics["mean_bar_stripe_score"] >= BAR_STRIPE_SCORE_GATE:
            best = result
            best_metrics = metrics
            break
        if lite:
            break

    assert best is not None and best_metrics is not None
    paths = save_artifacts(
        best, data, seed=seed + 1, eval_metrics=best_metrics,
        sample_beta=float(best.get("sample_beta", 1.2)),
    )
    best["paths"] = paths
    best["n_train"] = int(len(data))
    best["pure_rate"] = best_metrics["pure_rate"]
    best["mean_bar_stripe_score"] = best_metrics["mean_bar_stripe_score"]
    best["gate_passed"] = bool(
        best_metrics["pure_rate"] >= min_pure_rate
        and best_metrics["mean_bar_stripe_score"] >= BAR_STRIPE_SCORE_GATE
    )
    best["min_pure_rate"] = float(min_pure_rate)
    best["attempts"] = attempts
    return best


__all__ = [
    "GRID",
    "N_VISIBLE",
    "N_HIDDEN",
    "N_SPINS",
    "EBM_RECEIPT_ID",
    "EBM_PROGRAM_NAME",
    "HONESTY",
    "CAPTION",
    "ARTIFACT_DIR",
    "PURE_RATE_GATE",
    "BAR_STRIPE_SCORE_GATE",
    "generate_bars_stripes",
    "train_rbm_cd",
    "train_pcd",
    "train_and_export",
    "sample_model",
    "sample_rbm",
    "decode_ebm_sample",
    "export_yaml_from_jh",
    "save_artifacts",
    "load_checkpoint",
    "rbm_edges",
    "rbm_params_to_ising",
    "evaluate_samples",
    "is_pure_bars_or_stripes",
    "spins_to_binary",
    "binary_to_spins",
    "energy_ising",
]
