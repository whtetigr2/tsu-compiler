"""Distribution Lab, binary ordering alloy decode + batch score.

Original Observatory gift (not Extropic codon). Maps 0/1 lattice states to
occupancy, staggered order parameter, short-range order, concentration, and
configurational energy from YAML product_over_edges weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Default mild weights matching programs/alloy_ordering_8x8.yaml
DEFAULT_WEIGHTS = {
    (0, 0): 0.45,
    (1, 1): 0.45,
    (0, 1): -0.20,
    (1, 0): -0.20,
}

ALLOY_RECEIPT_ID = "prog_alloy_ordering_8x8"
ALLOY_PROGRAM_NAME = "alloy_ordering_8x8"


def _as_binary(state: list | np.ndarray) -> np.ndarray:
    """Accept 0/1 or ±1 (or bool); return flat int 0/1."""
    arr = np.asarray(state).ravel()
    if arr.size == 0:
        return arr.astype(np.int8)
    # ±1 → map -1→0, +1→1
    if np.any(arr < 0) or (np.issubdtype(arr.dtype, np.floating) and np.any(np.abs(arr - 1.0) < 1e-9) and np.any(np.abs(arr + 1.0) < 1e-9)):
        return (arr > 0).astype(np.int8)
    return (arr.astype(np.float64) > 0.5).astype(np.int8)


def occupancy_grid(state: list | np.ndarray, width: int, height: int) -> list[list[int]]:
    """8×8 (or WxH) occupancy: 0=A, 1=B."""
    flat = _as_binary(state)
    n = width * height
    if flat.size < n:
        pad = np.zeros(n, dtype=np.int8)
        pad[: flat.size] = flat
        flat = pad
    else:
        flat = flat[:n]
    return flat.reshape(height, width).tolist()


def staggered_order(state: list | np.ndarray, width: int, height: int) -> float:
    """m_s = mean_i (−1)^{x+y} · s_mapped, s_mapped=+1 for B, −1 for A."""
    flat = _as_binary(state)
    n = width * height
    flat = flat[:n] if flat.size >= n else np.pad(flat, (0, n - flat.size))
    s_mapped = 2.0 * flat.astype(np.float64) - 1.0  # 0→−1, 1→+1
    total = 0.0
    for y in range(height):
        for x in range(width):
            stagger = 1.0 if ((x + y) % 2 == 0) else -1.0
            total += stagger * s_mapped[y * width + x]
    return float(total / n)


def concentration_b(state: list | np.ndarray, width: int, height: int) -> float:
    flat = _as_binary(state)
    n = width * height
    flat = flat[:n] if flat.size >= n else np.pad(flat, (0, n - flat.size))
    return float(flat.mean())


def grid_edges(width: int, height: int) -> list[tuple[int, int]]:
    """Nearest-neighbour undirected edges on a square lattice (row-major)."""
    edges: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            i = y * width + x
            if x + 1 < width:
                edges.append((i, i + 1))
            if y + 1 < height:
                edges.append((i, i + width))
    return edges


def unlike_edge_fraction(
    state: list | np.ndarray, width: int, height: int, edges: list[tuple[int, int]] | None = None
) -> float:
    """Short-range order proxy: fraction of NN edges with unlike endpoints."""
    flat = _as_binary(state)
    n = width * height
    flat = flat[:n] if flat.size >= n else np.pad(flat, (0, n - flat.size))
    edges = edges or grid_edges(width, height)
    if not edges:
        return 0.0
    unlike = sum(1 for i, j in edges if flat[i] != flat[j])
    return float(unlike / len(edges))


def configurational_energy(
    state: list | np.ndarray,
    width: int,
    height: int,
    weights: dict[tuple[int, int], float] | None = None,
    edges: list[tuple[int, int]] | None = None,
) -> float:
    """Sum of product_over_edges style terms (AA/BB costly, AB optional reward)."""
    flat = _as_binary(state)
    n = width * height
    flat = flat[:n] if flat.size >= n else np.pad(flat, (0, n - flat.size))
    w = weights or DEFAULT_WEIGHTS
    edges = edges or grid_edges(width, height)
    e = 0.0
    for i, j in edges:
        a, b = int(flat[i]), int(flat[j])
        e += float(w.get((a, b), w.get((b, a), 0.0)))
    return float(e)


def weights_from_yaml_terms(terms: list[dict[str, Any]] | None) -> dict[tuple[int, int], float]:
    """Extract (a,b)→weight from product_over_edges terms."""
    out = dict(DEFAULT_WEIGHTS)
    if not terms:
        return out
    for t in terms:
        if t.get("kind") != "product_over_edges":
            continue
        a = int(t["a_value"])
        b = int(t["b_value"])
        w = float(t["weight"])
        out[(a, b)] = w
        out[(b, a)] = w
    return out


def decode_sample(
    state: list | np.ndarray,
    *,
    width: int = 8,
    height: int = 8,
    weights: dict[tuple[int, int], float] | None = None,
    ising_energy: float | None = None,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Decode one spin/state vector into Distribution Lab metrics + grid."""
    edges = grid_edges(width, height)
    w = weights or DEFAULT_WEIGHTS
    m_s = staggered_order(state, width, height)
    c_b = concentration_b(state, width, height)
    sro = unlike_edge_fraction(state, width, height, edges)
    e_cfg = configurational_energy(state, width, height, w, edges)
    grid = occupancy_grid(state, width, height)
    return {
        "occupancy": grid,
        "width": width,
        "height": height,
        "m_s": m_s,
        "abs_m_s": abs(m_s),
        "concentration_B": c_b,
        "unlike_edge_fraction": sro,
        "energy_config": e_cfg,
        "energy_ising": ising_energy,
        "sample_index": sample_index,
        "labels": {"0": "A", "1": "B"},
        "honesty": "JAX/THRML software sim, not Extropic silicon",
    }


def histogram_ms(values: list[float], n_bins: int = 21) -> dict[str, Any]:
    """Histogram of staggered order over [-1, 1]."""
    n_bins = max(5, min(int(n_bins), 51))
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bins": centers.tolist(),
        "counts": counts.astype(int).tolist(),
        "edges": edges.tolist(),
    }


def rank_decoded(
    decoded: list[dict[str, Any]],
    *,
    by: str = "abs_m_s",
    descending: bool = True,
) -> list[dict[str, Any]]:
    key = by if by in ("abs_m_s", "m_s", "energy_config", "unlike_edge_fraction") else "abs_m_s"
    reverse = descending
    if key == "energy_config":
        reverse = not descending  # lower energy better when ranking "best"
    return sorted(decoded, key=lambda d: float(d.get(key) or 0.0), reverse=reverse)


def batch_decode(
    states: list[list],
    *,
    width: int = 8,
    height: int = 8,
    weights: dict[tuple[int, int], float] | None = None,
    energies: list[float] | None = None,
    rank_by: str = "abs_m_s",
    top_k: int | None = None,
    n_bins: int = 21,
) -> dict[str, Any]:
    """Decode a batch, rank, and build m_s histogram."""
    decoded: list[dict[str, Any]] = []
    for i, st in enumerate(states):
        e_ising = float(energies[i]) if energies and i < len(energies) else None
        decoded.append(
            decode_sample(
                st,
                width=width,
                height=height,
                weights=weights,
                ising_energy=e_ising,
                sample_index=i,
            )
        )
    ranked = rank_decoded(decoded, by=rank_by)
    if top_k is not None:
        ranked = ranked[: max(1, int(top_k))]
    ms_vals = [float(d["m_s"]) for d in decoded]
    return {
        "n": len(decoded),
        "samples": ranked,
        "all_m_s": ms_vals,
        "histogram_m_s": histogram_ms(ms_vals, n_bins=n_bins),
        "best": ranked[0] if ranked else None,
        "rank_by": rank_by,
        "caption": "Distribution Lab · binary ordering alloy · THRML sim · not silicon",
        "honesty": "JAX/THRML software sim, not Extropic silicon",
    }


def load_alloy_weights_from_receipt(receipt_id: str = ALLOY_RECEIPT_ID) -> dict[tuple[int, int], float]:
    """Try to read product_over_edges weights from receipt spec.yaml / program."""
    try:
        from pathlib import Path as _Path

        # Prefer on-disk spec.yaml next to receipts
        roots = [
            _Path(__file__).resolve().parents[2] / "receipts" / receipt_id / "spec.yaml",
            _Path("/workspace/gibbs-observatory/receipts") / receipt_id / "spec.yaml",
        ]
        yaml_text = ""
        for path in roots:
            if path.is_file():
                yaml_text = path.read_text(encoding="utf-8")
                break
        if not yaml_text:
            from .program_service import read_spec_yaml

            spec = read_spec_yaml(receipt_id)
            yaml_text = spec.get("spec_yaml") or ""
        if not yaml_text:
            return dict(DEFAULT_WEIGHTS)
        try:
            import yaml  # type: ignore
        except ImportError:
            # Minimal fallback parser for product_over_edges lines
            out = dict(DEFAULT_WEIGHTS)
            import re
            for m in re.finditer(
                r"a_value:\s*(\d+).*?b_value:\s*(\d+).*?weight:\s*([-0-9.]+)",
                yaml_text,
            ):
                a, b, w = int(m.group(1)), int(m.group(2)), float(m.group(3))
                out[(a, b)] = w
                out[(b, a)] = w
            return out

        doc = yaml.safe_load(yaml_text) or {}
        return weights_from_yaml_terms(doc.get("terms") or [])
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_WEIGHTS)
