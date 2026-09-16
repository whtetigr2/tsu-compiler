"""Graph presets: 2D Ising lattice, 1D chain, degree-capped sparse Ising, receipt-backed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

PresetName = Literal["lattice2d", "chain1d", "sparse", "receipt"]


@dataclass
class GraphSpec:
    """Static graph structure + chromatic blocks (params are separate)."""

    name: str
    n_nodes: int
    edges: list[tuple[int, int]]  # index pairs
    # chromatic 2-coloring: node indices in each independent set
    color0: list[int]
    color1: list[int]
    # layout hints for frontend
    layout: str  # "grid" | "chain" | "force" | "receipt"
    shape: tuple[int, ...]  # e.g. (L, L) for grid
    positions: list[tuple[float, float]]  # normalized [0,1] coords
    # optional clamp region (node indices)
    clamp_indices: list[int]
    degree_cap: int | None = None
    # receipt-backed optional fields
    weights: list[float] | None = None
    biases: list[float] | None = None
    world_idx: list[int] = field(default_factory=list)
    mediator_idx: list[int] = field(default_factory=list)
    node_names: list[str] = field(default_factory=list)
    receipt_id: str | None = None
    beta_fixed: bool = False
    kernel: str | None = None


def _positions_grid(L: int) -> list[tuple[float, float]]:
    if L <= 1:
        return [(0.5, 0.5)]
    return [(c / (L - 1), r / (L - 1)) for r in range(L) for c in range(L)]


def _positions_chain(n: int) -> list[tuple[float, float]]:
    if n <= 1:
        return [(0.5, 0.5)]
    return [(i / (n - 1), 0.5) for i in range(n)]


def _positions_circle(n: int) -> list[tuple[float, float]]:
    if n <= 0:
        return []
    out = []
    for i in range(n):
        ang = 2 * np.pi * i / n - np.pi / 2
        out.append((0.5 + 0.42 * np.cos(ang), 0.5 + 0.42 * np.sin(ang)))
    return out


def build_lattice2d(L: int = 16) -> GraphSpec:
    """2D nearest-neighbor Ising lattice with checkerboard 2-coloring."""
    L = int(np.clip(L, 4, 48))
    n = L * L
    edges: list[tuple[int, int]] = []
    for r in range(L):
        for c in range(L):
            i = r * L + c
            if c + 1 < L:
                edges.append((i, i + 1))
            if r + 1 < L:
                edges.append((i, i + L))
    color0 = [r * L + c for r in range(L) for c in range(L) if (r + c) % 2 == 0]
    color1 = [r * L + c for r in range(L) for c in range(L) if (r + c) % 2 == 1]
    # clamp outer border
    clamp = []
    for r in range(L):
        for c in range(L):
            if r in (0, L - 1) or c in (0, L - 1):
                clamp.append(r * L + c)
    return GraphSpec(
        name="lattice2d",
        n_nodes=n,
        edges=edges,
        color0=color0,
        color1=color1,
        layout="grid",
        shape=(L, L),
        positions=_positions_grid(L),
        clamp_indices=clamp,
        world_idx=list(range(n)),
    )


def build_chain1d(n: int = 32) -> GraphSpec:
    """1D Ising chain, bipartite even/odd coloring."""
    n = int(np.clip(n, 4, 128))
    edges = [(i, i + 1) for i in range(n - 1)]
    color0 = list(range(0, n, 2))
    color1 = list(range(1, n, 2))
    clamp = [0, n - 1]
    return GraphSpec(
        name="chain1d",
        n_nodes=n,
        edges=edges,
        color0=color0,
        color1=color1,
        layout="chain",
        shape=(n,),
        positions=_positions_chain(n),
        clamp_indices=clamp,
        world_idx=list(range(n)),
    )


def build_sparse(n: int = 40, degree_cap: int = 16, seed: int = 0) -> GraphSpec:
    """Sparse random Ising graph, degree-capped, made bipartite for 2-color Gibbs."""
    n = int(np.clip(n, 8, 80))
    degree_cap = int(np.clip(degree_cap, 2, 16))
    rng = np.random.default_rng(seed)
    # bipartition ~half / half so chromatic 2-coloring is exact
    n0 = n // 2
    color0 = list(range(n0))
    color1 = list(range(n0, n))
    deg = np.zeros(n, dtype=np.int32)
    edges: list[tuple[int, int]] = []
    edge_set: set[tuple[int, int]] = set()
    # target ~ min(degree_cap, n1) average degree on smaller side
    target_edges = min(n * degree_cap // 2, n0 * len(color1))
    attempts = 0
    while len(edges) < target_edges and attempts < target_edges * 40:
        attempts += 1
        i = int(rng.choice(color0))
        j = int(rng.choice(color1))
        if deg[i] >= degree_cap or deg[j] >= degree_cap:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
        edges.append((a, b))
        deg[i] += 1
        deg[j] += 1
    # clamp a small "boundary", highest-degree nodes on color0 side
    order = sorted(color0, key=lambda u: -int(deg[u]))
    clamp = order[: max(2, n // 10)]
    return GraphSpec(
        name="sparse",
        n_nodes=n,
        edges=edges,
        color0=color0,
        color1=color1,
        layout="force",
        shape=(n,),
        positions=_positions_circle(n),
        clamp_indices=clamp,
        degree_cap=degree_cap,
        world_idx=list(range(n)),
    )


def build_from_receipt_arrays(data: dict) -> GraphSpec:
    """Build GraphSpec from receipt_to_graph_arrays() output."""
    n = int(data["n_nodes"])
    positions_raw = data.get("positions") or []
    positions: list[tuple[float, float]]
    if len(positions_raw) == n:
        positions = [(float(p[0]), float(p[1])) for p in positions_raw]
    else:
        positions = _positions_circle(n)

    color0 = list(data.get("color0") or [])
    color1 = list(data.get("color1") or [])
    if not color0 and not color1:
        # last-resort bipartite split for THRML blocks (even/odd), only if
        # receipt omitted blocks; prefer real chromatic blocks when present.
        color0 = list(range(0, n, 2))
        color1 = list(range(1, n, 2))

    world_idx = list(data.get("world_idx") or list(range(n)))
    mediator_idx = list(data.get("mediator_idx") or [])

    return GraphSpec(
        name="receipt",
        n_nodes=n,
        edges=list(data["edges"]),
        color0=color0,
        color1=color1,
        layout="receipt",
        shape=(n,),
        positions=positions,
        clamp_indices=[],
        weights=list(data["weights"]),
        biases=list(data["biases"]),
        world_idx=world_idx,
        mediator_idx=mediator_idx,
        node_names=list(data.get("node_names") or []),
        receipt_id=data.get("receipt_id"),
        beta_fixed=bool(data.get("beta_fixed", False)),
        kernel=data.get("kernel"),
    )


def build_preset(
    name: PresetName | str,
    size: int = 16,
    degree_cap: int = 16,
    seed: int = 0,
) -> GraphSpec:
    if name == "lattice2d":
        return build_lattice2d(size)
    if name == "chain1d":
        return build_chain1d(size)
    if name == "sparse":
        return build_sparse(size, degree_cap=degree_cap, seed=seed)
    raise ValueError(f"Unknown preset: {name}")
