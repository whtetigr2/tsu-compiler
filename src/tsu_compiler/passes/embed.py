"""Minor embedding: a logical spin becomes a CHAIN of physical p-bits.

`place.py` does direct embedding, one logical spin to one physical cell. R33
measures why that only works when the interaction graph is already a subgraph of
the hardware lattice: in a sparse lattice the cells legal from d already-placed
neighbours collapse fast --

    neighbours placed   cells legal from all of them, anywhere
                    1                                      16
                    2                                       4
                    3                                       2
                    4                                       1

-- so past four neighbours a node has one candidate, a collision ends the
embedding, and restarts meet the same wall somewhere else. Grid-shaped programs
place; the published sequence-design models above 31 spins do not.

Minor embedding dissolves that. A logical spin becomes a connected SET of cells
held together by strong ferromagnetic couplings so they act as one variable, and
a neighbour attaches to whichever cell of the chain is convenient. The logical
graph is then a *minor* of the hardware graph rather than a subgraph, which is a
far weaker requirement -- and notably one that a bipartite host can satisfy for
a non-bipartite logical graph, since contracting a 6-cycle yields a triangle.

WHY THIS CALLS A LIBRARY. Finding a good minor embedding is a hard search, and
`minorminer` (D-Wave, Apache-2.0, the same licence as this project) is the
mature implementation the field actually uses. A hand-rolled version was written
first and did not converge: on a 31-spin model it plateaued at 51 overlapping
cells after 30 rounds of tear-and-route with escalating penalties, while
`minorminer` embedded the same model in under a second. Shipping the worse one to
avoid a dependency would have been vanity, not engineering.

WHAT IS NOT DELEGATED. `minorminer` knows nothing about Z1. This module supplies
the host graph built from `target.offsets`, and -- more importantly -- it
verifies the result independently in `validate_embedding`: every chain connected
in the host, no cell used twice, and every logical edge realized by some pair of
adjacent cells. A library's success flag is not evidence, and the check here does
not ask `minorminer` whether it succeeded, it asks the embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ChainEmbedding",
    "EmbeddingUnavailable",
    "build_host",
    "find_chain_embedding",
    "validate_embedding",
]


class EmbeddingUnavailable(RuntimeError):
    """`minorminer` is not installed, so chain embedding cannot run."""


@dataclass(frozen=True)
class ChainEmbedding:
    """A logical-spin to physical-cell mapping, with what it cost."""

    chains: dict[int, tuple[tuple[int, int], ...]]
    #: Side of the square host patch the chains live in.
    host_side: int
    n_logical: int
    n_physical: int
    max_chain: int
    mean_chain: float
    seconds: float

    @property
    def overhead(self) -> float:
        """Physical cells per logical spin. 1.0 would be direct placement."""
        return self.n_physical / self.n_logical if self.n_logical else 0.0


def build_host(side: int, offsets: Sequence[tuple[int, int]]):
    """The hardware graph as a square patch of the offset lattice.

    A patch rather than the whole die: `minorminer`'s cost scales with the host,
    and a model needing 6,784 cells does not need 269,568 of them presented to
    the search. The node budget is a separate gate and is checked there.
    """
    import networkx as nx

    H = nx.Graph()
    for x in range(side):
        for y in range(side):
            H.add_node((x, y))
            for dx, dy in offsets:
                a, b = x + dx, y + dy
                if 0 <= a < side and 0 <= b < side:
                    H.add_edge((x, y), (a, b))
    return H


def validate_embedding(edges: Iterable[tuple[int, int]],
                       host,
                       chains: Mapping[int, Iterable[tuple[int, int]]],
                       nodes: Iterable[int] | None = None) -> tuple[str, ...]:
    """Check the embedding against its definition. Returns the failures found.

    Three conditions, and all three are load-bearing:

    **Every chain is connected in the host.** A disconnected chain is not one
    variable; the pieces are free to disagree and the logical spin has no
    defined value.

    **No cell belongs to two chains.** A shared cell means two logical spins are
    the same physical p-bit, which silently identifies variables the model
    intends to be distinct.

    **Every logical edge is realized.** Some cell of one chain must be adjacent
    to some cell of the other, or the coupling the model asked for does not
    exist on the die and the energy being sampled is not the energy compiled.

    A degree-0 logical spin has no edges to realize and is satisfied by any free
    cell; it is still required to HAVE one, because a spin with no location is
    not placed.
    """
    import networkx as nx

    failures: list[str] = []
    edges = list(edges)
    if nodes is None:
        nodes = sorted({v for e in edges for v in e})

    seen: dict[tuple[int, int], int] = {}
    for v in nodes:
        chain = tuple(chains.get(v, ()))
        if not chain:
            failures.append(f"logical spin {v} has no chain")
            continue
        sub = host.subgraph(chain)
        if sub.number_of_nodes() != len(set(chain)):
            failures.append(f"chain {v} uses a cell outside the host")
            continue
        if not nx.is_connected(sub):
            failures.append(
                f"chain {v} is disconnected ({len(chain)} cells, "
                f"{nx.number_connected_components(sub)} pieces)")
        for cell in chain:
            other = seen.get(cell)
            if other is not None:
                failures.append(
                    f"cell {cell} is used by both spin {other} and spin {v}")
            seen[cell] = v

    for u, v in edges:
        cu, cv = tuple(chains.get(u, ())), tuple(chains.get(v, ()))
        if not cu or not cv:
            continue          # already reported as a missing chain
        if not any(host.has_edge(a, b) for a in cu for b in cv):
            failures.append(f"logical edge {u}-{v} is not realized by any "
                            f"adjacent pair of cells")
    return tuple(failures)


def find_chain_embedding(ising: Any, target: Any, *, host_side: int | None = None,
                         seed: int = 1, timeout: float = 600.0) -> ChainEmbedding:
    """Embed `ising`'s interaction graph into `target`'s offset lattice.

    Raises `EmbeddingUnavailable` when `minorminer` is not installed and
    `RuntimeError` when no embedding is found or the one found does not
    validate. It never returns an embedding this module has not checked itself.
    """
    import time

    try:
        import minorminer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbeddingUnavailable(
            "chain embedding needs `minorminer` (Apache-2.0, "
            "`pip install minorminer`). Without it only direct placement is "
            "available, which R33 shows fails on graphs that are not already "
            "lattice-shaped.") from exc

    import networkx as nx

    offsets = list(target.offsets.value)
    if not offsets:
        raise RuntimeError(
            f"target {getattr(target, 'name', '?')} declares no lattice "
            f"offsets, so there is no host graph to embed into")

    n_logical = len(ising.nodes)
    edges = [(int(u), int(v)) for u, v in ising.edges]

    # Room to route in. Chain overhead measured on the published sequence-design
    # models is about 2.2x, and a search boxed in tightly fails for want of space
    # rather than for any reason about the model, so the patch is generous. It is
    # checked against the node budget below.
    if host_side is None:
        host_side = max(8, int((max(n_logical, 1) * 12) ** 0.5) + 4)
    host = build_host(host_side, offsets)

    t0 = time.time()
    found = minorminer.find_embedding(edges, list(host.edges()),
                                      random_seed=seed, timeout=timeout)
    seconds = time.time() - t0
    if not found:
        raise RuntimeError(
            f"no chain embedding found for {n_logical} spins in a "
            f"{host_side}x{host_side} patch after {seconds:.0f}s")

    chains = {int(v): tuple(sorted(tuple(c) for c in cells))
              for v, cells in found.items()}

    # A spin with no couplings never appears in an edge list, so the search
    # never sees it. It still needs somewhere to be.
    used = {c for ch in chains.values() for c in ch}
    for v in range(n_logical):
        if chains.get(v):
            continue
        free = next((c for c in host.nodes() if c not in used), None)
        if free is None:
            raise RuntimeError(
                f"no free cell for isolated spin {v}; host patch is full")
        chains[v] = (free,)
        used.add(free)

    failures = validate_embedding(edges, host, chains, nodes=range(n_logical))
    if failures:
        raise RuntimeError(
            f"minorminer returned an embedding that does not validate: "
            f"{len(failures)} failures, first is {failures[0]}")

    lens = [len(c) for c in chains.values()]
    n_physical = sum(lens)
    budget = getattr(target, "node_budget", None)
    if budget is not None and n_physical > budget.value:
        raise RuntimeError(
            f"chain embedding needs {n_physical:,} p-bits and the target "
            f"budget is {budget.value:,}")

    return ChainEmbedding(
        chains=chains, host_side=host_side, n_logical=n_logical,
        n_physical=n_physical, max_chain=max(lens),
        mean_chain=n_physical / len(lens), seconds=seconds,
    )
