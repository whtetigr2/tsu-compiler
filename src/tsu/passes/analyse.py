"""IsingModel -> GraphReport. Measures; claims nothing.

`mediators` is the connectivity residual |E| - MaxCut(G), the floor on mediator
count for a bipartite substrate. It is a NECESSARY condition and never a sufficient
one -- EXP-WL2 found geometry, not parity, binding for a 3D lattice. `place` decides
realizability (spec section 6.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .lower import IsingModel

MAXCUT_EXACT_LIMIT = 20


@dataclass(frozen=True)
class GraphReport:
    n_nodes: int
    n_edges: int
    max_degree: int
    bipartite: bool
    mediators: int          # -1 when the graph is too large for an exact MaxCut
    colouring: dict
    colour_blocks: int
    max_abs_J: float
    max_abs_b: float


def _max_cut(G: nx.Graph) -> int:
    nodes = list(G.nodes())
    best = 0
    for mask in range(1 << (len(nodes) - 1)):
        side = {nodes[i]: (mask >> i) & 1 for i in range(len(nodes) - 1)}
        side[nodes[-1]] = 0
        best = max(best, sum(1 for u, v in G.edges() if side[u] != side[v]))
    return best


def analyse(ising: IsingModel) -> GraphReport:
    G = nx.Graph()
    G.add_nodes_from(range(len(ising.nodes)))
    G.add_edges_from(ising.edges)

    bip = nx.is_bipartite(G)
    if bip:
        mediators = 0
        colouring = dict(nx.bipartite.color(G)) if G.number_of_edges() else \
            {n: 0 for n in G.nodes()}
    else:
        mediators = (G.number_of_edges() - _max_cut(G)
                     if G.number_of_nodes() <= MAXCUT_EXACT_LIMIT else -1)
        colouring = nx.coloring.greedy_color(G, strategy="DSATUR")

    return GraphReport(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        max_degree=max((d for _, d in G.degree()), default=0),
        bipartite=bip,
        mediators=mediators,
        colouring=colouring,
        colour_blocks=(max(colouring.values()) + 1 if colouring else 1),
        max_abs_J=float(np.abs(ising.weights).max()) if len(ising.weights) else 0.0,
        max_abs_b=float(np.abs(ising.biases).max()) if len(ising.biases) else 0.0,
    )
