"""Physical placement. A real pass whose verdict is binding.

EXP-WL1 rated a 3D cubic lattice GREEN with zero mediators on parity; EXP-WL2 then
found geometry, not parity, was binding. Parity is NECESSARY, never SUFFICIENT, so
this pass runs an actual embedding and names which failure class it hit.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from ..failures import CompileError, Offender, PlacementFailure, Remediation
from ..target import TargetProfile
from .analyse import GraphReport
from .lower import IsingModel


@dataclass(frozen=True)
class Placement:
    coords: dict           # node index -> (x, y); empty for targets with no lattice
    realized: tuple        # edges placed on a legal offset
    unrealized: tuple      # edges that could not be


def _anneal(G, offsets, side, iters, seed):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    coords = {n: (rng.randrange(side), rng.randrange(side)) for n in nodes}
    offs = set(offsets)

    def bad(c):
        return sum(1 for u, v in G.edges()
                   if (c[u][0] - c[v][0], c[u][1] - c[v][1]) not in offs)

    cur = bad(coords)
    for _ in range(iters):
        if cur == 0:
            break
        n = rng.choice(nodes)
        old = coords[n]
        coords[n] = (rng.randrange(side), rng.randrange(side))
        new = bad(coords)
        if new <= cur:
            cur = new
        else:
            coords[n] = old
    return coords, cur


def place(ising: IsingModel, report: GraphReport, target: TargetProfile) -> Placement:
    n = len(ising.nodes)

    if report.max_degree > target.degree.value:
        offenders = tuple(
            Offender("node", f"{ising.nodes[i]} (degree {d})")
            for i, d in nx.Graph(list(ising.edges)).degree()
            if d > target.degree.value)
        raise CompileError(
            "placement failed: degree exceeded",
            [PlacementFailure(
                failure_class="degree_exceeded",
                offending=offenders or (Offender("node", "unknown"),),
                measured=report.max_degree, limit=target.degree.value,
                assumed=target.is_assumed("degree"),
                remediations=(
                    Remediation("change encoding",
                                "a sparser encoding lowers per-node degree",
                                {"note": "estimate"}),
                    Remediation("relax target",
                                f"target.degree >= {report.max_degree}"),
                ))])

    if not target.offsets.value:            # no lattice: IDEAL
        return Placement({}, tuple(ising.edges), ())

    if n > target.node_budget.value:
        raise CompileError(
            "placement failed: node budget exceeded",
            [PlacementFailure("budget_exceeded",
                              (Offender("node", f"{n} nodes"),),
                              n, target.node_budget.value,
                              target.is_assumed("node_budget"), ())])

    if target.bipartite.value and not report.bipartite:
        G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(ising.edges)
        cycle = nx.find_cycle(G)
        raise CompileError(
            "placement failed: parity conflict",
            [PlacementFailure(
                failure_class="parity_conflict",
                offending=tuple(Offender("edge", f"{ising.nodes[u]}-{ising.nodes[v]}")
                                for u, v, *_ in cycle),
                measured="odd cycle present", limit="bipartite",
                assumed=target.is_assumed("bipartite"),
                remediations=(
                    Remediation("route through mediator",
                                "hidden-spin mediation is exact and adds one spin "
                                "per frustrated coupling",
                                {"extra_spins": report.mediators, "note": "estimate"}),
                ))])

    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(ising.edges)
    side = max(4, int(n ** 0.5) + 3)
    best, best_bad = None, None
    for seed in range(6):
        coords, nbad = _anneal(G, target.offsets.value, side, 40_000, seed)
        if best_bad is None or nbad < best_bad:
            best, best_bad = coords, nbad
        if nbad == 0:
            break

    offs = set(target.offsets.value)
    realized, unrealized = [], []
    for u, v in ising.edges:
        d = (best[u][0] - best[v][0], best[u][1] - best[v][1])
        (realized if d in offs else unrealized).append((u, v))

    if unrealized:
        raise CompileError(
            "placement failed: geometry unreachable",
            [PlacementFailure(
                failure_class="geometry_unreachable",
                offending=tuple(Offender("edge", f"{ising.nodes[u]}-{ising.nodes[v]}")
                                for u, v in unrealized),
                measured=len(unrealized), limit=0,
                assumed=target.is_assumed("offsets"),
                remediations=(
                    Remediation("increase placement effort",
                                "more restarts or a larger patch may find an embedding"),
                    Remediation("change encoding",
                                "a lower-degree encoding is easier to embed",
                                {"note": "estimate"}),
                ))])

    return Placement(best, tuple(realized), ())
