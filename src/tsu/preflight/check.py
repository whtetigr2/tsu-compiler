"""Will this model fit on the hardware?

Compiler passes only. No sampling, so this is fast enough to run on every edit.

THE BIPARTITE LINE IS THE ONE THAT MATTERS. Z1's lattice is a chessboard: every
coupled pair sits at an offset with dx+dy odd. A bipartite graph can therefore be
a DIRECT SUBGRAPH and place deterministically in milliseconds. A non-bipartite
one cannot, and enters an annealing search that on this project took 6-32 minutes
TO FAIL at sizes above 8x8. The report states which case the user is in, because
that is the difference between an afternoon and a coffee.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from tsu.passes.analyse import analyse
from tsu.passes.place import place
from tsu.passes.lower import IsingModel
from tsu.target import PROFILES, TargetProfile

WARN_FRACTION = 0.8
"""Fraction of a limit above which a gate warns. A model here still compiles;
it has simply spent most of its headroom, which a user scaling the problem later
needs to know before they discover it."""


@dataclass(frozen=True)
class Gate:
    name: str
    value: float
    limit: float
    status: str
    note: str


@dataclass(frozen=True)
class PreflightReport:
    n_spins: int
    n_couplings: int
    max_degree: int
    bipartite: bool
    embedding: str
    mediators: int
    place_seconds: float
    gates: tuple[Gate, ...]
    verdict: str


def _gate(name: str, value: float, limit: float, note: str) -> Gate:
    if value > limit:
        status = "fail"
    elif value >= limit * WARN_FRACTION:
        status = "warn"
    else:
        status = "ok"
    return Gate(name=name, value=float(value), limit=float(limit),
                status=status, note=note)


def preflight(ising: IsingModel, target: TargetProfile = PROFILES["z1"],
              *, restarts: int = 6, iters: int = 40_000) -> PreflightReport:
    rep = analyse(ising)

    t0 = time.time()
    placement = place(ising, rep, target, restarts=restarts, iters=iters)
    place_seconds = time.time() - t0

    mediated = placement.mediation is not None
    embedding = "annealed" if mediated or not rep.bipartite else "grid_embed"
    mediators = 0 if placement.mediation is None else len(
        placement.mediated_ising.mediator_nodes)

    gates = (
        _gate("max_abs_coupling", rep.max_abs_J, target.max_abs_coupling.value,
              "|J| against the hardware's coupling cap"),
        _gate("max_abs_bias", rep.max_abs_b, target.max_abs_bias.value,
              "|b| against the hardware's bias cap"),
        _gate("max_degree", rep.max_degree, target.degree.value,
              "peak node degree against the hardware's connectivity"),
        _gate("node_budget", rep.n_nodes + mediators, target.node_budget.value,
              "spins required, including any mediators, against the die"),
    )
    verdict = ("fail" if any(g.status == "fail" for g in gates)
               else "warn" if any(g.status == "warn" for g in gates) else "ok")

    return PreflightReport(
        n_spins=rep.n_nodes, n_couplings=rep.n_edges,
        max_degree=rep.max_degree, bipartite=rep.bipartite,
        embedding=embedding, mediators=mediators,
        place_seconds=place_seconds, gates=gates, verdict=verdict)
