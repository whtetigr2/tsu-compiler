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

import networkx as nx

from tsu.failures import CompileError
from tsu.passes.analyse import analyse
from tsu.passes.place import place, _try_grid_embed
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
    placed: bool
    place_error: str | None
    remediations: tuple[str, ...]
    gates: tuple[Gate, ...]
    verdict: str


def _remediations(exc: CompileError) -> tuple[str, ...]:
    """Flatten every `Remediation` the compiler already computed for this
    failure into readable strings, so a pre-flight report that says "degree
    exceeded" does not throw away the actionable half of the answer
    `place()` had already worked out (`Remediation.action` + `.detail`,
    e.g. "relax target: target.degree >= 19")."""
    return tuple(f"{r.action}: {r.detail}"
                 for failure in exc.failures
                 for r in getattr(failure, "remediations", ()))


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

    # Gates are computed FIRST, from the model alone. place() raises
    # CompileError on a degree or budget violation before it returns, so
    # computing them afterwards left two of the four gates unreachable except
    # as an uncaught traceback -- in a tool whose whole purpose is reporting
    # headroom rather than crashing.
    #
    # The budget gate uses analyse()'s mediator ESTIMATE (-1 when the graph is
    # too large for an exact max-cut, treated as 0 here) rather than the count
    # placement actually needed. The two can disagree, which is why the report
    # carries both.
    est_mediators = max(rep.mediators, 0)
    gates = (
        _gate("max_abs_coupling", rep.max_abs_J, target.max_abs_coupling.value,
              "|J| against the hardware's coupling cap"),
        _gate("max_abs_bias", rep.max_abs_b, target.max_abs_bias.value,
              "|b| against the hardware's bias cap"),
        _gate("max_degree", rep.max_degree, target.degree.value,
              "peak node degree against the hardware's connectivity"),
        _gate("node_budget", rep.n_nodes + est_mediators,
              target.node_budget.value,
              "spins required, including estimated mediators, against the die"),
    )

    # ASK which path placement took; do not infer it. Bipartiteness is
    # NECESSARY and not SUFFICIENT for a direct grid embed -- a bipartite graph
    # that is not a subgraph of the lattice falls through to the annealer, and
    # reporting that as `grid_embed` claims a deterministic millisecond placement
    # for a run that was neither.
    g = nx.Graph()
    g.add_nodes_from(range(len(ising.nodes)))
    g.add_edges_from(ising.edges)
    direct = _try_grid_embed(g) is not None

    placement = None
    place_error = None
    remediations: tuple[str, ...] = ()
    t0 = time.time()
    try:
        placement = place(ising, rep, target, restarts=restarts, iters=iters)
    except CompileError as exc:
        place_error = str(exc)
        remediations = _remediations(exc)
    place_seconds = time.time() - t0

    mediators = 0
    if placement is not None and placement.mediation is not None:
        mediators = len(placement.mediated_ising.mediator_nodes)
    embedding = ("failed" if placement is None
                 else "grid_embed" if direct and mediators == 0
                 else "annealed")

    verdict = ("fail" if place_error is not None
               or any(x.status == "fail" for x in gates)
               else "warn" if any(x.status == "warn" for x in gates) else "ok")

    return PreflightReport(
        n_spins=rep.n_nodes, n_couplings=rep.n_edges,
        max_degree=rep.max_degree, bipartite=rep.bipartite,
        embedding=embedding, mediators=mediators,
        place_seconds=place_seconds, placed=placement is not None,
        place_error=place_error, remediations=remediations,
        gates=gates, verdict=verdict)
