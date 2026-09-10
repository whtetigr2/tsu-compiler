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
from tsu.gates import GateCheck, gate_checks
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
    assumed: bool = False
    # True iff this gate would have FAILED but was overridden by
    # --allow-assumed (F2, branch review) -- distinct from `status`, which
    # is separately set to "downgraded" in that case so a reader sees it
    # in the same place as ok/warn/fail rather than needing to cross-
    # reference two fields.
    downgraded: bool = False


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


def _status(value: float, limit: float, downgraded: bool) -> str:
    if downgraded:
        return "downgraded"
    if limit <= 0:
        # A zero-limit gate (e.g. colouring: any violation at all fails)
        # has no headroom to "warn" about -- `value >= limit*WARN_FRACTION`
        # would degenerate to `value >= 0`, always true, warning even at
        # value=0 (no violation). Binary ok/fail is the only sound reading.
        return "fail" if value > limit else "ok"
    if value > limit:
        return "fail"
    if value >= limit * WARN_FRACTION:
        return "warn"
    return "ok"


def _gate(name: str, value: float, limit: float, note: str, *,
          assumed: bool = False, downgraded: bool = False) -> Gate:
    return Gate(name=name, value=float(value), limit=float(limit),
                status=_status(value, limit, downgraded), note=note,
                assumed=assumed, downgraded=downgraded)


# F2 (branch review): display name + base note text for the three gates
# delegated to tsu.gates.gate_checks() below. The base note for
# max_abs_bias deliberately never says "hardware" -- target.py marks that
# field source="assumed" ("project working value; NOT a sourced Extropic
# figure"), and the shipped note ("|b| against the hardware's bias cap")
# rendered IDENTICALLY to max_abs_coupling's (an Extropic-documented
# fact), exactly the conflation target.py's own docstring records fixing
# once already ("Never conflate these two fields again just because their
# VALUES happen to agree"). Every note below gets an explicit ASSUMED
# marker appended (see `_delegated_gate`) whenever `target.is_assumed(...)`
# says so, sourced from `gate_checks()` itself -- never hand-guessed here.
_DELEGATED_GATES = {
    "degree": ("max_degree", "peak node degree against the hardware's connectivity"),
    "coupling_cap": ("max_abs_coupling", "|J| against the hardware's coupling cap"),
    "field_cap": ("max_abs_bias", "|b| against the bias cap"),
}


def _delegated_gate(gc: GateCheck) -> Gate:
    display_name, base_note = _DELEGATED_GATES[gc.gate]
    note = (f"{base_note} (ASSUMED -- not a sourced hardware figure)"
           if gc.assumed else base_note)
    return _gate(display_name, gc.measured, gc.limit, note,
                assumed=gc.assumed, downgraded=gc.downgraded)


def preflight(ising: IsingModel, target: TargetProfile = PROFILES["z1"],
              *, allow_assumed: bool = False,
              restarts: int = 6, iters: int = 40_000) -> PreflightReport:
    rep = analyse(ising)

    # F2 (branch review): degree/coupling_cap/field_cap/colouring are
    # DELEGATED to tsu.gates.gate_checks() -- the SAME evaluation the real
    # compile pipeline (passes/search.py) uses -- rather than an
    # independent reimplementation. Before this, preflight's own gates (a)
    # never carried target.py's `assumed` provenance at all, so a sourced
    # Extropic fact and a genuine project guess rendered identically; (b)
    # had no colouring gate; (c) had no --allow-assumed downgrade. All
    # three are closed by consuming gate_checks()'s output directly.
    checks = gate_checks(ising, rep, target, allow_assumed)
    by_gate = {gc.gate: gc for gc in checks}

    gates = tuple(_delegated_gate(by_gate[g]) for g in
                 ("degree", "coupling_cap", "field_cap"))

    colouring = by_gate["colouring"]
    gates += (_gate(
        "colouring", 0.0 if colouring.passed else 1.0, 0.0,
        "adjacent nodes must never share a colour block (a structural "
        "invariant of analyse()'s own colouring, checked here for "
        "agreement with tsu.gates)",
        assumed=colouring.assumed, downgraded=colouring.downgraded),)

    # node_budget is DELIBERATELY NOT delegated: gate_checks() measures it
    # as report.n_nodes alone -- correct for the real compile pipeline,
    # which evaluates gates BEFORE placement runs (mediators are not known
    # yet there; a later mediator overage is caught downstream by
    # placement itself, which raises CompileError on it, not by
    # re-checking this gate). preflight runs standalone and can afford to
    # fold in analyse()'s pre-placement mediator ESTIMATE up front instead,
    # so a user sees the risk before waiting on the slower placement
    # search -- a deliberately DIFFERENT, more conservative measurement
    # for a different purpose, not an accidental disagreement. The note
    # below says so explicitly (F9, branch review: this project's own
    # comparison found node_budget's value and the `mediators` line below
    # contradicting each other, unlabelled, on a 25-node odd cycle).
    est_mediators = max(rep.mediators, 0)
    node_budget_assumed = target.is_assumed("node_budget")
    node_budget_note = (
        "spins required, INCLUDING analyse()'s pre-placement mediator "
        "ESTIMATE (0 when the graph exceeds the exact max-cut limit), "
        "against the die's node budget -- see the `mediators` line above "
        "for placement's ACTUAL count, which may differ")
    if node_budget_assumed:
        node_budget_note += " (ASSUMED -- not a sourced hardware figure)"
    gates += (_gate("node_budget", rep.n_nodes + est_mediators,
                    target.node_budget.value, node_budget_note,
                    assumed=node_budget_assumed),)

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

    # A "downgraded" gate (F2: --allow-assumed overrode a would-have-failed
    # ASSUMED gate) must not silently verdict "ok" -- that would hide from
    # a reader that a limit was overridden, not cleared -- so it counts
    # toward "warn" alongside genuine warn-fraction gates, same as before.
    verdict = ("fail" if place_error is not None
               or any(x.status == "fail" for x in gates)
               else "warn" if any(x.status in ("warn", "downgraded")
                                  for x in gates)
               else "ok")

    return PreflightReport(
        n_spins=rep.n_nodes, n_couplings=rep.n_edges,
        max_degree=rep.max_degree, bipartite=rep.bipartite,
        embedding=embedding, mediators=mediators,
        place_seconds=place_seconds, placed=placement is not None,
        place_error=place_error, remediations=remediations,
        gates=gates, verdict=verdict)
