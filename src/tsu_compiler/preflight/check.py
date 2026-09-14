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

import math
import time
from dataclasses import dataclass

import numpy as np
import networkx as nx

from tsu_compiler.failures import CompileError
from tsu_compiler.gates import GateCheck, gate_checks
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.place import place, _try_grid_embed
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.route import mediator_coupling
from tsu_compiler.target import PROFILES, TargetProfile

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
    # None for any model that fits inside one core -- the common case, and
    # the reason this carries a default: absence is normal, not an omission.
    fabric_note: str | None = None
    # Distinct |J| values the model needs, against the die's
    # programmable-parameter budget. None when the target has no
    # finite parameter count (the ideal control).
    distinct_couplings: int | None = None
    coupling_note: str | None = None



def _core_nodes(target) -> int | None:
    """One core's worth of pbits: the largest model whose placement cannot
    depend on the connected-fabric assumption. Z1 is documented as 269,568
    pbits in 8 cores."""
    budget = target.node_budget.value
    if budget in (None, float("inf")):
        return None
    return int(budget) // 8


def _fabric_note(n_spins: int, target) -> str | None:
    """Say that a placement verdict rests on an ASSUMED connected fabric --
    but ONLY for a model large enough for it to matter.

    Below one core's worth of pbits a model fits inside a single core however
    the cores are wired, so the assumption cannot affect its placement and
    saying so would be noise. Warning on every small model is how a disclosure
    becomes decoration: a reader who sees it every time stops reading it, and
    then it is not there when it counts.
    """
    if not target.is_assumed("connected_fabric"):
        return None
    core = _core_nodes(target)
    if core is None or n_spins <= core:
        return None
    return (f"this model needs {n_spins:,} spins, more than one core's "
            f"{core:,}, so its placement rests on the ASSUMED connected fabric "
            f"(target.connected_fabric): the published material never states "
            f"whether {target.name}'s cores form one connected lattice. See "
            f"that field's citation for the derivation and what it does not "
            f"licence.")


def _coupling_note(ising, target) -> tuple[int | None, str | None]:
    """How many distinct coupling values does this model need, and can the die
    hold that many?

    Z1 carries ~2,135,904 coupling EDGES against ~215,904 coupling PARAMETERS
    (both Extropic's own figures -- see target.coupling_parameters), so roughly
    ten edges share each programmable parameter. This compiler assumes one
    independent J per edge, which is an IDEALISATION on the physical die.

    The measure that matters is DISTINCT coupling values, not edge count: a
    model with 9,557 edges but only 32 distinct |J| needs 32 parameters, and
    sharing is irrelevant to it. Reported always, because it is cheap and it is
    the number that decides whether the count binds.

    What this CANNOT check is the MAPPING -- whether those distinct values can
    be ASSIGNED under Z1's real sharing structure, which is unpublished. A model
    well inside the count can still be unprogrammable if sharing is structured
    (say, per offset class) and the model needs two different couplings on edges
    that share a parameter. That is stated rather than silently passed, because
    a report that said only "215,904: OK" would imply a check nobody ran.
    """
    budget = target.coupling_parameters.value
    if budget in (None, float("inf")):
        return None, None
    import numpy as np
    distinct = int(np.unique(np.round(np.abs(np.asarray(ising.weights,
                                                        dtype=float)), 9)).size)
    note = (f"this model needs {distinct:,} distinct coupling values against "
            f"{int(budget):,} programmable parameters on the die, so the COUNT "
            f"is not binding. The MAPPING is unmodelled: ~{2135904/budget:.1f} "
            f"edges share each parameter on Z1 and the sharing structure is "
            f"unpublished, so whether these values can be ASSIGNED is not "
            f"checked here (target.per_edge_independent_J is an assumption).")
    if distinct > budget:
        note = (f"this model needs {distinct:,} distinct coupling values but "
                f"the die exposes only {int(budget):,} programmable parameters "
                f"-- it cannot be programmed on Z1 regardless of placement.")
    return distinct, note


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
# delegated to tsu_compiler.gates.gate_checks() below. The base note for
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
    # DELEGATED to tsu_compiler.gates.gate_checks() -- the SAME evaluation the real
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
        "agreement with tsu_compiler.gates)",
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

    # A-1 task 3 (external review, 2026-09-09): a non-bipartite model's
    # POST-mediation coupling is predictable in closed form -- A =
    # arccosh(exp(2*beta*|J|))/(2*beta) (route.py's own derivation) -- from
    # this model's own |J|max and beta ALONE, with no placement (and no
    # annealer run) required at all. Computed here, BEFORE the (possibly
    # slow) `place()` call below, precisely because preflight's whole
    # reason to exist is answering "will it fit" before paying that cost
    # (this module's own docstring). Mediation ALWAYS raises the coupling
    # it replaces (A > |J| for every J != 0), which is exactly why a model
    # can clear `max_abs_coupling`/`coupling_cap` pre-mediation and still
    # not fit once `place()` mediates it -- the gap A-1's compile-time fix
    # (tsu_compiler.passes.search._try) closes at compile time; this closes the same
    # gap here, at preflight time, before a user ever reaches a compile.
    #
    # Omitted entirely (not merely marked "ok") for an already-bipartite
    # model: `insert_mediators` never runs on one at all (route.py's own
    # early-out), so there is nothing to predict -- reporting a gate here
    # would imply a prediction that was never actually computed for
    # anything. Also omitted (rather than fabricating a comparison) if this
    # model's own beta or |J|max is non-finite -- `mediator_coupling` gives
    # no guarantee of a sane result off that domain, and this project never
    # reports a number it does not trust (see this module's own docstring).
    #
    # DELEGATES to `tsu_compiler.passes.route.mediator_coupling` -- the SAME
    # function `insert_mediators` itself calls -- rather than re-deriving
    # the formula a second time (this project has been bitten before by
    # exactly that kind of drift; see route.py's own docstring).
    if not rep.bipartite:
        peak_J = float(np.abs(ising.weights).max()) if len(ising.weights) else 0.0
        beta = float(ising.beta)
        if math.isfinite(peak_J) and math.isfinite(beta):
            predicted_A = mediator_coupling(peak_J, beta)
            cap_J = target.max_abs_coupling.value
            cap_J_assumed = target.is_assumed("max_abs_coupling")
            note = (
                "the |J| a mediator coupling would need at this model's own "
                "beta and |J|max (A = arccosh(exp(2*beta*|J|))/(2*beta), "
                "computed WITHOUT placing anything) -- mediation ALWAYS "
                "RAISES every coupling it touches (A > |J| for every J != 0), "
                "which is why a model can pass max_abs_coupling (the "
                "coupling_cap gate above) before placement and still not fit "
                "once `place()` actually mediates it, gated against the SAME "
                "hardware cap coupling_cap uses")
            if cap_J_assumed:
                note += " (ASSUMED -- not a sourced hardware figure)"
            gates += (_gate("mediated_coupling_cap", predicted_A, cap_J, note,
                            assumed=cap_J_assumed),)

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

    n_distinct, coupling_note = _coupling_note(ising, target)
    return PreflightReport(
        n_spins=rep.n_nodes, n_couplings=rep.n_edges,
        max_degree=rep.max_degree, bipartite=rep.bipartite,
        embedding=embedding, mediators=mediators,
        place_seconds=place_seconds, placed=placement is not None,
        place_error=place_error, remediations=remediations,
        gates=gates, verdict=verdict,
        fabric_note=_fabric_note(rep.n_nodes + mediators, target),
        distinct_couplings=n_distinct, coupling_note=coupling_note)
