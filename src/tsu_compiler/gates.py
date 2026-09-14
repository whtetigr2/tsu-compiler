"""Hard gates. A gate failing aborts the compile; a poor metric never does.

Gates whose threshold comes from a Sourced field marked "assumed" are tagged and
overridable via --allow-assumed. Gates backed by a real fact id are NOT overridable
(spec section 7.1).

The spec's hardware gate is "any |J| OR |b| exceeds its own cap" (section 7.1).
C2 of the final review found `check_gates` gating |J| only, so a model whose bias
alone blew past the cap compiled clean with no gate failure at all.
`_magnitude_gate` below is shared by both the `coupling_cap` and `field_cap`
checks so the two can never drift apart again the way they did the first time.

P-3/F-A5 + I-9a/F-R7: `coupling_cap` (|J|) and `field_cap` (|b|) read TWO
separate `TargetProfile` fields -- `max_abs_coupling` and `max_abs_bias` -- not
one shared cap. They happen to carry the identical numeric value (6.0) but
different provenance: |J|'s cap is an Extropic-documented Z1 hardware fact
(not overridable via --allow-assumed, same as `degree`); |b|'s cap remains a
genuine, unsourced project assumption (overridable, same as before). Before
this split there was exactly one `Sourced` field feeding both checks, which is
why both were disclaimed identically everywhere in the app even after |J|'s
provenance was independently confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .failures import GateFailure, Remediation
from .passes.analyse import GraphReport
from .passes.lower import IsingModel
from .target import TargetProfile


@dataclass(frozen=True)
class GateCheck:
    """Every gate this pass evaluates, whether it passed or failed -- unlike
    GateFailure (which exists only for failures), this is what lets a receipt
    show "every gate, passed/failed, with the measured value and threshold"
    (spec section 10), not just the ones that aborted the compile."""
    gate: str
    passed: bool
    measured: Any
    limit: Any
    assumed: bool
    downgraded: bool = False   # True iff this check would have failed but was
                                # overridden by --allow-assumed


def _magnitude_gate(gate: str, label: str, peak: float, cap: float,
                    assumed: bool, source: str, target_field: str,
                    allow_assumed: bool,
                    encoding_hint: str) -> tuple[GateCheck, GateFailure | None]:
    """One `|value| <= cap` gate, shared by the |J| and |b| checks (C2). Returns
    the GateCheck (always) and a GateFailure (only when the gate fails and is not
    downgraded). `target_field` is the NAME of the TargetProfile.Sourced field
    this particular call is checking ("max_abs_coupling" or "max_abs_bias",
    P-3/F-A5 + I-9a/F-R7) -- named explicitly, not hardcoded, so a remediation
    for one gate can never tell a user to relax the OTHER gate's field."""
    fails = peak > cap
    downgraded = fails and assumed and allow_assumed
    check = GateCheck(gate=gate, passed=not fails, measured=peak, limit=cap,
                      assumed=assumed, downgraded=downgraded)
    if not fails or downgraded:
        return check, None

    rems = [Remediation("change encoding", encoding_hint, {"note": "estimate"}),
            Remediation("relax target", f"target.{target_field} >= {peak}")]
    if assumed:
        rems.append(Remediation(
            "override",
            f"pass --allow-assumed to downgrade this gate; "
            f"{target_field} is source={source}"))
    failure = GateFailure(
        gate=gate,
        cause=f"{label} = {peak} exceeds {cap} (source={source})",
        measured=peak, limit=cap, assumed=assumed, remediations=tuple(rems))
    return check, failure


def _evaluate(ising: IsingModel, report: GraphReport, target: TargetProfile,
             allow_assumed: bool) -> tuple[tuple[GateCheck, ...], tuple[GateFailure, ...]]:
    """Single pass shared by `gate_checks` and `check_gates`, so the two can never
    disagree about what a gate measured or whether it passed."""
    checks: list[GateCheck] = []
    fails: list[GateFailure] = []

    degree_ok = report.max_degree <= target.degree.value
    checks.append(GateCheck("degree", degree_ok, report.max_degree,
                            target.degree.value, target.is_assumed("degree")))
    if not degree_ok:
        fails.append(GateFailure(
            gate="degree",
            cause=(f"a node requires {report.max_degree} ports; target {target.name} "
                   f"provides {target.degree.value}"),
            measured=report.max_degree, limit=target.degree.value,
            assumed=target.is_assumed("degree"),
            remediations=(
                Remediation("change encoding",
                            "a denser encoding may lower per-node degree",
                            {"note": "estimate"}),
                Remediation("relax target",
                            f"target.degree >= {report.max_degree}"),
            )))

    # P-3/F-A5 + I-9a/F-R7: coupling_cap and field_cap now read their OWN
    # Sourced field -- max_abs_coupling (|J|, Extropic-documented, not
    # overridable) and max_abs_bias (|b|, a genuine project assumption,
    # overridable). Both are gated unconditionally -- NEITHER is nested
    # inside a "does this model have any edges" check, because a model with
    # zero edges still has biases that must be gated (C2).
    cap_J = target.max_abs_coupling.value
    cap_J_assumed = target.is_assumed("max_abs_coupling")
    cap_J_source = target.max_abs_coupling.source

    peak_J = float(np.abs(ising.weights).max()) if len(ising.weights) else 0.0
    j_check, j_failure = _magnitude_gate(
        "coupling_cap", "|J|max", peak_J, cap_J, cap_J_assumed, cap_J_source,
        "max_abs_coupling", allow_assumed,
        "a local encoding keeps |J| bounded independent of size")
    checks.append(j_check)
    if j_failure:
        fails.append(j_failure)

    cap_b = target.max_abs_bias.value
    cap_b_assumed = target.is_assumed("max_abs_bias")
    cap_b_source = target.max_abs_bias.source

    peak_b = float(np.abs(ising.biases).max()) if len(ising.biases) else 0.0
    b_check, b_failure = _magnitude_gate(
        "field_cap", "|b|max", peak_b, cap_b, cap_b_assumed, cap_b_source,
        "max_abs_bias", allow_assumed,
        "a local encoding keeps |b| bounded independent of size")
    checks.append(b_check)
    if b_failure:
        fails.append(b_failure)

    budget_ok = report.n_nodes <= target.node_budget.value
    checks.append(GateCheck("node_budget", budget_ok, report.n_nodes,
                            target.node_budget.value,
                            target.is_assumed("node_budget")))
    if not budget_ok:
        fails.append(GateFailure(
            gate="node_budget",
            cause=f"{report.n_nodes} nodes exceeds budget {target.node_budget.value}",
            measured=report.n_nodes, limit=target.node_budget.value,
            assumed=target.is_assumed("node_budget"),
            remediations=()))

    colouring_violation = next(
        ((u, v) for u, v in ising.edges
         if report.colouring.get(u) == report.colouring.get(v)), None)
    checks.append(GateCheck("colouring", colouring_violation is None, None,
                            "distinct", False))
    if colouring_violation is not None:
        u, v = colouring_violation
        fails.append(GateFailure(
            gate="colouring",
            cause=f"nodes {u} and {v} are adjacent and share a colour block",
            measured=report.colouring.get(u), limit="distinct",
            assumed=False, remediations=()))

    return tuple(checks), tuple(fails)


def gate_checks(ising: IsingModel, report: GraphReport, target: TargetProfile,
                allow_assumed: bool = False) -> tuple[GateCheck, ...]:
    """Every gate this pass evaluates, pass or fail. `check_gates` is a thin
    filter over the same evaluation for the failures the compile pipeline acts
    on; the receipt records this full set so a COMPILED verdict shows what was
    checked, not only what was never violated (spec section 10)."""
    return _evaluate(ising, report, target, allow_assumed)[0]


def check_gates(ising: IsingModel, report: GraphReport, target: TargetProfile,
                allow_assumed: bool = False) -> tuple[GateFailure, ...]:
    return _evaluate(ising, report, target, allow_assumed)[1]
