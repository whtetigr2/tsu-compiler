"""Hard gates. A gate failing aborts the compile; a poor metric never does.

Gates whose threshold comes from a Sourced field marked "assumed" are tagged and
overridable via --allow-assumed. Gates backed by a real fact id are NOT overridable
(spec section 7.1).
"""
from __future__ import annotations

import numpy as np

from .failures import GateFailure, Remediation
from .passes.analyse import GraphReport
from .passes.lower import IsingModel
from .target import TargetProfile


def check_gates(ising: IsingModel, report: GraphReport, target: TargetProfile,
                allow_assumed: bool = False) -> tuple[GateFailure, ...]:
    out = []

    if report.max_degree > target.degree.value:
        out.append(GateFailure(
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

    if len(ising.weights):
        peak = float(np.abs(ising.weights).max())
        cap = target.max_abs_coupling.value
        if peak > cap:
            assumed = target.is_assumed("max_abs_coupling")
            rems = [Remediation("change encoding",
                                "a local encoding keeps |J| bounded independent of size",
                                {"note": "estimate"}),
                    Remediation("relax target", f"target.max_abs_coupling >= {peak}")]
            if assumed:
                rems.append(Remediation(
                    "override",
                    f"pass --allow-assumed to downgrade this gate; "
                    f"max_abs_coupling is source={target.max_abs_coupling.source}"))
            if not (assumed and allow_assumed):
                out.append(GateFailure(
                    gate="coupling_cap",
                    cause=(f"|J|max = {peak} exceeds {target.name} cap {cap} "
                           f"(source={target.max_abs_coupling.source})"),
                    measured=peak, limit=cap, assumed=assumed,
                    remediations=tuple(rems)))

    if report.n_nodes > target.node_budget.value:
        out.append(GateFailure(
            gate="node_budget",
            cause=f"{report.n_nodes} nodes exceeds budget {target.node_budget.value}",
            measured=report.n_nodes, limit=target.node_budget.value,
            assumed=target.is_assumed("node_budget"),
            remediations=()))

    for u, v in ising.edges:
        if report.colouring.get(u) == report.colouring.get(v):
            out.append(GateFailure(
                gate="colouring",
                cause=f"nodes {u} and {v} are adjacent and share a colour block",
                measured=report.colouring.get(u), limit="distinct",
                assumed=False, remediations=()))
            break

    return tuple(out)
