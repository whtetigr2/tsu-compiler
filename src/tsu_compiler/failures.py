"""Structured, diagnostic failures. A failure that says only "failed" is useless,
and one attributed to the wrong cause is worse (spec section 6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Offender:
    kind: Literal["node", "edge"]
    detail: str


@dataclass(frozen=True)
class Remediation:
    action: str
    detail: str
    estimated_cost: dict = field(default_factory=dict)   # estimates, labelled as such


@dataclass(frozen=True)
class GateFailure:
    gate: str
    cause: str
    measured: Any
    limit: Any
    assumed: bool
    remediations: tuple[Remediation, ...] = ()


@dataclass(frozen=True)
class PlacementFailure:
    failure_class: Literal["degree_exceeded", "parity_conflict",
                           "geometry_unreachable", "budget_exceeded",
                           "placement_effort_exhausted"]
    offending: tuple[Offender, ...]
    measured: Any
    limit: Any
    assumed: bool
    remediations: tuple[Remediation, ...] = ()
    # Task 6 (spec 5.3.7): the MediationReport a mediated-but-still-failing
    # placement attempt produced, if any -- e.g. `placement_effort_exhausted`
    # on a graph `insert_mediators` DID make bipartite, where the subsequent
    # geometric search still could not embed it within budget. None for a
    # failure that never reached mediation (degree_exceeded, budget_exceeded
    # on the pre-mediation graph, or a target that needed no mediation at
    # all) -- never fabricated, so real mediation evidence is not silently
    # dropped just because placement ultimately failed for a DIFFERENT
    # (geometric) reason.
    mediation: Any = None


class CompileError(Exception):
    def __init__(self, message: str, failures=()):
        super().__init__(message)
        self.failures = tuple(failures)
