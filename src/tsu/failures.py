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


class CompileError(Exception):
    def __init__(self, message: str, failures=()):
        super().__init__(message)
        self.failures = tuple(failures)
