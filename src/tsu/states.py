from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CandidateState(str, Enum):
    GENERATED = "GENERATED"
    SEMANTICALLY_INVALID = "SEMANTICALLY_INVALID"
    SEMANTICALLY_VALID = "SEMANTICALLY_VALID"
    HARDWARE_INFEASIBLE = "HARDWARE_INFEASIBLE"
    HARDWARE_FEASIBLE = "HARDWARE_FEASIBLE"
    SELECTED = "SELECTED"
    VIABLE_NOT_SELECTED = "VIABLE_NOT_SELECTED"


@dataclass(frozen=True)
class Candidate:
    encoding: str
    state: CandidateState
    reason: str | None = None
    failure: Any = None
    report: Any = None
    regime: Any = None
    placement: Any = None   # the Placement this candidate reached, if any (A5:
                              # `compare()` needs it for physical p-bit count)


@dataclass(frozen=True)
class RepresentationSet:
    candidates: tuple[Candidate, ...]
    selected: Candidate | None
    ordering_rationale: str
