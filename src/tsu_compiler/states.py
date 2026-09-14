from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CandidateState(str, Enum):
    GENERATED = "GENERATED"
    SEMANTICALLY_INVALID = "SEMANTICALLY_INVALID"
    SEMANTICALLY_VALID = "SEMANTICALLY_VALID"
    HARDWARE_INFEASIBLE = "HARDWARE_INFEASIBLE"
    # A fault in THIS COMPILER, not a property of the user's model.
    # HARDWARE_INFEASIBLE asserts "your model does not fit the chip";
    # reporting our own unexpected exception as that would tell a user to
    # change a model that was fine. External review (2026-09-10, C-5)
    # routed unexpected place/route exceptions into HARDWARE_INFEASIBLE;
    # this state exists so the two claims can never be confused again.
    COMPILER_ERROR = "COMPILER_ERROR"
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
