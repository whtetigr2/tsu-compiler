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
    # The placement SEARCH ran out of budget. The model may well fit; nothing
    # here says it does not. Same argument as COMPILER_ERROR above, applied to
    # a timer instead of an exception: HARDWARE_INFEASIBLE asserts "your model
    # does not fit the chip", and saying that because an annealer stopped early
    # tells a user to change a model that was fine.
    #
    # Measured (audit/findings/R27.md) on programs/seq_design_longer.yaml: the
    # domain_wall candidate fails to place at the default budget, carrying
    # failure_class 'placement_effort_exhausted' and a remediation reading
    # "increase placement effort". At 24 restarts / 250,000 iterations it
    # places and is SELECTED, at 24 physical spins against one-hot's 32. So the
    # cheaper search was shipping a model 25% larger while calling the smaller
    # one hardware-infeasible.
    #
    # This is NOT feasible for selection. It means "not found within budget",
    # which is a different sentence from "cannot exist", and the difference is
    # the whole point.
    PLACEMENT_EFFORT_EXHAUSTED = "PLACEMENT_EFFORT_EXHAUSTED"
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
