"""Hardware targets are DATA, and every constraint carries its evidence.

A target profile that silently presents an assumption as a fact is the failure the
`source` field exists to prevent (spec section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

RULES = ((1, 0), (2, 1), (2, 3), (4, 1))


def _z1_offsets() -> tuple[tuple[int, int], ...]:
    """Rotations only. Including reflections gives degree 28 and breaks bipartiteness."""
    off = set()
    for a, b in RULES:
        off |= {(a, b), (-b, a), (-a, -b), (b, -a)}
    return tuple(sorted(off))


@dataclass(frozen=True)
class Sourced:
    value: Any
    source: str          # a fact id like "F-14", or "assumed", or "user-supplied"
    quote: str = ""


@dataclass(frozen=True)
class TargetProfile:
    name: str
    degree: Sourced
    offsets: Sourced
    bipartite: Sourced
    schedule: Sourced
    max_abs_coupling: Sourced
    coupling_bits: Sourced
    node_budget: Sourced

    def is_assumed(self, field_name: str) -> bool:
        got = getattr(self, field_name)
        if not isinstance(got, Sourced):
            raise TypeError(f"{field_name} is not a Sourced field")
        return got.source == "assumed"

    def sourced_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self)
                     if isinstance(getattr(self, f.name), Sourced))


Z1 = TargetProfile(
    name="z1",
    degree=Sourced(16, "F-14",
                   "All nodes in Z1 have connection rules (1,0),(2,1),(2,3),(4,1), "
                   "giving degree 16"),
    offsets=Sourced(_z1_offsets(), "F-14/F-17"),
    bipartite=Sourced(True, "F-18", "Since the graph is 2-colorable"),
    schedule=Sourced("chromatic_block_gibbs", "F-19",
                     "resampling every node v in V1 ... The same is then done for V2"),
    max_abs_coupling=Sourced(6.0, "assumed",
                             "project working value; NOT a sourced Extropic figure"),
    coupling_bits=Sourced(6, "assumed",
                          "project working value; NOT a sourced Extropic figure"),
    node_budget=Sourced(250_000, "F-15", "the entire chip has ~250,000 nodes"),
)

IDEAL = TargetProfile(
    name="ideal",
    degree=Sourced(float("inf"), "control"),
    offsets=Sourced((), "control"),
    bipartite=Sourced(False, "control"),
    schedule=Sourced("chromatic_block_gibbs", "control"),
    max_abs_coupling=Sourced(float("inf"), "control"),
    coupling_bits=Sourced(float("inf"), "control"),
    node_budget=Sourced(float("inf"), "control"),
)

PROFILES = {"z1": Z1, "ideal": IDEAL}
