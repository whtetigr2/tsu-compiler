"""The rule schema: turns a rule from an implicit term-template shape into an
object the compiler can hold, inspect, and reason about -- including telling
a user it CANNOT express one (see `tsu_compiler.passes.expressibility`). Before this
module, a rule's meaning existed only as the terms a spec's `term` block
happened to emit; there was nothing to run an expressibility pass over.

`RULE_CLASSES` names twelve rule classes for MATHEMATICS, not domain --
`pairwise`, `neighbourhood`, `morphology`, `cluster`, `gradient`,
`topological`, `hydrological`, `climate`, `ecological`, `gameplay`,
`statistical`, `boundary`. Four of those twelve -- `hydrological`, `climate`,
`ecological`, `gameplay` -- read as domain vocabulary. That is permitted: a
class label sitting in a data tuple is DATA, not code. What is not permitted,
here or anywhere else under `src/`, is for one of those four words (or any
other workload noun) to become a Python identifier, an `if`/`elif` branch, or
any other piece of behaviour -- the instant a class label stops being a string
compared against a tuple and starts being a code path, it has become exactly
the workload-specific logic this compiler exists to keep out. `load_rules`
below treats all twelve names identically: each is looked up in
`RULE_CLASSES` and nothing about the lookup, or anything downstream of it,
varies by which of the twelve strings it happens to be.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


RULE_CLASSES: tuple[str, ...] = (
    "pairwise",
    "neighbourhood",
    "morphology",
    "cluster",
    "gradient",
    "topological",
    "hydrological",
    "climate",
    "ecological",
    "gameplay",
    "statistical",
    "boundary",
)


@dataclass(frozen=True)
class Rule:
    """A declared rule, independent of whether the pairwise IR can express it
    exactly. `measurement` is a free-form mapping (e.g. `{"kind":
    "squared_deviation", "target": 4, "value": 1}`) that
    `tsu_compiler.passes.expressibility.analyse_rule` dispatches on by its `"kind"`."""
    id: str
    rule_class: str
    scope: str
    hard: bool
    weight: float
    measurement: Mapping
    provenance: str


def load_rules(mapping) -> tuple[Rule, ...]:
    """Build `Rule`s from an iterable of plain mappings (e.g. parsed YAML/JSON).
    Rejects an unknown `rule_class` loudly rather than accepting it silently --
    a rule that slips past this check would skip expressibility analysis
    entirely, which is the one failure mode this schema exists to prevent."""
    rules = []
    for entry in mapping:
        rule_class = entry["rule_class"]
        if rule_class not in RULE_CLASSES:
            raise ValueError(
                f"unknown rule_class {rule_class!r}; known: {RULE_CLASSES}")
        rules.append(Rule(
            id=entry["id"],
            rule_class=rule_class,
            scope=entry["scope"],
            hard=entry["hard"],
            weight=entry["weight"],
            measurement=entry["measurement"],
            provenance=entry["provenance"],
        ))
    return tuple(rules)
