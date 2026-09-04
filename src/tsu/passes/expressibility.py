"""Per-rule expressibility verdicts against the pairwise substrate.

The IR (`tsu.ir`) admits exactly two term kinds, `Linear` and `Product`, and
every term lowers to a pairwise model -- that is not a style choice, it is
what Z1 is (plan section "The central constraint, already measured"). This
pass says, per declared `Rule`, what happens when its meaning is forced
through that substrate:

- EXACT         -- expressible as Linear and/or Product with no change of
                    meaning.
- DISTORTED      -- expressible only as a DIFFERENT rule. `distortion` names
                    the change in plain words, e.g. the measured case below:
                    a one-sided threshold's nearest pairwise form is a
                    symmetric quadratic, so "at least N" silently becomes
                    "exactly N" unless this pass says so out loud.
                    DISTORTED is never promoted to EXACT -- a rule whose
                    meaning changed is a different rule, and the verdict's
                    job is to tell the user which one they actually got.
- INEXPRESSIBLE -- this pass has actually analysed the measurement kind's
                    mathematical shape and can support the claim that no
                    pairwise form exists; `reason` carries that argument.
- UNKNOWN        -- this pass has no dispatch entry for the measurement
                    kind at all. This is NOT a claim about expressibility --
                    it is a claim about this pass's own coverage. (C4, code
                    review 2026-09-04-lattice-rule-taxonomy: an unregistered
                    kind used to fall through to INEXPRESSIBLE, asserting
                    "no pairwise form exists" for kinds -- including
                    `neighbourhood_count`, shipped and measured EXACT by
                    this same branch -- that were never analysed at all.
                    INEXPRESSIBLE must be reserved for a claim this module
                    can actually support; "I don't know this kind" gets its
                    own honest status instead of borrowing that one's
                    vocabulary.)

Dispatch is on `rule.measurement["kind"]` -- a string looked up against the
handful of kinds this module knows how to analyse, exactly the same shape as
`load_rules` dispatching on `rule_class` in `tsu.rules`. Nothing here reads
or branches on `rule.rule_class`: the twelve rule classes are a taxonomy for
humans (and, per that module's docstring, permitted domain-flavoured labels
among them); the pairwise substrate does not care what a rule is CALLED, only
what its `measurement` computes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..rules import Rule


@dataclass(frozen=True)
class Verdict:
    rule_id: str
    status: str          # "EXACT" | "DISTORTED" | "INEXPRESSIBLE" | "UNKNOWN"
    cost: Mapping
    distortion: str | None
    reason: str


def analyse_rule(rule: Rule) -> Verdict:
    kind = rule.measurement.get("kind")
    if kind == "squared_deviation":
        return _squared_deviation(rule)
    if kind == "one_sided_threshold":
        return _one_sided_threshold(rule)
    return Verdict(
        rule_id=rule.id,
        status="UNKNOWN",
        cost={},
        distortion=None,
        reason=(
            f"this pass has no dispatch entry for measurement kind {kind!r} "
            f"-- it has not analysed this kind's mathematical shape at all, "
            f"so it can make no claim either way about whether a pairwise "
            f"form exists; this is NOT an INEXPRESSIBLE verdict"),
    )


def _squared_deviation(rule: Rule) -> Verdict:
    """(target - value)^2 is a squared linear form: Product(L, L, weight),
    which is pairwise with no change of meaning -- the honest EXACT case."""
    return Verdict(
        rule_id=rule.id,
        status="EXACT",
        cost={"term_kind": "Product", "pairwise_terms": 1},
        distortion=None,
        reason=(
            "(target - value)^2 is a squared linear form -> "
            "Product(L, L, weight) -> pairwise, with no change of meaning"),
    )


def _one_sided_threshold(rule: Rule) -> Verdict:
    """max(0, target - N) has a kink: it is not a polynomial at any degree, so
    no pairwise form expresses it exactly. The nearest pairwise form is the
    symmetric (target - N)^2 -- measured, not assumed: it penalises N =
    target + k identically to N = target - k for every k, so a one-sided
    threshold ("at least target") is silently replaced by a two-sided one
    ("exactly target") unless this verdict names that distortion in plain
    words, which is the one thing DISTORTED exists to do."""
    target = rule.measurement["target"]
    over, under = target + 2, target - 2
    distortion = (
        f"one-sided max(0, {target} - N) becomes two-sided ({target} - N)^2, "
        f"which is symmetric: N={over} is penalised as hard as N={under}, so "
        f"'at least {target}' becomes 'exactly {target}'"
    )
    return Verdict(
        rule_id=rule.id,
        status="DISTORTED",
        cost={"term_kind": "Product", "pairwise_terms": 1},
        distortion=distortion,
        reason=(
            "max(0, target - N) has a kink and is not a polynomial at any "
            "degree, so no exact pairwise form exists; the nearest pairwise "
            "(Product) form changes the rule's meaning -- see distortion"),
    )
