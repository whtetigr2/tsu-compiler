"""Remove what cannot happen, before anything expensive runs.

R29. The compiler searched the encoding and analysed coefficient precision, but
nothing eliminated a variable that cannot take a value. Every serious
mixed-integer or SAT solver presolves first, because removing a variable costs
nothing downstream and shrinks every pass after it.

Two reductions, both exact, both derived only from the contract's DECLARED
rules:

  DOMAIN REDUCTION   A value with no support in some binary rule cannot appear
                     in any satisfying assignment, so it leaves the domain.
                     This is arc consistency, and it is sound: a value present
                     in a solution always has support, so it is never removed.

  VARIABLE MERGING   When a rule leaves a pair able to agree and unable to
                     differ, the two variables are forced equal. Equality is
                     transitive, so union-find collapses whole components into
                     one representative.

Merging is the reduction that matters in practice. `ecology_lotka_lite` forbids
its two species from sharing an edge, which on a connected grid forces every
cell to match: 16 variables become 1 and the reachable space is 2, matching what
the receipt's verification pass measures independently.

WHAT THIS CANNOT SEE. Presolve reads declared constraints. A structure expressed
only in the energy terms is invisible to it. R28's folding model is the standing
example: its parity structure comes from the lattice being bipartite, is not
stated as a contract rule, and no amount of propagation here would find it. That
elimination still has to be reasoned out by a person.

Soundness is the whole value. A presolve that drops a reachable configuration
turns every later result into a confident answer to the wrong question, and it
would look like a win throughout. `tests/test_presolve.py` enumerates small
specs exhaustively and requires every assignment the contract accepts to survive.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PresolveReport:
    """What presolve removed, and what it could not justify removing."""

    n_variables: int
    n_fixed: int
    """Variables whose domain collapsed to a single value."""

    n_merged: int
    """Variables absorbed into another variable by a forced equality."""

    domains: Mapping[str, tuple[int, ...]]
    groups: Mapping[str, str]
    """Merged variable -> the representative it must equal. Excludes
    representatives themselves, so `len(groups)` is `n_merged`."""

    upper_bound_states: int | None
    """Product of the surviving domain sizes. An UPPER BOUND on the reachable
    space, not a count: propagation is local and cannot see every interaction.
    None when the product would be unreasonably large to report."""

    infeasible: bool = False
    reason: str | None = None
    removals: tuple[str, ...] = field(default_factory=tuple)


def _find(parent: dict[str, str], x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def presolve_domains(domains: Mapping[str, Sequence[int]],
                     *,
                     equal_pairs: Iterable[tuple[str, str]] = (),
                     forbidden: Mapping[tuple[str, str], set[tuple[int, int]]]
                     | None = None) -> PresolveReport:
    """The reduction itself, over plain domains and pair constraints.

    Separated from spec parsing so it can be tested directly, including the
    contradictory case that must be reported infeasible rather than silently
    producing an empty model.
    """
    dom: dict[str, set[int]] = {k: set(v) for k, v in domains.items()}
    forbidden = dict(forbidden or {})
    parent = {k: k for k in dom}
    removals: list[str] = []

    for a, b in equal_pairs:
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[rb] = ra
            removals.append(f"{b} is forced equal to {a}, merged")

    # Propagate to a fixpoint: a forbidden pair can strip a value, and a
    # stripped value can force another pair.
    changed = True
    while changed:
        changed = False
        for (x, y), bad in forbidden.items():
            if x not in dom or y not in dom:
                continue
            for value in sorted(dom[x]):
                if all((value, other) in bad for other in dom[y]):
                    dom[x].discard(value)
                    removals.append(
                        f"{x}={value} has no support against {y}, removed")
                    changed = True
            for value in sorted(dom[y]):
                if all((other, value) in bad for other in dom[x]):
                    dom[y].discard(value)
                    removals.append(
                        f"{y}={value} has no support against {x}, removed")
                    changed = True

        # A merged pair must agree, so the representative keeps only the
        # values both can take.
        for name in list(dom):
            root = _find(parent, name)
            if root == name:
                continue
            shared = dom[root] & dom[name]
            if shared != dom[root]:
                dom[root] = set(shared)
                changed = True

    # A merged pair that can never agree is a contradiction, not a reduction.
    for (x, y), bad in forbidden.items():
        if x not in dom or y not in dom:
            continue
        if _find(parent, x) != _find(parent, y):
            continue
        if all((v, v) in bad for v in dom[x] & dom[y]):
            return PresolveReport(
                n_variables=len(domains), n_fixed=0, n_merged=0,
                domains={k: tuple(sorted(v)) for k, v in dom.items()},
                groups={}, upper_bound_states=0, infeasible=True,
                reason=(f"infeasible: {x} and {y} are forced equal by one rule "
                        f"and forbidden from agreeing by another"),
                removals=tuple(removals))

    empty = [k for k, v in dom.items() if not v]
    if empty:
        return PresolveReport(
            n_variables=len(domains), n_fixed=0, n_merged=0,
            domains={k: tuple(sorted(v)) for k, v in dom.items()},
            groups={}, upper_bound_states=0, infeasible=True,
            reason=(f"infeasible: no value survives for "
                    f"{', '.join(sorted(empty))}"),
            removals=tuple(removals))

    groups = {k: _find(parent, k) for k in dom if _find(parent, k) != k}
    live = [k for k in dom if k not in groups]
    bound: int | None = 1
    for k in live:
        bound *= len(dom[k])
        if bound > 10 ** 18:
            bound = None
            break

    return PresolveReport(
        n_variables=len(domains),
        n_fixed=sum(1 for k in live if len(dom[k]) == 1),
        n_merged=len(groups),
        domains={k: tuple(sorted(v)) for k, v in dom.items()},
        groups=groups,
        upper_bound_states=bound,
        removals=tuple(removals),
    )


def presolve(spec: Any) -> PresolveReport:
    """Read a workload spec's contract and reduce what it declares impossible."""
    domains: dict[str, tuple[int, ...]] = {}
    for v in spec.variables:
        k = getattr(getattr(v, "domain", None), "k", None)
        domains[v.name] = tuple(range(k)) if k else (0, 1)

    forbidden: dict[tuple[str, str], set[tuple[int, int]]] = {}
    equal_pairs: list[tuple[str, str]] = []

    def forbid(x: str, y: str, pair: tuple[int, int]) -> None:
        key = (x, y)
        forbidden.setdefault(key, set()).add(pair)

    names = [v.name for v in spec.variables]
    edges = tuple(getattr(spec.contract, "edges", ()) or ())

    for rule in getattr(spec.contract, "rules", ()) or ():
        kind = rule.get("rule")
        if kind == "forbid_both":
            x, y = rule["vars"]
            forbid(x, y, (1, 1))
        elif kind == "forbid_value_with":
            forbid(rule["var"], rule["other"], (rule["value"], 1))
        elif kind == "forbid_value_pair_over_edges":
            a, b = rule["a_value"], rule["b_value"]
            symmetric = rule.get("symmetric", True)
            for u, v in edges:
                forbid(u, v, (a, b))
                if symmetric:
                    forbid(u, v, (b, a))
        # An unknown rule is NOT ignored: reducing on a partial reading of the
        # contract could remove something the unread rule permits.
        else:
            return PresolveReport(
                n_variables=len(domains), n_fixed=0, n_merged=0,
                domains=domains, groups={}, upper_bound_states=None,
                reason=(f"not attempted: rule {kind!r} is not understood here, "
                        f"and reducing on a partial reading of the contract "
                        f"could remove a configuration it permits"),
                removals=(),
            )

    # A pair that can agree and cannot differ is forced equal.
    for (x, y), bad in forbidden.items():
        dx, dy = domains.get(x, ()), domains.get(y, ())
        shared = set(dx) & set(dy)
        if not shared:
            continue
        can_agree = any((v, v) not in bad for v in shared)
        can_differ = any((p, q) not in bad
                         for p, q in itertools.product(dx, dy) if p != q)
        if can_agree and not can_differ:
            equal_pairs.append((x, y))

    report = presolve_domains(domains, equal_pairs=equal_pairs,
                              forbidden=forbidden)
    if report.n_merged or report.n_fixed:
        return report
    return PresolveReport(
        n_variables=report.n_variables, n_fixed=report.n_fixed,
        n_merged=report.n_merged, domains=report.domains,
        groups=report.groups, upper_bound_states=report.upper_bound_states,
        infeasible=report.infeasible, reason=report.reason,
        removals=report.removals,
    )
