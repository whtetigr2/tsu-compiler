"""Workload layer: what is legal, preferred, forbidden, and what counts as valid.

Workload correctness is measured on DECODED samples and is never inferred from
energy (spec section 3.1). `validate` returns the specific violations, because a
verdict without them is not actionable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from .ir import (Binary, Categorical, Domain, Linear, LinearForm, Product, Var, VarRef)

_REF = re.compile(r"^(?P<name>[A-Za-z_]\w*)(?:=(?P<value>\d+))?$")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskContract:
    """Declarative validation rules. A Python callable form is permitted later and
    changes nothing about the layering.

    `edges` is the spec's generated edge set (empty for a hand-written spec with
    no `generate` block) -- it exists here, not just on WorkloadSpec, because an
    edge-defined rule (`forbid_value_pair_over_edges`) needs it at validate time
    and TaskContract is the layer that owns validation."""
    rules: tuple[Mapping[str, Any], ...]
    edges: tuple = ()

    def validate(self, assignment: Mapping[str, int]) -> ValidationResult:
        out = []
        for r in self.rules:
            kind = r["rule"]
            if kind == "forbid_both":
                x, y = r["vars"]
                if assignment[x] == 1 and assignment[y] == 1:
                    out.append(r["message"])
            elif kind == "forbid_value_with":
                if assignment[r["var"]] == r["value"] and assignment[r["other"]] == 1:
                    out.append(r["message"])
            elif kind == "forbid_value_pair_over_edges":
                # Generic: any edge-defined workload can express "these two
                # values must not sit next to each other" over its own edge
                # set. Undirected adjacency is the default (`symmetric: true`);
                # a spec wanting a one-way rule sets it false. Each violation
                # names its SPECIFIC edge -- a bare count is not actionable.
                a_val, b_val = r["a_value"], r["b_value"]
                symmetric = r.get("symmetric", True)
                for u, v in self.edges:
                    if assignment[u] == a_val and assignment[v] == b_val:
                        out.append(f"{r['message']}: {u}-{v}")
                    elif symmetric and assignment[u] == b_val and assignment[v] == a_val:
                        out.append(f"{r['message']}: {u}-{v}")
            else:
                raise ValueError(f"unknown validation rule {kind!r}")
        return ValidationResult(ok=not out, violations=tuple(out))


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    variables: tuple[Var, ...]
    terms: tuple
    contract: TaskContract
    source_text: str = ""
    # The generated edge set (pairs of variable NAMES), empty for a hand-written
    # spec with no `generate` block. A generic structural fact about the spec's
    # shape -- not a workload feature -- that edge-defined term templates and
    # validation rules are built over (spec section 9 precedent: `clique`
    # describes a graph shape, not a domain).
    edges: tuple = ()

    def var(self, name: str) -> Var:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def assignments(self):
        """Every logical assignment. Only used where the domain is small."""
        import itertools
        names = [v.name for v in self.variables]
        doms = [v.domain.values for v in self.variables]
        for combo in itertools.product(*doms):
            yield dict(zip(names, combo))


def _parse_ref(token: str) -> VarRef:
    m = _REF.match(token)
    if not m:
        raise ValueError(f"malformed variable reference {token!r}")
    v = m.group("value")
    return VarRef(m.group("name"), None if v is None else int(v))


def _parse_form(raw: Mapping[str, float]) -> LinearForm:
    coeffs, const = {}, 0.0
    for token, w in raw.items():
        if token == "const":
            const = float(w)
        else:
            coeffs[_parse_ref(token)] = float(w)
    return LinearForm(coeffs, const)


def _parse_domain(d: Mapping[str, Any]) -> Domain:
    kind = d["domain"]
    return Binary() if kind == "binary" else Categorical(int(d["k"]))


def _generated_terms(gen: Mapping[str, Any]):
    """Structural generators keep large synthetic specs out of YAML by hand.

    This is NOT a workload feature: `clique` and `grid` describe graph SHAPES,
    not a domain. Every generator returns (variables, terms, edges) -- `edges`
    is the generated pairwise structure itself, exposed so term templates and
    validation rules (A2/A3) can be built over it generically, for any shape.
    """
    kind = gen["kind"]
    if kind == "clique":
        n, w = int(gen["n"]), float(gen["weight"])
        variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
        edges = tuple((f"x{i}", f"x{j}") for i in range(n) for j in range(i + 1, n))
        terms = tuple(
            Product(LinearForm({VarRef(u): 1.0}), LinearForm({VarRef(v): 1.0}), w)
            for u, v in edges
        )
        return variables, terms, edges

    if kind == "grid":
        # A 2-D lattice SHAPE: variables at every (x, y) cell and the 4-neighbour
        # edge set between them. No term is generated here -- unlike `clique`,
        # which bakes a weight straight into its own edges, a grid's constraints
        # are declared separately (via `product_over_edges`/A2) over the edge
        # set this returns, so the same shape serves any pairwise rule.
        width, height = int(gen["width"]), int(gen["height"])
        domain = _parse_domain(gen["variable_domain"])

        def name(x: int, y: int) -> str:
            return f"g{x}_{y}"

        variables = tuple(
            Var(name(x, y), domain) for y in range(height) for x in range(width))
        edges = []
        for y in range(height):
            for x in range(width):
                if x + 1 < width:
                    edges.append((name(x, y), name(x + 1, y)))
                if y + 1 < height:
                    edges.append((name(x, y), name(x, y + 1)))
        return variables, (), tuple(edges)

    raise ValueError(f"unknown generator {kind!r}")


def _value_indicator(var: Var, value: int) -> LinearForm:
    """A LinearForm that is 1 when `var` == `value`, 0 otherwise -- generic over
    a variable's own domain (spec section 9 note in A2: not categorical-only).

    For a categorical variable this is exactly VarRef(name, value), per the IR's
    own definition ("VarRef('c', 0) -> the indicator [c == 0]"). A Binary
    variable has no `value`-tagged VarRef form in the IR -- its occupancy IS the
    indicator of ==1 (VarRef(name), per "VarRef('a') -> the occupancy of binary
    variable a"), and the indicator of ==0 is its complement (1 - occupancy),
    which is exactly as linear.
    """
    if isinstance(var.domain, Binary):
        if value == 1:
            return LinearForm({VarRef(var.name): 1.0})
        if value == 0:
            return LinearForm({VarRef(var.name): -1.0}, const=1.0)
        raise ValueError(
            f"binary variable {var.name!r} has no value {value}; binary domains "
            f"are {{0, 1}}")
    return LinearForm({VarRef(var.name, value): 1.0})


def _product_over_edges(t: Mapping[str, Any], edges, var_by_name: Mapping[str, Var]):
    """Term template (A2): one Product per edge of the spec's edge set --
    "apply this pairwise term across every edge", generic over any edge-defined
    workload. Adjacency is undirected by default (`symmetric: true`), so the
    mirrored orientation is also emitted unless the spec says otherwise."""
    a_val, b_val, weight = t["a_value"], t["b_value"], float(t["weight"])
    symmetric = t.get("symmetric", True)
    terms = []
    for u, v in edges:
        terms.append(Product(_value_indicator(var_by_name[u], a_val),
                             _value_indicator(var_by_name[v], b_val), weight))
        if symmetric:
            terms.append(Product(_value_indicator(var_by_name[u], b_val),
                                 _value_indicator(var_by_name[v], a_val), weight))
    return terms


def load_spec(path: str) -> WorkloadSpec:
    text = open(path, encoding="utf-8").read()
    raw = yaml.safe_load(text)

    if "generate" in raw:
        variables, generated_terms, edges = _generated_terms(raw["generate"])
    else:
        edges = ()
        generated_terms = ()
        variables = []
        for name, d in raw["variables"].items():
            # "const" is reserved: _parse_form treats the literal key "const" in
            # a term's `form`/`a`/`b` mapping as the form's CONSTANT, not a
            # variable reference. A declared variable named "const" would parse
            # silently -- the variable simply vanishes from every term that
            # references it (LinearForm(coeffs={}, const=<its weight>) instead
            # of a real coefficient entry) -- with no error and a wrong energy.
            # Same silent-wrongness class as the '__dw' chain-name collision in
            # encode.py, guarded the same way: raise loudly instead.
            if name == "const":
                raise ValueError(
                    "variable name 'const' is reserved: a term's `form`/`a`/`b` "
                    "mapping treats the key 'const' as the form's constant, not "
                    "a variable reference, so a declared variable named 'const' "
                    "would silently vanish from every term that names it -- "
                    "rename the variable")
            kind = d["domain"]
            dom = Binary() if kind == "binary" else Categorical(int(d["k"]))
            variables.append(Var(name, dom))
        variables = tuple(variables)

    var_by_name = {v.name: v for v in variables}

    # Hand-written terms are parsed regardless of whether the spec also has a
    # `generate` block, so a generated shape (grid, clique) can carry both its
    # own baked-in terms (clique's weighted edges) AND additional hand-declared
    # ones (e.g. `product_over_edges` over a generated grid's edge set) --
    # previously a `generate` block silently discarded any `terms:` key
    # entirely, which would have made A2 impossible to use alongside A1.
    extra_terms = []
    for t in raw.get("terms", []):
        kind = t["kind"]
        if kind == "linear":
            extra_terms.append(Linear(_parse_form(t["form"]), float(t["weight"])))
        elif kind == "product":
            extra_terms.append(Product(_parse_form(t["a"]), _parse_form(t["b"]),
                                       float(t["weight"])))
        elif kind == "product_over_edges":
            extra_terms.extend(_product_over_edges(t, edges, var_by_name))
        else:
            raise ValueError(f"unknown term kind {kind!r}")
    terms = tuple(generated_terms) + tuple(extra_terms)

    rules = tuple(raw.get("contract", {}).get("validate", []) or ())
    for r in rules:
        if "message" not in r:
            raise ValueError(
                f"validation rule {r.get('rule')!r} has no `message`; a verdict "
                f"without the violation that caused it is not actionable")

    return WorkloadSpec(name=raw["name"], variables=variables, terms=terms,
                        contract=TaskContract(rules, edges), source_text=text,
                        edges=edges)
