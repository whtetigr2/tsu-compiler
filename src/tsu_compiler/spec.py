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
class FormulationRationale:
    """WHY a spec construct expanded into the IR shape it did -- recorded HERE,
    at expansion time (inside spec.py's own term-generation code), not
    reconstructed after the fact from the finished IR (spec section on
    `tsuc explain`'s FORMULATION layer: a reader needs the reasoning, not just
    a glossary of shapes).

    `reason` is the STRUCTURAL reason this shape is used -- e.g. a
    conservation constraint is an equality, and squaring a linear form keeps
    it pairwise regardless of how many variables the form contains. When a
    construct genuinely has none to report (a hand-written `linear`/`product`
    term IS its own shape by direct declaration, not a template's design
    decision), `reason` is "" -- an absent rationale must print as absent,
    never as invented prose standing in for one."""
    construct: str    # the spec construct this is about: "linear", "product",
                        # "product_over_edges", "conserve_over_edges", "clique", ...
    ir_shape: str      # the IR shape it became: "Linear", "Product", "Product
                        # per edge", "Product(L, L, weight) -- squared linear form"
    count: int         # how many IR terms this expansion produced
    scope: str         # what it produced them OVER, e.g. "the spec's 4-edge
                        # generated set"; "" when there is no set to name (a
                        # single hand-written term)
    reason: str = ""   # the structural reason this shape is used; "" (never
                        # fabricated) when the construct has none


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
    # G1: WHY each construct's terms took the IR shape they did, recorded at
    # expansion time by the SAME code that built them (`_generated_terms`'s
    # clique branch, `_product_over_edges`, `_conserve_over_edges_terms`,
    # and `load_spec`'s own literal-term loop) -- empty for a spec whose
    # generator produces no terms at all (e.g. a bare `grid`, which only
    # declares shape).
    formulation: tuple = ()

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
    not a domain. Every generator returns (variables, terms, edges, rationale)
    -- `edges` is the generated pairwise structure itself, exposed so term
    templates and validation rules (A2/A3) can be built over it generically,
    for any shape. `rationale` (G1) is a tuple of `FormulationRationale`
    records, one per KIND of term this generator actually produced -- empty
    for a generator (like a bare `grid`) that produces no terms, since there
    is then nothing to explain.
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
        rationale = (FormulationRationale(
            construct="clique",
            ir_shape="Product per edge (direct product of the two endpoints' "
                     "own occupancy indicators)",
            count=len(terms),
            scope=f"the generated {n}-node clique's {len(edges)}-edge set",
            reason="a clique pairs every two nodes directly, so each edge "
                   "becomes one Product between their two occupancy "
                   "indicators -- a single pairwise interaction needs no "
                   "chain or auxiliary variable"),
        ) if terms else ()
        return variables, terms, edges, rationale

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
        return variables, (), tuple(edges), ()

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


def _flow_var_name(prefix: str, u: str, v: str) -> str:
    """The naming convention for a `conserve_over_edges` (C2) flow variable:
    one Binary spin per edge (u, v) of the spec's own edge set, read as "the
    flow from u to v" in the direction the edge is already declared. Shared
    by variable generation and term generation so the two can never name a
    variable differently."""
    return f"{prefix}__{u}__{v}"


def _conserve_over_edges_variables(t: Mapping[str, Any], edges, existing_names: set):
    """Term template (C2): a `conserve_over_edges` term needs one flow
    variable PER EDGE of the spec's edge set -- generated here, before the
    term itself, following `grid`'s own precedent of generating variables
    from a structural shape rather than requiring them hand-declared. Raises
    on a name collision rather than silently shadowing an existing variable,
    the same guard encode.py's chain-name generation already uses."""
    prefix = t["flow_prefix"]
    variables = []
    for u, v in edges:
        name = _flow_var_name(prefix, u, v)
        if name in existing_names:
            raise ValueError(
                f"generated conserve_over_edges flow variable {name!r} "
                f"collides with an existing variable; choose a different "
                f"flow_prefix")
        variables.append(Var(name, Binary()))
        existing_names.add(name)
    return variables


def _conserve_over_edges_terms(t: Mapping[str, Any], edges):
    """Term template (C2): flow conservation over a node's incident edges.

    For every node n of the edge set, emits `weight * (inflow - outflow -
    net)**2` -- a squared linear form (Product(L, L, weight)), hence PAIRWISE
    once lowered, exactly the shape one-hot's own exactly-one penalty already
    uses. `inflow`/`outflow` sum the incident flow variables by the edge's
    own declared direction; `net = sinks.get(n, 0) - sources.get(n, 0)` is
    the node's own source/sink value, 0 (ordinary conservation: inflow ==
    outflow) for a node named in neither map. A sink (net > 0) wants net
    inflow; a source (net < 0) wants net outflow -- the sign a reader would
    expect from "sinks accumulate, sources emit".

    Named for the MATHEMATICS -- conservation of a signed quantity over an
    edge set -- not for reachability or any workload; it serves any
    conservation-shaped constraint the same way `product_over_edges` serves
    any pairwise one.
    """
    prefix = t["flow_prefix"]
    weight = float(t["weight"])
    sources = t.get("sources", {}) or {}
    sinks = t.get("sinks", {}) or {}
    nodes = sorted({n for e in edges for n in e})
    node_set = set(nodes)

    for name in list(sources) + list(sinks):
        if name not in node_set:
            raise ValueError(
                f"conserve_over_edges: {name!r} (in sources/sinks) is not a "
                f"node of this edge set")

    incident_in: dict[str, list[str]] = {n: [] for n in nodes}
    incident_out: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v in edges:
        name = _flow_var_name(prefix, u, v)
        incident_out[u].append(name)
        incident_in[v].append(name)

    terms = []
    for n in nodes:
        net = float(sinks.get(n, 0.0)) - float(sources.get(n, 0.0))
        coeffs: dict[VarRef, float] = {}
        for name in incident_in[n]:
            r = VarRef(name)
            coeffs[r] = coeffs.get(r, 0.0) + 1.0
        for name in incident_out[n]:
            r = VarRef(name)
            coeffs[r] = coeffs.get(r, 0.0) - 1.0
        L = LinearForm(coeffs, const=-net)
        terms.append(Product(L, L, weight))
    rationale = FormulationRationale(
        construct="conserve_over_edges",
        ir_shape="Product(L, L, weight) -- a squared linear form",
        count=len(terms),
        scope=f"one per node of the spec's {len(edges)}-edge generated set "
              f"({len(nodes)} nodes)",
        reason="a conservation constraint is an equality (inflow - outflow "
               "== net); squaring a linear form keeps the penalty pairwise "
               "(a Product of two identical linear forms) no matter how "
               "many flow variables the form itself sums over")
    return terms, (rationale,) if terms else ()


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
    rationale = FormulationRationale(
        construct="product_over_edges",
        ir_shape="Product per edge" + (" (both orientations)" if symmetric else ""),
        count=len(terms),
        scope=f"the spec's {len(edges)}-edge generated set",
        reason="a pairwise value constraint between two connected variables "
               "becomes a Product of their two value-indicator linear forms "
               "-- one indicator per side of the edge, so the shape stays "
               "pairwise regardless of either variable's own domain size")
    return terms, (rationale,) if terms else ()


def _neighbourhood_count_terms(t: Mapping[str, Any], edges,
                               var_by_name: Mapping[str, Var]):
    """Term template: squared deviation of a cell's own neighbourhood count
    from a target -- "a cell's own neighbourhood indicators, minus a
    target". This is a SHAPE, not a domain rule (Task 5's own brief): for
    every node n of the spec's edge set, `L_n = (sum over n's neighbours m
    of the indicator [m == value]) - target`, and the term emits
    `Product(L_n, L_n, weight)` -- one per node, generic over any
    edge-defined graph, following `_conserve_over_edges_terms`'s own "one
    per node of the edge set" precedent rather than being grid-specific.

    `audit/expressibility_matrix.md` row #2 (`neighbourhood`) measured this
    EXACT shape as EXACT -- but only because that rule's own INTENT is
    already symmetric ("as close to target as possible"); squaring
    introduces no distortion there. A ONE-SIDED rule ("at least N") is a
    DIFFERENT rule, and the SAME shape DISTORTS it (matrix rows #3
    `morphology`, #9 `ecological`: "at least N" silently becomes "exactly
    N"). This template does not -- and cannot -- tell the two apart from
    `value`/`target`/`weight` alone; it is the caller's responsibility to
    reach for it only where the underlying rule is honestly symmetric (see
    `tsu_compiler.passes.expressibility._one_sided_threshold` for the DISTORTED
    case's own verdict machinery, which this template deliberately does not
    duplicate)."""
    value, target, weight = t["value"], float(t["target"]), float(t["weight"])
    neighbours: dict[str, list[str]] = {}
    for u, v in edges:
        neighbours.setdefault(u, []).append(v)
        neighbours.setdefault(v, []).append(u)
    terms = []
    for n in sorted(neighbours):
        coeffs: dict[VarRef, float] = {}
        const = -target
        for m in neighbours[n]:
            ind = _value_indicator(var_by_name[m], value)
            const += ind.const
            for ref, w in ind.coeffs.items():
                coeffs[ref] = coeffs.get(ref, 0.0) + w
        L = LinearForm(coeffs, const)
        terms.append(Product(L, L, weight))
    rationale = FormulationRationale(
        construct="neighbourhood_count",
        ir_shape="Product(L, L, weight) -- a squared linear form",
        count=len(terms),
        scope=f"one per node of the spec's {len(edges)}-edge generated set "
              f"({len(neighbours)} nodes with at least one neighbour)",
        reason="a neighbourhood COUNT's own intent is 'as close to target "
               "as possible', which is symmetric by definition -- squaring "
               "a linear form over the neighbour indicators keeps this "
               "pairwise (Product(L, L, weight)) with no change of "
               "meaning. This is a SHAPE, not a domain rule: it is the "
               "honest form only where the underlying rule's own intent "
               "is symmetric-in-count; a ONE-SIDED threshold ('at least "
               "N') is a different rule and distorts under this same "
               "shape (audit/expressibility_matrix.md rows #3, #9) -- this "
               "template does not detect that case, the caller must not "
               "reach for it there")
    return terms, (rationale,) if terms else ()


def load_spec(path: str) -> WorkloadSpec:
    text = open(path, encoding="utf-8").read()
    raw = yaml.safe_load(text)

    if "generate" in raw:
        variables, generated_terms, edges, gen_rationale = \
            _generated_terms(raw["generate"])
    else:
        edges = ()
        generated_terms = ()
        gen_rationale = ()
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

    # C2: conserve_over_edges generates FLOW VARIABLES from the edge set,
    # before term parsing (which needs them present in var_by_name/the model
    # even though nothing else ever declares them by hand) -- following
    # `grid`'s own precedent of variables generated from a structural shape.
    declared_names = {v.name for v in variables}
    for t in raw.get("terms", []):
        if t.get("kind") == "conserve_over_edges":
            if not edges:
                raise ValueError(
                    "conserve_over_edges requires a non-empty edge set (e.g. "
                    "from a `generate` block); this spec has none")
            variables = variables + tuple(
                _conserve_over_edges_variables(t, edges, declared_names))

    var_by_name = {v.name: v for v in variables}

    # Hand-written terms are parsed regardless of whether the spec also has a
    # `generate` block, so a generated shape (grid, clique) can carry both its
    # own baked-in terms (clique's weighted edges) AND additional hand-declared
    # ones (e.g. `product_over_edges` over a generated grid's edge set) --
    # previously a `generate` block silently discarded any `terms:` key
    # entirely, which would have made A2 impossible to use alongside A1.
    extra_terms = []
    # G1: literal `linear`/`product` terms are hand-authored -- each IS its
    # own shape by direct declaration, not a template's design decision, so
    # they are recorded as ONE aggregate FormulationRationale per kind (count
    # of how many, reason left honestly absent) rather than one record per
    # declared term, matching the aggregate-by-kind granularity `tsuc explain`
    # already used for its (now-removed) static glossary.
    literal_counts = {"linear": 0, "product": 0}
    formulation = list(gen_rationale)
    for t in raw.get("terms", []):
        kind = t["kind"]
        if kind == "linear":
            extra_terms.append(Linear(_parse_form(t["form"]), float(t["weight"])))
            literal_counts["linear"] += 1
        elif kind == "product":
            extra_terms.append(Product(_parse_form(t["a"]), _parse_form(t["b"]),
                                       float(t["weight"])))
            literal_counts["product"] += 1
        elif kind == "product_over_edges":
            poe_terms, poe_rationale = _product_over_edges(t, edges, var_by_name)
            extra_terms.extend(poe_terms)
            formulation.extend(poe_rationale)
        elif kind == "conserve_over_edges":
            coe_terms, coe_rationale = _conserve_over_edges_terms(t, edges)
            extra_terms.extend(coe_terms)
            formulation.extend(coe_rationale)
        elif kind == "neighbourhood_count":
            nc_terms, nc_rationale = _neighbourhood_count_terms(t, edges, var_by_name)
            extra_terms.extend(nc_terms)
            formulation.extend(nc_rationale)
        else:
            raise ValueError(f"unknown term kind {kind!r}")
    for kind, ir_shape in (("linear", "Linear"), ("product", "Product")):
        if literal_counts[kind]:
            formulation.append(FormulationRationale(
                construct=kind, ir_shape=ir_shape, count=literal_counts[kind],
                scope="", reason=""))
    terms = tuple(generated_terms) + tuple(extra_terms)

    rules = tuple(raw.get("contract", {}).get("validate", []) or ())
    for r in rules:
        if "message" not in r:
            raise ValueError(
                f"validation rule {r.get('rule')!r} has no `message`; a verdict "
                f"without the violation that caused it is not actionable")

    return WorkloadSpec(name=raw["name"], variables=variables, terms=terms,
                        contract=TaskContract(rules, edges), source_text=text,
                        edges=edges, formulation=tuple(formulation))
