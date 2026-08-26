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

from .ir import (Binary, Categorical, Linear, LinearForm, Product, Var, VarRef)

_REF = re.compile(r"^(?P<name>[A-Za-z_]\w*)(?:=(?P<value>\d+))?$")


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskContract:
    """Declarative validation rules. A Python callable form is permitted later and
    changes nothing about the layering."""
    rules: tuple[Mapping[str, Any], ...]

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


def _generated_terms(gen: Mapping[str, Any]):
    """Structural generators keep large synthetic specs out of YAML by hand.

    This is NOT a workload feature: `clique` describes a graph shape, not a domain.
    """
    if gen["kind"] != "clique":
        raise ValueError(f"unknown generator {gen['kind']!r}")
    n, w = int(gen["n"]), float(gen["weight"])
    variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
    terms = tuple(
        Product(LinearForm({VarRef(f"x{i}"): 1.0}), LinearForm({VarRef(f"x{j}"): 1.0}), w)
        for i in range(n) for j in range(i + 1, n)
    )
    return variables, terms


def load_spec(path: str) -> WorkloadSpec:
    text = open(path, encoding="utf-8").read()
    raw = yaml.safe_load(text)

    if "generate" in raw:
        variables, terms = _generated_terms(raw["generate"])
    else:
        variables = []
        for name, d in raw["variables"].items():
            kind = d["domain"]
            dom = Binary() if kind == "binary" else Categorical(int(d["k"]))
            variables.append(Var(name, dom))
        variables = tuple(variables)

        terms = []
        for t in raw.get("terms", []):
            if t["kind"] == "linear":
                terms.append(Linear(_parse_form(t["form"]), float(t["weight"])))
            elif t["kind"] == "product":
                terms.append(Product(_parse_form(t["a"]), _parse_form(t["b"]),
                                     float(t["weight"])))
            else:
                raise ValueError(f"unknown term kind {t['kind']!r}")
        terms = tuple(terms)

    rules = tuple(raw.get("contract", {}).get("validate", []) or ())
    for r in rules:
        if "message" not in r:
            raise ValueError(
                f"validation rule {r.get('rule')!r} has no `message`; a verdict "
                f"without the violation that caused it is not actionable")

    return WorkloadSpec(name=raw["name"], variables=variables, terms=terms,
                        contract=TaskContract(rules), source_text=text)
