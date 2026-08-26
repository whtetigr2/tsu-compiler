"""The workload-agnostic IR.

Two term kinds over linear forms, and nothing else. A third kind may not be added
without a documented workload that provably needs one (spec section 4).

Semantics: E(x) = sum over terms of weight * (term evaluated at x), and the induced
distribution is p(x) proportional to exp(-beta * E(x)). Lower energy is better, so a
PENALTY carries a POSITIVE weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Binary:
    """A variable taking values 0 or 1."""
    @property
    def values(self) -> tuple[int, ...]:
        return (0, 1)


@dataclass(frozen=True)
class Categorical:
    """A variable taking one of k values, 0..k-1. The ENCODING is not chosen here."""
    k: int

    def __post_init__(self):
        if self.k < 2:
            raise ValueError(f"Categorical needs k >= 2, got {self.k}")

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(range(self.k))


Domain = Binary | Categorical


@dataclass(frozen=True)
class Var:
    name: str
    domain: Domain


@dataclass(frozen=True)
class VarRef:
    """A reference to a variable, or to one VALUE of a categorical variable.

    VarRef("a")      -> the occupancy of binary variable a
    VarRef("c", 0)   -> the indicator [c == 0]
    """
    name: str
    value: int | None = None


@dataclass(frozen=True)
class LinearForm:
    coeffs: Mapping[VarRef, float] = field(default_factory=dict)
    const: float = 0.0

    def evaluate(self, assignment: Mapping[str, int]) -> float:
        total = self.const
        for ref, w in self.coeffs.items():
            if ref.value is None:
                total += w * float(assignment[ref.name])
            else:
                total += w * (1.0 if assignment[ref.name] == ref.value else 0.0)
        return total

    def refs(self) -> tuple[VarRef, ...]:
        return tuple(self.coeffs)


@dataclass(frozen=True)
class Linear:
    form: LinearForm
    weight: float

    def evaluate(self, assignment) -> float:
        return self.weight * self.form.evaluate(assignment)

    def refs(self):
        return self.form.refs()


@dataclass(frozen=True)
class Product:
    a: LinearForm
    b: LinearForm
    weight: float

    def evaluate(self, assignment) -> float:
        return self.weight * self.a.evaluate(assignment) * self.b.evaluate(assignment)

    def refs(self):
        return self.a.refs() + self.b.refs()


Term = Linear | Product


@dataclass(frozen=True)
class EnergyModel:
    """The compiler's meaning-carrier. Nothing else defines what a program means."""
    variables: tuple[Var, ...]
    terms: tuple[Term, ...]
    beta: float = 1.0

    def var(self, name: str) -> Var:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def energy(self, assignment: Mapping[str, int]) -> float:
        return sum(t.evaluate(assignment) for t in self.terms)
