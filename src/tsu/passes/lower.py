"""EnergyModel -> IsingModel. One shared symbolic expansion, never per-workload algebra.

A hand-written coefficient with a stray 1/2 voided EXP-TA1. A global quadratic
penalty produced a false "Z1 adds up to 7" hardware wall. Both are structurally
prevented here: the algebra is derived by sympy, and any surviving term of order > 2
raises rather than being dropped.

Convention: the IR means E(x). thrml means E_thrml = -beta(sum b s + sum J s s).
Therefore  sum b s + sum J s s  ==  -E(x).  The sign flip is applied ONCE, here.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import sympy as sp

from ..ir import Binary, EnergyModel, Linear, Product


class ThreeBodyError(ValueError):
    """A term of order > 2 survived expansion. The model is not pairwise."""


@dataclass(frozen=True)
class IsingModel:
    nodes: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    weights: np.ndarray      # J, aligned with edges
    biases: np.ndarray       # b, aligned with nodes
    beta: float
    offset: float            # constant energy, carried so E(x) is reproducible


def _term_expr(term, sym):
    if hasattr(term, "sympy_expr"):
        return term.weight * term.sympy_expr(sym)

    def form(f):
        e = sp.Float(f.const)
        for ref, w in f.coeffs.items():
            if ref.value is not None:
                raise ValueError(
                    "a categorical indicator reached `lower`; run `encode` first")
            e += sp.Float(w) * sym[ref.name]
        return e

    if isinstance(term, Linear):
        return sp.Float(term.weight) * form(term.form)
    if isinstance(term, Product):
        return sp.Float(term.weight) * form(term.a) * form(term.b)
    raise ValueError(f"unknown term type {type(term).__name__}")


def lower(model: EnergyModel) -> IsingModel:
    for v in model.variables:
        if not isinstance(v.domain, Binary):
            raise ValueError(
                f"variable {v.name!r} is not binary; run `encode` before `lower`")

    names = tuple(v.name for v in model.variables)
    spins = {n: sp.Symbol(f"s_{n}") for n in names}
    # occupancy n = (s + 1) / 2
    occ = {n: (spins[n] + 1) / 2 for n in names}

    E = sp.Integer(0)
    for t in model.terms:
        E += _term_expr(t, occ)
    E = sp.expand(E)

    # binary spins: s^2 == 1
    for n in names:
        E = E.subs(spins[n] ** 2, 1)
    E = sp.expand(E)

    # sum b s + sum J s s == -E
    target = sp.expand(-E)

    for tri in itertools.combinations(names, 3):
        c = target.coeff(spins[tri[0]] * spins[tri[1]] * spins[tri[2]])
        if sp.simplify(c) != 0:
            raise ThreeBodyError(
                f"order-3 term on {tri} survived expansion with coefficient {c}; "
                f"the model is not pairwise and cannot be placed on a pairwise target")

    idx = {n: i for i, n in enumerate(names)}
    edges, weights = [], []
    for a, b in itertools.combinations(names, 2):
        c = float(target.coeff(spins[a] * spins[b]))
        if c != 0.0:
            edges.append((idx[a], idx[b]))
            weights.append(c)

    biases = np.zeros(len(names))
    for n in names:
        e = target
        for other in names:
            if other != n:
                e = e.subs(spins[other], 0)
        biases[idx[n]] = float(sp.expand(e).coeff(spins[n]))

    const = target
    for n in names:
        const = const.subs(spins[n], 0)
    # offset makes E(x) reconstructible: E = -(b.s + sJs) + offset_correction
    offset = float(-sp.expand(const))

    return IsingModel(nodes=names, edges=tuple(edges),
                      weights=np.asarray(weights, dtype=float),
                      biases=biases, beta=model.beta, offset=offset)
