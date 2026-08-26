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


def _reduce_spin_powers(expr, spin_list):
    """Fold every spin exponent mod 2: s**k -> s for odd k, s**k -> 1 for even k
    (k > 0), because a binary spin satisfies s**2 == 1. Done by walking the fully
    expanded polynomial's monomials via sympy.Poly and reducing each exponent mod
    2 in one pass -- NOT a single `.subs(s**2, 1)`, which only cancels even powers
    and leaves e.g. s**3 (odd, >= 3) untouched. Reduced monomials that collide
    (e.g. s**3 and s**1 both reduce to s**1) have their coefficients summed.

    This is algebra, not a violation check: for a binary spin s**3 == s (a
    legitimate degree-1 term) and s**4 == 1 (folds into the constant). A high
    power of a SINGLE spin is always reducible and never a pairwise violation --
    only a product across three or more DISTINCT spins is. That distinction is
    why this reduction must run before `_assert_pairwise`, not after.
    """
    if not spin_list:
        return sp.expand(expr)
    poly = sp.Poly(sp.expand(expr), *spin_list)
    collected: dict[tuple[int, ...], sp.Expr] = {}
    for monom, coeff in poly.terms():
        reduced_monom = tuple(e % 2 for e in monom)
        collected[reduced_monom] = collected.get(reduced_monom, sp.Integer(0)) + coeff
    result = sp.Integer(0)
    for monom, coeff in collected.items():
        term = coeff
        for s, e in zip(spin_list, monom):
            if e:
                term *= s
        result += term
    return sp.expand(result)


def _assert_pairwise(target, names, spin_list):
    """Raise ThreeBodyError if ANY monomial of the REDUCED polynomial (every spin
    exponent already folded to 0 or 1 by `_reduce_spin_powers`) has total degree
    > 2 -- checked via sympy.Poly over the whole expression, not a per-triple
    `.coeff()` probe. A per-triple scan (`target.coeff(s_i*s_j*s_k)` for each
    combination of 3 distinct variable names) is blind to a monomial like
    s_a*s_b*s_c*s_d (order 4): `.coeff()` returns exactly 0 for that pattern, so
    the term would pass through silently.

    This MUST run on the reduced expression, not the raw pre-reduction one. In
    the reduced form every exponent is 0 or 1, so a monomial's total degree
    equals the number of DISTINCT spins it couples -- exactly what "pairwise"
    means. A high power of a single spin (s_a**3, s_a**4, ...) is not a
    violation: s**2 == 1 for a binary spin, so s_a**3 == s_a (a legitimate
    degree-1 bias term) and s_a**4 == 1 (folds into the constant offset).
    Checking on the raw, unreduced expansion would misclassify both as
    "order > 2" and reject a perfectly good pairwise (or lower) model -- the
    same shape of false positive this module exists to prevent (see the
    module docstring's "Z1 adds up to 7" story). Only a monomial that still
    couples 3+ distinct spins AFTER reduction is a genuine pairwise violation.
    """
    if not spin_list:
        return
    poly = sp.Poly(target, *spin_list)
    for monom, coeff in poly.terms():
        deg = sum(monom)
        if deg > 2 and sp.simplify(coeff) != 0:
            offending = " * ".join(
                f"s_{n}**{e}" if e > 1 else f"s_{n}"
                for n, e in zip(names, monom) if e > 0)
            raise ThreeBodyError(
                f"order-{deg} term {offending} survived expansion with "
                f"coefficient {coeff}; the model is not pairwise and cannot be "
                f"placed on a pairwise target")


def lower(model: EnergyModel) -> IsingModel:
    for v in model.variables:
        if not isinstance(v.domain, Binary):
            raise ValueError(
                f"variable {v.name!r} is not binary; run `encode` before `lower`")

    names = tuple(v.name for v in model.variables)
    spins = {n: sp.Symbol(f"s_{n}") for n in names}
    spin_list = [spins[n] for n in names]
    # occupancy n = (s + 1) / 2
    occ = {n: (spins[n] + 1) / 2 for n in names}

    E = sp.Integer(0)
    for t in model.terms:
        E += _term_expr(t, occ)
    E = sp.expand(E)

    # sum b s + sum J s s == -E
    target_raw = sp.expand(-E)

    # binary spins: s**2 == 1 (s**k -> s odd, -> 1 even). Must happen BEFORE the
    # pairwise check: a high power of a single spin (s_a**3, s_a**4, ...) is
    # legitimate algebra, not a violation, and only resolves to its true degree
    # (1, or 0 folded into the constant) once reduced.
    target = _reduce_spin_powers(target_raw, spin_list)

    # Reject any surviving term of degree > 2 on the REDUCED expression -- catches
    # every genuine violation shape (any product across 3+ distinct spins,
    # regardless of how many variables), without misflagging a reducible high
    # power of a single spin.
    _assert_pairwise(target, names, spin_list)

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
