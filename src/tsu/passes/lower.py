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


def _assert_pairwise(target_raw, names, spin_list):
    """Raise ThreeBodyError if ANY monomial of the RAW (pre-reduction) expanded
    polynomial has total degree > 2 -- checked via sympy.Poly over the whole
    expression, not a per-triple `.coeff()` probe. A per-triple scan
    (`target.coeff(s_i*s_j*s_k)` for each combination of 3 distinct variable
    names) is blind to a monomial like s_a**3*s_b: `.coeff()` returns exactly 0
    for that pattern, so the term passes through silently. Checking on the RAW
    expansion (before folding s**2 -> 1) also matters: reducing first would
    collapse a genuine violation like s_a**3 down to s_a and hide it entirely.
    """
    if not spin_list:
        return
    poly = sp.Poly(target_raw, *spin_list)
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

    # Reject any surviving term of degree > 2 on the RAW expansion -- catches
    # every shape: cross terms among 3+ distinct spins AND odd powers of a
    # single spin (s**3 etc.), neither of which a per-triple coeff scan sees.
    _assert_pairwise(target_raw, names, spin_list)

    # binary spins: s**2 == 1. Needed to canonicalize same-spin repeats (e.g. a
    # self-product occ_a*occ_a) into a proper linear/constant term, now that the
    # pairwise check above has already run on the unreduced expansion.
    target = _reduce_spin_powers(target_raw, spin_list)

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
