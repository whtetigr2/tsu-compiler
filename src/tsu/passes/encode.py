"""WorkloadSpec -> EnergyModel. Representation choice lives HERE, not in the spec.

Domain-wall (Chancellor 2019) for categoricals: k-1 spins on a PATH, so the
constraint graph is bipartite and needs no mediators (EXP-GK6). The value v is
encoded as n_j = 1 for j < v, so the indicator

    delta_v = n_{v-1} - n_v        (n_{-1} == 1, n_{k-1} == 0)

is a LINEAR form -- which is why this encoding fits an IR built on linear forms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..ir import (Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef)
from ..spec import WorkloadSpec

MONOTONE_PENALTY = 10.0

# The monotonicity penalty must dominate the workload's own term weights, or an
# illegal (non-monotone) chain state can become energetically favourable and the
# sampler will silently return samples that decode to nothing meaningful -- a wrong
# answer with no diagnostic, which is the exact failure class this compiler exists
# to prevent (see encode()'s guard below). A factor of 2 is the stated floor: it
# guarantees the deterministic per-link cost of a monotonicity violation
# (MONOTONE_PENALTY) always outweighs what any single term in the spec could gain
# by exploiting that violation. This is a first-order heuristic, not a certified
# bound over interactions between multiple concurrent terms -- workloads that stack
# many large-weight terms against the same chain should still check the legal-vs-
# illegal energy gap directly, the way toy.yaml's own test does.
#
# C5 (final review): the guard used to compare MONOTONE_PENALTY against
# `abs(term.weight)` alone, which is not a term's energy scale when its
# LinearForm carries large coefficients -- Product(LinearForm({c=0: 20.0}),
# LinearForm({c=2: 20.0}), -1.0) has weight 1.0 (guard sees "1.0, accept") but
# an actual maximum contribution of 1.0 * 20 * 20 = 400 (guard should reject).
# `_max_energy_contribution` computes the real bound: a LinearForm's maximum
# attainable |value| over any binary assignment is |const| + sum(|coeff|) (every
# coefficient's variable independently maxes out at 0 or 1); a Product's is the
# product of its two forms' bounds times |weight|; a Linear's is its form's bound
# times |weight|.
MONOTONE_MARGIN_FACTOR = 2.0


def _max_abs_form(form: LinearForm) -> float:
    """Maximum |value| a LinearForm can attain over any binary assignment: each
    coefficient contributes at most its own magnitude (an occupancy or a
    categorical indicator each range over {0, 1}), so the bound is |const| plus
    the sum of |coeff| across every term in the form."""
    return abs(form.const) + sum(abs(w) for w in form.coeffs.values())


def _max_energy_contribution(term) -> float:
    """The maximum |energy| a single spec term can contribute over any binary
    assignment -- the quantity MONOTONE_PENALTY must actually dominate (C5)."""
    if isinstance(term, Product):
        return abs(term.weight) * _max_abs_form(term.a) * _max_abs_form(term.b)
    if isinstance(term, Linear):
        return abs(term.weight) * _max_abs_form(term.form)
    raise ValueError(f"unknown term type {type(term).__name__}")


@dataclass(frozen=True)
class Encoded:
    model: EnergyModel
    binary_names: tuple[str, ...]
    categorical: Mapping[str, tuple[str, ...]]   # logical name -> chain spin names

    def encode_assignment(self, asg: Mapping[str, int]) -> dict[str, int]:
        out = {}
        for name, chain in self.categorical.items():
            v = asg[name]
            for j, spin in enumerate(chain):
                out[spin] = 1 if j < v else 0
        for n in self.binary_names:
            if n not in out:
                out[n] = int(asg[n])
        return out

    def decode(self, bits: Mapping[str, int]) -> dict[str, int]:
        out = {}
        for name, chain in self.categorical.items():
            out[name] = sum(int(bits[s]) for s in chain)
        for n in self.binary_names:
            if not any(n in chain for chain in self.categorical.values()):
                out[n] = int(bits[n])
        return out

    def is_codeword(self, bits: Mapping[str, int]) -> bool:
        """True iff every domain-wall chain in `bits` is monotone (n_{j-1} >=
        n_j: once a chain bit is 0, every later bit in the chain must be 0 too).

        `decode` is a PROJECTION (`sum(bits[s] for s in chain)`), not an inverse:
        handed a non-monotone chain -- not a valid codeword -- it still returns a
        legal-looking categorical value with no error and no flag (C5). A caller
        that needs to know whether the sample it is about to decode is actually
        meaningful must check this first; `decode`'s own output gives no signal
        either way.
        """
        for chain in self.categorical.values():
            for j in range(len(chain) - 1):
                if int(bits[chain[j]]) == 0 and int(bits[chain[j + 1]]) == 1:
                    return False
        return True


def _chain_names(name: str, k: int) -> tuple[str, ...]:
    return tuple(f"{name}__dw{j}" for j in range(k - 1))


def _rewrite_form(form: LinearForm, categorical) -> LinearForm:
    """Replace categorical indicators with their domain-wall linear expression."""
    coeffs, const = {}, form.const
    for ref, w in form.coeffs.items():
        if ref.value is None:
            coeffs[ref] = coeffs.get(ref, 0.0) + w
            continue
        chain = categorical[ref.name]
        v, k = ref.value, len(chain) + 1
        # delta_v = n_{v-1} - n_v, with n_{-1}=1 and n_{k-1}=0
        if v == 0:
            const += w                                   # 1 - n_0
            r = VarRef(chain[0])
            coeffs[r] = coeffs.get(r, 0.0) - w
        elif v == k - 1:
            r = VarRef(chain[v - 1])                     # n_{k-2}
            coeffs[r] = coeffs.get(r, 0.0) + w
        else:
            lo, hi = VarRef(chain[v - 1]), VarRef(chain[v])
            coeffs[lo] = coeffs.get(lo, 0.0) + w
            coeffs[hi] = coeffs.get(hi, 0.0) - w
    return LinearForm(coeffs, const)


def encode(spec: WorkloadSpec, encoding: str = "domain_wall") -> Encoded:
    if encoding != "domain_wall":
        raise ValueError(
            f"encoding {encoding!r} is not implemented in the vertical slice; "
            f"only 'domain_wall' is available")

    declared_names = {v.name for v in spec.variables}

    categorical, variables = {}, []
    for v in spec.variables:
        if isinstance(v.domain, Binary):
            variables.append(v)
        else:
            chain = _chain_names(v.name, v.domain.k)
            for cn in chain:
                if cn in declared_names:
                    raise ValueError(
                        f"generated domain-wall spin name {cn!r} for categorical "
                        f"{v.name!r} collides with a declared variable of the same "
                        f"name; the '__dw<j>' suffix is reserved for domain-wall "
                        f"chain spins -- rename the declared variable {cn!r}")
            categorical[v.name] = chain
            variables.extend(Var(n, Binary()) for n in chain)
    variables = tuple(variables)

    if categorical:
        max_contribution = max(
            (_max_energy_contribution(t) for t in spec.terms), default=0.0)
        if MONOTONE_PENALTY <= MONOTONE_MARGIN_FACTOR * max_contribution:
            raise ValueError(
                f"MONOTONE_PENALTY ({MONOTONE_PENALTY}) does not exceed "
                f"{MONOTONE_MARGIN_FACTOR}x the workload's largest term's maximum "
                f"energy contribution ({max_contribution}); the monotonicity "
                f"penalty must dominate the workload's own weights or an illegal "
                f"(non-monotone) chain state can become energetically favourable")

    terms = []
    for t in spec.terms:
        if isinstance(t, Linear):
            terms.append(Linear(_rewrite_form(t.form, categorical), t.weight))
        elif isinstance(t, Product):
            terms.append(Product(_rewrite_form(t.a, categorical),
                                 _rewrite_form(t.b, categorical), t.weight))
        else:
            raise ValueError(f"unknown term type {type(t).__name__}")

    # monotonicity: penalise n_j = 0 while n_{j+1} = 1, i.e. (1 - n_j) * n_{j+1}
    for chain in categorical.values():
        for j in range(len(chain) - 1):
            terms.append(Product(
                LinearForm({VarRef(chain[j]): -1.0}, const=1.0),
                LinearForm({VarRef(chain[j + 1]): 1.0}),
                MONOTONE_PENALTY))

    return Encoded(
        model=EnergyModel(variables, tuple(terms), spec_beta(spec)),
        binary_names=tuple(v.name for v in variables),
        categorical=categorical,
    )


def spec_beta(spec: WorkloadSpec) -> float:
    return 1.0
