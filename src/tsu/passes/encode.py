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

    categorical, variables = {}, []
    for v in spec.variables:
        if isinstance(v.domain, Binary):
            variables.append(v)
        else:
            chain = _chain_names(v.name, v.domain.k)
            categorical[v.name] = chain
            variables.extend(Var(n, Binary()) for n in chain)
    variables = tuple(variables)

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
