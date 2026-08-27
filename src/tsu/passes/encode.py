"""WorkloadSpec -> EnergyModel. Representation choice lives HERE, not in the spec.

Two encodings for a categorical variable, so the compiler can COMPARE
representations rather than assert a preference (A5 searches over both):

Domain-wall (Chancellor 2019): k-1 spins on a PATH, so the constraint graph is
bipartite and needs no mediators (EXP-GK6). The value v is encoded as
n_j = 1 for j < v, so the indicator

    delta_v = n_{v-1} - n_v        (n_{-1} == 1, n_{k-1} == 0)

is a LINEAR form -- which is why this encoding fits an IR built on linear forms.

One-hot: k spins, one per value, exactly one of which is 1. The indicator for
value v is simply VarRef(x_v) -- a single spin, linear by construction, no chain
algebra needed. Exactly-one is enforced by a squared-linear-form penalty
P*(sum_v x_v - 1)**2, itself a Product(L, L, P). This is expected to produce a
CLIQUE per categorical variable (every pair of its k spins interacts through
that squared form) and therefore a non-bipartite graph -- that is the honest
structural cost of this representation, reported rather than avoided.
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

# The one-hot exactly-one penalty must dominate the workload's own term weights
# for exactly the same reason MONOTONE_PENALTY must (see above): an illegal
# (not-exactly-one) pattern must never become energetically favourable. Reuses
# the SAME `_max_energy_contribution` bound and the SAME margin factor -- a
# second, weaker dominance computation is exactly what the brief (A4) forbids.
ONE_HOT_PENALTY = 10.0


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
    encoding: str = "domain_wall"

    def encode_assignment(self, asg: Mapping[str, int]) -> dict[str, int]:
        out = {}
        for name, chain in self.categorical.items():
            v = asg[name]
            if self.encoding == "domain_wall":
                for j, spin in enumerate(chain):
                    out[spin] = 1 if j < v else 0
            elif self.encoding == "one_hot":
                for j, spin in enumerate(chain):
                    out[spin] = 1 if j == v else 0
            else:
                raise ValueError(f"unknown encoding {self.encoding!r}")
        for n in self.binary_names:
            if n not in out:
                out[n] = int(asg[n])
        return out

    def decode(self, bits: Mapping[str, int]) -> dict[str, int]:
        """Domain-wall: a PROJECTION (`sum(bits[s] for s in chain)`), not an
        inverse -- handed a non-monotone (invalid) chain it still returns a
        legal-looking value with no error (C5); `is_codeword` is the required
        pre-check.

        One-hot: reads "the index of the single 1" literally. A pattern
        without exactly one 1 is not a codeword and has no well-defined index
        to report -- this raises rather than repairing it via an argmax-style
        guess, per A4. Call `is_codeword` first; every caller in this
        compiler already does.
        """
        out = {}
        for name, chain in self.categorical.items():
            if self.encoding == "domain_wall":
                out[name] = sum(int(bits[s]) for s in chain)
            elif self.encoding == "one_hot":
                ones = [j for j, s in enumerate(chain) if int(bits[s]) == 1]
                if len(ones) != 1:
                    raise ValueError(
                        f"one-hot decode: variable {name!r} has {len(ones)} "
                        f"spins set to 1 (expected exactly 1); this is not a "
                        f"codeword -- call is_codeword() first rather than "
                        f"decoding a pattern with no well-defined value")
                out[name] = ones[0]
            else:
                raise ValueError(f"unknown encoding {self.encoding!r}")
        for n in self.binary_names:
            if not any(n in chain for chain in self.categorical.values()):
                out[n] = int(bits[n])
        return out

    def is_codeword(self, bits: Mapping[str, int]) -> bool:
        """True iff every categorical's chain in `bits` is a valid codeword for
        this encoding: domain-wall requires monotonicity (n_{j-1} >= n_j); one-
        hot requires exactly one spin set to 1. Neither `decode` above can be
        trusted without this check first -- see their docstring.
        """
        for chain in self.categorical.values():
            if self.encoding == "domain_wall":
                for j in range(len(chain) - 1):
                    if int(bits[chain[j]]) == 0 and int(bits[chain[j + 1]]) == 1:
                        return False
            elif self.encoding == "one_hot":
                if sum(int(bits[s]) for s in chain) != 1:
                    return False
            else:
                raise ValueError(f"unknown encoding {self.encoding!r}")
        return True


def _chain_names_domain_wall(name: str, k: int) -> tuple[str, ...]:
    return tuple(f"{name}__dw{j}" for j in range(k - 1))


def _chain_names_one_hot(name: str, k: int) -> tuple[str, ...]:
    return tuple(f"{name}__oh{j}" for j in range(k))


def _rewrite_form_domain_wall(form: LinearForm, categorical) -> LinearForm:
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


def _rewrite_form_one_hot(form: LinearForm, categorical) -> LinearForm:
    """Replace categorical indicators with their one-hot linear expression: the
    indicator for value v is exactly VarRef(chain[v]) -- a single spin, weight
    1, no chain algebra."""
    coeffs, const = {}, form.const
    for ref, w in form.coeffs.items():
        if ref.value is None:
            coeffs[ref] = coeffs.get(ref, 0.0) + w
            continue
        chain = categorical[ref.name]
        r = VarRef(chain[ref.value])
        coeffs[r] = coeffs.get(r, 0.0) + w
    return LinearForm(coeffs, const)


_REWRITE = {"domain_wall": _rewrite_form_domain_wall, "one_hot": _rewrite_form_one_hot}
_CHAIN_NAMES = {"domain_wall": _chain_names_domain_wall, "one_hot": _chain_names_one_hot}


def _build_categorical(spec: WorkloadSpec, encoding: str):
    """Chain-spin generation and collision guard, shared by both encodings
    (only the naming suffix and spin count per categorical differ)."""
    suffix = {"domain_wall": "__dw<j>", "one_hot": "__oh<j>"}[encoding]
    declared_names = {v.name for v in spec.variables}
    chain_names = _CHAIN_NAMES[encoding]

    categorical, variables = {}, []
    for v in spec.variables:
        if isinstance(v.domain, Binary):
            variables.append(v)
        else:
            chain = chain_names(v.name, v.domain.k)
            for cn in chain:
                if cn in declared_names:
                    raise ValueError(
                        f"generated {encoding} spin name {cn!r} for categorical "
                        f"{v.name!r} collides with a declared variable of the same "
                        f"name; the {suffix!r} suffix is reserved for {encoding} "
                        f"chain spins -- rename the declared variable {cn!r}")
            categorical[v.name] = chain
            variables.extend(Var(n, Binary()) for n in chain)
    return categorical, tuple(variables)


def _guard_penalty_dominates(spec: WorkloadSpec, categorical, penalty: float,
                             penalty_name: str) -> None:
    """The SAME dominance check both encodings need: their structural penalty
    (monotonicity for domain-wall, exactly-one for one-hot) must exceed
    MONOTONE_MARGIN_FACTOR times the workload's own largest term contribution,
    or an illegal chain state can become energetically favourable. Reuses
    `_max_energy_contribution` -- a second, weaker computation is exactly what
    the brief (A4) forbids."""
    if not categorical:
        return
    max_contribution = max(
        (_max_energy_contribution(t) for t in spec.terms), default=0.0)
    if penalty <= MONOTONE_MARGIN_FACTOR * max_contribution:
        raise ValueError(
            f"{penalty_name} ({penalty}) does not exceed "
            f"{MONOTONE_MARGIN_FACTOR}x the workload's largest term's maximum "
            f"energy contribution ({max_contribution}); the {penalty_name} "
            f"must dominate the workload's own weights or an illegal chain "
            f"state can become energetically favourable")


def _rewrite_terms(spec: WorkloadSpec, categorical, encoding: str):
    rewrite = _REWRITE[encoding]
    terms = []
    for t in spec.terms:
        if isinstance(t, Linear):
            terms.append(Linear(rewrite(t.form, categorical), t.weight))
        elif isinstance(t, Product):
            terms.append(Product(rewrite(t.a, categorical),
                                 rewrite(t.b, categorical), t.weight))
        else:
            raise ValueError(f"unknown term type {type(t).__name__}")
    return terms


def _encode_domain_wall(spec: WorkloadSpec) -> Encoded:
    categorical, variables = _build_categorical(spec, "domain_wall")
    _guard_penalty_dominates(spec, categorical, MONOTONE_PENALTY, "MONOTONE_PENALTY")

    terms = _rewrite_terms(spec, categorical, "domain_wall")

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
        categorical=categorical, encoding="domain_wall")


def _encode_one_hot(spec: WorkloadSpec) -> Encoded:
    categorical, variables = _build_categorical(spec, "one_hot")
    _guard_penalty_dominates(spec, categorical, ONE_HOT_PENALTY, "ONE_HOT_PENALTY")

    terms = _rewrite_terms(spec, categorical, "one_hot")

    # exactly-one: P * (sum_v x_v - 1)**2 == Product(L, L, P)
    for chain in categorical.values():
        L = LinearForm({VarRef(s): 1.0 for s in chain}, const=-1.0)
        terms.append(Product(L, L, ONE_HOT_PENALTY))

    return Encoded(
        model=EnergyModel(variables, tuple(terms), spec_beta(spec)),
        binary_names=tuple(v.name for v in variables),
        categorical=categorical, encoding="one_hot")


_ENCODERS = {"domain_wall": _encode_domain_wall, "one_hot": _encode_one_hot}


def encode(spec: WorkloadSpec, encoding: str = "domain_wall") -> Encoded:
    if encoding not in _ENCODERS:
        raise ValueError(
            f"unknown encoding {encoding!r}; available: {sorted(_ENCODERS)}")
    return _ENCODERS[encoding](spec)


def spec_beta(spec: WorkloadSpec) -> float:
    return 1.0
