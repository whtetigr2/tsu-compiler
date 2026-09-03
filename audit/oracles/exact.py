"""audit/oracles/exact.py -- brute-force Boltzmann oracle, independent of src/tsu.

Plan Task A4: "Write the energy from the physics ... and do NOT import
src/tsu to compute it. An oracle sharing code with the thing under test
verifies nothing." Accordingly this module imports nothing from `tsu` --
every line below is written from the definition of a classical Ising model,
not by calling (or copying) the compiler's own `energy_of_draw`,
`tsu.passes.search._from_ising`, or `tsu.passes.lower`.

Convention (stated once, used consistently throughout this module):

    E(s) = -sum_{(i,j)} J_ij * s_i * s_j  -  sum_i b_i * s_i,   s_i in {-1, +1}
    p(s) ∝ exp(-beta * E(s))

This is the same sign convention `src/tsu/passes/lower.py`'s own module
docstring states ("sum b s + sum J s s == -E(x)") -- confirmed by reading
that file, not by importing it. Any state may be handed in as a {0,1}
occupancy vector (the compiler's own convention, spin = 2*occupancy - 1) or
as a {-1,+1} spin vector directly; both are accepted and treated identically.
"""
from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

Coupling = Mapping[tuple[int, int], float]


def _to_spins(state: Sequence[int]) -> list[int]:
    """Normalise a state to {-1,+1}. Accepts {0,1} (occupancy) or {-1,+1}
    (spin) entries, mixed or uniform; rejects anything else rather than
    silently coercing a malformed value."""
    spins = []
    for x in state:
        if x in (0, 1):
            spins.append(2 * x - 1)
        elif x in (-1, 1):
            spins.append(x)
        else:
            raise ValueError(
                f"state entries must be 0/1 (occupancy) or -1/+1 (spin), got {x!r}")
    return spins


def exact_energy(state: Sequence[int], J: Coupling, b: Sequence[float]) -> float:
    """E(s) = -sum J_ij*s_i*s_j - sum b_i*s_i, independent of src/tsu.

    `J` maps an (i, j) index pair to its coupling strength; only pairs
    present in `J` contribute (an absent pair is treated as J_ij=0, not an
    error -- callers pass only the edges that exist). `b` is the full bias
    vector, one entry per spin, so `len(b)` fixes the number of spins.
    """
    s = _to_spins(state)
    total = 0.0
    for (i, j), Jij in J.items():
        total -= Jij * s[i] * s[j]
    for i, bi in enumerate(b):
        total -= bi * s[i]
    return total


def exact_boltzmann(J: Coupling, b: Sequence[float], beta: float
                    ) -> tuple[list[tuple[int, ...]], list[float]]:
    """Brute-force enumeration of p(s) ∝ exp(-beta*E(s)) over all 2**n
    configurations, n = len(b). Returns (states, probs): `states` are
    {0,1}-valued tuples (the compiler's own occupancy convention, so a
    caller can compare directly against a decoded draw) in a fixed
    enumeration order, and `probs` sums to 1 to floating-point precision
    (normalised by the partition function Z = sum_s exp(-beta*E(s))).

    Exact and independent of src/tsu -- suitable only for n small enough to
    enumerate (a handful of spins); this is a verification oracle, not a
    sampler.
    """
    n = len(b)
    if n == 0:
        raise ValueError("b must be non-empty: exact_boltzmann needs at least one spin")
    states = list(itertools.product((0, 1), repeat=n))
    energies = [exact_energy(s, J, b) for s in states]
    weights = [math.exp(-beta * e) for e in energies]
    Z = sum(weights)
    if Z == 0.0 or not math.isfinite(Z):
        raise ValueError(
            f"partition function is not usable (Z={Z}); beta/J/b produced "
            f"weights that under/overflowed or summed to zero")
    probs = [w / Z for w in weights]
    return states, probs
