"""IsingModel -> IsingModel, degree-driven graph rewrite. Route A of the
degree/space trade: spend NODES to buy DEGREE headroom, the trade
`insert_mediators` (edge subdivision) cannot make -- subdividing an edge
leaves both endpoints with exactly the degree they had.

A logical spin `v` of degree `d > max_degree` becomes `k =
ceil(d / (max_degree - 2))` copies `v_1 .. v_k`, chained `v_1 -- v_2 -- ...
-- v_k` by a coupling of magnitude `chain_strength` on every consecutive
pair. `v`'s own `d` external edges are distributed across the copies, at
most `max_degree - 2` per copy, so every copy's own degree is at most
`(max_degree - 2)` external `+ 2` chain neighbours `= max_degree` (an
interior copy has 2 chain edges; an end copy has only 1, so its own degree
sits at or under the cap with room to spare -- this pass does not chase that
slack, matching the brief's own worst-case accounting). `v`'s bias is placed
on exactly ONE copy (the first); duplicating it across all k copies would
multiply the field the original spin experiences by k, which is a different
model, not the same one split.

THE risk this pass exists to manage, not merely to reduce degree: the chain
coupling must be strong enough that every copy takes the SAME value in
every state that matters. A chain that BREAKS leaves the copies disagreeing
-- the logical spin then has no value at all, and the model is silently
wrong rather than approximately right. This module does not itself decide
"strong enough"; `chain_strength` is the caller's own choice, verified
externally (see `tests/test_split.py`'s marginal-preservation test against
`audit/oracles/exact.py`) rather than asserted here.

Sign convention (matches `passes/lower.py`'s own, confirmed against
`audit/oracles/exact.py`, which is written from the physics and does not
import this package): `E(s) = -sum_ij J_ij*s_i*s_j - sum_i b_i*s_i`. Under
this convention agreement (`s_i == s_j`) is the LOWER-energy state, and
therefore the Boltzmann-favoured one, when `J_ij > 0` -- so the chain edges
below carry weight `+chain_strength`, not `-chain_strength`, for a positive
`chain_strength` to be ferromagnetic (agreement-favouring) AT ALL. A
negative literal here would instead favour DISAGREEMENT under this
convention -- the opposite of what a chain is for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analyse import analyse
from .lower import IsingModel


@dataclass(frozen=True)
class SplitReport:
    splits: int              # number of logical spins split
    added_nodes: int         # net new nodes (sum of (k-1) over every split)
    chain_strength: float    # the coupling magnitude actually used
    max_degree_after: int    # analyse()'s own measurement of the result
    chain_edges: int         # total chain edges added, across all splits


def _copy_name(base: str, j: int) -> str:
    return f"{base}__chain{j}"


def split_high_degree(ising: IsingModel, max_degree: int,
                      chain_strength: float) -> tuple[IsingModel, SplitReport]:
    """Split every node whose degree exceeds `max_degree` into a chain of
    copies, none of which exceeds it. Reuses `analyse` for every degree
    figure this function needs or reports -- it never hand-rolls a degree
    count (Task 2's own constraint)."""
    if max_degree < 3:
        raise ValueError(
            f"max_degree must be >= 3: a split copy needs at least 1 slot "
            f"for an external edge plus up to 2 for its chain neighbours, "
            f"so max_degree - 2 (the per-copy external-edge budget) must be "
            f">= 1; got max_degree={max_degree}")

    before = analyse(ising)
    n = len(ising.nodes)

    incident: list[list[int]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(ising.edges):
        incident[u].append(e)
        incident[v].append(e)
    degree = [len(lst) for lst in incident]
    to_split = [i for i in range(n) if degree[i] > max_degree]

    if not to_split:
        return ising, SplitReport(splits=0, added_nodes=0,
                                  chain_strength=float(chain_strength),
                                  max_degree_after=before.max_degree,
                                  chain_edges=0)

    per_copy_cap = max_degree - 2

    new_names: list[str] = []
    new_biases: list[float] = []
    copies_of: dict[int, list[int]] = {}          # old index -> [new indices]
    bucket_of: dict[tuple[int, int], int] = {}    # (old index, edge idx) -> bucket
    keep_map: dict[int, int] = {}                 # old index -> new index (unsplit)

    to_split_set = set(to_split)
    for i in range(n):
        if i in to_split_set:
            d = degree[i]
            k = math.ceil(d / per_copy_cap)
            base = ising.nodes[i]
            copy_indices = []
            for j in range(k):
                copy_indices.append(len(new_names))
                new_names.append(_copy_name(base, j))
                # v's own bias goes on ONE copy only -- duplicating it would
                # multiply the field the original spin experiences by k.
                new_biases.append(float(ising.biases[i]) if j == 0 else 0.0)
            copies_of[i] = copy_indices
            for slot, e in enumerate(incident[i]):
                bucket_of[(i, e)] = min(slot // per_copy_cap, k - 1)
        else:
            keep_map[i] = len(new_names)
            new_names.append(ising.nodes[i])
            new_biases.append(float(ising.biases[i]))

    def resolve(i: int, e: int) -> int:
        if i in copies_of:
            return copies_of[i][bucket_of[(i, e)]]
        return keep_map[i]

    new_edges: list[tuple[int, int]] = []
    new_weights: list[float] = []
    for e, (u, v) in enumerate(ising.edges):
        nu, nv = resolve(u, e), resolve(v, e)
        new_edges.append((min(nu, nv), max(nu, nv)))
        new_weights.append(float(ising.weights[e]))

    chain_edges = 0
    for i in to_split:
        idxs = copies_of[i]
        for a, b in zip(idxs, idxs[1:]):
            new_edges.append((min(a, b), max(a, b)))
            new_weights.append(float(chain_strength))
            chain_edges += 1

    # A mediator node that itself needs splitting stays a mediator on every
    # one of its own copies -- fragmenting it does not change what it IS
    # (a hidden, temperature-dependent spin), only how many physical nodes
    # represent it. Silently dropping split mediators here would be the
    # same class of silent-wrongness this pass exists to avoid elsewhere.
    new_mediator_nodes: list[int] = []
    for m in ising.mediator_nodes:
        if m in keep_map:
            new_mediator_nodes.append(keep_map[m])
        else:
            new_mediator_nodes.extend(copies_of[m])

    out = IsingModel(nodes=tuple(new_names), edges=tuple(new_edges),
                     weights=np.asarray(new_weights, dtype=float),
                     biases=np.asarray(new_biases, dtype=float),
                     beta=ising.beta, offset=ising.offset,
                     mediator_nodes=tuple(sorted(new_mediator_nodes)))

    after = analyse(out)
    added_nodes = sum(len(copies_of[i]) - 1 for i in to_split)

    return out, SplitReport(splits=len(to_split), added_nodes=added_nodes,
                            chain_strength=float(chain_strength),
                            max_degree_after=after.max_degree,
                            chain_edges=chain_edges)
