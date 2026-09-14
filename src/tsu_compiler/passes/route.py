"""Mediate couplings the substrate cannot place directly.

`insert_mediators` (Task 6, spec section 5.3) is the real implementation: it
subdivides every edge that a bipartite target cannot realize directly through a
hidden MEDIATOR spin, so a non-bipartite logical graph becomes bipartite without
changing the marginal distribution over its original spins. `place.py:181`'s own
remediation names this; `place()` calls it before ever reporting a
`parity_conflict` (spec 5.3.3/5.3.6 -- see place.py).

The mathematics (spec 5.3.2), verified by hand on a frustrated triangle
(J=-1.25 on all three edges, beta=4.0) to 3.3e-16 against the real thrml exact
distribution before this module existed:

    A = arccosh(exp(2*beta*|J|)) / (2*beta)
    couplings: (A, A) reproduces J > 0, (A, -A) reproduces J < 0
    mediator bias: 0

Derivation of the two facts this module leans on:

    sum_{m=+-1} exp(beta*A*m*s_u + beta*A2*m*s_v) = 2*cosh(beta*A*(s_u +- s_v))

reduces, at s_u==s_v, to 2*cosh(2*beta*A) and at s_u!=s_v to 2 -- matching
C*exp(beta*J*s_u*s_v) (up to the same state-independent constant C for every
configuration) exactly when 2*beta*A = arccosh(exp(2*beta*|J|)), with
C = 2*exp(beta*|J|). Folding ln(C)/beta into `offset` per mediated edge is what
keeps E(x) reconstructible after marginalizing the mediator out (a state-
independent additive shift to every configuration's energy does not, by itself,
change the induced Boltzmann distribution -- softmax is shift-invariant -- but
without it the JOINT model's own energy no longer equals the original edge's
energy plus the rest of the model, which is the reconstructibility property the
brief names).

Edge selection (spec 5.3.3): partition nodes into two sides via BFS depth
parity (each connected component's own BFS tree, root at depth 0, alternating
parity thereafter) -- O(V+E) -- then run a bounded local-search (greedy,
worklist-driven) improvement pass, also O(V+E) amortized (each flip strictly
decreases the total within-side edge count, which is bounded by |E|, so the
total number of flips across the whole pass is bounded by |E| regardless of how
many sweeps it takes). `networkx.algorithms.approximation.one_exchange` is
NOT used here -- measured on the L1 16x16 spec's 8320-edge graph, it did not
finish in 10 minutes. Correctness never depends on the cut being optimal: ANY
2-partition works, because subdividing a within-side edge through a fresh
mediator always produces two cross-side edges (the mediator takes the OPPOSITE
side from the edge's own two endpoints). A better cut only lowers the mediator
count.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..target import TargetProfile
from .analyse import GraphReport, analyse
from .lower import IsingModel


class BetaMismatchError(ValueError):
    """Raised when a model carrying mediator spins would be sampled at a beta
    other than the one its mediator couplings were computed at (spec 5.3.5).
    `A` is temperature-dependent (it is a function of `beta` as well as `J`),
    so a mediated program sampled at a different beta silently reproduces the
    WRONG couplings -- this must be refused, never silently sampled."""


@dataclass(frozen=True)
class MediationReport:
    """What `insert_mediators` actually did, kept distinct from `GraphReport`
    (which describes the LOGICAL, pre-mediation graph) so a reader can always
    tell "how many spins does the workload need" from "how many spins does
    the substrate actually see" without the two being conflated under one
    field."""
    mediator_count: int
    partition_method: str
    bipartite_after: bool
    beta_used: float


def assert_beta_consistent(ising: IsingModel, requested_beta: float) -> None:
    """Spec 5.3.5: refuse rather than silently sample. A model with no
    mediator spins (`mediator_nodes` empty) has nothing temperature-dependent
    baked into its couplings, so any beta is fine for it -- this only fires
    for a model `insert_mediators` actually touched."""
    if not ising.mediator_nodes:
        return
    if not math.isclose(float(requested_beta), float(ising.beta),
                        rel_tol=1e-9, abs_tol=1e-12):
        raise BetaMismatchError(
            f"this model carries {len(ising.mediator_nodes)} mediator "
            f"spin(s) whose couplings were computed at beta={ising.beta!r}; "
            f"sampling it at beta={requested_beta!r} would silently "
            f"reproduce the WRONG couplings (A depends on beta -- spec "
            f"5.3.5), so this is refused rather than sampled")


def _bfs_parity_colouring(n: int, adj: list[list[int]]) -> list[int]:
    """One BFS per connected component, O(V+E) total: root at parity 0,
    every neighbour the opposite parity of its discoverer. This is NOT a
    proper 2-colouring of `adj` in general (an odd cycle forces some edge to
    land within one parity class) -- that is the whole point: those within-
    side edges are exactly the ones `insert_mediators` subdivides."""
    colour = [-1] * n
    for start in range(n):
        if colour[start] != -1:
            continue
        colour[start] = 0
        q = deque((start,))
        while q:
            u = q.popleft()
            for v in adj[u]:
                if colour[v] == -1:
                    colour[v] = 1 - colour[u]
                    q.append(v)
    return colour


def _greedy_local_search(n: int, adj: list[list[int]], colour: list[int]) -> None:
    """Worklist-driven single-node-flip local search over the SAME potential
    the BFS colouring already reduces (number of within-side edges): flip a
    node to the side most of its neighbours are NOT on, whenever that
    strictly helps. Each flip strictly decreases the total within-side edge
    count (bounded below by 0, above by |E|), so the total number of flips
    over the WHOLE run -- regardless of how many nodes get re-checked -- is
    bounded by |E|; each flip only re-queues its own neighbours (not the
    whole graph), so this is the standard efficient implementation of this
    local search, not a fixed-point loop that rescans every node per pass."""
    in_queue = [True] * n
    q = deque(range(n))
    while q:
        u = q.popleft()
        in_queue[u] = False
        if not adj[u]:
            continue
        same = sum(1 for v in adj[u] if colour[v] == colour[u])
        other = len(adj[u]) - same
        if same > other:
            colour[u] = 1 - colour[u]
            for v in adj[u]:
                if not in_queue[v]:
                    in_queue[v] = True
                    q.append(v)


def mediator_coupling(absJ: float, beta: float) -> float:
    """A = arccosh(exp(2*beta*|J|)) / (2*beta) -- see this module's own
    docstring for the derivation. This is the SOLE implementation of that
    formula: `insert_mediators` below calls it (rather than re-deriving the
    expression inline), and so does `tsu_compiler.preflight.check.preflight`, which
    predicts a non-bipartite model's post-mediation coupling from its own
    beta and |J|max WITHOUT ever running placement (A-1, external review
    2026-09-09) -- this project has been bitten before by a constant
    re-derived in a second place and quietly drifting from the first, so
    this closed form now lives in exactly one place, not two.

    No finiteness guards here: `insert_mediators` already checks `beta` once
    (for the whole model) and `absJ` per-edge, at the exact points of use,
    with error messages naming the specific model/edge involved -- richer
    context than this small, general-purpose function could give on its
    own. A caller that skips those guards (as `preflight` must, since it is
    predicting ahead of any per-edge iteration) is responsible for its own
    finiteness check; see `preflight`'s own call site."""
    return math.acosh(math.exp(2.0 * beta * absJ)) / (2.0 * beta)


PARTITION_METHOD = "bfs_depth_parity_with_greedy_local_search"


def insert_mediators(ising: IsingModel, report: GraphReport
                     ) -> tuple[IsingModel, MediationReport]:
    """spec 5.3: subdivide every edge lying within one side of a 2-partition
    through a fresh hidden mediator spin, so the returned model is bipartite
    and its marginal distribution over `ising`'s own nodes is unchanged.

    `report` is accepted (per the task's own interface) as a cheap early-out:
    a graph that is already bipartite needs no mediators at all. It is not
    otherwise consulted -- `report.colouring` is a DSATUR colouring (however
    many colours a non-bipartite graph needs), not the 2-partition this pass
    needs, so that partition is always computed fresh here.
    """
    n = len(ising.nodes)
    beta = float(ising.beta)

    if report.bipartite:
        return ising, MediationReport(
            mediator_count=0, partition_method="already_bipartite",
            bipartite_after=True, beta_used=beta)

    # N-2 (R14 / F-R14): the same silent-NaN-into-"COMPILED" failure class
    # as lower.py's N-1, one file over -- IsingModel is a frozen dataclass
    # constructible directly (this file's own fixtures do exactly that),
    # so lower()'s guard on `beta` does not by itself protect this entry
    # point. Checked HERE, past the already-bipartite early-out, because
    # beta is never actually used in any formula on that path -- it would
    # be an unnecessary rejection, not a real guard. Every model that
    # reaches this point WILL mediate at least one edge (a non-bipartite
    # graph cannot be validly 2-coloured, so at least one within-side edge
    # survives the partition below), so beta is guaranteed to feed the
    # gadget formula from here on.
    if not math.isfinite(beta):
        raise ValueError(
            f"insert_mediators cannot compute the mediator gadget "
            f"(A = arccosh(exp(2*beta*|J|))/(2*beta)) at a non-finite "
            f"beta={beta!r}; refusing rather than emitting a NaN/inf "
            f"mediator coupling into a model that would still look "
            f"successfully compiled")

    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in ising.edges:
        adj[u].append(v)
        adj[v].append(u)

    colour = _bfs_parity_colouring(n, adj)
    _greedy_local_search(n, adj, colour)

    kept_edges: list[tuple[int, int]] = []
    kept_weights: list[float] = []
    new_edges: list[tuple[int, int]] = []
    new_weights: list[float] = []
    new_names: list[str] = []
    offset_correction = 0.0
    next_index = n

    for (u, v), J in zip(ising.edges, ising.weights):
        J = float(J)
        if colour[u] != colour[v]:
            kept_edges.append((u, v))
            kept_weights.append(J)
            continue

        # within-side edge: subdivide through a fresh mediator spin.
        absJ = abs(J)
        # N-2 (R14 / F-R14): the coupling reaching the gadget formula,
        # checked at the exact point of use -- see the beta guard above
        # for the sibling half of this same finding.
        if not math.isfinite(absJ):
            raise ValueError(
                f"insert_mediators cannot mediate edge "
                f"({ising.nodes[u]!r}, {ising.nodes[v]!r}) with non-finite "
                f"coupling J={J!r}; the gadget formula "
                f"(A = arccosh(exp(2*beta*|J|))/(2*beta)) would silently "
                f"produce a NaN/inf mediator coupling into a model that "
                f"would still look successfully compiled")
        A = mediator_coupling(absJ, beta)
        m = next_index
        next_index += 1
        new_names.append(f"__mediator_{m - n}")
        new_edges.append((u, m))
        new_weights.append(A)
        new_edges.append((m, v))
        new_weights.append(A if J > 0 else -A)
        # C = 2*exp(beta*|J|); folding ln(C)/beta into offset per mediated
        # edge keeps E(x) reconstructible after marginalizing the mediator
        # out (see this module's own docstring for the derivation).
        offset_correction += math.log(2.0 * math.exp(beta * absJ)) / beta

    mediator_count = len(new_names)
    mediator_nodes = tuple(range(n, n + mediator_count))

    med_ising = IsingModel(
        nodes=ising.nodes + tuple(new_names),
        edges=tuple(kept_edges + new_edges),
        weights=np.asarray(kept_weights + new_weights, dtype=float),
        biases=np.concatenate([ising.biases, np.zeros(mediator_count)])
              if mediator_count else np.asarray(ising.biases, dtype=float),
        beta=ising.beta,
        offset=ising.offset + offset_correction,
        mediator_nodes=mediator_nodes,
    )

    # A real measurement, not an assertion of the construction's own proof --
    # cheap (O(V+E)) even at the L1 16x16 scale, and this is exactly the kind
    # of claim this codebase never fabricates.
    G = nx.Graph()
    G.add_nodes_from(range(len(med_ising.nodes)))
    G.add_edges_from(med_ising.edges)
    bipartite_after = nx.is_bipartite(G)

    return med_ising, MediationReport(
        mediator_count=mediator_count, partition_method=PARTITION_METHOD,
        bipartite_after=bipartite_after, beta_used=beta)


def route(ising: IsingModel, report: GraphReport, target: TargetProfile) -> IsingModel:
    """The pipeline's own post-`place` hook. `place()` (spec 5.3.6) already
    performs mediation itself when a bipartite target needs it -- by the time
    `route` runs, `report` reflects whatever `place` actually placed, so this
    stays the identity for every candidate that reaches it. It still raises
    rather than silently passing through a graph that DOES need routing and
    somehow reached here un-mediated, so a gap can never be mistaken for
    support."""
    if target.bipartite.value and not report.bipartite:
        raise NotImplementedError(
            "this graph needs mediator routing and reached `route` still "
            "non-bipartite; `place` should have mediated it or raised "
            "parity_conflict first")
    return ising
