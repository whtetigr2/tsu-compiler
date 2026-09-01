"""Physical placement. A real pass whose verdict is binding.

EXP-WL1 rated a 3D cubic lattice GREEN with zero mediators on parity; EXP-WL2 then
found geometry, not parity, was binding. Parity is NECESSARY, never SUFFICIENT, so
this pass runs an actual embedding and names which failure class it hit.

`_anneal` is a seeded, budgeted heuristic search, not a decision procedure. When it
does not find a legal embedding within its budget, that is a fact about the search,
not about the substrate -- see `placement_effort_exhausted` below. C1 of the final
review found this pass reporting three provably-embeddable grids (an identity map
`v(x,y) -> (x,y)` realizes every edge) as `geometry_unreachable`, which is the exact
representation-vs-hardware conflation this project exists to prevent. `_try_grid_embed`
is the structural fix: it attempts an exact, verified unit-step embedding before ever
falling back to the annealer, so a graph the annealer would have needed luck for is
placed deterministically instead.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from ..failures import CompileError, Offender, PlacementFailure, Remediation
from ..target import TargetProfile
from .analyse import GraphReport, analyse
from .lower import IsingModel
from .route import MediationReport, insert_mediators

# The four unit steps of an axis-aligned 2D grid. Z1's offset set (rotations of
# (1,0)) always contains these, so any interaction graph that is a subgraph of the
# infinite grid under this adjacency embeds directly via the identity map.
_GRID_UNIT = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _try_grid_embed(G: nx.Graph, max_steps: int = 500_000):
    """Attempt to place every node of G on Z^2 such that every edge is exactly one
    of the four axis-unit steps. Returns a coords dict covering every node when
    such a placement exists and this search finds it, else None.

    This is deterministic backtracking, not a heuristic: whenever a node has two or
    more already-placed neighbours its position is forced (the intersection of
    each neighbour's four candidate cells), so the only real choice points are
    frontier nodes with exactly one known neighbour, and any wrong choice among
    those is caught (and backtracked out of) as soon as a later edge closes the
    loop. The result is verified against every edge of G before being returned, so
    a returned coords dict is always correct -- this never claims a false
    positive. It may fail to find an embedding that exists in principle (the
    max_steps budget), in which case the caller falls back to the annealer; it
    never claims the graph is NOT grid-embeddable.
    """
    if G.number_of_nodes() == 0:
        return {}

    all_coords: dict = {}
    x_shift = 0

    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        root = next(iter(comp))
        order = list(nx.bfs_tree(sub, root))          # order[0] == root
        n_comp = len(order)
        coords = {root: (0, 0)}
        used = {(0, 0)}
        cand_lists: list = [None] * n_comp
        cand_idx = [0] * n_comp
        i = 1
        steps = 0
        while 1 <= i < n_comp:
            steps += 1
            if steps > max_steps:
                return None
            v = order[i]
            if cand_lists[i] is None:
                known = [u for u in sub.neighbors(v) if u in coords]
                sets = [{(coords[u][0] + dx, coords[u][1] + dy) for dx, dy in _GRID_UNIT}
                        for u in known]
                cand_lists[i] = sorted(set.intersection(*sets) - used) if sets else []
                cand_idx[i] = 0
            if cand_idx[i] < len(cand_lists[i]):
                pos = cand_lists[i][cand_idx[i]]
                cand_idx[i] += 1
                coords[v] = pos
                used.add(pos)
                i += 1
            else:
                cand_lists[i] = None          # abandoning this node: force recompute
                i -= 1
                if i >= 1:
                    used.discard(coords.pop(order[i]))
        if i == 0:
            return None                       # this component has no grid embedding

        # shift this component clear of every previously placed one
        shift = x_shift - min(c[0] for c in coords.values())
        coords = {k: (x + shift, y) for k, (x, y) in coords.items()}
        x_shift = max(c[0] for c in coords.values()) + 2
        all_coords.update(coords)

    # Safety net: verify EVERY edge, not just the ones the search reasoned about.
    # A bug in the incremental logic must never surface as a false PLACED.
    for u, v in G.edges():
        d = (all_coords[u][0] - all_coords[v][0], all_coords[u][1] - all_coords[v][1])
        if d not in _GRID_UNIT:
            return None
    return all_coords


@dataclass(frozen=True)
class Placement:
    coords: dict           # node index -> (x, y); empty for targets with no lattice
    realized: tuple        # edges placed on a legal offset
    unrealized: tuple      # edges that could not be
    # Task 6 (spec 5.3.6): set exactly when `place` had to mediate a
    # non-bipartite graph to reach this placement. `mediated_ising`/
    # `mediated_report` are the model/report every downstream pass (route,
    # build_program, and this candidate's own regime/receipt) must use from
    # this point on -- coords/realized/unrealized above are already indexed
    # against `mediated_ising`'s (larger) node set whenever this is set, not
    # the pre-mediation `ising` the caller passed in. None/None for a
    # placement that needed no mediation at all (already bipartite, or the
    # target does not require it).
    mediation: MediationReport | None = None
    mediated_ising: IsingModel | None = None
    mediated_report: GraphReport | None = None


def _anneal(G, offsets, side, iters, seed):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    coords = {n: (rng.randrange(side), rng.randrange(side)) for n in nodes}
    offs = set(offsets)

    def bad(c):
        return sum(1 for u, v in G.edges()
                   if (c[u][0] - c[v][0], c[u][1] - c[v][1]) not in offs)

    cur = bad(coords)
    for _ in range(iters):
        if cur == 0:
            break
        n = rng.choice(nodes)
        old = coords[n]
        coords[n] = (rng.randrange(side), rng.randrange(side))
        new = bad(coords)
        if new <= cur:
            cur = new
        else:
            coords[n] = old
    return coords, cur


def _budget_check(n: int, target: TargetProfile) -> None:
    if n > target.node_budget.value:
        raise CompileError(
            "placement failed: node budget exceeded",
            [PlacementFailure("budget_exceeded",
                              (Offender("node", f"{n} nodes"),),
                              n, target.node_budget.value,
                              target.is_assumed("node_budget"), ())])


def _embed_on_lattice(ising: IsingModel, target: TargetProfile,
                      restarts: int, iters: int,
                      mediation=None) -> Placement:
    """The geometric search itself (structured grid embed, then the annealer
    fallback): place every node of `ising` on Z^2 so that every edge is a
    legal `target.offsets` step. `ising` here is ALREADY known bipartite (or
    the target does not require it) -- this function has no opinion on
    parity, only geometry.

    `mediation`: the MediationReport that produced THIS `ising` (None when
    it needed no mediation at all). Threaded through only so a
    `placement_effort_exhausted` failure can carry it -- real, already-
    computed mediation evidence (mediator count, bipartiteness achieved)
    must not be silently dropped just because the SEPARATE geometric search
    that follows it then ran out of budget. Never used to decide anything
    about the geometry itself."""
    n = len(ising.nodes)
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(ising.edges)

    # Try a structured, exact embedding before ever spending annealer budget.
    # Z1's offset set always contains the four axis-unit steps, so any interaction
    # graph that is itself a subgraph of the grid (the identity map v(x,y)->(x,y))
    # places directly and deterministically -- no search, no seed dependence, no
    # possibility of a false `geometry_unreachable`/`placement_effort_exhausted`.
    if set(_GRID_UNIT) <= set(target.offsets.value):
        grid_coords = _try_grid_embed(G)
        if grid_coords is not None:
            return Placement(grid_coords, tuple(ising.edges), ())

    side = max(4, int(n ** 0.5) + 3)
    best, best_bad = None, None
    for seed in range(restarts):
        coords, nbad = _anneal(G, target.offsets.value, side, iters, seed)
        if best_bad is None or nbad < best_bad:
            best, best_bad = coords, nbad
        if nbad == 0:
            break

    if best is None:
        # No restarts were attempted (restarts=0, used by tests that want to force
        # reliance on the structured embedding above). Treat this as "nothing
        # realized" rather than crashing on a None coords dict below.
        best = {node: (0, 0) for node in G.nodes()}

    offs = set(target.offsets.value)
    realized, unrealized = [], []
    for u, v in ising.edges:
        d = (best[u][0] - best[v][0], best[u][1] - best[v][1])
        (realized if d in offs else unrealized).append((u, v))

    if unrealized:
        # This is a search running out of budget, not a proof that no legal
        # offset spans these edges (that would require establishing unreachability
        # cheaply, which we cannot do in general -- see the module docstring and
        # C1 in the final review). `geometry_unreachable` is reserved for a case
        # this pass can actually PROVE; this pass makes no such proof, so it must
        # never raise that failure_class here.
        raise CompileError(
            "placement failed: placement effort exhausted",
            [PlacementFailure(
                failure_class="placement_effort_exhausted",
                offending=tuple(Offender("edge", f"{ising.nodes[u]}-{ising.nodes[v]}")
                                for u, v in unrealized),
                measured=len(unrealized), limit=0,
                assumed=target.is_assumed("offsets"),
                remediations=(
                    Remediation("increase placement effort",
                                "more restarts, more iterations per restart, or a "
                                "larger patch may find an embedding within budget"),
                    Remediation("change encoding",
                                "a lower-degree encoding is easier to embed",
                                {"note": "estimate"}),
                ), mediation=mediation)])

    return Placement(best, tuple(realized), ())


def place(ising: IsingModel, report: GraphReport, target: TargetProfile,
         *, restarts: int = 6, iters: int = 40_000) -> Placement:
    """`restarts`/`iters` tune ONLY the annealer fallback's effort budget (default
    6 x 40,000, the production budget). They exist so a test can force a small,
    deterministic effort budget without touching production behaviour -- see
    `placement_effort_exhausted` below, which names running out of THIS budget,
    not a claim about the substrate.

    Task 6 (spec 5.3.6): a non-bipartite graph against a bipartite target is no
    longer an automatic `parity_conflict`. This pass attempts hidden-spin
    mediation FIRST (`insert_mediators`, spec 5.3) -- mathematically, ANY
    2-partition-based subdivision makes ANY graph bipartite, so mediation
    always succeeds at fixing parity; `parity_conflict` is reserved for the
    (should-be-unreachable-in-practice) case where the mediated graph is
    somehow still not bipartite, a defensive check this pass makes rather
    than assumes. The node-budget gate is re-checked against the LARGER,
    mediated node count too -- mediation can push a graph that fit the
    original budget past it, and that must be reported as `budget_exceeded`,
    not silently ignored.
    """
    n = len(ising.nodes)

    if report.max_degree > target.degree.value:
        offenders = tuple(
            Offender("node", f"{ising.nodes[i]} (degree {d})")
            for i, d in nx.Graph(list(ising.edges)).degree()
            if d > target.degree.value)
        raise CompileError(
            "placement failed: degree exceeded",
            [PlacementFailure(
                failure_class="degree_exceeded",
                offending=offenders or (Offender("node", "unknown"),),
                measured=report.max_degree, limit=target.degree.value,
                assumed=target.is_assumed("degree"),
                remediations=(
                    Remediation("change encoding",
                                "a sparser encoding lowers per-node degree",
                                {"note": "estimate"}),
                    Remediation("relax target",
                                f"target.degree >= {report.max_degree}"),
                ))])

    if not target.offsets.value:            # no lattice: IDEAL
        return Placement({}, tuple(ising.edges), ())

    _budget_check(n, target)

    if target.bipartite.value and not report.bipartite:
        med_ising, mediation = insert_mediators(ising, report)
        if not mediation.bipartite_after:
            # Defensive only: `insert_mediators`' own construction proves
            # every within-side edge becomes cross-side once subdivided, so
            # this should be unreachable. It is checked, not assumed, so a
            # future change to that construction can never silently regress
            # into a graph this pass claims to have fixed but has not.
            G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(ising.edges)
            cycle = nx.find_cycle(G)
            known_cost = report.mediators >= 0
            cost = ({"extra_spins": report.mediators, "note": "estimate"} if known_cost
                   else {"note": "extra_spins not computed: graph exceeds the "
                                 "exact max-cut limit"})
            raise CompileError(
                "placement failed: parity conflict",
                [PlacementFailure(
                    failure_class="parity_conflict",
                    offending=tuple(Offender("edge", f"{ising.nodes[u]}-{ising.nodes[v]}")
                                    for u, v, *_ in cycle),
                    measured="odd cycle present after mediation was attempted",
                    limit="bipartite", assumed=target.is_assumed("bipartite"),
                    remediations=(
                        Remediation("route through mediator",
                                    "hidden-spin mediation is exact and adds one "
                                    "spin per frustrated coupling; it was attempted "
                                    "and did not resolve this graph", cost),
                    ))])

        med_report = analyse(med_ising)
        _budget_check(len(med_ising.nodes), target)
        placement = _embed_on_lattice(med_ising, target, restarts, iters,
                                          mediation=mediation)
        return Placement(placement.coords, placement.realized,
                         placement.unrealized, mediation=mediation,
                         mediated_ising=med_ising, mediated_report=med_report)

    return _embed_on_lattice(ising, target, restarts, iters)
