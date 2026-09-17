"""Fold a protein on a thermodynamic sampler, against a known answer.

================================ PROTOCOL ================================
SYSTEM DEFINITION   The HP lattice protein model: a chain of hydrophobic (H)
                    and polar (P) monomers folding on a 2D square lattice.
                    Encoded as a QUBO over binary occupancy variables
                    x[i][s] = "monomer i sits on lattice site s", lowered to
                    an Ising model and checked against Z1's published limits.
STATE VARIABLES     8 monomers on a 3x3 grid, so 72 binary variables. The
                    sequence is HHPPHPPH.
TRANSITION RULES    THRML block Gibbs over the compiler's chromatic schedule.
ALLOWED OPERATIONS  Building the QUBO; lowering it; `preflight`; sampling
                    through `tsu_compiler.backends.thrml_backend`; decoding a
                    draw back to a chain; comparing against an INDEPENDENT
                    exhaustive enumeration written in this file.
FORBIDDEN OPERATIONS
                    No energy claim. No claim about protein structure
                    prediction in general: this is a lattice toy model, and the
                    distance between it and a real folding prediction is the
                    distance between a wind tunnel model and an airplane. No
                    treating the |J| or |b| caps as established beyond what
                    target.py records, and `max_abs_bias` is ASSUMED there.
ASSUMPTIONS         Z1's |b| <= 6.0. This project's working value, not a
                    sourced Extropic figure. The constraint-versus-objective
                    headroom below turns on it.
INVARIANTS          The optimum found by enumeration is a property of the
                    sequence and the lattice. It does not depend on the
                    encoding, the compiler, or the sampler, so it is computed
                    first and never adjusted afterwards.
MEASUREMENTS        Compile verdict, gates, degree, mediators, placement cost;
                    then the best contact count any sampled draw achieves.
NULL HYPOTHESES     "A thermodynamic sampler finds the known optimal fold."
                    Refuted if sampling never reaches 3 contacts while the
                    enumeration says 3 is reachable.
SUCCESS CRITERIA    A clear answer either way, with the binding constraint
                    named if it does not compile.
FAILURE CRITERIA    Reporting a fold that violates self-avoidance or chain
                    connectivity as a success. Those are checked on the
                    decoded chain, not assumed from the energy.
PROVENANCE          HP model: Lau and Dill 1989. The QUBO encoding follows the
                    occupancy formulation used by Perdomo-Ortiz et al., "Finding
                    low-energy conformations of lattice protein models by
                    quantum annealing", Sci. Rep. 2 (2012), which ran the same
                    problem on D-Wave hardware. Ground truth here is computed by
                    exhaustive enumeration in this file, not taken from a paper.
SCOPE OF VALIDITY   One sequence, 8 monomers, one lattice, one grid size. It
                    says whether this instance fits Z1's published limits and
                    whether sampling reaches the known optimum. It says nothing
                    about larger instances, and nothing about energy.
==========================================================================
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEQUENCE = "HHPPHPPH"
GRID_W = GRID_H = 3

# QUBO coefficients. The conversion to Ising divides couplings by 4, so a
# coefficient of 24 is the largest that keeps |J| inside Z1's published 6.0
# cap. Everything here is expressed relative to a single H-H contact being
# worth 1, which is what makes the headroom question measurable.
W_ONE_SITE = 8.0     # each monomer occupies exactly one site
W_EXCLUSION = 8.0    # each site holds at most one monomer
W_CHAIN = 8.0        # consecutive monomers must be lattice neighbours
W_CONTACT = -1.0     # reward: two H monomers touching but not chain-adjacent


def sites() -> list[tuple[int, int]]:
    return [(x, y) for y in range(GRID_H) for x in range(GRID_W)]


def adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# --------------------------------------------------------------------------
# Ground truth. Computed first, by exhaustive enumeration, and never adjusted.
# Depends on nothing else in this repository.
# --------------------------------------------------------------------------

def all_self_avoiding_walks(n: int) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []

    def walk(path: list[tuple[int, int]], seen: set[tuple[int, int]]) -> None:
        if len(path) == n:
            out.append(tuple(path))
            return
        x, y = path[-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            step = (x + dx, y + dy)
            if step in seen:
                continue
            seen.add(step)
            path.append(step)
            walk(path, seen)
            path.pop()
            seen.discard(step)

    walk([(0, 0)], {(0, 0)})
    return out


def contact_count(seq: str, path: tuple[tuple[int, int], ...]) -> int:
    """H-H pairs touching on the lattice but NOT adjacent along the chain."""
    where = {p: i for i, p in enumerate(path)}
    total = 0
    for i, p in enumerate(path):
        if seq[i] != "H":
            continue
        for q in ((p[0] + 1, p[1]), (p[0], p[1] + 1)):
            j = where.get(q)
            if j is not None and seq[j] == "H" and abs(i - j) > 1:
                total += 1
    return total


def known_optimum(seq: str) -> tuple[int, int, int]:
    """(best contacts, folds achieving it, total walks). Enumeration only."""
    walks = all_self_avoiding_walks(len(seq))
    counts = [contact_count(seq, w) for w in walks]
    best = max(counts)
    return best, counts.count(best), len(walks)


# --------------------------------------------------------------------------
# The thermodynamic program.
# --------------------------------------------------------------------------

def build_qubo(seq: str) -> tuple[dict[int, float], dict[tuple[int, int], float],
                                  dict[tuple[int, int], int]]:
    """Return (linear, quadratic, index) for the HP occupancy QUBO.

    `index[(monomer, site)] -> variable id`. Penalties are positive, the H-H
    contact reward is negative, and a configuration satisfying every constraint
    has energy equal to minus its contact count.
    """
    S = sites()
    idx = {(i, s): i * len(S) + s for i in range(len(seq)) for s in range(len(S))}
    lin: dict[int, float] = {v: 0.0 for v in idx.values()}
    quad: dict[tuple[int, int], float] = {}

    def add(a: int, b: int, w: float) -> None:
        key = (a, b) if a < b else (b, a)
        quad[key] = quad.get(key, 0.0) + w

    n = len(seq)
    m = len(S)

    # Each monomer on exactly one site: (sum_s x - 1)^2, expanded for binary x.
    for i in range(n):
        for s in range(m):
            lin[idx[(i, s)]] -= W_ONE_SITE
        for s, t in itertools.combinations(range(m), 2):
            add(idx[(i, s)], idx[(i, t)], 2.0 * W_ONE_SITE)

    # Each site holds at most one monomer.
    for s in range(m):
        for i, j in itertools.combinations(range(n), 2):
            add(idx[(i, s)], idx[(j, s)], W_EXCLUSION)

    # Consecutive monomers must land on neighbouring sites: penalise every
    # placement of the pair that is not adjacent, including the same site.
    for i in range(n - 1):
        for s in range(m):
            for t in range(m):
                if adjacent(S[s], S[t]):
                    continue
                add(idx[(i, s)], idx[(i + 1, t)], W_CHAIN)

    # The objective: H monomers touching on the lattice, not along the chain.
    for i, j in itertools.combinations(range(n), 2):
        if abs(i - j) < 2 or seq[i] != "H" or seq[j] != "H":
            continue
        for s in range(m):
            for t in range(m):
                if adjacent(S[s], S[t]):
                    add(idx[(i, s)], idx[(j, t)], W_CONTACT)

    return lin, quad, idx


def qubo_to_ising(lin: dict[int, float], quad: dict[tuple[int, int], float],
                  n_vars: int) -> tuple[list[list[float]], list[float]]:
    """Binary QUBO to the oracle's Ising convention, E = -sum J s s - sum b s.

    Verified against brute-force enumeration before use: substituting
    x = (s + 1) / 2 gives J_ij = -Q_ij / 4 and
    b_i = -(Q_ii / 2 + sum_j Q_ij / 4).
    """
    edges = [[i, j, -w / 4.0] for (i, j), w in quad.items() if w != 0.0]
    biases = [0.0] * n_vars
    for i in range(n_vars):
        off = sum(w for (a, b), w in quad.items() if a == i or b == i)
        biases[i] = -(lin.get(i, 0.0) / 2.0 + off / 4.0)
    return edges, biases


def decode(state: list[int], idx: dict[tuple[int, int], int], seq: str
           ) -> tuple[list[tuple[int, int] | None], list[str]]:
    """A draw becomes a chain, and every violation it carries is named.

    A draw is NOT assumed to be a valid fold because its energy looks good.
    Self-avoidance and connectivity are checked here, on the decoded chain.
    """
    S = sites()
    placed: list[tuple[int, int] | None] = []
    problems: list[str] = []
    for i in range(len(seq)):
        occupied = [s for s in range(len(S)) if state[idx[(i, s)]] == 1]
        if len(occupied) != 1:
            problems.append(f"monomer {i} occupies {len(occupied)} sites")
            placed.append(S[occupied[0]] if occupied else None)
        else:
            placed.append(S[occupied[0]])

    seen: dict[tuple[int, int], int] = {}
    for i, p in enumerate(placed):
        if p is None:
            continue
        if p in seen:
            problems.append(f"monomers {seen[p]} and {i} share site {p}")
        seen[p] = i

    for i in range(len(placed) - 1):
        a, b = placed[i], placed[i + 1]
        if a is None or b is None:
            continue
        if not adjacent(a, b):
            problems.append(f"chain breaks between monomer {i} and {i + 1}")

    return placed, problems


# --------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------

def scaled(edges: list[list[float]], biases: list[float], cap: float = 6.0
           ) -> tuple[list[list[float]], list[float], float]:
    """Bring the largest coefficient to `cap`, and say what beta must become.

    Scaling every coefficient by s and beta by 1/s leaves the distribution
    identical, so a bias cap failure is a units problem rather than a hardware
    limit. Worth separating, because the DEGREE failure below is not.
    """
    biggest = max(max(abs(w) for _, _, w in edges), max(abs(b) for b in biases))
    s = cap / biggest
    return ([[i, j, w * s] for i, j, w in edges], [b * s for b in biases], s)


def main() -> int:
    import json
    import tempfile

    import jax
    import jax.numpy as jnp
    import networkx as nx
    import numpy as np
    from thrml import Block, SamplingSchedule, SpinNode, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

    from tsu_compiler.preflight.check import preflight
    from tsu_compiler.preflight.model import load_model

    best, n_opt, n_walks = known_optimum(SEQUENCE)
    print("GROUND TRUTH, by exhaustive enumeration in this file")
    print(f"  sequence {SEQUENCE}, {len(SEQUENCE)} monomers on {GRID_W}x{GRID_H}")
    print(f"  optimal contacts {best}, reached by {n_opt} of {n_walks} walks "
          f"({100 * n_opt / n_walks:.2f}%)")

    lin, quad, idx = build_qubo(SEQUENCE)
    n = len(idx)
    edges, biases = qubo_to_ising(lin, quad, n)
    degree: dict[int, int] = {}
    for i, j, _ in edges:
        degree[int(i)] = degree.get(int(i), 0) + 1
        degree[int(j)] = degree.get(int(j), 0) + 1

    print("\nTHE PROGRAM")
    print(f"  {n} variables, {len(edges)} couplings")
    print(f"  max degree {max(degree.values())}")
    print(f"  clique floor {(GRID_W * GRID_H - 1) + (len(SEQUENCE) - 1)}: "
          f"one-hot over sites plus exclusion over monomers, and they add")

    e_s, b_s, s = scaled(edges, biases)
    path = Path(tempfile.mkdtemp()) / "hp.json"
    path.write_text(json.dumps({"nodes": n, "edges": e_s, "biases": b_s,
                                "beta": 1.0 / s}), encoding="utf-8")
    rep = preflight(load_model(edges=str(path)))
    print(f"\nAGAINST Z1  (coefficients scaled by {s:.4f}, beta by {1 / s:.1f})")
    print(f"  verdict {rep.verdict}")
    for g in rep.gates:
        print(f"    {g.name:22} {g.status}")
    if rep.place_error:
        print(f"  {rep.place_error}")

    # Sample it anyway, on no hardware target, to separate two questions:
    # is the MODEL right, and does Z1 take it? They have different answers.
    nodes = [SpinNode() for _ in range(n)]
    pairs = [(int(i), int(j)) for i, j, _ in edges]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(pairs)
    colouring = nx.coloring.greedy_color(G, strategy="DSATUR")
    blocks = [Block([nodes[v] for v, c in colouring.items() if c == col])
              for col in range(max(colouring.values()) + 1)]

    model = IsingEBM(
        nodes, [(nodes[i], nodes[j]) for i, j in pairs],
        jnp.array(b_s, dtype=jnp.float32),
        jnp.array([w for _, _, w in e_s], dtype=jnp.float32),
        jnp.array(1.0 / s, dtype=jnp.float32))
    program = IsingSamplingProgram(model, blocks, [])
    schedule = SamplingSchedule(n_warmup=4000, n_samples=400, steps_per_sample=8)

    key = jax.random.key(0)
    key, k_init = jax.random.split(key)
    chains = 24
    init = hinton_init(k_init, model, blocks, (chains,))
    run = jax.jit(jax.vmap(lambda i0, kk: sample_states(
        kk, program, schedule, i0, [], [Block(nodes)])))
    draws = np.asarray(run(init, jax.random.split(key, chains))[0])

    flat = draws.reshape(-1, n).astype(int)
    valid = 0
    seen = -1
    fold = None
    for row in flat:
        placed, problems = decode(list(row), idx, SEQUENCE)
        if problems:
            continue
        valid += 1
        c = contact_count(SEQUENCE, tuple(placed))
        if c > seen:
            seen, fold = c, tuple(placed)

    print(f"\nSAMPLED THROUGH THRML  ({len(blocks)} colours, {chains} chains)")
    print(f"  valid folds {valid} of {len(flat)} draws "
          f"({100 * valid / len(flat):.2f}%)")
    print(f"  best contacts {seen}, optimum {best}")
    if fold:
        print(f"  best fold {fold}")

    ok = seen == best
    print("\n  " + ("REACHED THE KNOWN OPTIMUM"
                    if ok else "DID NOT REACH THE KNOWN OPTIMUM"))
    print("  Simulation on CPU through THRML. Not Extropic silicon.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
