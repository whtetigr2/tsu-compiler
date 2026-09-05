"""C1 (code review, 2026-09-04-lattice-rule-taxonomy): `aux_variable_probe.py`
(Task 4) showed that a SINGLE auxiliary `z` entering only as `Product(z,
target-N, weight)` cannot rescue a one-sided threshold, and its Step 3
generalised that failure to "any finite set of auxiliaries, each entering
only as a factor in a product term". THAT GENERALISATION IS FALSE. It
compares ABSOLUTE energies, but an energy model is defined only up to an
additive constant (`IsingModel.offset` carries exactly this; `p(x) ~
exp(-beta*E)` cancels it) -- so "min_z E <= 0 for every x" constrains
nothing about whether a one-sided rule is expressible, because the intended
function only needs to be matched up to that same constant.

This script builds and verifies the COUNTEREXAMPLE: an assignment/matching
gadget where EVERY added term still contains an auxiliary as a factor
(exactly the shape Step 3 claimed was impossible), and which reproduces
max(0, target-N) EXACTLY, up to an additive constant.

    z[s][i] = "slot s is assigned to cell i",  s in 0..target-1, i in 0..n-1

    term1:  sum_{s,i}         z[s][i] * (-x_i)         (weight -1 each)
    term2:  P * sum_s         sum_{i<j} z[s][i]*z[s][j]  (each slot used <=1x)
    term3:  P * sum_i         sum_{s<s'} z[s][i]*z[s'][i] (each cell used <=1x)

Claim: min_z E(x, z) = -min(N, target), where N = sum(x) -- i.e.
min_z E(x,z) + target = max(0, target-N) EXACTLY, for every x.

Why (the exchange argument, stated once, used at every scale below): with
P > 1, any Y (0/1 matrix over (s,i)) that is NOT a partial matching (some
row or column sum >= 2) is STRICTLY worse than the best partial matching
using the same or fewer raised cells -- because a single collision costs P
> 1 but the maximum possible marginal reward from causing it is exactly 1
(one more raised cell covered). Among partial matchings, the T slots and n
cells form a COMPLETE bipartite graph (every (s,i) pair is a legal edge:
neither term1, term2, nor term3 excludes any pair), so a matching of size
min(N, target) using ONLY raised cells always exists (match any target' =
min(N,target) of the N raised cells to target' of the T slots) and is
optimal (using an unraised cell in a matching only wastes a slot for zero
reward, never an improvement). Hence the global minimum over Y is
achieved by a pure matching of size exactly min(N, target), giving
E_min = -min(N, target). This script uses P=1.5 (the natural minimal safe
choice: the single dangerous marginal case -- one extra raised cell taking
an already-used slot's second binding -- gains reward 1 at a cost of
exactly one new collision, so any P > 1 blocks it; 1.5 gives comfortable
margin without inflating |J|/|b| unnecessarily).

Verification performed here, honestly labelled by method:
  1. FULL brute force (over BOTH x and z jointly) at (target, n) in
     {(1,2), (2,3), (3,4)} -- small enough (<=2**16 states) to enumerate
     exhaustively via the real IR's own `EnergyModel.energy`.
  2. At (target, n) = (4, 6) (T*n=24 aux, 2**30 total states -- NOT
     exhaustively enumerable in reasonable time): the SAME claimed minimum
     is checked two ways instead: (a) exhaustive enumeration over every
     VALID matching (small: at most C(n,m)*C(target,m)*m! for m<=min(n,target),
     a few hundred here) confirms the best matching achieves -min(N,target)
     exactly; (b) a broad random sample of non-matching Y (20,000 draws per
     x) confirms none beats it. This is NOT full brute force, and is
     reported as such -- the exchange argument above is what actually
     carries the claim at this scale; the checks here corroborate it rather
     than substitute for it.
  3. The plan's own morphology scale (target=4, Moore-8 neighbourhood,
     n=8) run through the REAL `tsu.passes.lower.lower` /
     `tsu.passes.analyse.analyse` pipeline (not hand-rolled), measuring
     n_nodes, max_degree, |J|max, |b|max against the Z1 gates.
  4. A finite-beta distribution comparison at a small tractable scale
     (target=2, n=3): the INTENDED distribution (built directly from
     max(0,target-N), no pairwise form at all), the assignment-gadget's
     marginal (aux variables summed out), and the (target-N)^2 reference's
     marginal, all via the independent oracle
     (`audit/oracles/exact.py`, does not import `src/tsu`) -- reporting
     total variation distance of each against the intended distribution.

Run with the project's pinned interpreter, from the repo root:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
        audit/assignment_gadget_probe.py
"""
from __future__ import annotations

import itertools
import random
import sys
from math import comb, factorial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "audit"))

from tsu.ir import Binary, EnergyModel, LinearForm, Product, Var, VarRef  # noqa: E402
from tsu.passes.lower import lower  # noqa: E402
from tsu.passes.analyse import analyse  # noqa: E402
from oracles.exact import exact_boltzmann, exact_energy  # noqa: E402

P = 1.5  # penalty weight; see module docstring for why P > 1 suffices


def cell_name(i: int) -> str:
    return f"x{i}"


def z_name(s: int, i: int) -> str:
    return f"z{s}_{i}"


def build_gadget_model(target: int, n: int, weight: float = P) -> EnergyModel:
    """The assignment gadget over an n-cell neighbourhood with `target`
    slots. Every added term contains a z as a factor -- the exact shape
    Step 3 of aux_variable_probe.py claimed could never reach a positive
    minimum energy."""
    xs = [Var(cell_name(i), Binary()) for i in range(n)]
    zs = [[Var(z_name(s, i), Binary()) for i in range(n)] for s in range(target)]

    terms = []
    # term1: sum_{s,i} z[s][i] * (-x_i)
    for s in range(target):
        for i in range(n):
            terms.append(Product(
                LinearForm({VarRef(z_name(s, i)): 1.0}),
                LinearForm({VarRef(cell_name(i)): 1.0}),
                -1.0))
    # term2: P * sum_s sum_{i<j} z[s][i]*z[s][j]  (each slot used at most once)
    for s in range(target):
        for i, j in itertools.combinations(range(n), 2):
            terms.append(Product(
                LinearForm({VarRef(z_name(s, i)): 1.0}),
                LinearForm({VarRef(z_name(s, j)): 1.0}),
                weight))
    # term3: P * sum_i sum_{s<s'} z[s][i]*z[s'][i]  (each cell used at most once)
    for i in range(n):
        for s, sp in itertools.combinations(range(target), 2):
            terms.append(Product(
                LinearForm({VarRef(z_name(s, i)): 1.0}),
                LinearForm({VarRef(z_name(sp, i)): 1.0}),
                weight))

    variables = tuple(xs) + tuple(z for row in zs for z in row)
    return EnergyModel(variables, tuple(terms), 1.0)


def intended(N: int, target: int) -> float:
    return max(0.0, float(target - N))


# ---------------------------------------------------------------------------
# 1 & 2: ground-state verification at four scales
# ---------------------------------------------------------------------------
def _brute_force_min_z(model: EnergyModel, xvals: tuple[int, ...], n: int, target: int) -> float:
    names_x = [cell_name(i) for i in range(n)]
    names_z = [z_name(s, i) for s in range(target) for i in range(n)]
    best = None
    for zcombo in itertools.product((0, 1), repeat=len(names_z)):
        asg = dict(zip(names_x, xvals))
        asg.update(zip(names_z, zcombo))
        e = model.energy(asg)
        if best is None or e < best:
            best = e
    return best


def _matching_based_min_z(model: EnergyModel, xvals: tuple[int, ...], n: int, target: int) -> float:
    """Exact minimum via the exchange argument: enumerate every VALID partial
    matching (not every Y) and take the best; corroborated (not replaced) by
    a broad random sample of non-matching Y at the largest scale."""
    N = sum(xvals)
    raised = [i for i, v in enumerate(xvals) if v == 1]
    m_max = min(len(raised), target)
    names_x = [cell_name(i) for i in range(n)]
    best = None
    # every matching of size m, for m = 0..m_max: choose m raised cells and
    # assign them to m distinct slots (order matters -- which slot is cosmetic
    # here since all slots are interchangeable in cost, but enumerate fully
    # for an honest exhaustive check over the matching space).
    for m in range(0, m_max + 1):
        for cells in itertools.combinations(raised, m):
            for slots in itertools.permutations(range(target), m):
                asg = dict(zip(names_x, xvals))
                for i in range(n):
                    for s in range(target):
                        asg[z_name(s, i)] = 0
                for cell, slot in zip(cells, slots):
                    asg[z_name(slot, cell)] = 1
                e = model.energy(asg)
                if best is None or e < best:
                    best = e
    # corroborate with a broad random sample of ARBITRARY (non-matching-
    # restricted) z, to catch anything the matching restriction might miss.
    names_z = [z_name(s, i) for s in range(target) for i in range(n)]
    rng = random.Random(20260904)
    for _ in range(20_000):
        zcombo = [rng.randint(0, 1) for _ in names_z]
        asg = dict(zip(names_x, xvals))
        asg.update(zip(names_z, zcombo))
        e = model.energy(asg)
        if e < best:
            best = e   # would indicate the matching restriction MISSED the true optimum
    return best


def step1_small_scale_exhaustive():
    print("=== Step 1: FULL brute force over (x, z) jointly, small scales ===")
    all_ok = True
    for target, n in ((1, 2), (2, 3), (3, 4)):
        model = build_gadget_model(target, n)
        n_aux = target * n
        print(f"\n  (target={target}, n={n}): {n} cells + {n_aux} aux = "
              f"{n + n_aux} vars, 2**{n + n_aux} = {2 ** (n + n_aux)} joint states")
        ok = True
        rows = []
        for xvals in itertools.product((0, 1), repeat=n):
            N = sum(xvals)
            min_e = _brute_force_min_z(model, xvals, n, target)
            want = intended(N, target)
            got = min_e + target  # additive constant -- see module docstring
            match = abs(got - want) < 1e-9
            ok &= match
            rows.append((xvals, N, min_e, want, got, match))
        for xvals, N, min_e, want, got, match in rows:
            tag = "MATCH" if match else "DISAGREE"
            print(f"    x={xvals} N={N}  min_z(E)={min_e:+.1f}  "
                  f"min_z(E)+target={got:+.1f}  intended max(0,target-N)={want:.1f}  {tag}")
        print(f"  ALL {2**n} states match max(0,target-N) exactly "
              f"(up to the +target additive constant): {ok}")
        all_ok &= ok
    return all_ok


def step2_large_scale_matching_argument():
    print("\n=== Step 2: (target=4, n=6) -- 2**30 joint states, NOT brute-forced ===")
    print("    Verified instead via: (a) exhaustive search over every VALID")
    print("    matching (a few hundred candidates, not 2**24 arbitrary z),")
    print("    (b) 20,000 random non-matching z per x as a broad corroborating")
    print("    sample. See module docstring for the exchange argument that")
    print("    actually carries this claim at this scale.")
    target, n = 4, 6
    model = build_gadget_model(target, n)
    ok = True
    for xvals in itertools.product((0, 1), repeat=n):
        N = sum(xvals)
        min_e = _matching_based_min_z(model, xvals, n, target)
        want = intended(N, target)
        got = min_e + target
        match = abs(got - want) < 1e-9
        ok &= match
        if not match:
            print(f"    DISAGREE at x={xvals}: min_z(E)+target={got}, want={want}")
    print(f"    all {2**n} x-configurations match max(0,target-N) exactly: {ok}")
    return ok


# ---------------------------------------------------------------------------
# 3: the plan's own morphology scale, through the REAL compiler pipeline
# ---------------------------------------------------------------------------
def step3_morphology_scale_through_real_pipeline():
    print("\n=== Step 3: target=4, Moore-8 (n=8) -- through REAL lower()/analyse() ===")
    target, n = 4, 8
    model = build_gadget_model(target, n)
    n_aux = target * n
    print(f"    {n} neighbour cells + {n_aux} aux spins = {n + n_aux} total variables")
    ising = lower(model)
    report = analyse(ising)
    print(f"    n_nodes={report.n_nodes}  n_edges={report.n_edges}  "
          f"max_degree={report.max_degree}  bipartite={report.bipartite}  "
          f"mediators={report.mediators}")
    print(f"    max_abs_J={report.max_abs_J}  max_abs_b={report.max_abs_b}")

    from tsu.target import Z1
    gates = {
        "degree <= 16": (report.max_degree, 16, report.max_degree <= 16),
        "|J| <= 6.0": (report.max_abs_J, 6.0, report.max_abs_J <= 6.0),
        "|b| <= 6.0": (report.max_abs_b, 6.0, report.max_abs_b <= 6.0),
    }
    print("    Z1 gate check:")
    all_pass = True
    for gate, (measured, cap, passed) in gates.items():
        print(f"      {gate}: measured={measured}  ->  {'PASS' if passed else 'FAIL'}")
        all_pass &= passed
    print(f"    passes all three Z1 gates: {all_pass}")
    return report, all_pass


# ---------------------------------------------------------------------------
# 4: finite-beta distribution comparison at a small tractable scale
# ---------------------------------------------------------------------------
def _to_oracle_JB(ising):
    J = {ising.edges[i]: float(ising.weights[i]) for i in range(len(ising.edges))}
    b = [float(x) for x in ising.biases]
    return J, b, float(ising.offset)


def step4_distribution_comparison():
    target, n, beta = 2, 3, 1.0
    print(f"\n=== Step 4: finite-beta distribution comparison, target={target}, n={n} ===")

    names_x = [cell_name(i) for i in range(n)]

    # intended distribution: built DIRECTLY from max(0,target-N), no pairwise
    # form at all -- the ground truth every other model is judged against.
    intended_energy = {}
    for xvals in itertools.product((0, 1), repeat=n):
        intended_energy[xvals] = intended(sum(xvals), target)
    Z_int = sum(__import__("math").exp(-beta * e) for e in intended_energy.values())
    p_intended = {x: __import__("math").exp(-beta * e) / Z_int for x, e in intended_energy.items()}

    # assignment-gadget model, marginalised over z via the independent oracle
    gmodel = build_gadget_model(target, n)
    g_ising = lower(gmodel)
    Jg, bg, offg = _to_oracle_JB(g_ising)
    ok = True
    for combo in itertools.product((0, 1), repeat=len(g_ising.nodes)):
        asg = dict(zip(g_ising.nodes, combo))
        e_model = gmodel.energy(asg)
        e_oracle = exact_energy(list(combo), Jg, bg) + offg
        ok &= abs(e_model - e_oracle) < 1e-9
    print(f"    lower()-derived (J,b,offset) matches model.energy() over all "
          f"{2**len(g_ising.nodes)} gadget states: {ok}")
    assert ok

    g_states, g_probs = exact_boltzmann(Jg, bg, beta)
    xi_g = {name: i for i, name in enumerate(g_ising.nodes)}
    p_gadget = {}
    for s, p in zip(g_states, g_probs):
        key = tuple(s[xi_g[name]] for name in names_x)
        p_gadget[key] = p_gadget.get(key, 0.0) + p

    # reference: (target-N)^2, same as elsewhere in this plan
    xs2 = [Var(name, Binary()) for name in names_x]
    L2 = LinearForm({VarRef(name): -1.0 for name in names_x}, const=float(target))
    ref_model = EnergyModel(tuple(xs2), (Product(L2, L2, 1.0),), 1.0)
    ref_ising = lower(ref_model)
    Jr, br, offr = _to_oracle_JB(ref_ising)
    r_states, r_probs = exact_boltzmann(Jr, br, beta)
    xi_r = {name: i for i, name in enumerate(ref_ising.nodes)}
    p_ref = {}
    for s, p in zip(r_states, r_probs):
        key = tuple(s[xi_r[name]] for name in names_x)
        p_ref[key] = p_ref.get(key, 0.0) + p

    def tv(pa, pb, keys):
        return 0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys)

    keys = list(itertools.product((0, 1), repeat=n))
    tv_gadget = tv(p_intended, p_gadget, keys)
    tv_ref = tv(p_intended, p_ref, keys)

    print(f"    P(x) intended:  { {k: round(v,4) for k,v in sorted(p_intended.items())} }")
    print(f"    P(x) gadget:    { {k: round(v,4) for k,v in sorted(p_gadget.items())} }")
    print(f"    P(x) (t-N)^2:   { {k: round(v,4) for k,v in sorted(p_ref.items())} }")
    print(f"    TV(intended, assignment-gadget)  = {tv_gadget:.4f}")
    print(f"    TV(intended, (target-N)^2 ref)   = {tv_ref:.4f}")
    print(f"    gadget closer to intended than the two-sided reference: "
          f"{tv_gadget < tv_ref}")
    return tv_gadget, tv_ref


# ---------------------------------------------------------------------------
# 5: the real prize -- does the gadget fit at 8x8 GRID scale, one gadget per
#    cell, not just one neighbourhood in isolation? "One-sided rules are
#    EXPRESSIBLE" (steps 1-4 above) and "one-sided rules are AFFORDABLE at
#    grid scale" are different claims; this step measures the second one
#    honestly rather than letting the first stand in for it.
# ---------------------------------------------------------------------------
def _grid_cell(r: int, c: int) -> str:
    return f"g{r}_{c}"


def _grid_z_name(r: int, c: int, s: int, i: int) -> str:
    return f"z_{r}_{c}_{s}_{i}"


def _moore8_neighbours(r: int, c: int, width: int, height: int) -> list[tuple[int, int]]:
    """Boundary-TRUNCATED Moore-8 (no wraparound): corner cells get 3
    neighbours, edge cells 5, interior cells 8 -- the realistic case, not a
    toroidal idealisation."""
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                out.append((nr, nc))
    return out


def build_grid_gadget_model(width: int, height: int, target: int, weight: float = P) -> EnergyModel:
    """One independent assignment gadget PER CELL of a width x height grid,
    each site's own gadget scoped to that site's own (boundary-truncated)
    Moore-8 neighbours, with SITE-LOCAL aux-variable names (no sharing of z
    across sites) -- the direct, unoptimised cost of applying the C1
    counterexample everywhere morphology/ecological would want it, not a
    cherry-picked single instance."""
    cell_vars = [Var(_grid_cell(r, c), Binary()) for r in range(height) for c in range(width)]
    terms = []
    z_vars = []
    total_aux = 0
    for r in range(height):
        for c in range(width):
            nbrs = _moore8_neighbours(r, c, width, height)
            n_local = len(nbrs)
            for s in range(target):
                z_vars.extend(Var(_grid_z_name(r, c, s, i), Binary()) for i in range(n_local))
            total_aux += target * n_local
            # term1
            for s in range(target):
                for i, (nr, nc) in enumerate(nbrs):
                    terms.append(Product(
                        LinearForm({VarRef(_grid_z_name(r, c, s, i)): 1.0}),
                        LinearForm({VarRef(_grid_cell(nr, nc)): 1.0}),
                        -1.0))
            # term2: same slot, distinct neighbours
            for s in range(target):
                for i, j in itertools.combinations(range(n_local), 2):
                    terms.append(Product(
                        LinearForm({VarRef(_grid_z_name(r, c, s, i)): 1.0}),
                        LinearForm({VarRef(_grid_z_name(r, c, s, j)): 1.0}),
                        weight))
            # term3: same neighbour, distinct slots
            for i in range(n_local):
                for s, sp in itertools.combinations(range(target), 2):
                    terms.append(Product(
                        LinearForm({VarRef(_grid_z_name(r, c, s, i)): 1.0}),
                        LinearForm({VarRef(_grid_z_name(r, c, sp, i)): 1.0}),
                        weight))
    variables = tuple(cell_vars) + tuple(z_vars)
    return EnergyModel(variables, tuple(terms), 1.0), total_aux


def step5_grid_scale_ceiling():
    print("\n=== Step 5: THE REAL PRIZE -- does this fit at 8x8 grid scale? ===")
    width = height = 8
    target = 4
    model, total_aux = build_grid_gadget_model(width, height, target)
    n_cells = width * height
    print(f"    {n_cells} grid cells, target={target} slots/cell, Moore-8 "
          f"(boundary-truncated) neighbourhoods")
    print(f"    total auxiliary spins across the whole grid: {total_aux}")
    print(f"    total variables (cells + aux): {n_cells + total_aux}")
    print(f"    total IR terms: {len(model.terms)}")

    ising = lower(model)
    report = analyse(ising)
    print(f"    n_nodes={report.n_nodes}  n_edges={report.n_edges}  "
          f"max_degree={report.max_degree}  bipartite={report.bipartite}")
    print(f"    max_abs_J={report.max_abs_J}  max_abs_b={report.max_abs_b}")

    from tsu.target import Z1
    gates = {
        "degree <= 16": (report.max_degree, 16, report.max_degree <= 16),
        "|J| <= 6.0": (report.max_abs_J, 6.0, report.max_abs_J <= 6.0),
        "|b| <= 6.0": (report.max_abs_b, 6.0, report.max_abs_b <= 6.0),
        "node_budget <= 250000": (report.n_nodes, 250_000, report.n_nodes <= 250_000),
    }
    print("    Z1 gate check (grid scale):")
    all_pass = True
    for gate, (measured, cap, passed) in gates.items():
        print(f"      {gate}: measured={measured}  ->  {'PASS' if passed else 'FAIL'}")
        all_pass &= passed
    print(f"    HONEST CEILING: {n_cells} cells x this rule fits Z1 at this "
          f"scale: {all_pass}")
    return report, total_aux, all_pass


def main():
    ok1 = step1_small_scale_exhaustive()
    ok2 = step2_large_scale_matching_argument()
    report, gate_ok = step3_morphology_scale_through_real_pipeline()
    tv_gadget, tv_ref = step4_distribution_comparison()
    grid_report, grid_aux, grid_gate_ok = step5_grid_scale_ceiling()
    print("\n=== SUMMARY ===")
    print(f"  small-scale exhaustive match (target,n in "
          f"(1,2),(2,3),(3,4)): {ok1}")
    print(f"  (target=4,n=6) matching-argument match: {ok2}")
    print(f"  morphology-scale (target=4, Moore-8) passes all Z1 gates: {gate_ok}")
    print(f"  TV(intended,gadget)={tv_gadget:.4f} vs "
          f"TV(intended,(target-N)^2)={tv_ref:.4f} "
          f"-> gadget closer: {tv_gadget < tv_ref}")
    print(f"  8x8 GRID SCALE: {grid_aux} aux spins, n_nodes={grid_report.n_nodes}, "
          f"max_degree={grid_report.max_degree}, max_abs_J={grid_report.max_abs_J}, "
          f"max_abs_b={grid_report.max_abs_b} -> fits Z1: {grid_gate_ok}")


if __name__ == "__main__":
    main()
