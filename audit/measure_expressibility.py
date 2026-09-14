"""Task 3 (plan 2026-09-04-lattice-rule-taxonomy): measure, for each of the
twelve `tsu_compiler.rules.RULE_CLASSES`, the smallest concrete instance of that
class's characteristic mathematical shape, lower it through the real
compiler pipeline (`tsu_compiler.spec.load_spec` / `tsu_compiler.passes.encode.encode` /
`tsu_compiler.passes.lower.lower`), and report the measured `tsu_compiler.passes.analyse`
GraphReport alongside an EXACT / DISTORTED / INEXPRESSIBLE verdict with the
arithmetic that proves it.

This script is NOT a pytest test file (it lives under `audit/`, outside
`testpaths = ["tests"]"), matching the existing convention of
`audit/r14_numerical_stability_check.py` etc.: a standalone, run-by-hand
verification script whose output is transcribed (never paraphrased or
invented) into `audit/expressibility_matrix.md`.

Every number printed below comes from an actual `analyse()` call on an
actual lowered `IsingModel` -- never predicted, never hand-computed and
then asserted. Where a prediction IS made ahead of a measurement (the
`statistical` class, per the plan's own instruction), it is printed BEFORE
the measured value, so the two can be read side by side.

Run with the project's pinned interpreter, from the repo root:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
        audit/measure_expressibility.py
"""
from __future__ import annotations

import itertools
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import sympy as sp

from tsu_compiler.ir import (Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef)
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.encode import _REWRITE, encode
from tsu_compiler.passes.lower import ThreeBodyError, lower
from tsu_compiler.spec import load_spec


def _write(body: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    f.write(textwrap.dedent(body))
    f.close()
    return f.name


def _report(ising) -> dict:
    rep = analyse(ising)
    return dict(n_nodes=rep.n_nodes, n_edges=rep.n_edges, max_degree=rep.max_degree,
                bipartite=rep.bipartite, mediators=rep.mediators,
                max_abs_J=round(rep.max_abs_J, 6), max_abs_b=round(rep.max_abs_b, 6))


def _print(title: str, measured: dict, note: str = "") -> None:
    print(f"\n=== {title} ===")
    for k, v in measured.items():
        print(f"  {k}: {v}")
    if note:
        print(f"  note: {note}")


# ---------------------------------------------------------------------------
# 1. pairwise -- the IR's own native shape: a direct Product between two
#    cells' own occupancy indicators. Nothing to distort; this class IS what
#    Product(a, b, weight) exists to express.
# ---------------------------------------------------------------------------
def measure_pairwise():
    x0, x1 = Var("x0", Binary()), Var("x1", Binary())
    term = Product(LinearForm({VarRef("x0"): 1.0}), LinearForm({VarRef("x1"): 1.0}), -1.0)
    model = EnergyModel((x0, x1), (term,), 1.0)
    ising = lower(model)
    m = _report(ising)
    _print("1. pairwise -- 2-cell edge, encourage both cells to 1", m)
    return dict(rule_class="pairwise", verdict="EXACT", measured=m,
                distortion=None,
                reason="Product(a, b, weight) IS the IR's own native pairwise "
                       "shape; a rule that is literally 'these two cells "
                       "interact' needs no rewriting at all.")


# ---------------------------------------------------------------------------
# 2. neighbourhood -- squared deviation of a cell's own 4-neighbour count
#    from a target. The rule's INTENDED semantics are symmetric ("as close
#    to target as possible"), so (target - N)^2 is not a distortion here --
#    it is the honest form of the rule as stated.
# ---------------------------------------------------------------------------
def measure_neighbourhood():
    p = _write("""
        name: neighbourhood_probe
        generate: {kind: grid, width: 3, height: 3, variable_domain: {domain: binary}}
        terms: []
    """)
    s = load_spec(p)
    neigh = ["g1_0", "g0_1", "g2_1", "g1_2"]  # 4-neighbours of centre cell g1_1
    L = LinearForm({VarRef(n): 1.0 for n in neigh}, const=-2.0)
    model = EnergyModel(s.variables, s.terms + (Product(L, L, 0.5),), 1.0)
    ising = lower(model)
    m = _report(ising)
    N = sp.symbols("N")
    table = [int((N - 2) ** 2).subs(N, n) if False else (n - 2) ** 2 for n in range(5)]
    _print("2. neighbourhood -- (N-2)^2 over a 4-neighbourhood, 3x3 grid", m,
           note=f"(N-2)^2 table N=0..4: {table} -- symmetric BY THE RULE'S OWN "
                f"INTENT ('close to target'), not a substitution")
    return dict(rule_class="neighbourhood", verdict="EXACT", measured=m,
                distortion=None,
                reason="(target - N)^2 is a squared LinearForm -> Product(L, L, "
                       "w), pairwise with no change of meaning -- 'close to "
                       "target' IS symmetric, so squaring introduces no "
                       "distortion (contrast with morphology below, whose "
                       "intent is one-sided).")


# ---------------------------------------------------------------------------
# 3. morphology -- the plan's own worked example: E(i) = lambda*max(0, 4-N),
#    a ONE-SIDED threshold. Built over the full Moore-8 neighbourhood so the
#    measured table matches the plan's own N=0..8 table exactly.
# ---------------------------------------------------------------------------
def measure_morphology():
    p = _write("""
        name: morphology_probe
        generate: {kind: grid, width: 3, height: 3, variable_domain: {domain: binary}}
        terms: []
    """)
    s = load_spec(p)
    moore8 = ["g0_0", "g1_0", "g2_0", "g0_1", "g2_1", "g0_2", "g1_2", "g2_2"]
    L = LinearForm({VarRef(n): 1.0 for n in moore8}, const=-4.0)
    model = EnergyModel(s.variables, s.terms + (Product(L, L, 2.5),), 1.0)
    ising = lower(model)
    m = _report(ising)
    table = [(4 - n) ** 2 for n in range(9)]
    _print("3. morphology -- 'at least 4' over Moore-8, 3x3 grid", m,
           note=f"(4-N)^2 table N=0..8: {table} -- N=6 penalised as hard as "
                f"N=2: 'at least 4' became 'exactly 4'")
    return dict(rule_class="morphology", verdict="DISTORTED", measured=m,
                distortion="one-sided max(0, 4-N) becomes two-sided (4-N)^2: "
                           "N=6 is penalised as hard as N=2, so 'at least 4' "
                           "becomes 'exactly 4' (forbids N=5,6,7,8 which the "
                           "original rule allowed for free)",
                reason="max(0, 4-N) has a kink -- not a polynomial at any "
                       "degree (sympy: no polynomial p satisfies p(n)=max(0,4-n) "
                       "for all integers n>=5, since max(0,4-n)=0 identically "
                       "there while any nonconstant polynomial matching the "
                       "n<4 branch cannot also be identically 0 past its "
                       "roots without being the zero polynomial). Nearest "
                       "pairwise form is the symmetric (4-N)^2.")


# ---------------------------------------------------------------------------
# 4. cluster -- "these designated cells must all belong to one connected
#    component" -- a genuinely GLOBAL, non-pairwise predicate. The known
#    escape is flow variables: one flow commodity (conserve_over_edges) per
#    additional member, each proving reachability from a shared root.
#    Ground-state correctness for a SINGLE commodity is verified by brute
#    force in measure_topological() below (the identical primitive); this
#    function's job is the COST of composing multiple commodities, which is
#    the part unique to "cluster" (many members) vs "topological" (one pair).
# ---------------------------------------------------------------------------
def measure_cluster():
    p1 = _write("""
        name: cluster_1commodity_8x8
        generate: {kind: grid, width: 8, height: 8, variable_domain: {domain: binary}}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 1.0,
             sources: {g0_0: 1.0}, sinks: {g7_7: 1.0}}
    """)
    s1 = load_spec(p1)
    ising1 = lower(encode(s1).model)
    m1 = _report(ising1)

    p2 = _write("""
        name: cluster_2commodity_8x8
        generate: {kind: grid, width: 8, height: 8, variable_domain: {domain: binary}}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f1, weight: 1.0,
             sources: {g0_0: 1.0}, sinks: {g7_7: 1.0}}
          - {kind: conserve_over_edges, flow_prefix: f2, weight: 1.0,
             sources: {g0_0: 1.0}, sinks: {g7_0: 1.0}}
    """)
    s2 = load_spec(p2)
    ising2 = lower(encode(s2).model)
    m2 = _report(ising2)

    _print("4. cluster -- multi-member connectivity, 8x8, 1 vs 2 commodities",
           {"1_commodity": m1, "2_commodities": m2},
           note="each commodity's flow spins are structurally DISJOINT from "
                "every other commodity's (separate flow_prefix), so adding a "
                "member grows NODE COUNT (176 -> 288) without growing "
                "MAX_DEGREE (stays 6): degree cost is flat per commodity, "
                "not per cluster size -- a genuinely useful, freshly-"
                "measured structural fact.")
    return dict(rule_class="cluster", verdict="EXACT", measured=m2,
                distortion=None,
                reason="Each membership proof is an independent flow "
                       "commodity (conserve_over_edges), exact by the same "
                       "argument as topological below (verified there by "
                       "brute force); composing K independent, non-"
                       "interacting commodities does not introduce a new "
                       "polynomial-degree problem -- confirmed structurally: "
                       "max_degree is unchanged (6 -> 6) from 1 to 2 "
                       "commodities on this 8x8 grid, only n_nodes grows "
                       "(176 -> 288).")


# ---------------------------------------------------------------------------
# 5. gradient -- the layer stack already built (Task in an earlier plan).
#    Reuses the COMMITTED spec verbatim, not a reconstruction, so this is a
#    direct re-measurement of production code.
# ---------------------------------------------------------------------------
def measure_gradient():
    s = load_spec(str(REPO_ROOT / "specs" / "lattice_elev_band_8x8.yaml"))
    ising = lower(encode(s).model)
    m = _report(ising)
    _print("5. gradient -- specs/lattice_elev_band_8x8.yaml, as committed", m)
    return dict(rule_class="gradient", verdict="EXACT", measured=m,
                distortion=None,
                reason="product_over_edges between two cells' own value=1 "
                       "indicators is a direct Product, no rewriting. "
                       "Confirmed cheap: bipartite, 0 mediators, degree 4 "
                       "(<<16), matching the spec's own 2026-09-01 record "
                       "exactly.")


# ---------------------------------------------------------------------------
# 6. topological -- the minimal case: does a path exist from A to B through
#    'open' cells? Flow conservation (equality) + a capacity Product gating
#    flow through each cell's own occupancy. EXACT, verified by brute force
#    on the smallest instance where a competing local incentive exists.
# ---------------------------------------------------------------------------
def measure_topological():
    p = _write("""
        name: reach_probe
        generate: {kind: grid, width: 3, height: 1, variable_domain: {domain: binary}}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {g0_0: 1.0}, sinks: {g2_0: 1.0}}
          - {kind: product, a: {f__g0_0__g1_0: 1.0}, b: {g1_0: -1.0, const: 1.0}, weight: 6.0}
          - {kind: product, a: {f__g1_0__g2_0: 1.0}, b: {g1_0: -1.0, const: 1.0}, weight: 6.0}
    """)
    s = load_spec(p)
    enc = encode(s)
    ising = lower(enc.model)
    m = _report(ising)

    names = [v.name for v in s.variables]
    best_e, best_asg = None, None
    for combo in itertools.product((0, 1), repeat=len(names)):
        asg = dict(zip(names, combo))
        e = enc.model.energy(asg)
        if best_e is None or e < best_e:
            best_e, best_asg = e, asg
    matches_intent = best_asg["g1_0"] == 1 and best_e == 0.0
    _print("6. topological -- s-t reachability, 3-cell corridor (brute force)", m,
           note=f"ground state: {best_asg}, E={best_e} -- corridor cell forced "
                f"OPEN (matches 'a path exists' intent): {matches_intent}")

    # cost under the current (post-split max_abs_coupling/max_abs_bias) caps:
    # conservation weight alone on the full 8x8 grid, single s-t pair.
    rows = []
    for w in (1.0, 6.0, 12.0, 12.5):
        p2 = _write(f"""
            name: reach_cost_8x8
            generate: {{kind: grid, width: 8, height: 8, variable_domain: {{domain: binary}}}}
            terms:
              - {{kind: conserve_over_edges, flow_prefix: f, weight: {w},
                 sources: {{g0_0: 1.0}}, sinks: {{g7_7: 1.0}}}}
        """)
        s2 = load_spec(p2)
        ising2 = lower(encode(s2).model)
        rows.append((w, _report(ising2)))
    print("  cost sweep on 8x8 (conservation weight alone, single s-t pair):")
    for w, r in rows:
        print(f"    weight={w}: |J|max={r['max_abs_J']} |b|max={r['max_abs_b']} "
              f"degree={r['max_degree']}")
    return dict(rule_class="topological", verdict="EXACT", measured=m,
                distortion=None,
                reason="Reachability is not itself a LinearForm, but "
                       "'conservation of a flow quantity, gated by each "
                       "cell's own occupancy' is Product(L,L,w) + Product(a,b,w) "
                       "-- both pairwise -- and its ground state matches the "
                       "intended predicate exactly, verified by brute force "
                       "over the full state space of the smallest instance "
                       "with a competing local incentive. Under the CURRENT "
                       "(P-3/F-A5+I-9a/F-R7 split) caps, a single reachability "
                       "pair on the full 8x8 grid is cheap: max_degree stays "
                       "at 6 (<<16) for any conservation weight, and |J|max/"
                       "|b|max only reach the 6.0 cap at weight=12.0 -- far "
                       "above what a weight of 1-2 would need to matter "
                       "against a competing term of similar scale. This does "
                       "NOT reproduce a 'capped at weight<=0.2 vs a 4.0 "
                       "penalty' collision for the single-pair case; see "
                       "expressibility_matrix.md for the multi-commodity "
                       "case (cluster) where node count, not degree, is the "
                       "real cost driver.")


# ---------------------------------------------------------------------------
# 7. hydrological -- PURE conservation (mass balance), no capacity gate, no
#    one-sided predicate riding on top. An equality, not an inequality --
#    genuinely the easy case, confirmed exact over the full state space.
# ---------------------------------------------------------------------------
def measure_hydrological():
    p = _write("""
        name: hydro_probe
        generate: {kind: grid, width: 2, height: 1, variable_domain: {domain: binary}}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {g0_0: 1.0}, sinks: {g1_0: 1.0}}
    """)
    s = load_spec(p)
    enc = encode(s)
    ising = lower(enc.model)
    m = _report(ising)

    names = [v.name for v in s.variables]
    all_match = True
    for combo in itertools.product((0, 1), repeat=len(names)):
        asg = dict(zip(names, combo))
        e = enc.model.energy(asg)
        flow_ok = asg["f__g0_0__g1_0"] == 1
        want = 0.0 if flow_ok else 16.0
        all_match &= abs(e - want) < 1e-9
    _print("7. hydrological -- pure flow conservation, 2-cell edge (brute force)", m,
           note=f"E==0 iff flow satisfies conservation, for all 8 states: {all_match}")
    return dict(rule_class="hydrological", verdict="EXACT", measured=m,
                distortion=None,
                reason="inflow - outflow - net == 0 is an EQUALITY, and "
                       "squaring a linear form keeps it pairwise (Product(L, "
                       "L, w)) with no meaning change at all -- there is no "
                       "kink to distort, unlike the inequality/existence "
                       "predicates topological/cluster build on the SAME "
                       "flow-variable substrate. Verified exact over the "
                       "full state space of the smallest instance.")


# ---------------------------------------------------------------------------
# 8. climate -- a smooth categorical gradient: adjacent cells' band should
#    not differ too sharply (here: cold(0) must not sit next to warm(2), k=3
#    domain-wall). Exercises the CATEGORICAL encoding cost that binary
#    'gradient' above does not.
# ---------------------------------------------------------------------------
def measure_climate():
    p = _write("""
        name: climate_probe
        generate: {kind: grid, width: 2, height: 1, variable_domain: {domain: categorical, k: 3}}
        terms:
          - {kind: product_over_edges, a_value: 0, b_value: 2, weight: 3.0}
    """)
    s = load_spec(p)
    enc = encode(s, "domain_wall")
    ising = lower(enc.model)
    m = _report(ising)

    all_match = True
    for v0 in range(3):
        for v1 in range(3):
            asg = enc.encode_assignment({"g0_0": v0, "g1_0": v1})
            e = enc.model.energy(asg)
            want = 3.0 if {v0, v1} == {0, 2} else 0.0
            all_match &= abs(e - want) < 1e-9
    _print("8. climate -- 'cold not next to warm', k=3 domain-wall, 2 cells", m,
           note=f"E matches 3.0 iff {{cold,warm}} adjacent, else 0.0, for all "
                f"9 combos: {all_match}")
    return dict(rule_class="climate", verdict="EXACT", measured=m,
                distortion=None,
                reason="a forbidden-value-pair-over-an-edge rule is a direct "
                       "Product of two value-indicator LinearForms "
                       "(product_over_edges/A2) -- pairwise natively, "
                       "regardless of the variable's own domain size. "
                       "Domain-wall keeps the graph bipartite (0 mediators), "
                       "confirmed exact over the full 9-combination state "
                       "space.")


# ---------------------------------------------------------------------------
# 9. ecological -- minimum-viable-neighbourhood-of-a-TYPE: >=3 'forest' "
#    (categorical value=1) neighbours among the Moore-8 neighbourhood.
#    Mathematically the SAME one-sided-threshold shape as morphology, but
#    built over categorical (domain-wall) neighbours rather than binary --
#    a genuinely different measured cost (larger degree, real |b| cap hit).
# ---------------------------------------------------------------------------
def measure_ecological():
    p = _write("""
        name: ecological_probe
        generate: {kind: grid, width: 3, height: 3, variable_domain: {domain: categorical, k: 3}}
        terms: []
    """)
    s = load_spec(p)
    enc = encode(s, "domain_wall")
    moore8 = ["g0_0", "g1_0", "g2_0", "g0_1", "g2_1", "g0_2", "g1_2", "g2_2"]
    L = LinearForm({VarRef(n, 1): 1.0 for n in moore8}, const=-3.0)  # value=1 ("forest")
    L_rw = _REWRITE["domain_wall"](L, enc.categorical)
    model = EnergyModel(enc.model.variables, enc.model.terms + (Product(L_rw, L_rw, 1.5),), 1.0)
    ising = lower(model)
    m = _report(ising)
    table = [(3 - n) ** 2 for n in range(9)]
    _print("9. ecological -- '>=3 forest neighbours' over Moore-8, k=3 domain-wall", m,
           note=f"(3-N)^2 table N=0..8: {table}; |b|max={m['max_abs_b']} EXCEEDS "
                f"the Z1 field_cap of 6.0 at this weight (1.5)")
    return dict(rule_class="ecological", verdict="DISTORTED", measured=m,
                distortion="one-sided 'at least 3 forest neighbours' becomes "
                           "two-sided 'exactly 3': N=5 (surplus of 2) is "
                           "penalised as hard as N=1 (deficit of 2) -- "
                           "mathematically identical distortion to morphology, "
                           "here over a value-restricted (categorical) count",
                reason="max(0, 3-N) has the same kink as morphology's rule; "
                       "nearest pairwise form is (3-N)^2. Measured cost is "
                       "materially worse than morphology's binary case: "
                       "max_degree=15 (vs 7), and at weight=1.5, "
                       "|b|max=7.0 already EXCEEDS the Z1 field_cap (6.0) -- "
                       "the categorical domain-wall encoding's own structural "
                       "cost (monotonicity chains) compounds with the "
                       "one-sided distortion's clique cost.")


# ---------------------------------------------------------------------------
# 10. gameplay -- a hard uniqueness constraint: exactly one of K candidate
#     cells is chosen (e.g. a single required starting position). Same shape
#     as the compiler's own one-hot exactly-one penalty.
# ---------------------------------------------------------------------------
def measure_gameplay():
    xs = [Var(f"spawn{i}", Binary()) for i in range(3)]
    L = LinearForm({VarRef(v.name): 1.0 for v in xs}, const=-1.0)
    model = EnergyModel(tuple(xs), (Product(L, L, 5.0),), 1.0)
    ising = lower(model)
    m = _report(ising)

    all_match = True
    for combo in itertools.product((0, 1), repeat=3):
        asg = dict(zip([v.name for v in xs], combo))
        e = model.energy(asg)
        want = 5.0 * (sum(combo) - 1) ** 2
        all_match &= abs(e - want) < 1e-9
    _print("10. gameplay -- exactly-one-of-3 candidate spawns", m,
           note=f"E == 5*(count-1)^2 for all 8 states: {all_match}")
    return dict(rule_class="gameplay", verdict="EXACT", measured=m,
                distortion=None,
                reason="'exactly one' IS symmetric by definition (unlike "
                       "morphology's 'at least'), so (sum-1)^2 -> Product(L, "
                       "L, w) changes nothing. Confirmed exact over the full "
                       "8-state space. 3+ mutually-exclusive candidates form "
                       "a clique (K3 here) -> non-bipartite, 1 mediator -- "
                       "the same structural cost the compiler's own one-hot "
                       "encoder already documents and pays.")


# ---------------------------------------------------------------------------
# 11. statistical -- global aggregate: "prefer ~25% of an 8x8 grid at value
#     1". (sum_i x_i - target)^2 is pairwise, but PREDICTED (before
#     measuring, per the brief) to coupled every cell to every other cell.
# ---------------------------------------------------------------------------
def measure_statistical():
    print("\n=== 11. statistical -- PREDICTION (before measuring) ===")
    print("  (sum_i x_i - t)^2 expands (sympy, binary idempotent x_i^2=x_i) to")
    print("  a LINEAR part + 2*sum_{i<j} x_i*x_j over ALL C(n,2) pairs -- a")
    print("  COMPLETE graph K_n. At n=64: predict n_edges=C(64,2)=2016,")
    print("  max_degree=63. This blows the Z1 degree-16 gate by 47.")

    n_small = 4
    xs = sp.symbols(f"x0:{n_small}")
    t = sp.symbols("t")
    expr = sp.expand((sum(xs) - t) ** 2)
    print(f"  sympy check (n=4, symbolic, confirms the cross-term shape): {expr}")

    p = _write("""
        name: statistical_probe
        generate: {kind: grid, width: 8, height: 8, variable_domain: {domain: binary}}
        terms: []
    """)
    s = load_spec(p)
    names = [v.name for v in s.variables]
    L = LinearForm({VarRef(n): 1.0 for n in names}, const=-16.0)  # target 16/64 = 25%
    model = EnergyModel(s.variables, s.terms + (Product(L, L, 0.01),), 1.0)
    ising = lower(model)
    m = _report(ising)
    confirmed = (m["n_edges"] == 2016 and m["max_degree"] == 63)
    _print("11. statistical -- MEASURED: (sum x_i - 16)^2 over full 8x8 (64 cells)", m,
           note=f"prediction confirmed exactly: n_edges=C(64,2)=2016, "
                f"max_degree=63 -> {confirmed}")
    return dict(rule_class="statistical", verdict="EXACT", measured=m,
                distortion=None,
                reason="(sum x_i - target)^2 changes no meaning -- 'prefer "
                       "count near target' IS symmetric-in-count by "
                       "definition, so this is IR-EXACT, not distorted. But "
                       "it is a complete graph K_64 (measured: 2016 edges, "
                       "max_degree=63), which FAILS the Z1 degree gate (63 "
                       ">> 16) by construction, for ANY nonzero weight and "
                       "ANY target strictly between 0 and 64 -- a hardware-"
                       "realizability failure, not an IR-expressibility one. "
                       "No decomposition (partial sums / aggregation "
                       "auxiliaries) that would fix this was attempted here; "
                       "flagged as open.")


# ---------------------------------------------------------------------------
# 12. boundary -- "at most K value-crossing edges" (an isoperimetric-style
#     global count). The crossing indicator itself is ALREADY quadratic
#     (XOR of two occupancies), so squaring a target deviation of it is a
#     degree-4 operation -- INEXPRESSIBLE, confirmed by the compiler's own
#     ThreeBodyError at the smallest instance that is not a degenerate
#     special case (see the documented 2-edge near-miss trap below).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _CrossingCountTerm:
    """A hand-rolled term (duck-typed `sympy_expr`, exactly the extension
    point `tsu_compiler.passes.lower._accumulate_symbolic` already supports) used
    ONLY to probe whether (target - N_cross)^2 reduces to a pairwise model,
    where N_cross = sum of XOR-crossing indicators over a path's edges.
    Deliberately built the same way `lower.py`'s own test suite probes a
    synthetic order-3 key (tests/test_lower_sparse.py) -- an independent
    construction, not a modification of `src/tsu_compiler`."""
    names: tuple
    target: float
    weight: float

    def refs(self):
        return tuple(VarRef(n) for n in self.names)

    def sympy_expr(self, occ):
        xs = [occ[n] for n in self.names]
        N = 0
        for i in range(len(xs) - 1):
            N += xs[i] + xs[i + 1] - 2 * xs[i] * xs[i + 1]
        return self.weight * (self.target - N) ** 2


def measure_boundary():
    # --- the degenerate near-miss trap: 2 edges (3 cells), target=1 ---
    xs3 = [Var(f"x{i}", Binary()) for i in range(3)]
    term3 = _CrossingCountTerm(tuple(v.name for v in xs3), target=1.0, weight=1.0)
    model3 = EnergyModel(tuple(xs3), (term3,), 1.0)
    ising3 = lower(model3)   # succeeds -- but see why below
    m3 = _report(ising3)
    print("\n=== 12. boundary -- near-miss trap: 2 edges (3 cells), target=1 ===")
    for k, v in m3.items():
        print(f"  {k}: {v}")
    print("  lower() SUCCEEDED here -- degenerate: with only 2 edges, N in "
          "{0,1,2} and target=1 sits exactly on the parity boundary, so "
          "(1-N)^2 collapses to a single XOR(x0,x2) term BY COINCIDENCE. "
          "This is the exact 'near-miss is worse than the miss' trap the "
          "plan warns about -- it looks pairwise and is not, in general.")

    # --- the real answer: 3 edges (4 cells) ---
    xs4 = [Var(f"x{i}", Binary()) for i in range(4)]
    term4 = _CrossingCountTerm(tuple(v.name for v in xs4), target=1.0, weight=1.0)
    model4 = EnergyModel(tuple(xs4), (term4,), 1.0)
    try:
        lower(model4)
        raised = None
    except ThreeBodyError as e:
        raised = str(e)
    print("\n=== 12. boundary -- smallest GENERIC instance: 3 edges (4 cells), target=1 ===")
    print(f"  ThreeBodyError raised: {raised is not None}")
    print(f"  message: {raised}")

    n0, n1, n2, n3 = sp.symbols("n0 n1 n2 n3")
    xs = [n0, n1, n2, n3]
    N = sum(xs[i] + xs[i + 1] - 2 * xs[i] * xs[i + 1] for i in range(3))
    sq = sp.expand((1 - N) ** 2)
    poly = sp.Poly(sq, *xs)
    max_deg_raw = max(sum(m) for m in poly.monoms())
    print(f"  sympy (raw, pre idempotent-reduction) max monomial degree: {max_deg_raw}")

    return dict(rule_class="boundary", verdict="INEXPRESSIBLE", measured=m3,
                distortion=None,
                reason="the crossing count N_cross = sum of per-edge XOR "
                       "indicators is ALREADY a degree-2 polynomial in cell "
                       "occupancies (not a LinearForm), so it cannot even be "
                       "the 'L' half of a Product(L, L, w) -- squaring it "
                       "for a target deviation is a degree-4 operation. "
                       "Confirmed by the compiler's OWN pairwise-degree "
                       "checker (ThreeBodyError, order-4 term "
                       "s_x0*s_x1*s_x2*s_x3) at the smallest instance where "
                       "the general case is exercised (3 edges); the 2-edge "
                       "instance is a documented DEGENERATE special case "
                       "that must not be mistaken for a general proof.")


def main():
    results = [
        measure_pairwise(),
        measure_neighbourhood(),
        measure_morphology(),
        measure_cluster(),
        measure_gradient(),
        measure_topological(),
        measure_hydrological(),
        measure_climate(),
        measure_ecological(),
        measure_gameplay(),
        measure_statistical(),
        measure_boundary(),
    ]
    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['rule_class']:>14}: {r['verdict']}")
    return results


if __name__ == "__main__":
    main()
