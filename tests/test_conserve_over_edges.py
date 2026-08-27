"""C2: `conserve_over_edges` -- flow conservation as a generic term template
over a spec's edge set. Named for the MATHEMATICS (conservation of a signed
quantity over a node's incident edges), not for reachability or any workload
-- it serves any conservation-shaped constraint the same way `product_over_edges`
serves any pairwise one.

Convention (documented here and in spec.py): for a `conserve_over_edges` term
with `flow_prefix: f`, one Binary flow variable is generated per edge (u, v)
of the spec's edge set, named `f"{flow_prefix}__{u}__{v}"`, read as "the flow
from u to v" in the direction the edge is already declared. For every node n
of the edge set, the term emits `weight * (inflow - outflow - net)**2` where
inflow/outflow sum the incident flow variables by direction and
`net = sinks.get(n, 0) - sources.get(n, 0)` (0 for a node named in neither
map) -- a squared linear form, hence PAIRWISE once lowered, exactly the same
shape one-hot's own exactly-one penalty already uses.
"""
import os
import tempfile
import textwrap

import pytest

from tsu.ir import Binary, EnergyModel, LinearForm, Product, Var, VarRef
from tsu.passes.analyse import analyse
from tsu.passes.encode import encode
from tsu.passes.lower import ThreeBodyError, lower
from tsu.spec import load_spec


def _write(body: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        f.write(textwrap.dedent(body))
        return f.name


# --------------------------------------------------------------------------
# variable/term generation
# --------------------------------------------------------------------------

def test_conserve_over_edges_generates_one_flow_variable_per_edge():
    p = _write("""
        name: flow_vars
        generate:
          kind: grid
          width: 3
          height: 1
          variable_domain: {domain: binary}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {}, sinks: {}}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        # 3 grid cells + 2 flow variables (2 edges in a 3x1 strip)
        assert len(s.variables) == 5
        names = {v.name for v in s.variables}
        assert "f__g0_0__g1_0" in names and "f__g1_0__g2_0" in names
        assert all(isinstance(v.domain, Binary) for v in s.variables
                  if v.name.startswith("f__"))
    finally:
        os.unlink(p)


def test_conserve_over_edges_requires_a_non_empty_edge_set():
    p = _write("""
        name: no_edges
        variables:
          a: {domain: binary}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {}, sinks: {}}
        contract:
          validate: []
    """)
    try:
        with pytest.raises(ValueError, match="edge set"):
            load_spec(p)
    finally:
        os.unlink(p)


def test_conserve_over_edges_rejects_a_source_or_sink_naming_an_unknown_node():
    p = _write("""
        name: bad_node
        generate:
          kind: grid
          width: 2
          height: 1
          variable_domain: {domain: binary}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {"nowhere": 1.0}, sinks: {}}
        contract:
          validate: []
    """)
    try:
        with pytest.raises(ValueError, match="nowhere"):
            load_spec(p)
    finally:
        os.unlink(p)


def test_conserve_over_edges_emits_one_squared_term_per_node():
    p = _write("""
        name: term_count
        generate:
          kind: grid
          width: 3
          height: 1
          variable_domain: {domain: binary}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {"g0_0": 1.0}, sinks: {"g2_0": 1.0}}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        # 3 nodes -> 3 conservation terms, all Product(L, L, weight)
        assert len(s.terms) == 3
        for t in s.terms:
            assert isinstance(t, Product)
            assert t.a == t.b
            assert t.weight == 8.0
    finally:
        os.unlink(p)


# --------------------------------------------------------------------------
# semantics: the penalty fires exactly when conservation is violated
# --------------------------------------------------------------------------

def _two_node_source_sink_spec():
    return _write("""
        name: two_node_flow
        generate:
          kind: grid
          width: 2
          height: 1
          variable_domain: {domain: binary}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {"g0_0": 1.0}, sinks: {"g1_0": 1.0}}
        contract:
          validate: []
    """)


def test_conserve_over_edges_ground_truth_energy_is_zero_when_flow_satisfies_it():
    p = _two_node_source_sink_spec()
    try:
        s = load_spec(p)
        enc = encode(s)
        # only real variable beyond the flow one is g0_0/g1_0 themselves,
        # which the conservation term never references -- fix them arbitrarily
        asg = {"g0_0": 0, "g1_0": 0, "f__g0_0__g1_0": 1}
        assert enc.model.energy(asg) == pytest.approx(0.0, abs=1e-9)
    finally:
        os.unlink(p)


def test_conserve_over_edges_penalty_fires_when_flow_violates_conservation():
    p = _two_node_source_sink_spec()
    try:
        s = load_spec(p)
        enc = encode(s)
        violating = {"g0_0": 0, "g1_0": 0, "f__g0_0__g1_0": 0}
        satisfying = {"g0_0": 0, "g1_0": 0, "f__g0_0__g1_0": 1}
        e_bad = enc.model.energy(violating)
        e_good = enc.model.energy(satisfying)
        assert e_bad > e_good
        assert e_bad == pytest.approx(16.0, abs=1e-9), \
            "source and sink each carry an unmet unit of net flow: 8*(1)**2 twice"
    finally:
        os.unlink(p)


# --------------------------------------------------------------------------
# the corridor: ground state keeps the path open even when "blocking" is
# otherwise locally cheaper (the exact claim C2's brief makes)
# --------------------------------------------------------------------------

def test_corridor_ground_state_keeps_the_path_open_under_a_competing_incentive():
    """3 grid cells in a row; the middle cell doubles as an "open" indicator
    gating BOTH of its incident flow edges (capacity coupling: `f * (1 -
    open)`, a Product -- an ordinary, already-generic term template, composed
    here with conserve_over_edges, not a new mechanism). A separate linear
    term makes being "open" locally expensive (+1.0) and "blocked" locally
    free. Blocking with no flow costs 16.0 in unmet conservation (worse);
    blocking while flow is forced through costs 12.0 in capacity violation
    (also worse). The GLOBAL optimum keeps the corridor open, paying the
    local 1.0 cost, because every alternative is more expensive -- exactly
    the claim: "the ground state correctly keeps the corridor open ... even
    when blocking it is otherwise cheaper".

    Verified here by BRUTE FORCE over the whole (small) state space, using
    the IR's own `model.energy` -- the same technique test_backends.py uses
    for its own hand-checked reference, never a hand-rolled stand-in for a
    backend result.
    """
    p = _write("""
        name: corridor_probe
        generate:
          kind: grid
          width: 3
          height: 1
          variable_domain: {domain: binary}
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {"g0_0": 1.0}, sinks: {"g2_0": 1.0}}
          - {kind: product, a: {f__g0_0__g1_0: 1.0}, b: {g1_0: -1.0, const: 1.0},
             weight: 6.0}
          - {kind: product, a: {f__g1_0__g2_0: 1.0}, b: {g1_0: -1.0, const: 1.0},
             weight: 6.0}
          - {kind: linear, form: {g1_0: 1.0}, weight: 1.0}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        enc = encode(s)

        import itertools
        names = [v.name for v in s.variables]
        best_e, best_asg = None, None
        for combo in itertools.product((0, 1), repeat=len(names)):
            asg = dict(zip(names, combo))
            e = enc.model.energy(asg)
            if best_e is None or e < best_e:
                best_e, best_asg = e, asg

        assert best_e == pytest.approx(1.0, abs=1e-9)
        assert best_asg["g1_0"] == 1, "the ground state must keep the corridor OPEN"
        assert best_asg["f__g0_0__g1_0"] == 1
        assert best_asg["f__g1_0__g2_0"] == 1
    finally:
        os.unlink(p)


# --------------------------------------------------------------------------
# lowering: pairwise even when a node's clique (its incident flow variables)
# has more than 2 members
# --------------------------------------------------------------------------

def test_conserve_over_edges_lowers_to_pairwise_at_a_degree_4_node():
    """5 nodes, every pair connected (`clique`, weight 0 so the clique's own
    baked edges contribute nothing and do not confound this check) -- every
    node has degree 4, so every node's conservation term forms a clique among
    4 flow variables. `lower` must still succeed without ThreeBodyError: a
    squared linear form is pairwise by construction, at any degree."""
    p = _write("""
        name: high_degree_flow
        generate:
          kind: clique
          n: 5
          weight: 0.0
        terms:
          - {kind: conserve_over_edges, flow_prefix: f, weight: 8.0,
             sources: {"x0": 1.0}, sinks: {"x4": 1.0}}
        contract:
          validate: []
    """)
    try:
        s = load_spec(p)
        enc = encode(s)
        ising = lower(enc.model)   # must not raise ThreeBodyError
        assert len(ising.nodes) == len(s.variables)
    finally:
        os.unlink(p)


def test_conserve_over_edges_is_reported_non_bipartite_with_the_honest_clique_cost():
    """The claim from the brief, isolated and hand-verifiable: a single
    degree-4 node's conservation term forms a K4 among its 4 incident flow
    variables (Product(L, L, w) with |L|'s support = 4 -- the SAME clique
    shape one-hot's exactly-one penalty already produces at k=4). K4 has 6
    edges; its max-cut is 4 (any 2-2 bipartition cuts all 4 cross edges, and
    only those); mediators = |E| - maxcut = 2, and the graph is non-bipartite
    (K4 contains a triangle/odd cycle). Built directly from the IR to isolate
    exactly this one clique, independent of any larger spec's structure."""
    flows = [Var(f"f{i}", Binary()) for i in range(4)]
    L = LinearForm({VarRef(v.name): 1.0 for v in flows}, const=-1.0)
    model = EnergyModel(tuple(flows), (Product(L, L, 8.0),), 1.0)
    ising = lower(model)
    report = analyse(ising)

    assert report.n_edges == 6, "K4 among the 4 incident flow variables"
    assert report.bipartite is False
    assert report.mediators == 2, "6 edges - max-cut(K4)=4"
