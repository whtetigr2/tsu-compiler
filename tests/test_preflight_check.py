"""Task 2: the pre-flight report. Compiler passes only -- no sampling."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu.passes.lower import IsingModel
from tsu.preflight.check import Gate, PreflightReport, preflight, WARN_FRACTION
from tsu.target import PROFILES


def grid(n: int, j: float = 0.4, b: float = 0.0) -> IsingModel:
    """A 4-neighbour grid: bipartite, and a direct subgraph of Z1."""
    idx = {(x, y): y * n + x for y in range(n) for x in range(n)}
    e = []
    for (x, y), i in idx.items():
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in idx:
                e.append((i, idx[(x + dx, y + dy)]))
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n * n)),
                      edges=tuple(e), weights=np.full(len(e), j),
                      biases=np.full(n * n, b), beta=1.0, offset=0.0)


def triangle() -> IsingModel:
    """An odd cycle: the smallest non-bipartite graph."""
    return IsingModel(nodes=("a", "b", "c"), edges=((0, 1), (1, 2), (0, 2)),
                      weights=np.full(3, 0.4), biases=np.zeros(3),
                      beta=1.0, offset=0.0)


def test_a_grid_is_reported_bipartite_and_directly_embeddable():
    r = preflight(grid(8))
    assert r.bipartite is True
    assert r.mediators == 0
    assert r.embedding == "grid_embed"
    assert r.n_spins == 64


def test_an_odd_cycle_is_reported_non_bipartite():
    """The decisive line in the whole report. A non-bipartite graph cannot be a
    direct subgraph of a chessboard lattice, whatever else is true of it."""
    r = preflight(triangle())
    assert r.bipartite is False
    assert r.embedding != "grid_embed"


def test_a_coupling_over_the_cap_fails_its_gate():
    r = preflight(grid(6, j=9.0))
    g = {x.name: x for x in r.gates}["max_abs_coupling"]
    assert g.status == "fail"
    assert g.limit == PROFILES["z1"].max_abs_coupling.value
    assert r.verdict == "fail"


def test_a_coupling_near_the_cap_warns_but_does_not_fail():
    """A model at 90% of the cap still compiles, but a user should know it has
    no headroom left for any later scaling."""
    cap = PROFILES["z1"].max_abs_coupling.value
    r = preflight(grid(6, j=cap * 0.9))
    g = {x.name: x for x in r.gates}["max_abs_coupling"]
    assert g.status == "warn"
    assert r.verdict != "fail"


def test_gate_statuses_use_the_stated_warn_fraction():
    """The boundary itself is asserted, not points near it: `_gate` uses
    `value >= limit * WARN_FRACTION`, so a value exactly at the boundary must
    warn. Sampling only either side would let a `>=` become a `>` unnoticed."""
    cap = PROFILES["z1"].max_abs_coupling.value
    at = preflight(grid(4, j=cap * WARN_FRACTION))
    assert {x.name: x for x in at.gates}["max_abs_coupling"].status == "warn"
    below = preflight(grid(4, j=cap * WARN_FRACTION * 0.99))
    assert {x.name: x for x in below.gates}["max_abs_coupling"].status == "ok"


def test_the_node_budget_gate_reports_the_share_used():
    r = preflight(grid(8))
    g = {x.name: x for x in r.gates}["node_budget"]
    assert g.value == 64
    assert g.limit == PROFILES["z1"].node_budget.value
    assert g.status == "ok"


def test_every_gate_carries_its_limit_and_a_note():
    """A gate without its limit is a number the reader cannot act on."""
    for g in preflight(grid(6)).gates:
        assert g.limit > 0
        assert g.note.strip()


def test_preflight_imports_no_sampler_at_all():
    """Pre-flight must be fast enough to run on every edit, and the guarantee is
    structural rather than behavioural: this module must not import the sampler
    under ANY spelling. A monkeypatch on the backend module would miss
    `from ... import sample` at module scope, which binds the real function
    before the patch runs -- the footgun this project documents in
    audit/findings/R1.md."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path("src/tsu/preflight/check.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert "thrml" not in a.name, a.name
        elif isinstance(node, ast.ImportFrom):
            assert "thrml" not in (node.module or ""), node.module
            for a in node.names:
                assert a.name != "sample", "check.py must not import a sampler"


def test_a_degree_violation_is_REPORTED_not_raised():
    """place() raises CompileError on a degree violation. A tool meant to run on
    every edit must report that as a failed gate, not hand the user a traceback."""
    n = 20
    e = [(i, j) for i in range(n) for j in range(i + 1, n)]
    im = IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=tuple(e),
                    weights=np.full(len(e), 0.1), biases=np.zeros(n),
                    beta=1.0, offset=0.0)
    r = preflight(im)
    assert r.verdict == "fail"
    assert {x.name: x for x in r.gates}["max_degree"].status == "fail"
    assert r.placed is False and r.place_error


def test_a_bipartite_graph_that_is_not_a_grid_subgraph_is_not_called_grid_embed():
    """Bipartiteness is necessary and NOT sufficient for a direct embed. This
    fixture is bipartite and is not a subgraph of the lattice; reporting it as
    grid_embed would claim a deterministic millisecond placement for a run that
    used the annealer."""
    import networkx as nx
    g = nx.bipartite.random_graph(30, 30, 0.15, seed=7)
    e = tuple(sorted((min(u, v), max(u, v))) for u, v in g.edges())
    im = IsingModel(nodes=tuple(f"n{i}" for i in range(60)), edges=e,
                    weights=np.full(len(e), 0.1), biases=np.zeros(60),
                    beta=1.0, offset=0.0)
    r = preflight(im)
    assert r.bipartite is True
    assert r.embedding != "grid_embed"


def test_the_mediated_fixture_reports_its_mediator_count():
    """The triangle is the only mediated fixture and nothing asserted what it
    cost, so a mediator-accounting change would pass unnoticed."""
    r = preflight(triangle())
    assert r.mediators > 0
    assert {x.name: x for x in r.gates}["node_budget"].value >= 3
