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
    """The threshold is a named constant, not a magic number scattered through
    the checks -- and it is asserted here so a change to it cannot pass silently."""
    cap = PROFILES["z1"].max_abs_coupling.value
    below = preflight(grid(4, j=cap * (WARN_FRACTION - 0.05)))
    above = preflight(grid(4, j=cap * (WARN_FRACTION + 0.05)))
    assert {x.name: x for x in below.gates}["max_abs_coupling"].status == "ok"
    assert {x.name: x for x in above.gates}["max_abs_coupling"].status == "warn"


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


def test_preflight_does_not_sample(monkeypatch):
    """Pre-flight must be fast enough to run on every edit. Patching where the
    name is BOUND, without raising=False, so a missing attribute fails loudly."""
    import tsu.backends.thrml_backend as bk

    def _forbidden(*a, **k):
        raise AssertionError("preflight sampled; it must use compiler passes only")

    monkeypatch.setattr(bk, "sample", _forbidden)
    preflight(grid(6))
