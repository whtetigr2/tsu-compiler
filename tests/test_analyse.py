import numpy as np
import pytest
from tsu.passes.lower import IsingModel
from tsu.passes.analyse import analyse, MAXCUT_EXACT_LIMIT


def ising(nodes, edges, w=None, b=None):
    return IsingModel(tuple(nodes), tuple(edges),
                      np.array(w if w is not None else [1.0] * len(edges)),
                      np.array(b if b is not None else [0.0] * len(nodes)),
                      1.0, 0.0)


def test_path_is_bipartite_with_no_mediators():
    im = ising(("x", "y", "z"), ((0, 1), (1, 2)))
    r = analyse(im)
    assert r.bipartite is True
    assert r.mediators == 0
    assert r.colour_blocks == 2
    assert r.max_degree == 2


def test_triangle_is_not_bipartite_and_needs_one_mediator():
    """|E| - MaxCut = 3 - 2 = 1."""
    im = ising(("x", "y", "z"), ((0, 1), (1, 2), (0, 2)))
    r = analyse(im)
    assert r.bipartite is False
    assert r.mediators == 1
    assert r.colour_blocks == 3


def test_colouring_never_places_adjacent_nodes_in_one_block():
    im = ising(("a", "b", "c", "d"), ((0, 1), (1, 2), (2, 3), (3, 0)))
    r = analyse(im)
    for u, v in im.edges:
        assert r.colouring[u] != r.colouring[v]


def test_reports_parameter_magnitudes():
    im = ising(("x", "y"), ((0, 1),), w=[-4.5], b=[2.0, -7.0])
    r = analyse(im)
    assert r.max_abs_J == pytest.approx(4.5)
    assert r.max_abs_b == pytest.approx(7.0)


def test_mediators_is_the_minus_one_sentinel_above_the_exact_maxcut_limit():
    """I5: -1 means 'not computed' (exact max-cut is exponential and this graph
    exceeds the limit), not zero mediators -- a genuinely different fact from
    'this bipartite-broken graph needs zero mediators', which is impossible.
    Consumers (place.py's remediation, receipt.py's metrics.json) must never
    treat -1 as a spin count."""
    n = MAXCUT_EXACT_LIMIT + 5    # an odd ring, so it is never bipartite
    edges = [(i, (i + 1) % n) for i in range(n)]
    im = ising(range(n), edges)
    r = analyse(im)
    assert r.bipartite is False
    assert r.mediators == -1, "mediators must read the 'not computed' sentinel"
