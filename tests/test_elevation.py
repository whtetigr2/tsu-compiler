"""Task 2/3: the elevation overlay spec and the thermometer logic on top of it.

Task 2 (this section): specs/lattice_elev_band_8x8.yaml compiles to a
bipartite, unmediated 64-spin overlay -- spec section 1.2's claim that beta
stays free on an overlay (never welded to compile time the way the base
layer's mediated beta is) rests on bipartiteness, so this test pins it.
"""
from tsu.spec import load_spec
from tsu.passes.encode import encode
from tsu.passes.lower import lower
from tsu.passes.analyse import analyse


def test_overlay_is_bipartite_and_needs_no_mediators():
    """Spec section 1.2: a bipartite overlay is never mediated, which is why
    beta stays free on it while the base layer's beta is welded to compile time."""
    m = lower(encode(load_spec("specs/lattice_elev_band_8x8.yaml"), "domain_wall").model)
    rep = analyse(m)
    assert rep.bipartite is True
    assert rep.max_degree <= 16
    assert max(abs(x) for x in m.biases) <= 6.0
