"""Task 2/3: the elevation overlay spec and the thermometer logic on top of it.

Task 2 (this section): specs/lattice_elev_band_8x8.yaml compiles to a
bipartite, unmediated 64-spin overlay -- spec section 1.2's claim that beta
stays free on an overlay (never welded to compile time the way the base
layer's mediated beta is) rests on bipartiteness, so this test pins it.
"""
import sys

from tsu.spec import load_spec
from tsu.passes.encode import encode
from tsu.passes.lower import lower
from tsu.passes.analyse import analyse

sys.path.insert(0, "demo")


def test_overlay_is_bipartite_and_needs_no_mediators():
    """Spec section 1.2: a bipartite overlay is never mediated, which is why
    beta stays free on it while the base layer's beta is welded to compile time."""
    m = lower(encode(load_spec("specs/lattice_elev_band_8x8.yaml"), "domain_wall").model)
    rep = analyse(m)
    assert rep.bipartite is True
    assert rep.max_degree <= 16
    assert max(abs(x) for x in m.biases) <= 6.0


# --------------------------------------------------------------------------
# Task 3: thermometer elevation -- N binary overlays counted give N+1
# ordinal elevation levels. See demo/elevation.py's module docstring.
# --------------------------------------------------------------------------

def test_thermometer_level_counts_true_bands():
    from elevation import thermometer_level
    bands = [{"g0_0": 1}, {"g0_0": 1}, {"g0_0": 0}]
    assert thermometer_level(bands, "g0_0") == 2


def test_level_zero_when_no_band_is_true():
    from elevation import thermometer_level
    assert thermometer_level([{"g0_0": 0}, {"g0_0": 0}], "g0_0") == 0


def test_monotonicity_violation_is_detected_not_repaired():
    """Band 1 true where band 0 is false is a broken thermometer code. It must
    be REPORTED -- silently sorting the bands would hide the very rate the
    spec (section 3.1) asks us to measure."""
    from elevation import monotonicity_violations
    bands = [{"g0_0": 0, "g1_0": 1}, {"g0_0": 1, "g1_0": 1}]
    assert monotonicity_violations(bands) == [("g0_0", 1)]


def test_band_patch_encourages_cells_where_the_band_below_is_true():
    from elevation import band_patch
    prev = {"g0_0": 1, "g1_0": 0}
    patch = band_patch(prev, base={"g0_0": 2, "g1_0": 2}, strength=0.05)
    assert patch[("g0_0", 1)] > 0     # below is high -> encourage
    assert patch[("g1_0", 1)] < 0     # below is low  -> discourage
