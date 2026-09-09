"""Task 1: parameterised spline entry points. The existing scalar `height` and
`terrain_index` stay exactly as they are; these are additive array paths that a
live slider can afford to call."""
import sys
import itertools

import numpy as np
import pytest

sys.path.insert(0, "demo")

from world.spline import (LEVELS, TERRAINS, DEFAULT_BOUNDS, BOUND_EPS,
                          height, terrain_index, shifted_bounds,
                          height_array, terrain_array)


def test_height_array_matches_the_scalar_spline_exactly():
    """The array path is an optimisation, not a second implementation, and
    BIT-IDENTICAL is the requirement rather than merely close: the result feeds
    an integer terrain classification, where a last-ULP difference at a band
    boundary flips a cell to the neighbouring terrain. An earlier np.interp
    version was equal to ~2e-16 and did exactly that."""
    vals = [i / LEVELS for i in range(LEVELS + 1)] + [0.17, 0.5, 0.83]
    c, e, pv = (np.array(x, dtype=float) for x in zip(*itertools.product(vals, repeat=3)))
    fast = height_array(c, e, pv)
    slow = np.array([height(a, b, d) for a, b, d in zip(c, e, pv)])
    assert np.array_equal(fast, slow), "array and scalar paths must be bit-identical"


def test_terrain_array_matches_the_scalar_index_exactly():
    h = np.linspace(-1.6, 1.6, 401)
    fast = terrain_array(h, DEFAULT_BOUNDS)
    slow = np.array([terrain_index(v) for v in h])
    assert np.array_equal(fast, slow)


def test_relief_scales_the_amplitude_not_the_base():
    """relief must multiply the erosion x peaks-and-valleys term ONLY, leaving
    the continentalness base untouched.

    Exact values are asserted because an inequality cannot separate the two.
    At c=0.5, e=0.0, pv=0.5 the parts are base=0.05, amp=1.00, relief knot=0.10,
    so the correct height is 0.05 + relief*1.00*0.10 -- 0.15 at relief 1 and
    0.25 at relief 2. A version that scaled the BASE instead would give 0.15 and
    0.20: it also changes with relief, which is why `assert not isclose` passed
    under the very bug this test is named for."""
    c, e, pv = np.array([0.5]), np.array([0.0]), np.array([0.5])
    assert height_array(c, e, pv, relief=1.0)[0] == pytest.approx(0.15)
    assert height_array(c, e, pv, relief=2.0)[0] == pytest.approx(0.25)


def test_relief_of_one_is_the_shipped_behaviour():
    c = np.array([0.0, 0.5, 1.0]); e = np.array([0.2, 0.5, 0.9]); pv = np.array([1.0, 0.3, 0.0])
    assert np.allclose(height_array(c, e, pv, relief=1.0),
                       [height(a, b, d) for a, b, d in zip(c, e, pv)], atol=1e-12)


def test_sea_level_raises_the_water_bounds():
    """Raising sea level floods: the water bounds move up, so more cells fall
    below them."""
    up = shifted_bounds(sea_level=0.2)
    assert up[0] > DEFAULT_BOUNDS[0]
    assert up[1] > DEFAULT_BOUNDS[1]


def test_mountain_line_slider_up_yields_more_mountain():
    """Mountain is everything ABOVE the last bound, so more mountain means the
    bound moves DOWN as the slider goes up."""
    assert shifted_bounds(mountain_line=0.3)[4] < DEFAULT_BOUNDS[4]


def test_bounds_stay_strictly_increasing_under_extreme_sliders():
    """A reordered band table produces terrain that no longer means what its
    name says. Every reachable slider combination must stay monotone."""
    for sea in np.linspace(-0.5, 0.5, 11):
        for mtn in np.linspace(-0.5, 0.5, 11):
            b = shifted_bounds(float(sea), float(mtn))
            assert all(b[i + 1] >= b[i] + BOUND_EPS for i in range(len(b) - 1)), (sea, mtn)


def test_defaults_reproduce_the_shipped_bounds():
    assert shifted_bounds() == DEFAULT_BOUNDS


def test_terrain_array_with_shifted_bounds_still_covers_every_index():
    h = np.linspace(-1.6, 1.6, 401)
    t = terrain_array(h, shifted_bounds(0.3, 0.3))
    assert t.min() >= 0 and t.max() < len(TERRAINS)
