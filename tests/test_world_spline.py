"""Task 1: the deterministic spline. Pure functions -- no sampling, no
randomness, no I/O -- so they are tested exhaustively over every attainable
input rather than sampled, the same way tests/test_cascade.py tests
`upsample` as ordinary Python logic."""
import sys
import itertools

import pytest

sys.path.insert(0, "demo")

from world.spline import LEVELS, TERRAINS, height, terrain_index, cost_of


def test_continentalness_raises_average_height():
    """The spec's whole claim about continentalness: larger c means higher
    terrain, holding the other two parameters fixed."""
    low = height(0.0, 0.5, 0.5)
    high = height(1.0, 0.5, 0.5)
    assert high > low


def test_erosion_flattens_relief():
    """Erosion enters MULTIPLICATIVELY, so a heavily eroded cell varies less
    with peaks-and-valleys than an uneroded one. This is the behaviour that
    would silently vanish if erosion were made additive."""
    uneroded = height(0.5, 0.0, 1.0) - height(0.5, 0.0, 0.0)
    eroded = height(0.5, 1.0, 1.0) - height(0.5, 1.0, 0.0)
    assert uneroded > eroded
    assert eroded >= 0.0


def test_peaks_and_valleys_raises_height():
    low = height(0.5, 0.0, 0.0)
    high = height(0.5, 0.0, 1.0)
    assert high > low


def test_every_attainable_level_triple_lands_in_a_terrain():
    """All 4^3 = 64 combinations of integer levels are reachable from real
    sampled fields, so every one must map to a valid terrain index."""
    n = 0
    for c, e, pv in itertools.product(range(LEVELS + 1), repeat=3):
        h = height(c / LEVELS, e / LEVELS, pv / LEVELS)
        idx = terrain_index(h)
        assert 0 <= idx < len(TERRAINS)
        assert cost_of(h) == TERRAINS[idx].cost
        n += 1
    assert n == 64


def test_height_stays_inside_the_designed_range():
    """base in [-1.00, 0.70] plus amp*relief in [-0.50, 0.80] -> [-1.50, 1.50].
    A height outside that means a spline knot was mistyped."""
    for c, e, pv in itertools.product(range(LEVELS + 1), repeat=3):
        h = height(c / LEVELS, e / LEVELS, pv / LEVELS)
        assert -1.5 <= h <= 1.5


def test_all_six_terrains_are_reachable_from_integer_levels():
    """A band nothing can reach is a dead band -- the thresholds and the
    spline range would disagree and no world would ever show that terrain."""
    seen = set()
    for c, e, pv in itertools.product(range(LEVELS + 1), repeat=3):
        seen.add(terrain_index(height(c / LEVELS, e / LEVELS, pv / LEVELS)))
    assert seen == set(range(len(TERRAINS)))


def test_every_terrain_is_walkable():
    """The design decision that makes every world traversable by construction:
    cost varies, permission never does. A zero or negative cost would break
    Dijkstra; an infinite one would reintroduce impassable terrain."""
    for t in TERRAINS:
        assert t.cost >= 1


def test_bilinear_upsampling_produces_fractional_levels_that_still_work():
    """Bilinear upsampling yields non-integer parameters, so the spline must
    INTERPOLATE between knots rather than index them. The exact value is
    asserted because a nearest-knot lookup also returns a float in a valid
    band -- so type and range checks cannot tell the two apart. Hand-computed:
    base(0.5) = 0.05, amp(0.5) = 0.55, relief(0.5) = 0.10,
    height = 0.05 + 0.55 * 0.10 = 0.105. A quantizing _spline gives 0.42."""
    assert height(0.5, 0.5, 0.5) == pytest.approx(0.105)
    assert 0 <= terrain_index(height(0.5, 0.5, 0.5)) < len(TERRAINS)
