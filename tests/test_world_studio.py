"""Task 2: the readout state machine. These tests build ONE world (slow, real
sampler) in a module fixture and then exercise the instant path against it."""
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate
from world.spline import TERRAINS, DEFAULT_BOUNDS, height_array, terrain_array
from world.studio import Studio, Readout, View, COST_TABLES


@pytest.fixture(scope="module")
def studio():
    s = Studio()
    s.load(generate(size=64, seed=3, warmup=200))
    return s


def test_default_readout_reproduces_the_generators_own_terrain(studio):
    """The studio must agree with `generate` when no slider has moved. If it
    does not, every later comparison is against a different world than the one
    the generator and the tests elsewhere produced."""
    v = studio.set_readout(Readout(relief=1.0, sea_level=0.0, mountain_line=0.0))
    assert np.array_equal(v.terrain, studio.world.terrain)


def test_sea_level_up_makes_more_water(studio):
    dry = studio.set_readout(Readout(sea_level=-0.3))
    wet = studio.set_readout(Readout(sea_level=+0.3))
    water = lambda v: int((v.terrain <= 1).sum())
    assert water(wet) > water(dry)


def test_mountain_line_up_makes_more_mountain(studio):
    low = studio.set_readout(Readout(mountain_line=-0.3))
    high = studio.set_readout(Readout(mountain_line=+0.3))
    peak = len(TERRAINS) - 1
    assert int((high.terrain == peak).sum()) > int((low.terrain == peak).sum())


def test_relief_changes_the_height_spread(studio):
    flat = studio.set_readout(Readout(relief=0.5))
    rugged = studio.set_readout(Readout(relief=2.0))
    assert rugged.height.std() > flat.height.std()


def test_cost_table_changes_cost_but_never_terrain(studio):
    """Cost is a separate lookup from terrain. A cost table that moved terrain
    would mean the palette and the routing disagree about the same world."""
    a = studio.set_readout(Readout(cost_table="shipped"))
    ta, ca = a.terrain.copy(), a.cost.copy()
    b = studio.set_readout(Readout(cost_table="wide"))
    assert np.array_equal(b.terrain, ta)
    assert not np.array_equal(b.cost, ca)


def test_every_cost_table_keeps_every_cell_walkable(studio):
    """The property the whole design rests on: terrain sets cost, never
    permission. A zero or negative cost would break Dijkstra."""
    for name in COST_TABLES:
        assert studio.set_readout(Readout(cost_table=name)).cost.min() >= 1


def test_readout_never_resamples(studio, monkeypatch):
    """THE load-bearing performance property. If a slider reaches the sampler
    the whole design is defeated, and it would be invisible except as lag."""
    import world.studio as mod
    called = []
    monkeypatch.setattr(mod, "sample_field",
                        lambda *a, **k: called.append(1), raising=False)
    studio.set_readout(Readout(sea_level=0.25, relief=1.4, mountain_line=-0.2))
    assert called == []


def test_readout_is_fast_enough_for_a_slider(studio):
    """Not a microbenchmark -- a floor. Anything near the 3.3s sample time means
    the cache boundary is in the wrong place."""
    studio.set_readout(Readout(sea_level=0.1))
    t0 = time.time()
    for i in range(5):
        studio.set_readout(Readout(sea_level=0.02 * i))
    assert (time.time() - t0) / 5 < 0.25


def test_view_bounds_reflect_the_sliders(studio):
    v = studio.set_readout(Readout(sea_level=0.2))
    assert v.bounds[0] > DEFAULT_BOUNDS[0]


def test_an_unknown_cost_table_is_refused(studio):
    with pytest.raises(ValueError):
        studio.set_readout(Readout(cost_table="lunar"))


def test_view_before_load_is_refused():
    with pytest.raises(RuntimeError):
        Studio().view()
