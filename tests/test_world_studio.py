"""Task 2: the readout state machine. These tests build ONE world (slow, real
sampler) in a module fixture and then exercise the instant path against it."""
import dataclasses
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate, FIELD_PLAN
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
    """THE load-bearing performance property: if a readout slider reaches the
    sampler, the whole design is defeated and the only symptom is ~3.3s of lag
    per drag.

    The patch targets `world.generate.sample_field`, which is where the name is
    actually BOUND (generate.py does `from world.fields import sample_field` at
    import time, so patching `world.fields` would not affect it either). An
    earlier version patched `world.studio.sample_field` with `raising=False` --
    a name studio.py does not import, so it created an inert attribute and the
    test could never fail.

    No `raising=False`: if the attribute ever stops existing, this must fail
    loudly rather than pass silently, since a silent pass is the bug being
    fixed."""
    import world.generate as gen_mod

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "a readout control reached the sampler -- readout must recompute "
            "from the cached fields, never resample")

    monkeypatch.setattr(gen_mod, "sample_field", _forbidden)
    studio.set_readout(Readout(sea_level=0.25, relief=1.4, mountain_line=-0.2))
    studio.set_readout(Readout(cost_table="wide"))
    studio.view()


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


from world.studio import Sampled, BAND


def test_generate_samples_and_loads_in_one_call():
    s = Studio()
    v = s.generate(Sampled(seed=11, size=64, warmup=200))
    assert v.terrain.shape == (64, 64)
    assert s.sampled.seed == 11


def test_per_field_timings_are_apportioned_by_spin_count():
    """The apportionment CLAIM, not merely its presence.

    The previous version asserted each timing was > 0 and no larger than the
    total. Both follow algebraically from splitting a positive number into
    fractions summing to 1, so it passed identically for an equal split
    (0.333 each) or a reversed one. Measured spin shares at size 64 are
    0.01449 / 0.05797 / 0.92754, so a ratio assertion separates all three."""
    s = Studio()
    s.generate(Sampled(seed=14, size=64, warmup=200))
    spins = {name: int(s.world.fields[name].size) for name, _src in FIELD_PLAN}
    total_spins = sum(spins.values())
    for name, _src in FIELD_PLAN:
        assert s.timings[name] / s.timings["total"] == pytest.approx(
            spins[name] / total_spins), f"{name} not apportioned by spin count"
    assert sum(s.timings[n] for n, _src in FIELD_PLAN) == pytest.approx(
        s.timings["total"])


def test_the_measured_band_is_exposed_for_the_ui_to_mark():
    """The usable band is narrow and a world sampled outside it is noise or a
    single blob. The UI marks it, so the value lives here rather than in Tk."""
    assert BAND == (0.38, 0.45)


def test_warmup_is_reported_and_cannot_be_mutated():
    """"Reported, not a control" mechanically means: it has a fixed default, it
    cannot be reassigned, and nothing on Studio offers to change it.

    The previous version asserted only the default value while its docstring
    claimed the absence of a setter was the point -- so a mutable dataclass with
    a slider bound to it would have passed."""
    s = Sampled()
    assert s.warmup == 4000
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.warmup = 8000
    assert [a for a in dir(Studio) if "warmup" in a.lower()] == []


def test_generate_handles_a_size_other_than_64():
    """Task 2's cache path derives each field's upsample factor from the raw
    field's own shape rather than recomputing it, so a non-64 size is the case
    that would expose a mismatch between what generate() sampled and what the
    studio thinks it sampled. 128 is the only other size the UI offers."""
    s = Studio()
    v = s.generate(Sampled(seed=13, size=128, warmup=200))
    assert v.terrain.shape == (128, 128)
    assert v.cost.min() >= 1
