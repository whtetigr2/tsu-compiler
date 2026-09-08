"""Task 5: assembling a world. These call the real sampler."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate, World, FIELD_PLAN
from world.fields import sample_field
from world.spline import TERRAINS, LEVELS, terrain_index, cost_of


def test_generate_returns_arrays_at_the_output_size():
    w = generate(size=64, seed=0, warmup=200)
    assert isinstance(w, World)
    assert w.height.shape == (64, 64)
    assert w.terrain.shape == (64, 64)
    assert w.cost.shape == (64, 64)


def test_all_three_parameter_fields_are_sampled_at_their_own_size():
    w = generate(size=64, seed=0, warmup=200)
    assert set(w.fields) == {name for name, _ in FIELD_PLAN}
    for name, src in FIELD_PLAN:
        assert w.fields[name].shape == (src, src)


def test_every_cell_is_walkable():
    """The property that makes traversability free. A cost of 0 or below would
    break Dijkstra; anything unreachable would reintroduce impassable terrain."""
    w = generate(size=64, seed=0, warmup=200)
    assert w.cost.min() >= 1


def test_terrain_indices_are_valid():
    w = generate(size=64, seed=0, warmup=200)
    assert w.terrain.min() >= 0
    assert w.terrain.max() < len(TERRAINS)


def test_terrain_and_cost_are_both_derived_from_the_worlds_own_height():
    """Recompute BOTH independently from `w.height`, by a different path than
    the generator used. The generator goes fields -> height -> terrain -> cost;
    this goes height -> terrain_index/cost_of. The previous version of this test
    built its expectation with the SAME expression generate.py uses on the SAME
    array, so it could only fail on memory corruption -- it never checked that
    terrain corresponds to the height the world reports, and would have passed
    with terrain derived from a stale or un-upsampled array."""
    w = generate(size=64, seed=0, warmup=200)
    for y in (0, 17, 63):
        for x in (0, 31, 63):
            h = float(w.height[y, x])
            assert w.terrain[y, x] == terrain_index(h)
            assert w.cost[y, x] == cost_of(h)


def test_same_seed_reproduces_the_same_world():
    a = generate(size=64, seed=3, warmup=200)
    b = generate(size=64, seed=3, warmup=200)
    assert np.array_equal(a.terrain, b.terrain)


def test_different_seeds_give_different_worlds():
    a = generate(size=64, seed=1, warmup=200)
    b = generate(size=64, seed=2, warmup=200)
    assert not np.array_equal(a.terrain, b.terrain)


def test_the_world_is_not_one_terrain():
    """A single-terrain world passes every shape check while being useless."""
    w = generate(size=64, seed=0, warmup=200)
    assert len(np.unique(w.terrain)) > 1


def test_a_size_that_is_not_a_multiple_of_64_is_refused():
    with pytest.raises(ValueError):
        generate(size=100, seed=0, warmup=200)


def test_the_three_fields_use_distinct_derived_seeds():
    """If all three fields shared the caller's seed, continentalness and erosion
    would be perfectly correlated and two of the three parameters would carry no
    information. Every other test in this file passes under that regression,
    because the fields have different GRID SIZES and so differ as arrays
    whatever the seed -- this is the only guard."""
    w = generate(size=64, seed=5, warmup=200)
    for i, (name, src) in enumerate(FIELD_PLAN):
        assert np.array_equal(
            w.fields[name],
            sample_field(src, LEVELS, w.beta_j, 5 * 1000 + i, warmup=200))
    first_name, first_src = FIELD_PLAN[0]
    assert not np.array_equal(
        w.fields[first_name],
        sample_field(first_src, LEVELS, w.beta_j, 5, warmup=200))
