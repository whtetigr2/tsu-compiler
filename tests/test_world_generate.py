"""Task 5: assembling a world. These call the real sampler."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate, World, FIELD_PLAN
from world.spline import TERRAINS


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


def test_cost_agrees_with_terrain():
    """cost and terrain are derived from the same height and must not drift."""
    w = generate(size=64, seed=0, warmup=200)
    expected = np.array([[TERRAINS[i].cost for i in row] for row in w.terrain])
    assert np.array_equal(w.cost, expected)


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
