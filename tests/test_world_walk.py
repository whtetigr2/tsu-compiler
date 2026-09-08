"""Task 8: walking a world. Movement is pure, so it is tested directly; only
the key loop is interactive and stays untested."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate
from world.export import write
from world_walk import load, step, render, PlayerState


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    write(generate(size=64, seed=0, warmup=200), d)
    return load(d / "world.json")


def test_load_reads_an_exported_world(loaded):
    assert loaded.size == 64
    assert loaded.terrain.shape == (64, 64)
    assert loaded.cost.shape == (64, 64)


def test_step_moves_and_charges_the_entered_cell(loaded):
    s = PlayerState(x=10, y=10, spent=0, steps=0)
    t = step(s, 1, 0, loaded)
    assert (t.x, t.y) == (11, 10)
    assert t.spent == int(loaded.cost[10, 11])
    assert t.steps == 1


def test_step_clamps_at_the_edge_and_charges_nothing(loaded):
    """Walking into the boundary must not move, spend, or count a step --
    charging for a move that did not happen would corrupt the cost readout."""
    s = PlayerState(x=0, y=5, spent=7, steps=3)
    t = step(s, -1, 0, loaded)
    assert (t.x, t.y, t.spent, t.steps) == (0, 5, 7, 3)


def test_every_cell_is_enterable(loaded):
    """The property the whole design rests on: no move is ever refused."""
    assert loaded.cost.min() >= 1


def test_render_marks_the_player_and_fits_the_window(loaded):
    s = PlayerState(x=32, y=32, spent=0, steps=0)
    out = render(s, loaded, radius=5)
    lines = out.splitlines()
    assert len(lines) == 11
    assert all(len(ln) == 11 for ln in lines)
    assert lines[5][5] == "@"
