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


def test_step_charges_the_cell_entered_not_the_cell_left(loaded):
    """Search for an adjacent pair whose costs DIFFER, so that charging the
    origin and charging the destination give different answers. The previous
    version moved between two cells that both cost 1, where a regression
    charging the cell left behind produced the identical total and passed.

    Raises rather than skipping if no such pair exists -- a fixture that went
    uniform would make this test silently non-discriminating, which is the
    exact failure mode being fixed."""
    for y in range(loaded.size):
        for x in range(loaded.size - 1):
            here, there = int(loaded.cost[y, x]), int(loaded.cost[y, x + 1])
            if here != there:
                t = step(PlayerState(x=x, y=y, spent=0, steps=0), 1, 0, loaded)
                assert (t.x, t.y) == (x + 1, y)
                assert t.spent == there, "must charge the cell ENTERED"
                assert t.spent != here, "and not the cell left behind"
                assert t.steps == 1
                return
    raise AssertionError(
        "no adjacent pair with differing cost in this fixture -- the test "
        "cannot discriminate and must not silently pass")


def test_step_clamps_at_the_edge_and_charges_nothing(loaded):
    """Walking into the boundary must not move, spend, or count a step --
    charging for a move that did not happen would corrupt the cost readout."""
    s = PlayerState(x=0, y=5, spent=7, steps=3)
    t = step(s, -1, 0, loaded)
    assert (t.x, t.y, t.spent, t.steps) == (0, 5, 7, 3)


def test_every_cell_is_enterable(loaded):
    """The property the whole design rests on: no move is ever refused."""
    assert loaded.cost.min() >= 1


def test_render_is_not_transposed_and_shows_the_real_terrain(loaded):
    """x != y deliberately. At x == y a transposed render produces an identical
    window, so the previous diagonal fixture could not see the bug. This also
    compares every cell of the window against the world, so a reversed or
    misordered glyph list fails here too -- the length and marker checks alone
    could not tell.

    Position and radius are chosen so the window does not touch an edge, which
    keeps the offset arithmetic exact; the corner case is covered separately."""
    x, y, r = 40, 17, 4
    lines = render(PlayerState(x=x, y=y, spent=0, steps=0), loaded,
                   radius=r).splitlines()
    assert len(lines) == 2 * r + 1
    assert all(len(ln) == 2 * r + 1 for ln in lines)
    assert lines[r][r] == "@"
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx, dy) == (0, 0):
                continue
            assert lines[r + dy][r + dx] == loaded.glyphs[
                loaded.terrain[y + dy, x + dx]], f"mismatch at offset {(dx, dy)}"


def test_render_at_a_corner_keeps_the_window_square(loaded):
    """The clamp must keep the window exactly (2r+1) square at a corner rather
    than truncating it, with the player at its true offset rather than recentred."""
    r = 6
    lines = render(PlayerState(x=0, y=0, spent=0, steps=0), loaded,
                   radius=r).splitlines()
    assert len(lines) == 2 * r + 1
    assert all(len(ln) == 2 * r + 1 for ln in lines)
    assert lines[0][0] == "@"
