"""Headless tests for demo/lattice_app.py's pure logic units: ClampState
(pin bookkeeping), cell_at (canvas click -> grid cell), and the infeasible-
clamp detector. NONE of these construct a Tk window -- importing
demo/lattice_app.py at module scope is safe (it only creates a Tk() instance
inside LatticeApp.__init__ / main(), never at import time), and every test
here calls only pure functions/classes, never LatticeApp itself.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import lattice_app as la  # noqa: E402

WATER, ROCK, GRASS = la.WATER, la.ROCK, la.GRASS


# ---------------------------------------------------------------------------
# ClampState: add / remove / cycle / clear / as_dict
# ---------------------------------------------------------------------------

def test_new_clamp_state_has_no_pins():
    cs = la.ClampState()
    assert cs.as_dict() == {}
    assert len(cs) == 0


def test_cycle_unpinned_to_water():
    cs = la.ClampState()
    result = cs.cycle(2, 3)
    assert result == WATER
    assert cs.get(2, 3) == WATER


def test_cycle_water_to_rock():
    cs = la.ClampState()
    cs.cycle(2, 3)  # -> water
    result = cs.cycle(2, 3)  # -> rock
    assert result == ROCK
    assert cs.get(2, 3) == ROCK


def test_cycle_rock_to_grass():
    cs = la.ClampState()
    cs.cycle(2, 3)  # water
    cs.cycle(2, 3)  # rock
    result = cs.cycle(2, 3)  # grass
    assert result == GRASS
    assert cs.get(2, 3) == GRASS


def test_cycle_grass_back_to_unpinned():
    cs = la.ClampState()
    cs.cycle(2, 3)  # water
    cs.cycle(2, 3)  # rock
    cs.cycle(2, 3)  # grass
    result = cs.cycle(2, 3)  # -> unpinned
    assert result is None
    assert cs.get(2, 3) is None


def test_cycle_full_loop_matches_spec_order():
    cs = la.ClampState()
    seq = [cs.cycle(0, 0) for _ in range(5)]
    assert seq == [WATER, ROCK, GRASS, None, WATER]


def test_add_sets_a_pin_directly():
    cs = la.ClampState()
    cs.add(4, 5, ROCK)
    assert cs.get(4, 5) == ROCK


def test_remove_clears_a_single_pin():
    cs = la.ClampState()
    cs.add(1, 1, WATER)
    cs.add(2, 2, GRASS)
    cs.remove(1, 1)
    assert cs.get(1, 1) is None
    assert cs.get(2, 2) == GRASS


def test_remove_on_unpinned_cell_is_a_no_op():
    cs = la.ClampState()
    cs.remove(6, 6)  # must not raise
    assert cs.get(6, 6) is None


def test_clear_removes_every_pin():
    cs = la.ClampState()
    cs.add(0, 0, WATER)
    cs.add(1, 1, ROCK)
    cs.add(2, 2, GRASS)
    cs.clear()
    assert cs.as_dict() == {}
    assert len(cs) == 0


def test_as_dict_uses_workload_variable_names():
    cs = la.ClampState()
    cs.add(2, 3, WATER)
    cs.add(7, 0, GRASS)
    assert cs.as_dict() == {"g2_3": WATER, "g7_0": GRASS}


def test_as_dict_omits_unpinned_cells_entirely_not_as_nulls():
    cs = la.ClampState()
    cs.add(1, 1, WATER)
    cs.cycle(1, 1)  # water -> rock, still pinned
    cs.add(3, 3, ROCK)
    cs.cycle(3, 3)  # rock -> grass
    cs.cycle(3, 3)  # grass -> unpinned
    d = cs.as_dict()
    assert "g3_3" not in d
    assert d == {"g1_1": ROCK}
    assert None not in d.values()


# ---------------------------------------------------------------------------
# cell_at: canvas pixel -> grid cell
# ---------------------------------------------------------------------------

def test_cell_at_click_inside_a_known_cell():
    # 8x8 grid, 40px cells, origin at (0, 0) -> cell (3, 5) spans
    # px [120,160), py [200,240)
    assert la.cell_at(130, 210, 0, 0, 40, 320, 320) == (3, 5)


def test_cell_at_click_inside_a_known_cell_with_nonzero_origin():
    assert la.cell_at(10 + 130, 20 + 210, 10, 20, 40, 320, 320) == (3, 5)


def test_cell_at_click_outside_grid_returns_none_negative():
    assert la.cell_at(-1, 10, 0, 0, 40, 320, 320) is None


def test_cell_at_click_outside_grid_returns_none_beyond_width():
    assert la.cell_at(320, 10, 0, 0, 40, 320, 320) is None


def test_cell_at_click_outside_grid_returns_none_beyond_height():
    assert la.cell_at(10, 320, 0, 0, 40, 320, 320) is None


def test_cell_at_click_before_origin_returns_none():
    assert la.cell_at(5, 5, 10, 10, 40, 320, 320) is None


def test_cell_at_boundary_pixels_land_in_exactly_one_cell():
    # The shared boundary between cell (2, y) and cell (3, y) is px=120.
    # px=119 must be cell 2's; px=120 must be cell 3's -- never both.
    assert la.cell_at(119, 10, 0, 0, 40, 320, 320)[0] == 2
    assert la.cell_at(120, 10, 0, 0, 40, 320, 320)[0] == 3


def test_cell_at_last_pixel_of_grid_lands_in_last_cell():
    assert la.cell_at(319, 319, 0, 0, 40, 320, 320) == (7, 7)


# ---------------------------------------------------------------------------
# infeasible-clamp detection: a pure function over classified draws
# ---------------------------------------------------------------------------

def _draw(kind):
    return {"kind": kind, "raw": [], "seed": 0}


def test_batch_feasibility_all_invalid_is_infeasible():
    draws = [_draw("contract-fail"), _draw("non-codeword"), _draw("contract-fail")]
    infeasible, reason = la.batch_feasibility(draws)
    assert infeasible is True
    assert "0 valid" in reason
    assert "3 draws" in reason


def test_batch_feasibility_some_valid_is_not_infeasible():
    draws = [_draw("valid"), _draw("contract-fail"), _draw("valid"), _draw("non-codeword")]
    infeasible, reason = la.batch_feasibility(draws)
    assert infeasible is False
    assert "2" in reason and "4" in reason


def test_batch_feasibility_empty_batch_is_not_asserted_infeasible():
    infeasible, reason = la.batch_feasibility([])
    assert infeasible is False
