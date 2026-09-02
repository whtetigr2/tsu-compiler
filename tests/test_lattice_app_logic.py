"""Headless tests for demo/lattice_app.py's pure logic units: ClampState
(pin bookkeeping), cell_at (canvas click -> grid cell), the infeasible-
clamp detector, and (UI2) the speed-control bookkeeping / responsive-pause
decision (SPEED_LEVELS, speed_level, should_abort_batch, SampleWorker's
speed/step attributes). NONE of these construct a Tk window -- importing
demo/lattice_app.py at module scope is safe (it only creates a Tk() instance
inside LatticeApp.__init__ / main(), never at import time), and every test
here calls only pure functions/classes, never LatticeApp itself.
"""
import queue
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
# spin_cell_position: physical spin index -> (cell_x, cell_y, slot), or None
# for a mediator spin. This is the fix for the LIVE LATTICE / DECODED WORLD
# misalignment: each cell owns `spins_per_cell` consecutive spins, and cells
# are emitted in row-major order, so this is the single source of truth the
# canvas build must use instead of a bare `divmod(i, cols)`.
# ---------------------------------------------------------------------------

def test_spin_maps_to_its_own_cell_block():
    """Spins g0_0__dw0 and g0_0__dw1 are indices 0 and 1 and both belong to
    cell (0,0); g1_0__dw0 is index 2 and belongs to cell (1,0)."""
    from lattice_app import spin_cell_position
    assert spin_cell_position(0, 128, 2, 8) == (0, 0, 0)
    assert spin_cell_position(1, 128, 2, 8) == (0, 0, 1)
    assert spin_cell_position(2, 128, 2, 8) == (1, 0, 0)
    assert spin_cell_position(16, 128, 2, 8) == (0, 1, 0)   # start of row y=1
    assert spin_cell_position(127, 128, 2, 8) == (7, 7, 1)  # last world spin


def test_mediator_spins_belong_to_no_cell():
    """Mediators are hidden spins introduced by edge subdivision. They are not
    part of any cell and must not be drawn as if they were."""
    from lattice_app import spin_cell_position
    assert spin_cell_position(128, 128, 2, 8) is None
    assert spin_cell_position(191, 128, 2, 8) is None


# ---------------------------------------------------------------------------
# Canvas geometry for the redrawn LIVE LATTICE panel -- verified
# programmatically (no GUI/screenshot access in this environment) instead of
# by launching the app. The load-bearing claim is that cell (x, y)'s block
# lands at the SAME pixel origin as DECODED WORLD's cell (x, y) via cell_at,
# because both use WORLD_CELL_PX as the pitch -- that congruence is exactly
# what makes a pinned shape read the same in both panels.
# ---------------------------------------------------------------------------

def test_cell_block_bounds_uses_world_panel_pitch():
    from lattice_app import cell_block_bounds, WORLD_CELL_PX
    # cell (2, 5) at 40px/cell -> [80,120) x [200,240), matching the
    # DECODED WORLD panel's own cell (2, 5) pixel bounds exactly.
    assert cell_block_bounds(2, 5, WORLD_CELL_PX) == (80, 200, 120, 240)


def test_cell_block_bounds_lines_up_with_decoded_world_cell_at():
    """A click anywhere inside cell (2, 5)'s LIVE LATTICE block, if it were
    thrown at the DECODED WORLD canvas at the same pixel offset, resolves to
    the SAME grid cell -- i.e. no mirror/transpose between the two panels."""
    from lattice_app import cell_block_bounds, WORLD_CELL_PX, cell_at
    x0, y0, x1, y1 = cell_block_bounds(2, 5, WORLD_CELL_PX)
    midpoint = ((x0 + x1) // 2, (y0 + y1) // 2)
    assert cell_at(*midpoint, 0, 0, WORLD_CELL_PX, 320, 320) == (2, 5)


def test_cell_block_bounds_origin_cell():
    from lattice_app import cell_block_bounds, WORLD_CELL_PX
    assert cell_block_bounds(0, 0, WORLD_CELL_PX) == (0, 0, 40, 40)


def test_spin_slot_rect_places_two_slots_side_by_side_not_stacked():
    from lattice_app import spin_slot_rect, WORLD_CELL_PX
    slot0 = spin_slot_rect(0, 0, 0, 2, WORLD_CELL_PX)
    slot1 = spin_slot_rect(0, 0, 1, 2, WORLD_CELL_PX)
    # same vertical band (not stacked)...
    assert slot0[1] == slot1[1] and slot0[3] == slot1[3]
    # ...but slot 1 starts where slot 0 ends (side by side, left to right).
    assert slot1[0] >= slot0[2]
    # both slots stay fully inside the cell's own block.
    bx0, by0, bx1, by1 = 0, 0, WORLD_CELL_PX, WORLD_CELL_PX
    for x0, y0, x1, y1 in (slot0, slot1):
        assert bx0 <= x0 < x1 <= bx1
        assert by0 <= y0 < y1 <= by1


def test_spin_slot_rect_last_cell_last_slot_matches_cell_block_bounds():
    """Cell (7, 7)'s last slot must stay inside the panel -- the case that
    would previously have been drawn as extra columns spilling rightward."""
    from lattice_app import spin_slot_rect, cell_block_bounds, WORLD_CELL_PX
    x0, y0, x1, y1 = spin_slot_rect(7, 7, 1, 2, WORLD_CELL_PX)
    bx0, by0, bx1, by1 = cell_block_bounds(7, 7, WORLD_CELL_PX)
    assert bx0 <= x0 < x1 <= bx1
    assert by0 <= y0 < y1 <= by1


def test_mediator_slot_rect_is_row_major_not_the_worlds_8_wide_grid():
    from lattice_app import mediator_slot_rect
    # slot 16 (index 16, 16 cols) starts a new row, at row-major (0, 1).
    x0, y0, x1, y1 = mediator_slot_rect(16, 16, 16)
    assert x0 < 16  # back at column 0
    assert y0 >= 16  # one row down


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


# ---------------------------------------------------------------------------
# UI2: SPEED_LEVELS / speed_level -- the speed control's data, and the
# structural guarantee that it cannot smuggle in an n_warmup or
# steps_per_sample override (which would change what is sampled, not just
# how fast it displays).
# ---------------------------------------------------------------------------

def test_speed_levels_nonempty_and_well_formed():
    assert len(la.SPEED_LEVELS) >= 2
    for lvl in la.SPEED_LEVELS:
        assert set(lvl.keys()) == {"label", "chains", "clamp_samples", "delay_s"}
        assert isinstance(lvl["label"], str) and lvl["label"]
        assert lvl["chains"] >= 1
        assert lvl["clamp_samples"] >= 1
        assert lvl["delay_s"] >= 0


def test_speed_levels_never_names_warmup_or_steps():
    # The structural guarantee: no speed level can touch n_warmup or
    # steps_per_sample, because the key isn't even there to read.
    for lvl in la.SPEED_LEVELS:
        assert "n_warmup" not in lvl
        assert "steps_per_sample" not in lvl


def test_speed_levels_last_entry_is_full_speed_default():
    assert la.speed_level(la.DEFAULT_SPEED_IDX)["label"] == "Full speed"
    assert la.DEFAULT_SPEED_IDX == len(la.SPEED_LEVELS) - 1


def test_speed_level_clamps_negative_index():
    assert la.speed_level(-5) == la.SPEED_LEVELS[0]


def test_speed_level_clamps_too_large_index():
    assert la.speed_level(999) == la.SPEED_LEVELS[-1]


def test_speed_level_in_range_returns_that_entry():
    for i, lvl in enumerate(la.SPEED_LEVELS):
        assert la.speed_level(i) == lvl


def test_slow_end_has_more_delay_and_smaller_batches_than_full_speed():
    slow = la.SPEED_LEVELS[0]
    full = la.speed_level(la.DEFAULT_SPEED_IDX)
    assert slow["delay_s"] > full["delay_s"]
    assert slow["chains"] <= full["chains"]
    assert slow["clamp_samples"] <= full["clamp_samples"]


# ---------------------------------------------------------------------------
# UI2: should_abort_batch -- the responsive-pause decision. Pure, no
# threads, no worker construction needed.
# ---------------------------------------------------------------------------

def test_should_abort_batch_when_stopping_regardless_of_anything_else():
    assert la.should_abort_batch(stopping=True, paused=False, is_step=False) is True
    assert la.should_abort_batch(stopping=True, paused=True, is_step=True) is True


def test_should_abort_batch_when_paused_and_not_a_step():
    assert la.should_abort_batch(stopping=False, paused=True, is_step=False) is True


def test_should_not_abort_batch_when_paused_but_this_call_is_a_step():
    # Step's entire point: push its one batch even though the worker is
    # sitting paused when the step is requested.
    assert la.should_abort_batch(stopping=False, paused=True, is_step=True) is False


def test_should_not_abort_batch_when_running_normally():
    assert la.should_abort_batch(stopping=False, paused=False, is_step=False) is False
    assert la.should_abort_batch(stopping=False, paused=False, is_step=True) is False


# ---------------------------------------------------------------------------
# UI2: SampleWorker's speed/step bookkeeping -- constructible without a real
# Receipt (its __init__ only stores references; nothing in this section
# starts the thread or touches thrml).
# ---------------------------------------------------------------------------

def _worker(**kw):
    return la.SampleWorker(receipt=None, out_q=queue.Queue(), seed_base=0, **kw)


def test_worker_defaults_to_full_speed():
    w = _worker()
    assert w.speed_idx == la.DEFAULT_SPEED_IDX


def test_worker_accepts_explicit_speed_idx():
    w = _worker(speed_idx=0)
    assert w.speed_idx == 0


def test_worker_set_speed_updates_live():
    w = _worker()
    w.set_speed(1)
    assert w.speed_idx == 1


def test_worker_step_evt_starts_clear():
    w = _worker()
    assert not w.step_evt.is_set()


def test_worker_request_step_sets_the_event():
    w = _worker()
    w.request_step()
    assert w.step_evt.is_set()
