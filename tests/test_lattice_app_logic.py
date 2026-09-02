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


# ---------------------------------------------------------------------------
# Task 7: LAYERS panel pure logic -- band_index_from_name, overlay_pin_patch,
# composite_missing_layers, grid_to_decoded, and ClampState's per-instance
# `cycle` override. No Tk. (layer_supports_temperature, formerly tested
# here, was removed as dead code -- superseded by demo/scope.py's
# temperature_control_state, see Task 8; nothing in the app called it.)
# ---------------------------------------------------------------------------

def test_band_index_from_name_parses_the_trailing_digit():
    assert la.band_index_from_name("band0") == 0
    assert la.band_index_from_name("band2") == 2


def test_band_index_from_name_rejects_non_band_names():
    import pytest
    with pytest.raises(ValueError):
        la.band_index_from_name("base")
    with pytest.raises(ValueError):
        la.band_index_from_name("composite")


def test_overlay_pin_patch_empty_for_no_pins():
    assert la.overlay_pin_patch([]) == {}


def test_overlay_pin_patch_signs_match_pinned_value():
    patch = la.overlay_pin_patch([((0, 0), 1), ((2, 3), 0)], strength=2.0)
    assert patch[("g0_0", 1)] == 2.0    # pinned "above" -> encourage value 1
    assert patch[("g2_3", 1)] == -2.0   # pinned "below" -> discourage value 1


def test_composite_missing_layers_lists_base_and_every_missing_band():
    missing = la.composite_missing_layers(None, [None, "decoded", None])
    assert missing == ["base", "band0", "band2"]


def test_composite_missing_layers_empty_when_everything_present():
    assert la.composite_missing_layers("base-grid", ["b0", "b1", "b2"]) == []


def test_grid_to_decoded_uses_gx_y_row_major_convention():
    import numpy as np
    grid = np.array([[0, 1], [2, 3]])  # grid[y, x]
    d = la.grid_to_decoded(grid)
    assert d == {"g0_0": 0, "g1_0": 1, "g0_1": 2, "g1_1": 3}


def test_clamp_state_default_cycle_is_unchanged():
    cs = la.ClampState()
    assert [cs.cycle(0, 0) for _ in range(4)] == [WATER, ROCK, GRASS, None]


def test_clamp_state_custom_cycle_for_a_binary_overlay_layer():
    cs = la.ClampState(cycle=(0, 1))
    assert [cs.cycle(0, 0) for _ in range(3)] == [0, 1, None]


# ---------------------------------------------------------------------------
# I6 (fix-round-2): Receipt must load an unmediated (bipartite) overlay
# receipt cleanly. demo/receipts/elev_band/passes.json has a "mediation"
# key present with value null (place() ran the mediation pass and recorded
# "nothing to mediate", not "mediation didn't run") -- `.get("mediation")
# or {}` folds that null the same honest way as a genuinely absent key, so
# `med.get(...)` never runs on None. Before this test, no test anywhere in
# this suite constructed lattice_app.Receipt at all (`grep -rn
# "lattice_app.Receipt\|import Receipt" tests/` returned nothing) -- this
# fix was one revert away from silently restoring an app that crashes on
# every unmediated overlay layer, every band, every session.
# ---------------------------------------------------------------------------

def test_receipt_loads_cleanly_for_an_unmediated_bipartite_overlay():
    r = la.Receipt(la.OVERLAY_RECEIPT_DIR)
    assert r.mediator_count == 0


# ---------------------------------------------------------------------------
# Task 0: lattice_app's own palette constants must trace back to
# demo/theme.py's tokens, applied per the SEMANTIC rule (blue = cold /
# locked / mediator ONLY, never decorative) -- not a second, hand-copied
# palette that can drift from theme.py. The pre-Task-0 palette was flat
# grey/cyan and, worse, used blue DECORATIVELY for WORLD_ON/WORLD_OFF (the
# LIVE LATTICE panel's own live/lit world p-bit colour) while a separate
# purple pair (MEDIATOR_ON/OFF) stood in for the mediator (cold) channel --
# exactly backwards from the mockup's own construction note ("Lit gold
# nodes are world p-bits in state 1. Cold-blue nodes are hidden mediators
# -- frozen helpers, not terrain."). These tests pin the corrected mapping.
# ---------------------------------------------------------------------------

def test_window_and_panel_grounds_use_theme_tokens():
    import theme
    assert la.BG == theme.PAGE
    assert la.PANEL_BG == theme.PANEL
    assert la.BORDER == theme.BEZEL
    assert la.FG == theme.CREAM
    assert la.DIM == theme.CREAM_DIM


def test_status_colours_stay_off_the_decorative_accent_channel():
    """PASS/FAIL/WARN are semantic status colour, kept distinct from the
    gold/orange/blue 'this is live/hot/cold data' accent channel (olive
    PASS / orange warn / red FAIL, per the mockup's own legend)."""
    import theme
    assert la.GOOD == theme.STATUS_PASS == theme.OLIVE
    assert la.BAD == theme.STATUS_FAIL == theme.RED
    assert la.WARN == theme.STATUS_WARN == theme.ORANGE
    assert la.GOOD != theme.GOLD  # PASS must never read as "live data"


def test_accent_label_colour_is_gold_not_decorative_blue():
    import theme
    assert la.ACCENT == theme.GOLD
    assert la.ACCENT != theme.BLUE


def test_world_pbits_are_gold_lit_never_blue():
    """THE fix: world p-bits (LIVE LATTICE's own live/lit channel) must be
    gold when on, never blue -- blue is reserved for cold/mediator/locked
    only. Before Task 0, WORLD_ON was "#6fb3ff" (blue) -- a stray
    decorative blue the tokens spec calls out by name as the exact mistake
    that destroys the scheme's learnability."""
    import theme
    assert la.WORLD_ON == theme.GOLD
    assert la.WORLD_OFF == theme.GOLD_GHOST
    assert la.WORLD_ON != theme.BLUE
    assert la.WORLD_OFF != theme.BLUE
    assert la.WORLD_OFF != theme.BLUE_DEEP


def test_mediator_spins_are_the_cold_blue_channel():
    """Mediator spins are COLD by definition (spec: 'mediator spins' is one
    of the three things blue means) -- lit and unlit mediator both stay in
    the blue family, never purple/violet as before Task 0."""
    import theme
    assert la.MEDIATOR_ON == theme.BLUE_LIT
    assert la.MEDIATOR_OFF == theme.BLUE_DEEP


def test_terrain_palette_matches_the_tokens_spec_verbatim():
    import theme
    expected = [theme.WATER, theme.ROCK, theme.GRASS]
    for (r, g, b), hexval in zip(la.PAL, expected):
        h = hexval.lstrip("#")
        assert (int(r), int(g), int(b)) == (
            int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def test_inset_recessed_areas_use_the_inset_token():
    """The old hand-picked "#111218" (log box / frontier box / SCOPE plot
    canvases / PLOT_BG) must become theme.INSET -- one recessed-area colour,
    not a second literal copied by hand into 6 call sites."""
    import theme
    assert la.PLOT_BG == tuple(
        int(theme.INSET.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Task 8: render_temperature_track -- the beta slider TRACK, drawn on a
# PIL image (numpy/PIL blitted to a Canvas) rather than a native tk.Scale.
# The tokens spec calls this out by name as a trap: a native Scale cannot
# be gold, and in the locked state it always LOOKS disabled -- exactly the
# wrong message for a control that is refusing a change on purpose, not
# broken. These tests check actual pixel content: the adjustable state's
# thermal ramp (cold -> gold -> hot) and the locked state's solid cold
# field with a fixed-centre thumb that does NOT move with value_frac.
# ---------------------------------------------------------------------------

def _hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _close(a, b, tol=24):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_temperature_track_is_the_requested_size():
    import theme
    img = la.render_temperature_track(240, 28, adjustable=True, value_frac=0.5)
    assert img.size == (240, 28)


def test_adjustable_track_is_a_cold_to_hot_thermal_ramp():
    """Verbatim the tokens spec's own 3-stop gradient: blue_deep -> gold
    at 55% -> orange. value_frac=0.5 puts the thumb at CENTRE, so both
    edges and the 55% stop are sampled well clear of it -- the thumb
    overlay never contaminates the ramp-colour check."""
    import theme
    img = la.render_temperature_track(240, 28, adjustable=True, value_frac=0.5)
    w, h = img.size
    mid_y = h // 2
    cold_px = img.getpixel((2, mid_y))
    gold_px = img.getpixel((int(0.55 * (w - 1)), mid_y))
    hot_px = img.getpixel((w - 3, mid_y))
    assert _close(cold_px, _hexrgb(theme.BLUE_DEEP))
    assert _close(gold_px, _hexrgb(theme.GOLD))
    assert _close(hot_px, _hexrgb(theme.ORANGE))


def test_adjustable_thumb_tracks_value_frac():
    import theme
    img_cold = la.render_temperature_track(240, 28, adjustable=True, value_frac=0.0)
    img_hot = la.render_temperature_track(240, 28, adjustable=True, value_frac=1.0)
    mid_y = 28 // 2
    # A thumb at value_frac=0.0 puts gold-thumb pixels near x=0; at
    # value_frac=1.0 the SAME near-x=0 region is instead cold ramp colour
    # (no thumb there any more) -- proof the thumb actually moved.
    assert _close(img_cold.getpixel((2, mid_y)), _hexrgb(theme.GOLD))
    assert not _close(img_hot.getpixel((2, mid_y)), _hexrgb(theme.GOLD))


def test_locked_track_is_a_solid_cold_field_never_the_live_ramp():
    """No 'useful window' exists when locked -- the whole track is one
    solid blue_deep field, not a greyed-out copy of the gradient."""
    import theme
    img = la.render_temperature_track(240, 28, adjustable=False, value_frac=0.9)
    w, h = img.size
    mid_y = h // 2
    # Sample away from both the thumb (frozen at centre) and the lock-bar
    # (drawn across the middle row) -- e.g. near the left edge, one row
    # off centre.
    bg_px = img.getpixel((6, mid_y - 4))
    assert _close(bg_px, _hexrgb(theme.BLUE_DEEP))


def test_locked_thumb_is_frozen_at_centre_regardless_of_value_frac():
    """This is the whole point of Task 8's locked state: the thumb does
    NOT read `value_frac` (there is nothing to show a live position of).
    A value_frac near 0 or 1 must not move it -- it must stay at 50%."""
    import theme
    w = 240
    center_x = w // 2
    mid_y = 28 // 2
    for value_frac in (0.0, 0.5, 1.0):
        img = la.render_temperature_track(w, 28, adjustable=False, value_frac=value_frac)
        assert _close(img.getpixel((center_x, mid_y)), _hexrgb(theme.BLUE_LIT))


def test_locked_and_adjustable_tracks_are_visually_distinct_at_the_same_value():
    """The two states must be distinguishable at a glance -- same
    value_frac, different pixels, at a point on the track away from the
    thumb (the thumb colour alone already differs, but the TRACK itself
    must too: solid blue vs. a gradient)."""
    import theme
    live = la.render_temperature_track(240, 28, adjustable=True, value_frac=0.5)
    locked = la.render_temperature_track(240, 28, adjustable=False, value_frac=0.5)
    # Near the left edge (well clear of either state's thumb/lock-bar
    # geometry): live shows the cold end of the ramp, locked shows the
    # solid field -- both nominally "blue_deep-ish" at x=2, so instead
    # compare a point further right where live has visibly warmed toward
    # gold but locked is still flat blue_deep.
    x = int(0.35 * 240)
    mid_y = 28 // 2
    live_px = live.getpixel((x, mid_y))
    locked_px = locked.getpixel((x, mid_y))
    assert not _close(live_px, locked_px, tol=10)


# ---------------------------------------------------------------------------
# Task 8: get_beta_override / set_beta_override -- WHERE a layer's chosen
# beta override lives, generalised over base and bands. A prior review
# flagged `_refresh_temperature_control` reaching straight into
# `self.bands[self.active_layer]`, which raises KeyError the instant
# "base" is the active layer, because base has no LayerState. These pure
# functions replace that direct indexing everywhere in the app.
# ---------------------------------------------------------------------------

class _FakeAppForBase:
    def __init__(self, base_beta_override=None):
        self.base_beta_override = base_beta_override


def test_get_beta_override_reads_the_base_slot_for_base():
    app = _FakeAppForBase(base_beta_override=2.5)
    assert la.get_beta_override("base", app.base_beta_override, {}) == 2.5


def test_get_beta_override_does_not_keyerror_when_base_has_no_bands_entry():
    """THE fix: before this task, the equivalent inline code indexed
    `self.bands["base"]` directly and raised KeyError -- base is
    deliberately absent from `bands` (see LayerState's own docstring).
    This must return cleanly instead."""
    bands = {"band0": la.LayerState("band0", la.ClampState(cycle=(0, 1)))}
    result = la.get_beta_override("base", None, bands)
    assert result is None  # no exception


def test_get_beta_override_reads_a_bands_layer_state():
    band = la.LayerState("band0", la.ClampState(cycle=(0, 1)))
    band.beta_override = 1.75
    assert la.get_beta_override("band0", None, {"band0": band}) == 1.75


def test_get_beta_override_returns_none_for_an_unknown_layer():
    assert la.get_beta_override("composite", None, {}) is None


def test_set_beta_override_writes_the_base_slot_for_base():
    app = _FakeAppForBase()
    la.set_beta_override("base", 3.0, app, {})
    assert app.base_beta_override == 3.0


def test_set_beta_override_writes_a_bands_layer_state_not_the_base_slot():
    app = _FakeAppForBase(base_beta_override=None)
    band = la.LayerState("band0", la.ClampState(cycle=(0, 1)))
    la.set_beta_override("band0", 0.6, app, {"band0": band})
    assert band.beta_override == 0.6
    assert app.base_beta_override is None  # untouched


def test_temperature_value_frac_clamps_into_zero_one():
    assert la.temperature_value_frac(0.3, 0.3, 3.0) == 0.0
    assert la.temperature_value_frac(3.0, 0.3, 3.0) == 1.0
    assert la.temperature_value_frac(-10.0, 0.3, 3.0) == 0.0   # clamped, not negative
    assert la.temperature_value_frac(999.0, 0.3, 3.0) == 1.0   # clamped, not >1
    mid = la.temperature_value_frac(1.65, 0.3, 3.0)
    assert 0.0 < mid < 1.0


def test_temperature_value_frac_rejects_a_degenerate_range():
    import pytest
    with pytest.raises(ValueError):
        la.temperature_value_frac(1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Task 10: DetachRegistry -- detaching a plot into its own Toplevel then
# closing that window must leave no orphaned window and no live-update
# callback still scheduled (task-10-brief.md's own leak scenario: "a closed
# window still receiving after() updates is a slow leak that only shows up
# after a long session, which is exactly when a demo is being given").
# Pure Python here -- no real tk.Toplevel is constructed; a fake window/
# cancel stand-in is enough to exercise the registry's own bookkeeping
# (same "no Tk in pure logic" convention this file's own module docstring
# states). The Tk-layer wiring itself (LatticeApp._open_detach, which
# builds a real Toplevel and schedules real self.after() jobs) is not
# unit-tested -- it is checked by hand, launching the real app, the same
# way every other geometry/Tk-wiring concern in this app already is.
# ---------------------------------------------------------------------------

class _FakeDetachWindow:
    """Stand-in for a real tk.Toplevel -- just enough surface for
    DetachRegistry's own contract (which never calls a Tk method itself)."""

    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def test_detachable_plots_lists_exactly_the_five_named_in_the_brief():
    """node-and-edge lattice, relaxation strip, per-cell heatmap, sigmoid
    response, energy histogram -- task-10-brief.md's own "Detachable"
    list, no more, no fewer. The energy/valid-fraction/magnetization
    traces are explicitly "stays inline" in the same brief and must NOT
    appear here."""
    assert set(la.DETACHABLE_PLOTS) == {
        "lattice_graph", "relaxation", "heatmap", "sigmoid", "hist"}
    assert len(la.DETACHABLE_PLOTS) == 5


def test_detach_registry_starts_with_nothing_open():
    reg = la.DetachRegistry()
    for key in la.DETACHABLE_PLOTS:
        assert not reg.is_open(key)
        assert reg.window_for(key) is None


def test_detaching_then_closing_leaves_no_orphaned_window_and_cancels_the_callback():
    reg = la.DetachRegistry()
    window = _FakeDetachWindow()
    cancelled = {"n": 0}
    reg.open_window("sigmoid", window,
                     lambda: cancelled.__setitem__("n", cancelled["n"] + 1))
    assert reg.is_open("sigmoid")
    assert reg.window_for("sigmoid") is window

    reg.close("sigmoid")

    assert not reg.is_open("sigmoid")
    assert reg.window_for("sigmoid") is None
    # the live update callback was cancelled -- exactly once, not zero:
    assert cancelled["n"] == 1


def test_closing_twice_cancels_the_callback_only_once():
    """A second close() on an already-closed key must be a harmless
    no-op, not a double-cancel -- Tk's own after_cancel is not documented
    as side-effect-free on an id that's already been cancelled, so the
    registry itself (not luck) is what makes double-cancel impossible."""
    reg = la.DetachRegistry()
    cancelled = {"n": 0}
    reg.open_window("hist", _FakeDetachWindow(),
                     lambda: cancelled.__setitem__("n", cancelled["n"] + 1))
    reg.close("hist")
    reg.close("hist")
    assert cancelled["n"] == 1


def test_close_on_a_never_opened_key_is_a_harmless_noop():
    reg = la.DetachRegistry()
    reg.close("heatmap")  # never opened -- must not raise, must not call anything
    assert not reg.is_open("heatmap")


def test_detach_registry_rejects_reopening_an_already_open_key():
    """The Tk layer is responsible for LIFTING an existing window (the
    same singleton pattern demo/explainer.py's own _on_show_explainer
    already uses) -- the registry itself refuses a second open_window for
    an already-open key, so that path can never silently leak the first
    window's own handle/cancel callable."""
    import pytest
    reg = la.DetachRegistry()
    reg.open_window("lattice_graph", _FakeDetachWindow(), lambda: None)
    with pytest.raises(RuntimeError):
        reg.open_window("lattice_graph", _FakeDetachWindow(), lambda: None)


def test_detach_registry_rejects_unknown_plot_keys():
    import pytest
    reg = la.DetachRegistry()
    with pytest.raises(KeyError):
        reg.is_open("not_a_real_plot")
    with pytest.raises(KeyError):
        reg.open_window("not_a_real_plot", _FakeDetachWindow(), lambda: None)
    with pytest.raises(KeyError):
        reg.close("not_a_real_plot")


# ---------------------------------------------------------------------------
# Task 10: the three PIL renderers with no pre-existing test coverage
# (render_lattice_graph_image, render_relaxation_strip_image,
# render_heatmap_image / _thermal_ramp_rgb) -- pure PIL, no Tk, so
# headlessly testable the same way render_acf_plot etc. already are
# (indirectly, via the app) though this file adds the first DIRECT tests
# of the render_* functions themselves. Smoke-level: correct image size,
# and the honest "not enough data" / "no topology" fallback text, not a
# pixel-exact rendering check (this app's own convention per
# task-10-brief.md: "Geometry is not meaningfully unit-testable ...
# every geometry bug ... found by eye").
# ---------------------------------------------------------------------------

class _FakeIsingForGraph:
    """Just enough surface for render_lattice_graph_image (edges only)."""

    def __init__(self, edges):
        self.edges = edges


def test_render_lattice_graph_image_is_the_requested_size():
    im = _FakeIsingForGraph(edges=[(0, 1), (1, 2), (2, 3)])
    img, caption = la.render_lattice_graph_image(
        300, 300, im, world_idx=[0, 1, 2, 3], mediator_idx=[4, 5],
        spins_per_cell=2, grid_w=2)
    assert img.size == (300, 300)
    assert "4" in caption and "world" in caption.lower()
    assert "2" in caption and "mediator" in caption.lower()


def test_render_lattice_graph_image_handles_no_world_spins_honestly():
    im = _FakeIsingForGraph(edges=[])
    img, caption = la.render_lattice_graph_image(
        200, 200, im, world_idx=[], mediator_idx=[], spins_per_cell=2, grid_w=2)
    assert img.size == (200, 200)
    assert "unavailable" in caption.lower()


def test_render_relaxation_strip_image_is_the_requested_size():
    draws = [[1, 1, 0, 0]] * 3 + [[0, 0, 1, 1]] * 3
    img, caption = la.render_relaxation_strip_image(
        400, 150, draws, n_frames=4, n_world_spins=4, spins_per_cell=2, grid_w=2)
    assert img.size == (400, 150)
    assert "buffer" in caption.lower()


def test_render_relaxation_strip_image_handles_too_few_draws_honestly():
    img, caption = la.render_relaxation_strip_image(
        300, 100, [[1, 0]], n_frames=4, n_world_spins=2, spins_per_cell=2, grid_w=1)
    assert img.size == (300, 100)
    assert "no data" in caption.lower() or "waiting" in caption.lower()


def test_thermal_ramp_rgb_endpoints_match_blue_deep_and_orange():
    import numpy as np
    out = la._thermal_ramp_rgb(np.array([0.0, 1.0]))
    assert tuple(int(c) for c in out[0]) == la._rgb(la.theme.BLUE_DEEP)
    assert tuple(int(c) for c in out[1]) == la._rgb(la.theme.ORANGE)


def test_render_heatmap_image_is_the_requested_size():
    import numpy as np
    from scope import per_cell_occupancy
    draws = [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]
    occ = per_cell_occupancy(draws, n_world_spins=6, spins_per_cell=3, grid_w=2)
    img, caption = la.render_heatmap_image(250, 250, occ)
    assert img.size == (250, 250)
    assert "occupancy" in caption.lower()


def test_render_heatmap_image_handles_all_nan_honestly():
    import numpy as np
    occ = np.full((1, 2), np.nan)
    img, caption = la.render_heatmap_image(200, 200, occ)
    assert img.size == (200, 200)
    assert "unavailable" in caption.lower()
