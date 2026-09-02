"""A2: save/load a world.

A saved world is evidence, so it carries its provenance (spec_name,
receipt_dir, seed, clamp, sampler_params, saved_at) alongside the grid
itself. TDD, per the brief: write each test, watch it fail for the right
reason, then implement demo/worldfile.py.

No sampling here -- this is pure save/load logic over hand-built grids, so
none of this touches thrml/torx or `simulate()`.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import worldfile as wf  # noqa: E402

VALUE_NAMES = ["water", "rock", "grass"]


def _grid4():
    # 4x4, row-major (row = y, len 16)
    return [0, 1, 2, 0,
            1, 1, 2, 0,
            2, 2, 2, 1,
            0, 0, 1, 2]


# ---------------------------------------------------------------------------
# Step 1/2: round-trip -- save then load returns an identical grid and
# identical provenance.
# ---------------------------------------------------------------------------

def test_round_trip_grid_and_provenance(tmp_path):
    values = _grid4()
    clamp = {"g0_0": 0, "g3_3": 2}
    sampler_params = {"n_chains": 6, "n_samples": 30, "n_warmup": 600, "seed": 3}
    path = tmp_path / "world1.json"

    ret = wf.save_world(
        path, spec_name="lattice_terrain", receipt_dir="demo/receipts/small",
        seed=3, clamp=clamp, width=4, height=4, values=values,
        value_names=VALUE_NAMES, task_valid=True, violations=[],
        sampler_params=sampler_params, saved_at="2026-09-01T12:00:00Z")
    assert Path(ret) == path
    assert path.exists()

    world = wf.load_world(path)
    assert world.values == values
    assert world.width == 4
    assert world.height == 4
    assert world.spec_name == "lattice_terrain"
    assert world.receipt_dir == "demo/receipts/small"
    assert world.seed == 3
    assert world.clamp == clamp
    assert world.value_names == VALUE_NAMES
    assert world.task_valid is True
    assert world.violations == []
    assert world.sampler_params == sampler_params
    assert world.saved_at == "2026-09-01T12:00:00Z"


def test_round_trip_grid_helper_reshapes_row_major(tmp_path):
    values = _grid4()
    path = tmp_path / "world_grid.json"
    wf.save_world(
        path, spec_name="lattice_terrain", receipt_dir="demo/receipts/small",
        seed=1, clamp={}, width=4, height=4, values=values,
        value_names=VALUE_NAMES, task_valid=True, violations=[],
        sampler_params={}, saved_at="2026-09-01T12:00:00Z")
    world = wf.load_world(path)
    grid = world.grid()
    assert grid.shape == (4, 4)
    assert grid[0].tolist() == [0, 1, 2, 0]
    assert grid[3].tolist() == [0, 0, 1, 2]
    assert int(grid[2, 1]) == 2  # row y=2, col x=1 -> values[2*4+1] == 2


# ---------------------------------------------------------------------------
# Step 4/5: loading a world whose recorded task_valid is False must raise
# unless allow_invalid=True -- a saved invalid world must never silently
# reload as a good one.
# ---------------------------------------------------------------------------

def test_load_invalid_world_raises_by_default(tmp_path):
    path = tmp_path / "bad_world.json"
    wf.save_world(
        path, spec_name="lattice_terrain", receipt_dir="demo/receipts/small",
        seed=9, clamp={}, width=2, height=2, values=[0, 1, 1, 2],
        value_names=VALUE_NAMES, task_valid=False,
        violations=["g0_0/g1_0: water directly adjacent to rock"],
        sampler_params={}, saved_at="2026-09-01T12:05:00Z")

    with pytest.raises(ValueError, match="task_valid"):
        wf.load_world(path)


def test_load_invalid_world_succeeds_with_allow_invalid(tmp_path):
    path = tmp_path / "bad_world2.json"
    wf.save_world(
        path, spec_name="lattice_terrain", receipt_dir="demo/receipts/small",
        seed=9, clamp={}, width=2, height=2, values=[0, 1, 1, 2],
        value_names=VALUE_NAMES, task_valid=False,
        violations=["g0_0/g1_0: water directly adjacent to rock"],
        sampler_params={}, saved_at="2026-09-01T12:05:00Z")

    world = wf.load_world(path, allow_invalid=True)
    assert world.task_valid is False
    assert world.violations == ["g0_0/g1_0: water directly adjacent to rock"]


def test_load_valid_world_does_not_require_allow_invalid(tmp_path):
    path = tmp_path / "good_world.json"
    wf.save_world(
        path, spec_name="lattice_terrain", receipt_dir="demo/receipts/small",
        seed=2, clamp={}, width=2, height=2, values=[0, 2, 2, 1],
        value_names=VALUE_NAMES, task_valid=True, violations=[],
        sampler_params={}, saved_at="2026-09-01T12:06:00Z")
    world = wf.load_world(path)  # must NOT raise
    assert world.task_valid is True


# ---------------------------------------------------------------------------
# Defensive shape check -- values length must match width*height, so a
# corrupted/hand-edited file is caught at save time, not silently misread
# at load time.
# ---------------------------------------------------------------------------

def test_save_rejects_values_length_mismatch(tmp_path):
    path = tmp_path / "mismatch.json"
    with pytest.raises(ValueError, match="width.*height|length"):
        wf.save_world(
            path, spec_name="lattice_terrain", receipt_dir="demo/receipts/small",
            seed=1, clamp={}, width=4, height=4, values=[0, 1, 2],  # too short
            value_names=VALUE_NAMES, task_valid=True, violations=[],
            sampler_params={}, saved_at="2026-09-01T12:00:00Z")


# ---------------------------------------------------------------------------
# Task 4: EXTEND worldfile with a STACK format (base + N overlay layers),
# alongside -- never replacing -- the single-grid save/load above. Every
# test above this line must keep passing unmodified (see this module's own
# scope note and the plan's preflight risk flag: T4 must extend, not
# replace, the existing single-grid format).
#
# A stack layer carries the SAME provenance a single saved world does
# (receipt_dir, seed, clamp, sampler_params, task_valid, violations) PLUS
# two things a layer specifically needs: `name` (which layer this is --
# "base", "band0", "band1", ...) and `conditioning_patch` (the
# {(cell_name, value): weight} bias patch -- see demo/layers.bias_patch --
# this layer was sampled under; empty for the base layer, which is never
# conditioned on anything).
# ---------------------------------------------------------------------------

BAND_VALUE_NAMES = ["below", "above"]


def _base_layer(values=None):
    return dict(
        name="base", spec_name="lattice_terrain",
        receipt_dir="demo/receipts/small", seed=7, clamp={},
        conditioning_patch={}, width=4, height=4,
        values=values if values is not None else _grid4(),
        value_names=VALUE_NAMES, task_valid=True, violations=[],
        sampler_params={"n_chains": 8, "n_samples": 60, "n_warmup": 1200,
                         "steps_per_sample": 4})


def _band0_layer(values):
    return dict(
        name="band0", spec_name="lattice_elev_band_8x8",
        receipt_dir="demo/receipts/elev_band", seed=8, clamp={},
        conditioning_patch={("g0_0", 1): 0.06, ("g1_0", 1): -0.03},
        width=4, height=4, values=values, value_names=BAND_VALUE_NAMES,
        task_valid=True, violations=[],
        sampler_params={"n_chains": 8, "n_samples": 60, "n_warmup": 1200,
                         "steps_per_sample": 4})


def test_stack_round_trips_every_layer_with_its_own_provenance(tmp_path):
    base_values = _grid4()
    band0_values = [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    path = tmp_path / "stack1.json"

    ret = wf.save_stack(
        path, layers=[_base_layer(base_values), _band0_layer(band0_values)],
        saved_at="2026-09-01T12:10:00Z")
    assert Path(ret) == path
    assert path.exists()

    stack = wf.load_stack(path)
    assert stack.saved_at == "2026-09-01T12:10:00Z"
    assert len(stack.layers) == 2

    base, band0 = stack.layers
    assert base.name == "base"
    assert base.receipt_dir == "demo/receipts/small"
    assert base.seed == 7
    assert base.values == base_values
    assert base.conditioning_patch == {}
    assert base.grid().shape == (4, 4)

    assert band0.name == "band0"
    assert band0.receipt_dir == "demo/receipts/elev_band"
    assert band0.seed == 8
    assert band0.values == band0_values
    assert band0.value_names == BAND_VALUE_NAMES
    # conditioning_patch round-trips with its (cell, value) tuple keys and
    # float weights intact -- JSON has no tuple keys, so this is the part
    # save_stack/load_stack must convert, not just pass through.
    assert band0.conditioning_patch == {("g0_0", 1): 0.06, ("g1_0", 1): -0.03}


def test_stack_layer_with_invalid_task_raises_by_default(tmp_path):
    path = tmp_path / "stack_bad.json"
    bad_band = _band0_layer([0] * 16)
    bad_band["task_valid"] = False
    bad_band["violations"] = ["monotonicity: g2_2 true while band below false"]
    wf.save_stack(path, layers=[_base_layer(), bad_band],
                   saved_at="2026-09-01T12:11:00Z")

    with pytest.raises(ValueError, match="task_valid"):
        wf.load_stack(path)


def test_stack_layer_with_invalid_task_succeeds_with_allow_invalid(tmp_path):
    path = tmp_path / "stack_bad2.json"
    bad_band = _band0_layer([0] * 16)
    bad_band["task_valid"] = False
    bad_band["violations"] = ["monotonicity: g2_2 true while band below false"]
    wf.save_stack(path, layers=[_base_layer(), bad_band],
                   saved_at="2026-09-01T12:12:00Z")

    stack = wf.load_stack(path, allow_invalid=True)
    assert stack.layers[0].task_valid is True
    assert stack.layers[1].task_valid is False
    assert stack.layers[1].violations == [
        "monotonicity: g2_2 true while band below false"]


def test_stack_rejects_a_layer_with_mismatched_values_length(tmp_path):
    path = tmp_path / "stack_mismatch.json"
    bad_band = _band0_layer([0, 1, 1])  # too short for 4x4
    with pytest.raises(ValueError, match="width.*height|length"):
        wf.save_stack(path, layers=[_base_layer(), bad_band],
                       saved_at="2026-09-01T12:13:00Z")
