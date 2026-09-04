"""Desktop demo: a live view of a world sampled from an ALREADY-COMPILED
tsu program.

Everything on screen is either read verbatim from demo/receipts/small (a
receipt whose `place` pass alone took ~201s to produce -- read once at
startup here, NEVER regenerated: this file never calls `tsu compile`) or
computed live by sampling that already-compiled program with
`tsu.backends.thrml_backend.sample` and decoding with the compiler's OWN
`enc.is_codeword` / `enc.decode` / `spec.contract.validate` -- the same three
calls demo/render_world.py uses, reused here rather than reimplemented.
`render_world_image` below is that same file's rendering math, lifted
verbatim and parameterised on cell size, because task S1 requires the
decoration to stay a deterministic function of the decoded cell, not a
freshly invented one.

Nothing here is fabricated. A receipt value this app cannot find prints
`unavailable: <the reason string from the receipt>` -- never a blank, a
zero, a dash, or a guess. Three honesty commitments this file holds to:

  1. NO beta slider. This model carries 64 mediator spins whose couplings
     were baked in at beta=1 (see program.json / passes.json). Sampling a
     mediated model at any other beta is refused by
     `tsu.passes.route.assert_beta_consistent` (raises BetaMismatchError,
     spec 5.3.5) because the mediator couplings are themselves a function of
     beta. This app never calls `simulate(beta=...)`, so it never exercises
     that refusal path directly -- but it displays beta as FIXED and states
     why, rather than hiding the reason behind a control that would just
     raise if touched.
  2. Displayed worlds are CONDITIONAL. Roughly a quarter of draws are valid
     (codeword AND contract-passing); the rest are discarded. What's shown
     in DECODED WORLD and counted in DECODED MIX is therefore drawn from
     p(x | valid), not p(x). The sample log panel says this in plain words.
  3. A non-codeword or contract-failing draw is NEVER rendered as a world.
     The previous valid world stays on screen and the log line says why.
"""
from __future__ import annotations

import dataclasses
import json
import math
import queue
import random
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
DEMO_DIR = REPO_ROOT / "demo"
RECEIPT_DIR = DEMO_DIR / "receipts" / "small"
OVERLAY_RECEIPT_DIR = DEMO_DIR / "receipts" / "elev_band"  # Task 7: elevation bands
WORLDS_DIR = DEMO_DIR / "worlds"

# RP-1: demo/receipts/small (RECEIPT_DIR above) is git-tracked, frozen
# compile-time evidence -- program.json, verification.json, etc, written
# ONCE by `tsu compile` and never touched again. SampleWorker's clamped
# path calls `tsu.simulate.simulate()` once per pin change, and that call
# writes a simulation.json purely as a side effect this app never reads
# back (it uses simulate()'s returned samples directly) -- so that output
# belongs nowhere near RECEIPT_DIR. tempfile.gettempdir() (not a repo
# path) with a fixed, reused name so a long interactive session doesn't
# accumulate one throwaway directory per pin change/reseed (a fresh
# SampleWorker is constructed on every one of those, see _start_worker).
SIM_OUTPUT_DIR = Path(tempfile.gettempdir()) / "tsu_lattice_app_sim_output"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DEMO_DIR))

from tsu.spec import load_spec  # noqa: E402
from tsu.passes.encode import encode  # noqa: E402
from tsu.simulate import reconstruct_program, _selected_encoding  # noqa: E402
from tsu.backends.thrml_backend import sample as thrml_sample  # noqa: E402
from tsu.passes.route import assert_beta_consistent, BetaMismatchError  # noqa: E402 -- Task 7
from worldfile import save_world  # noqa: E402 -- A2: save/load provenance-carrying worlds
import frontier as frontier_mod  # noqa: E402 -- B1: capacity frontier panel
import theme  # noqa: E402 -- Task 0: theme foundation, see theme.py's own docstring
from scope import (beta_to_temperature, temperature_control_state,  # noqa: E402
                    magnetization, energy_histogram, local_field_response,
                    sigmoid, per_cell_occupancy,
                    MIN_LOCAL_FIELD_BIN_COUNT)  # Task 5/6/8/10
# C-1 fix: tsu.ess.integrated_autocorrelation_time / RELIABILITY_MIN_N_OVER_TAU
# and scope.autocorrelation are deliberately NOT imported here any more --
# render_acf_plot no longer calls tsu.ess's single-chain estimator on this
# app's live energy_trace (see that function's own docstring for why: the
# trace can never be one chain's own successive draws).
from layers import FIELD_CAP, FieldCapExceeded, bias_patch  # noqa: E402 -- Task 7
from elevation import band_patch, thermometer_level, monotonicity_violations  # noqa: E402
from elevation_world import render_elevation_world_image  # noqa: E402 -- composite view, reused verbatim
import explainer  # noqa: E402 -- Task 9: "What is this?" explainer window


def _rgb(hex_str: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b) ints -- so terrain/plot colours below share
    demo/theme.py's own tokens instead of a second, hand-copied palette
    that could drift from it. Moved up here (Task 0) so PAL, below, can use
    it too -- it used to live only just above the SCOPE panel's PLOT_*
    constants, too late for PAL's own definition."""
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# --------------------------------------------------------------------------
# constants shared by the raw-lattice grid and the decode/render path
# --------------------------------------------------------------------------
W = H = 8                      # decoded world is an 8x8 grid
WATER, ROCK, GRASS = 0, 1, 2
TERRAIN_NAMES = {WATER: "water", ROCK: "rock", GRASS: "grass"}
TERRAIN_ORDER = (WATER, ROCK, GRASS)
# Task 0: terrain palette verbatim from theme.py (water/rock/grass tokens),
# not a second hand-picked set of RGB triples.
PAL = np.array([_rgb(theme.WATER), _rgb(theme.ROCK), _rgb(theme.GRASS)], float)
CELL_UP = 32                    # px per grid cell in the rendered world image (render_world.py uses 64; halved here so a redraw fits inside one animation tick)
WORLD_DISPLAY_PX = 320          # DECODED WORLD canvas is square, this many px/side
WORLD_CELL_PX = WORLD_DISPLAY_PX // W   # px per grid cell ON SCREEN (for clicks + pin markers)

# Live sampling parameters used BY THIS APP (not receipt values -- disclosed
# as such in the SAMPLER panel). Chosen from a measured timing check: a
# single (n_chains=6, n_samples=1, n_warmup=300) call on this 192-node
# program takes ~0.2-0.3s on this machine, so a batch comfortably lands
# inside one UI tick.
BATCH_CHAINS = 6
N_SAMPLES_PER_CALL = 1
N_WARMUP = 300
STEPS_PER_SAMPLE = 4

PASS_ORDER = ("encode", "lower", "analyse", "gate_checks", "place",
              "route", "build_program", "regime", "verify")

FOOTER_TEXT = ("Simulated on CPU via thrml. No TSU silicon. No hardware "
               "speed or energy claims. |J| <= 6.0 is an Extropic-documented "
               "Z1 hardware cap (Thermalizers paper, Fig. 12 cap-sweep); "
               "|b| <= 6.0 is an assumed project value, not a sourced "
               "Extropic figure.")

# Live sampling parameters used for a CLAMPED batch (after a pin changes).
# Verified working: simulate("demo/receipts/small", n_chains=6, n_samples=30,
# n_warmup=600, seed=3, clamp={...}) returns in ~1.4s with all pins honoured
# in the decoded output. Bigger than the unclamped per-tick batch (below)
# because a pin change is a deliberate, infrequent gesture -- the app can
# afford one solid batch's worth of draws to answer "does this constraint
# even have a valid world" honestly, rather than trickling in one at a time.
CLAMP_N_CHAINS = 6
CLAMP_N_SAMPLES = 30
CLAMP_N_WARMUP = 600

# --------------------------------------------------------------------------
# Task 7: LAYERS panel -- a base layer plus N_BANDS elevation-band overlay
# layers (all N_BANDS bands sample from the SAME compiled demo/receipts/
# elev_band program, patched differently per band -- one receipt, no
# recompile, matching demo/elevation_world.py's own N_BANDS). "composite" is
# a fifth, view-only pseudo-layer: it samples nothing of its own, it is
# DERIVED from base's and every band's current last-valid decode.
# --------------------------------------------------------------------------
N_BANDS = 3   # matches demo/elevation_world.py's own N_BANDS
BAND_NAMES = tuple(f"band{i}" for i in range(N_BANDS))
LAYER_NAMES = ("base",) + BAND_NAMES + ("composite",)
BAND_VALUE_NAMES = ("below", "above")   # a band cell is 0 (below threshold) or 1 (above)

# A band has no continuous background worker (unlike base): it samples on
# an explicit "Regenerate this layer" click, one batch, same scale as a
# base pin-change batch (CLAMP_N_CHAINS/CLAMP_N_SAMPLES/CLAMP_N_WARMUP
# above) -- reused rather than a third set of numbers invented for a third
# sampling mode.
BAND_SAMPLE_PARAMS = dict(n_chains=CLAMP_N_CHAINS, n_samples=CLAMP_N_SAMPLES,
                          n_warmup=CLAMP_N_WARMUP, steps_per_sample=STEPS_PER_SAMPLE)

# A pin on an OVERLAY cell (see ClampState/LayerState below) is NOT a hard
# workload clamp the way a base-layer pin is (base pins route through
# `tsu.simulate.simulate(clamp=...)`, which repartitions and FIXES those
# physical spins -- see SampleWorker._run_clamped_batch). An overlay pin is
# instead folded into the SAME bias_patch conditioning every band already
# goes through, at a magnitude large relative to band_patch's own terms
# (order 1-1.5, see demo/elevation_world.py's STRENGTH calibration note)
# but kept under FIELD_CAP once combined with them. This is a DELIBERATE,
# DISCLOSED difference, not an oversight: "pinning in the base and in an
# overlay are different operations" (task 7 brief) -- a base pin is exact,
# an overlay pin is a strong probabilistic nudge, and the UI states this
# rather than presenting both as the same kind of control.
OVERLAY_PIN_STRENGTH = 3.5

# Reference dose-response table (spec section 2.1), MEASURED on
# demo/stacked_world.py's own ROCK-vs-distance-from-water conditioning
# sweep against demo/receipts/small -- reused here as-is (not re-measured)
# to show the SAME bias_patch mechanism's known degenerate-zone shape. The
# CONDITIONING STRENGTH control below drives demo/elevation.band_patch's
# own `strength` parameter, a DIFFERENT quantity on a different scale (a
# raw per-cell bias magnitude, not an alpha multiplying a distance field)
# -- the table is shown as qualitative reference for "conditioning strength
# controls have a degenerate zone," never claimed to read out this app's
# own alpha numerically. Stated on screen, not just here.
DOSE_RESPONSE_TABLE = (
    (0.00, 15.3), (0.05, 23.6), (0.10, 33.1), (0.20, 52.8), (0.35, 66.6))
DOSE_RESPONSE_NOTE = (
    "Reference only, NOT this control's own calibration: measured on "
    "demo/stacked_world.py's ROCK-vs-distance-from-water sweep (spec "
    "2.1) against demo/receipts/small, alpha scaling a DISTANCE field "
    "-- a different quantity on a different scale than this slider's "
    "STRENGTH (a raw per-cell bias on demo/receipts/elev_band). Shown to "
    "make the SHAPE visible: conditioning strength controls have a "
    "degenerate zone (here, above alpha~0.2 the base rules were "
    "overwhelmed) rather than a linear response -- discover that shape "
    "here, not by accident on this app's own slider.")

# --------------------------------------------------------------------------
# UI2: speed control. This is a DISPLAY-RATE knob only. It varies n_chains
# and n_samples per worker call (how much of a batch gets pulled and pushed
# to the UI before the worker looks at pause/stop again) plus an explicit
# inter-batch delay -- never n_warmup or steps_per_sample. Those two are
# what determine the equilibrium distribution being sampled from (spec
# 5.3); n_chains/n_samples are just how many i.i.d.-ish draws are taken per
# call, a pure batching knob. Measured directly on this receipt
# (PYTHONIOENCODING=utf-8 ... -- see ui2-report.md): with n_warmup=600
# fixed, a clamped call's wall time is ~0.4s steady-state (after JAX's
# one-time JIT compile per distinct (n_chains, n_samples) shape, itself
# ~1.4-1.7s the FIRST time that shape is used) REGARDLESS of n_samples
# from 2 to 30 -- warmup dominates, not sample count. So shrinking
# n_samples here does not, by itself, shrink worst-case Pause latency; it
# controls how many rows land in the queue per completed call (display
# density) and, combined with delay_s, how fast the world visibly fills
# in. The actual Pause-latency fix is structural (see should_abort_batch
# below): each row is pushed only after re-checking pause, so a completed
# batch stops contributing to the display within, at most, the single row
# that was already mid-push when Pause was clicked -- not the rest of the
# batch trailing out afterward.
SPEED_LEVELS = (
    {"label": "Slow (~1 batch/s, for watching)",
     "chains": 1, "clamp_samples": 3, "delay_s": 1.00},
    {"label": "Medium",
     "chains": 2, "clamp_samples": 8, "delay_s": 0.35},
    {"label": "Fast",
     "chains": 4, "clamp_samples": 16, "delay_s": 0.05},
    {"label": "Full speed",
     "chains": BATCH_CHAINS, "clamp_samples": CLAMP_N_SAMPLES, "delay_s": 0.0},
)
DEFAULT_SPEED_IDX = len(SPEED_LEVELS) - 1  # Full speed -- matches pre-UI2 behaviour

# --------------------------------------------------------------------------
# Task 6: SCOPE panel -- sizing/cadence constants. None of these change what
# is SAMPLED (that's N_WARMUP/STEPS_PER_SAMPLE, untouched above); they only
# bound how much of the live session's own history this panel looks at and
# how often it redraws.
# --------------------------------------------------------------------------
SCOPE_PLOT_W, SCOPE_PLOT_H = 300, 150       # px, one sub-plot's image size
# Task 8: temperature control -- the on-screen T range (unchanged from the
# tk.Scale this replaces: from_=0.3, to=3.0), and the Canvas track's own
# pixel size.
TEMP_T_MIN, TEMP_T_MAX = 0.3, 3.0
TEMP_TRACK_W, TEMP_TRACK_H = 240, 20
SCOPE_ENERGY_TRACE_MAXLEN = 400             # same ring-buffer length B2's
# energy_trace already used before this task; named here so the ACF plot's
# own caption can state it rather than hardcoding a second "400" that could
# drift from the Trace(...) construction below.
SCOPE_RAW_DRAWS_MAXLEN = 600                # ring buffer of raw spin rows,
# feeds local_field_response -- 600 draws * up to ~200 spins/draw pools
# comfortably past MIN_LOCAL_FIELD_BIN_COUNT per bin without growing
# unbounded over a long session.
SCOPE_ENERGY_HIST_BINS = 24
SCOPE_LOCAL_FIELD_BINS = 16
SCOPE_REDRAW_EVERY_N_DRAWS = 5              # throttle: local_field_response
# pools SCOPE_RAW_DRAWS_MAXLEN draws * every spin each redraw -- cheap per
# call, but recomputing on literally every one of many draws/sec is wasted
# work the display rate (poll cadence, ~80ms) doesn't need.

# --------------------------------------------------------------------------
# Task 10: detached-plot windows. Title/window-size/plot-size per key --
# static layout only, no statistics here. sigmoid's plot area is shorter
# than its window because its caption is long BY REQUIREMENT (carries the
# same reconstruction-vs-defect disclosure as the inline cell, see
# render_sigmoid_plot); relaxation's window is wide-short (a filmstrip);
# lattice_graph's is tall (grid + mediator strip + legend all stack).
# --------------------------------------------------------------------------
DETACH_WINDOW_SPECS: dict[str, dict] = {
    "lattice_graph": {
        "title": "LATTICE -- node-edge graph (role-coloured)",
        "win": (560, 640), "plot": (520, 500)},
    "relaxation": {
        "title": "LATTICE -- relaxation strip (raw physical state, recent draws)",
        "win": (820, 300), "plot": (780, 190)},
    "heatmap": {
        "title": "LATTICE -- per-cell occupancy heatmap",
        "win": (440, 540), "plot": (400, 400)},
    "sigmoid": {
        "title": "LATTICE -- local field: empirical P(s=1) vs sigmoid (REFERENCE ONLY)",
        "win": (620, 640), "plot": (580, 380)},
    "hist": {
        "title": "LATTICE -- energy histogram",
        "win": (620, 460), "plot": (580, 340)},
}
DETACH_REFRESH_MS = 250   # detached windows redraw on their OWN timer,
# independent of the inline SCOPE panel's message-driven redraw cadence --
# see LatticeApp._open_detach's own docstring for why.


def speed_level(idx: int) -> dict:
    """SPEED_LEVELS[idx], clamping idx into range -- pure, headlessly
    tested (tests/test_lattice_app_logic.py). Every entry's keys are
    exactly {"label", "chains", "clamp_samples", "delay_s"}: no entry can
    smuggle in an n_warmup or steps_per_sample override, which is the
    structural guarantee that this control cannot touch the statistics."""
    idx = max(0, min(idx, len(SPEED_LEVELS) - 1))
    return SPEED_LEVELS[idx]


# --------------------------------------------------------------------------
# pure logic: pin bookkeeping, canvas-click -> grid-cell mapping, and
# infeasible-clamp detection. No Tk here -- these are the units the headless
# test suite (tests/test_lattice_app_logic.py) exercises directly.
# --------------------------------------------------------------------------
class ClampState:
    """Which grid cells are pinned, and to what value. Workload-level only
    (g{x}_{y} -> 0/1/2) -- this is the object whose `as_dict()` is handed
    straight to `simulate(..., clamp=...)`; nothing downstream of it
    re-derives or hand-rolls clamping."""

    _CYCLE = (WATER, ROCK, GRASS)  # unpinned -> water -> rock -> grass -> unpinned

    def __init__(self, cycle: tuple[int, ...] | None = None):
        # Task 7: an overlay band's pin cycle is (0, 1) -- below/above --
        # not the base's 3-value terrain cycle. `cycle` overrides the class
        # default per-instance; every EXISTING call site (`ClampState()`,
        # no argument) is unaffected, same 3-value terrain cycle as before.
        self._cycle = cycle if cycle is not None else self._CYCLE
        self._pins: dict[tuple[int, int], int] = {}

    def get(self, x: int, y: int) -> int | None:
        return self._pins.get((x, y))

    def add(self, x: int, y: int, value: int) -> None:
        self._pins[(x, y)] = value

    def remove(self, x: int, y: int) -> None:
        self._pins.pop((x, y), None)

    def clear(self) -> None:
        self._pins.clear()

    def cycle(self, x: int, y: int) -> int | None:
        """Advance one cell's pin through this instance's own `_cycle`
        (default: unpinned -> water(0) -> rock(1) -> grass(2) -> unpinned;
        (0, 1) for an overlay band -- see __init__). Returns the new value
        (or None if now unpinned)."""
        cur = self._pins.get((x, y))
        if cur is None:
            nxt = self._cycle[0]
        else:
            i = self._cycle.index(cur)
            nxt = self._cycle[i + 1] if i + 1 < len(self._cycle) else None
        if nxt is None:
            self._pins.pop((x, y), None)
        else:
            self._pins[(x, y)] = nxt
        return nxt

    def as_dict(self) -> dict[str, int]:
        """{name: value} for `simulate(clamp=...)` -- unpinned cells are
        simply absent, never emitted as null/None."""
        return {f"g{x}_{y}": v for (x, y), v in self._pins.items()}

    def items(self):
        return dict(self._pins).items()

    def __len__(self) -> int:
        return len(self._pins)

    def __bool__(self) -> bool:
        return bool(self._pins)


# --------------------------------------------------------------------------
# Task 7: pure LAYERS-panel logic -- no Tk here, see
# tests/test_lattice_app_logic.py's own "no Tk in pure logic" convention.
# --------------------------------------------------------------------------

def get_beta_override(layer_name: str, base_override: float | None,
                       bands: "dict[str, LayerState]") -> float | None:
    """Task 8 structural fix: WHERE a layer's chosen beta override lives,
    generalised over base AND bands -- a prior review flagged
    `_refresh_temperature_control`/`_on_temperature_change` reaching
    straight into `self.bands[self.active_layer]`, which raises KeyError
    the moment "base" is the active layer (base has no LayerState -- see
    LayerState's own docstring). That never fired in practice only because
    base has always compiled WITH mediator spins (locked, no override
    settable) -- a defect waiting for the day base ever compiles
    unmediated, not a hardcoded-safe path. `base_override` is threaded in
    explicitly (LatticeApp.base_beta_override) rather than assuming
    `bands["base"]` exists."""
    if layer_name == "base":
        return base_override
    if layer_name in bands:
        return bands[layer_name].beta_override
    return None


def set_beta_override(layer_name: str, value: float,
                       app_for_base, bands: "dict[str, LayerState]") -> None:
    """Setter counterpart to get_beta_override -- see its docstring. Base
    has no LayerState to hold a per-layer override, so its slot lives on
    the app itself (`app_for_base.base_beta_override`); a band's lives on
    its own LayerState, unchanged from before this task."""
    if layer_name == "base":
        app_for_base.base_beta_override = value
    elif layer_name in bands:
        bands[layer_name].beta_override = value


# --------------------------------------------------------------------------
# Task 10: the detach registry -- which of the five DETACHABLE_PLOTS
# (task-10-brief.md's own list: the node-and-edge lattice, the relaxation
# strip, the per-cell heatmap, the sigmoid response, the energy histogram)
# currently has its own open Toplevel, and how to shut each one down
# cleanly. Pure Python -- no Tk import here at all, no Tk object is ever
# touched by this class (see the "no Tk in pure logic" convention this
# file's own module docstring/tests/test_lattice_app_logic.py's docstring
# both state) -- it only holds an opaque `window` handle (a real
# tk.Toplevel from LatticeApp._open_detach, or a fake stand-in in tests)
# and an opaque `cancel` callable the CALLER supplies to stop whatever
# live-update job that window's own host is running.
#
# The energy/valid-fraction/magnetization traces are explicitly NOT in
# DETACHABLE_PLOTS -- the brief's own "stays inline" list -- there is no
# detach path for them at all, so they can never leak one.
#
# This class exists specifically to make the brief's own leak scenario
# structurally impossible rather than merely "handled by care": "a closed
# window still receiving after() updates is a slow leak that only shows
# up after a long session, which is exactly when a demo is being given."
# `close(key)` is the ONLY place `cancel` is ever invoked, and it is
# invoked EXACTLY ONCE no matter how many times close() is called for the
# same key -- see test_closing_twice_cancels_the_callback_only_once.
# --------------------------------------------------------------------------
DETACHABLE_PLOTS = ("lattice_graph", "relaxation", "heatmap", "sigmoid", "hist")


class DetachRegistry:
    """At most one open window per key in DETACHABLE_PLOTS. See the block
    comment above for why this is pure Python with no Tk dependency."""

    def __init__(self, keys: tuple[str, ...] = DETACHABLE_PLOTS):
        self._keys = frozenset(keys)
        self._open: dict[str, tuple[object, Callable[[], None]]] = {}

    def _check_key(self, key: str) -> None:
        if key not in self._keys:
            raise KeyError(
                f"{key!r} is not a detachable plot -- must be one of "
                f"{sorted(self._keys)}")

    def is_open(self, key: str) -> bool:
        self._check_key(key)
        return key in self._open

    def window_for(self, key: str) -> object | None:
        """The open window handle for `key`, or None if not open -- the
        caller (LatticeApp._open_detach) uses this to decide whether to
        LIFT an existing window (the same singleton pattern
        demo/explainer.py's own _on_show_explainer uses) rather than
        opening a second one."""
        self._check_key(key)
        entry = self._open.get(key)
        return entry[0] if entry is not None else None

    def open_window(self, key: str, window: object, cancel) -> None:
        """Registers `window` as `key`'s own open window, with `cancel`
        as the ONE callable that will later stop its live-update job.
        Refuses to reopen an already-open key (RuntimeError) -- the
        caller must check is_open()/window_for() and LIFT the existing
        window instead, never silently drop the first handle by
        overwriting it here."""
        self._check_key(key)
        if key in self._open:
            raise RuntimeError(
                f"{key!r} is already open -- lift the existing window "
                f"(window_for({key!r})) instead of reopening it")
        self._open[key] = (window, cancel)

    def close(self, key: str) -> None:
        """Pops `key`'s entry (if any) and invokes its `cancel` callable
        exactly once. A no-op, not an error, if `key` was never opened or
        was already closed -- the caller (a WM_DELETE_WINDOW handler) must
        be safe to invoke more than once without double-cancelling
        whatever after() job `cancel` stops."""
        self._check_key(key)
        entry = self._open.pop(key, None)
        if entry is not None:
            _, cancel = entry
            cancel()


def temperature_value_frac(t: float, t_min: float, t_max: float) -> float:
    """Map a temperature T (T=1/beta) to [0, 1] for the slider track's
    thumb position -- clamped, since a caller can hold a value outside
    [t_min, t_max] (e.g. a receipt's own compiled beta) that the on-screen
    range doesn't span."""
    if t_max <= t_min:
        raise ValueError(f"t_max ({t_max!r}) must exceed t_min ({t_min!r})")
    frac = (t - t_min) / (t_max - t_min)
    return min(1.0, max(0.0, frac))


def render_temperature_track(width: int, height: int, adjustable: bool,
                              value_frac: float = 0.5) -> Image.Image:
    """Task 8: the beta-temperature slider TRACK, IMAGE per the tokens
    spec (numpy/PIL blitted to a Canvas), not a native Scale -- the spec
    calls this out by name as a trap: a native Scale (`ttk.Scale`, and the
    plain `tk.Scale` this app used before Task 8) cannot be gold, and in
    the LOCKED state it always LOOKS disabled -- exactly the wrong message
    for a control that is refusing a change ON PURPOSE, not broken.

    ADJUSTABLE (the useful window a viewer can move through): a cold ->
    gold -> hot thermal ramp, verbatim the tokens spec's own 3-stop
    gradient (blue_deep -> gold at 55% -> orange), with a gold thumb at
    `value_frac` (0 = coldest, 1 = hottest).

    FIXED (the "lock plate"): a SOLID cold-blue field -- never a copy of
    the live ramp, greyed or otherwise, since there IS no useful window to
    show (the model has exactly one valid beta) -- with a blue_lit
    lock-bar across the middle and a blue_lit thumb FROZEN at the centre
    regardless of `value_frac`. Deliberate and authoritative, per the
    brief's own framing: a point of pride, not an apology.

    Buildability: flat rectangles only, 1px outlines, zero corner radius,
    no shadow/blur/glow -- see the tokens spec's own buildability rules.
    """
    def hexrgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    img = Image.new("RGB", (width, height), hexrgb(theme.PANEL))
    draw = ImageDraw.Draw(img)
    track_top, track_bot = 3, height - 4
    track_h = max(track_bot - track_top, 1)

    if adjustable:
        cold = np.array(hexrgb(theme.BLUE_DEEP), dtype=float)
        gold = np.array(hexrgb(theme.GOLD), dtype=float)
        hot = np.array(hexrgb(theme.ORANGE), dtype=float)
        mid = 0.55  # verbatim the tokens spec's CSS gradient stop
        xs = np.arange(width, dtype=float)
        frac = xs / max(width - 1, 1)
        colors = np.empty((width, 3), dtype=float)
        left = frac <= mid
        t_left = frac[left] / mid if mid > 0 else np.zeros(left.sum())
        colors[left] = cold[None, :] * (1 - t_left[:, None]) + gold[None, :] * t_left[:, None]
        right = ~left
        t_right = (frac[right] - mid) / max(1 - mid, 1e-9)
        colors[right] = gold[None, :] * (1 - t_right[:, None]) + hot[None, :] * t_right[:, None]
        colors = colors.clip(0, 255).astype(np.uint8)
        row = np.tile(colors[None, :, :], (track_h, 1, 1))
        img.paste(Image.fromarray(row, "RGB"), (0, track_top))
        draw.rectangle([0, track_top, width - 1, track_bot - 1],
                       outline=hexrgb(theme.GOLD_DIM), width=1)
        thumb_color = hexrgb(theme.GOLD)
        thumb_x = int(round(value_frac * (width - 1)))
    else:
        draw.rectangle([0, track_top, width - 1, track_bot - 1],
                       fill=hexrgb(theme.BLUE_DEEP), outline=hexrgb(theme.BLUE), width=1)
        mid_y = (track_top + track_bot) // 2
        draw.line([(2, mid_y), (width - 3, mid_y)], fill=hexrgb(theme.BLUE_LIT), width=1)
        thumb_color = hexrgb(theme.BLUE_LIT)
        thumb_x = width // 2  # frozen -- does NOT read value_frac, see docstring

    thumb_w = 6
    x0 = max(thumb_x - thumb_w // 2, 0)
    x1 = min(x0 + thumb_w, width - 1)
    draw.rectangle([x0, 1, x1, height - 2], fill=thumb_color, outline=hexrgb(theme.CREAM), width=1)
    return img


# ==========================================================================
# Task 11: FRONTIER redesign -- load gauges, not a wall of monospace.
#
# PRESENTATION ONLY. Every number below is read verbatim from a
# frontier_mod.FrontierReport that demo/frontier.py already computed (same
# object `frontier_mod.render_text` renders to the plain-text panel this
# replaces) -- nothing here re-derives a prediction or a measurement.
# frontier_gauge_specs performs exactly ONE interpretive step beyond
# copying fields: which gate is "highest current utilisation" (argmax of
# already-computed pct_used) and which gate frontier_mod's own
# BindingForecast headline names as binding FIRST (an exact substring
# match against that SAME string, via _binding_gate_name -- never a
# re-walk of the law). Both are SELECTIONS among numbers frontier.py
# already produced, not new arithmetic.
#
# Every predicted value stays labelled with its source law in
# GaugeSpec.predicted_law/predicted_label -- the tokens spec's own
# requirement that a prediction never be presented as a measurement.
# ==========================================================================

FRONTIER_GAUGE_LABELS = {
    "degree": "DEGREE",
    "coupling_cap": "COUPLING CAP",
    "field_cap": "FIELD CAP",
    "node_budget": "NODE BUDGET",
}

# P-3/F-A5 + I-9a/F-R7: |J| and |b| no longer share one disclosure -- |J| is
# Extropic-documented (Thermalizers Fig. 12 cap-sweep, "6 (Z1)"), |b| remains
# a genuine, unsourced project assumption. Was one ASSUMED_CAP_NOTE constant
# reused for both gauges (a second, drifting phrasing of FOOTER_TEXT's own
# claim) -- reusing one string for two now-differently-sourced facts is
# exactly the bug this split exists to fix, so there are two constants now,
# each read by only its own gate below.
COUPLING_CAP_NOTE = ("documented Z1 hardware limit -- |J| <= 6.0 is an "
                     "Extropic-documented cap (Thermalizers paper, Fig. 12 "
                     "cap-sweep, annotated \"6 (Z1)\"), not a project "
                     "assumption")
FIELD_CAP_NOTE = ("assumed project limit -- |b| <= 6.0 is an assumed "
                  "project value, not a sourced Extropic figure")

# When a gate's predicted next value sits PAST its own cap, the track is
# compressed to this fraction of the gauge's width (0..cap) and a hatched
# red region fills the remainder -- verbatim the tokens spec's own
# buildability note ("FRONTIER gauges with hatched overrun... Predicted
# hairline and hatched overrun are the design. Do not flatten to a
# progressbar.").
GAUGE_OVERRUN_TRACK_FRAC = 0.62


@dataclass(frozen=True)
class GaugeSpec:
    """One FRONTIER load gauge's presentation data -- everything a renderer
    needs to draw one gate's row, with no further arithmetic required of
    the renderer. See the module section docstring above for what "tag" and
    "predicted_*" are derived from."""
    gate: str
    label: str
    measured: float
    limit: float
    pct_used: float | None
    tag_kind: str | None       # "bind_first" | "highest_util" | "ok" | None
                               # (Minor #5, final review: renamed from
                               # "over"/"bind" -- those names were inverted
                               # relative to their own meaning, a trap for
                               # the next editor. "bind_first" tags PREDICTED
                               # TO BIND FIRST; "highest_util" tags HIGHEST
                               # CURRENT UTILISATION.)
    tag_text: str | None
    predicted_value: float | None
    predicted_label: str | None   # e.g. "predicted degree 11 after k -> 4"
    predicted_law: str | None     # e.g. "one-hot law, spec section 4.2"
    predicted_exceeds_cap: bool
    foot_text: str

    @property
    def frac_current(self) -> float:
        return _safe_frac(self.measured, self.limit)

    @property
    def frac_predicted(self) -> float | None:
        if self.predicted_value is None:
            return None
        return _safe_frac(self.predicted_value, self.limit)


def _safe_frac(value: float, limit: float) -> float:
    """value/limit, clamped to [0, 1] -- never raises on a zero/inf/NaN
    limit (headroom_from_receipt already refuses to compute a PERCENTAGE
    for those, see GateHeadroom.pct_used; this is the same guard applied to
    a gauge's fill fraction, which the renderer needs even when pct_used
    is None)."""
    if not isinstance(limit, (int, float)):
        return 0.0
    if limit in (0, float("inf")) or limit != limit:  # zero, inf, NaN
        return 0.0
    return min(max(float(value) / float(limit), 0.0), 1.0)


def _binding_gate_name(headline: str, headroom: Sequence[Any]) -> str | None:
    """Which headroom gate frontier_mod.BindingForecast's own headline names
    as binding FIRST -- an EXACT substring match against the same string
    predict_first_binding_gate generated (".../ {gate} is predicted to bind
    FIRST, ..."), never a re-derivation of the forecast. Returns None for
    the "no gate is predicted to bind" headline, or if the headline's
    phrasing ever changes out from under this match (fails safe to "no
    tag" rather than guessing)."""
    for h in headroom:
        if f"{h.gate} is predicted to bind FIRST" in headline:
            return h.gate
    return None


def frontier_gauge_specs(report: Any) -> list[GaugeSpec]:
    """Build one GaugeSpec per report.headroom entry (degree, coupling_cap,
    field_cap, node_budget -- headroom_from_receipt's own fixed order).
    `report` is a frontier_mod.FrontierReport; every field read below
    already exists on it (see demo/frontier.py) -- this function only
    selects and labels, never computes a new prediction."""
    headroom = report.headroom
    bind_first = _binding_gate_name(report.binding.headline, headroom)
    ranked = [h for h in headroom if h.pct_used is not None]
    highest = max(ranked, key=lambda h: h.pct_used) if ranked else None

    # C1 (final review, fix round): report.verified[0] is ALWAYS
    # verify_k_increment run against THIS receipt's own SELECTED encoding
    # (build_frontier_report's first `verified` entry, unconditionally --
    # see demo/frontier.py). Every predicted hairline above is a one-hot-law
    # prediction; when the selected encoding isn't one_hot, that prediction
    # can DIVERGE from what this model's own encoding actually does (the
    # domain_wall case named in report.encoding_note). That divergence must
    # sit beside the prediction it refutes, not several lines below a fold.
    selected_verified = report.verified[0] if report.verified else None

    specs = []
    for h in headroom:
        # Minor #5 (final review): this used to be if/elif, which silently
        # DROPPED the "HIGHEST CURRENT UTILISATION" tag whenever a single
        # gate happened to also be the one predicted to bind first --
        # neither test receipt exercises that overlap, so it went unnoticed.
        # Both tags are now kept when both apply, joined into one label;
        # tag_kind (which only selects a colour) follows whichever is the
        # more urgent signal.
        tag_kind = tag_text = None
        tag_labels = []
        if h.gate == bind_first:
            tag_labels.append("PREDICTED TO BIND FIRST")
        if highest is not None and h.gate == highest.gate:
            tag_labels.append("HIGHEST CURRENT UTILISATION")
        if tag_labels:
            tag_kind = "bind_first" if h.gate == bind_first else "highest_util"
            tag_text = "  +  ".join(tag_labels)
        elif h.gate == "node_budget" and h.pct_used is not None and h.pct_used < 5.0:
            tag_kind, tag_text = "ok", "UNBOUND ON Z1-CLASS"

        predicted_value = predicted_label = predicted_law = None
        predicted_exceeds_cap = False
        if h.gate == "degree":
            predicted_value = report.law_predicted_degree_next_k
            predicted_label = (f"predicted degree {predicted_value} after "
                               f"k -> {report.shape.k + 1}")
            predicted_law = "one-hot law, spec section 4.2"
            predicted_exceeds_cap = predicted_value > h.limit
        elif h.gate == "field_cap":
            predicted_value = report.law_predicted_field_floor_next_k
            predicted_label = (f"predicted |b| floor {predicted_value:.2f} "
                               f"after k -> {report.shape.k + 1}")
            predicted_law = "one-hot law, spec section 4.8"
            predicted_exceeds_cap = predicted_value > h.limit

        foot_parts = []
        if h.gate == "degree":
            ticks = ", ".join(str(int(round(h.limit * f)))
                              for f in (0, 0.25, 0.5, 0.75, 1.0))
            foot_parts.append(f"ticks at {ticks}")
        if predicted_value is not None:
            over_note = "OVER CAP" if predicted_exceeds_cap else "still under cap"
            foot_parts.append(f"dashed/hatch = {predicted_label} "
                              f"({predicted_law}) -- {over_note}")
            # C1: the observed value for THIS receipt's own selected
            # encoding, named beside the prediction it either confirms or
            # refutes -- never left for the Text box alone to carry.
            if selected_verified is not None and h.gate == "degree":
                match = ("MATCHES" if selected_verified.degree_matches
                         else "DIVERGES")
                foot_parts.append(
                    f"OBSERVED ({report.encoding}): degree "
                    f"{selected_verified.observed_degree} ({match} the law)")
            elif (selected_verified is not None and h.gate == "field_cap"
                  and selected_verified.predicted_field_floor is not None):
                floor_ok = ("at/above floor"
                            if selected_verified.field_at_or_above_floor
                            else "BELOW floor")
                foot_parts.append(
                    f"OBSERVED ({report.encoding}): |b|max "
                    f"{selected_verified.observed_field:.2f} ({floor_ok})")
        if h.gate == "coupling_cap":
            foot_parts.append(COUPLING_CAP_NOTE)
        elif h.gate == "field_cap":
            foot_parts.append(FIELD_CAP_NOTE)
        if h.gate == "node_budget":
            if tag_kind == "ok":
                foot_parts.append("p-bits are not the constraint here")
            elif h.pct_used is not None:
                foot_parts.append(f"{h.pct_used:.2f}% of node budget used")
            else:
                foot_parts.append("unavailable: node budget percentage "
                                  "could not be computed")
        foot_text = "  |  ".join(foot_parts)

        specs.append(GaugeSpec(
            gate=h.gate, label=FRONTIER_GAUGE_LABELS[h.gate],
            measured=h.measured, limit=h.limit, pct_used=h.pct_used,
            tag_kind=tag_kind, tag_text=tag_text,
            predicted_value=predicted_value, predicted_label=predicted_label,
            predicted_law=predicted_law,
            predicted_exceeds_cap=predicted_exceeds_cap, foot_text=foot_text))
    return specs


def _dashed_vline(draw: "ImageDraw.ImageDraw", x: int, height: int,
                  color: tuple[int, int, int], dash: int = 3, gap: int = 2) -> None:
    y = 0
    while y < height:
        y2 = min(y + dash, height)
        draw.line([(x, y), (x, y2)], fill=color, width=1)
        y = y2 + gap


def _hatch_fill(draw: "ImageDraw.ImageDraw", x0: int, y0: int, x1: int, y1: int,
                fg: tuple[int, int, int], bg: tuple[int, int, int],
                spacing: int = 6, stripe_w: int = 2) -> None:
    """Diagonal hatch inside [x0,x1]x[y0,y1] -- the tokens spec's own
    "hatched overrun" treatment, drawn as repeated 45-degree stripes (PIL
    has no repeating-pattern fill primitive)."""
    draw.rectangle([x0, y0, x1, y1], fill=bg)
    h = y1 - y0
    offset = -h
    while offset < (x1 - x0) + h:
        draw.line([(x0 + offset, y1), (x0 + offset + h, y0)], fill=fg, width=stripe_w)
        offset += spacing


def render_frontier_gauge_track(width: int, height: int, frac_current: float,
                                frac_predicted: float | None = None,
                                overrun_ratio: float | None = None) -> Image.Image:
    """Task 11: one FRONTIER gauge's track, IMAGE per the tokens spec's own
    buildability table ("FRONTIER gauges with hatched overrun" is listed
    IMAGE, PhotoImage/Canvas). Draws only what GaugeSpec already computed --
    frac_current/frac_predicted are measured/limit and predicted/limit,
    already-safe fractions (see _safe_frac); this function never reads a
    report or a limit itself.

    overrun_ratio is None: a plain track, full width, gold_dim fill to
    frac_current, plus an optional gold DASHED hairline at frac_predicted
    (predicted <= cap in this branch, so it fits inside the track).

    overrun_ratio is not None (predicted_value / limit for a gate whose
    prediction EXCEEDS its cap): the track compresses to
    GAUGE_OVERRUN_TRACK_FRAC of the image width (still 0..cap, same fill
    scale as the plain branch) and a hatched red region fills the rest,
    its own width scaled by how far past 1.0 overrun_ratio sits (clamped
    so a very large overrun still fits the image) -- visually distinct
    from the measured gold fill, never a second gold bar."""
    def hexrgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    img = Image.new("RGB", (width, height), hexrgb(theme.INSET))
    draw = ImageDraw.Draw(img)

    track_frac = GAUGE_OVERRUN_TRACK_FRAC if overrun_ratio is not None else 1.0
    track_w = max(int(round(width * track_frac)), 2)

    draw.rectangle([0, 0, track_w - 1, height - 1], outline=hexrgb(theme.GOLD_GHOST), width=1)
    fill_w = int(round(track_w * min(max(frac_current, 0.0), 1.0)))
    if fill_w > 1:
        draw.rectangle([1, 1, max(fill_w - 1, 1), height - 2], fill=hexrgb(theme.GOLD_DIM))

    if overrun_ratio is None:
        if frac_predicted is not None:
            x = int(round(min(max(frac_predicted, 0.0), 1.0) * (track_w - 1)))
            _dashed_vline(draw, x, height, hexrgb(theme.GOLD))
    else:
        over_span = min(max(overrun_ratio - 1.0, 0.05), 1.5)
        avail = max(width - track_w, 6)
        hatch_w = max(int(round(avail * min(over_span / 0.5, 1.0))), 6)
        hatch_w = min(hatch_w, avail)
        x0, x1 = track_w, min(track_w + hatch_w, width - 1)
        _hatch_fill(draw, x0, 0, x1, height - 1, hexrgb(theme.RED), hexrgb(theme.RED_DEEP))
        draw.rectangle([x0, 0, x1, height - 1], outline=hexrgb(theme.RED), width=1)
    return img


def band_index_from_name(layer_name: str) -> int:
    """"band2" -> 2. Raises ValueError for anything else -- a caller
    passing "base"/"composite" here is a bug, not a value to guess at."""
    if not layer_name.startswith("band"):
        raise ValueError(f"{layer_name!r} is not a band layer name")
    return int(layer_name[len("band"):])


def overlay_pin_patch(pins: Sequence[tuple[tuple[int, int], int]],
                      strength: float = OVERLAY_PIN_STRENGTH
                      ) -> dict[tuple[str, int], float]:
    """{(cell_name, 1): weight} for every pinned cell in `pins` (an
    iterable of ((x, y), value) pairs, i.e. ClampState.items()) -- value 1
    (above) encourages at +strength, value 0 (below) encourages at
    -strength (discourages "above"). This is NOT the same operation as a
    base-layer pin (see OVERLAY_PIN_STRENGTH's own module-level docstring
    note): it is a strong bias_patch nudge, added into the SAME patch dict
    band_patch produces, not a hard clamp -- so a pinned overlay cell can
    still, rarely, sample the other way, which base's hard-clamped pins
    cannot. Returns {} for no pins -- an empty patch contribution, not a
    special case the caller needs to branch on."""
    return {(f"g{x}_{y}", 1): (strength if v == 1 else -strength)
            for (x, y), v in pins}


def composite_missing_layers(base_decoded, band_decodeds: Sequence) -> list[str]:
    """Which of "base"/"band0"/"band1"/... have no valid decode yet --
    the composite view is UNAVAILABLE (never fabricated from a partial
    stack) until this returns []."""
    missing = []
    if base_decoded is None:
        missing.append("base")
    missing.extend(f"band{i}" for i, d in enumerate(band_decodeds) if d is None)
    return missing


# F-R12/R13: demo/layers.py's own MANDATORY CAVEAT -- stacking samples
# p(base)*p(band|base), a DIRECTED/ANCESTRAL factorization, NOT the joint
# Boltzmann distribution over both layers at once -- used to exist only in
# a source docstring (demo/layers.py, demo/stacked_world.py) and never
# reached the screen where a composite is actually displayed. Stated here,
# reusing layers.py's own wording rather than inventing new phrasing that
# could drift from it, and kept short enough to survive the layout.
COMPOSITE_ANCESTRAL_CAVEAT = (
    "MANDATORY: this stack samples p(base)*p(band|base) -- a directed/"
    "ancestral factorization, NOT one joint Boltzmann sample.")


def composite_status_text(base_is_stale: bool) -> str:
    """The world-status text shown while the composite layer is on screen
    -- extracted to a pure function so the C4 requirement (an existing
    disclosure may not lose prominence) and the new mandatory caveat above
    are both independently testable without a live Tk app. `base_is_stale`
    is whether composite_base_grid (the base decode the most-recently-
    regenerated band was conditioned on) still matches base's current live
    decode -- see the call site's own long comment for why that, and not
    self.last_valid_grid, is the correct terrain source.

    The pre-existing staleness disclosure (fix-round-3) is reproduced
    verbatim below, unchanged and undiminished -- C4 forbids an existing
    disclosure losing prominence, so the caveat is APPENDED, never
    substituted for it."""
    staleness_note = (
        " -- base has advanced since (streaming continuously); "
        "this terrain is NOT base's current live decode"
        if base_is_stale else
        " -- currently matches base's live decode too")
    return ("composite: elevation-driven hillshade over the "
            "base decode the MOST RECENTLY REGENERATED band "
            "was conditioned on" + staleness_note +
            " -- the OTHER bands may have been conditioned on "
            "a DIFFERENT base draw (each band regenerates "
            "independently; see COMPOSITE VALIDATION below "
            "for the cross-layer check). " + COMPOSITE_ANCESTRAL_CAVEAT)


class LayerState:
    """Live, mutable per-layer sampling state for ONE band (or the base --
    see LatticeApp's own base-layer attributes, which stay as they were
    before this task; a base LayerState is not constructed). Composite has
    no LayerState of its own -- it is derived from every band's (and
    base's) `last_valid_decoded`, recomputed on demand, never stored."""

    def __init__(self, name: str, clamp: "ClampState"):
        self.name = name
        self.clamp = clamp
        self.conditioning_patch: dict[tuple[str, int], float] = {}
        self.beta_override: float | None = None   # None = use the receipt's own compiled beta
        self.last_valid_decoded: dict | None = None
        self.last_valid_grid: np.ndarray | None = None
        self.valid_count = 0
        self.total_draws = 0
        self.batch_infeasible = False
        self.batch_reason = ""


def cell_at(px: int, py: int, origin_x: int, origin_y: int, cell_px: int,
            width: int, height: int) -> tuple[int, int] | None:
    """Map a canvas click at (px, py) to a grid cell (x, y), given the
    on-canvas top-left of the grid (origin_x, origin_y), one cell's pixel
    size, and the grid's total pixel span (width, height). Half-open cell
    bounds -- [origin + i*cell_px, origin + (i+1)*cell_px) -- so a pixel
    exactly on a shared edge belongs to the cell that starts there, never
    both neighbours. Returns None for a click outside the grid entirely."""
    dx, dy = px - origin_x, py - origin_y
    if dx < 0 or dy < 0 or dx >= width or dy >= height:
        return None
    return int(dx // cell_px), int(dy // cell_px)


def spin_cell_position(index: int, n_world_spins: int, spins_per_cell: int,
                        grid_w: int) -> tuple[int, int, int] | None:
    """Map a physical spin index to (cell_x, cell_y, slot), or None if the
    spin is a mediator and belongs to no cell.

    The encoder emits each categorical variable's chain spins consecutively
    (`g{x}_{y}__dw{p}`), and the grid generator emits variables in row-major
    order, so cell = index // spins_per_cell and slot = index % spins_per_cell.
    `spins_per_cell` must be passed by the caller, derived from the receipt
    (world spin count / cell count), never hardcoded here -- this function
    only implements the arithmetic, not the assumption that it's 2."""
    if index >= n_world_spins:
        return None
    cell, slot = divmod(index, spins_per_cell)
    return (cell % grid_w, cell // grid_w, slot)


def cell_block_bounds(cell_x: int, cell_y: int, cell_px: int) -> tuple[int, int, int, int]:
    """Pixel bounds (x0, y0, x1, y1) of one cell's whole block on the LIVE
    LATTICE canvas, at `cell_px` per cell -- the SAME pitch the DECODED
    WORLD panel uses (WORLD_CELL_PX), so cell (x, y) lands at the identical
    on-screen origin in both panels and a pinned shape reads congruently in
    both. No margin here -- this is the block's OUTER border; sub-spin
    rectangles inside it are inset by spin_slot_rect below."""
    x0, y0 = cell_x * cell_px, cell_y * cell_px
    return (x0, y0, x0 + cell_px, y0 + cell_px)


def spin_slot_rect(cell_x: int, cell_y: int, slot: int, spins_per_cell: int,
                    cell_px: int, inset: int = 2) -> tuple[int, int, int, int]:
    """Pixel bounds of ONE sub-spin's fill rectangle inside its cell's
    block: `spins_per_cell` slots laid out side by side (never stacked --
    stacking would put slot 1 outside the 1:1 aspect this whole fix exists
    to restore), each `inset` px clear of its neighbours and of the block's
    own border so the border reads as a border, not just another seam."""
    bx0, by0, bx1, by1 = cell_block_bounds(cell_x, cell_y, cell_px)
    slot_w = max(1, (bx1 - bx0) // spins_per_cell)
    x0 = bx0 + slot * slot_w
    x1 = bx0 + (slot + 1) * slot_w if slot < spins_per_cell - 1 else bx1
    return (x0 + inset, by0 + inset, x1 - inset, by1 - inset)


def mediator_slot_rect(slot_index: int, cols: int, cw: int,
                        inset: int = 1) -> tuple[int, int, int, int]:
    """Pixel bounds of one mediator's rectangle within the mediator strip,
    relative to the strip's own top-left (0, 0) -- the caller offsets by
    the strip's canvas y-origin. Mediators are laid out row-major in their
    own `cols`-wide grid, deliberately NOT the world's 8-wide grid: they
    belong to no cell, so nothing about their layout should suggest one."""
    row, col = divmod(slot_index, cols)
    x0, y0 = col * cw, row * cw
    return (x0 + inset, y0 + inset, x0 + cw - inset, y0 + cw - inset)


def batch_feasibility(draws: list[dict]) -> tuple[bool, str]:
    """Given a batch of classify_draw() results, say whether the clamp that
    produced them is infeasible (zero valid worlds among the draws) and why,
    in a sentence the app can show verbatim. An empty batch is NOT asserted
    infeasible -- there is no evidence yet either way."""
    total = len(draws)
    if total == 0:
        return False, "no draws yet"
    valid = sum(1 for d in draws if d["kind"] == "valid")
    if valid == 0:
        return True, f"no valid world satisfies these pins -- {total} draws, 0 valid"
    return False, f"{valid}/{total} draws valid"


def fmt_duration(seconds: float) -> str:
    """Format a pass_durations value honestly -- these span ~5 orders of
    magnitude (place: 201s, route: 3.1us) and are shown as such, never
    normalised into a common unit or bar scale that would hide the spread."""
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.1f} us"


def onsager_betac(j_max: float) -> float:
    """Onsager's critical coupling for the UNIFORM 2-D square-lattice Ising
    model: sinh(2*Kc) = 1, so Kc = arcsinh(1) = ln(1+sqrt(2)) and
    betac = Kc / j_max. Computed from the formula every call, never a
    hardcoded decimal -- a copy-pasted constant is exactly the kind of
    unverified figure this project's whole ethos exists to refuse.

    THIS IS AN ORIENTING ESTIMATE, NOT A DERIVED CRITICAL POINT FOR THIS
    GRAPH (see `beta_regime` below, which is what the app actually
    displays and where this caveat is shown on screen, not just in a
    comment) -- Onsager's result is exact for uniform coupling on an
    infinite 2-D square lattice with no field; this model has non-uniform
    couplings (workload weights differ per rule), hidden mediator spins,
    and a different graph entirely."""
    if j_max <= 0:
        raise ValueError(
            f"onsager_betac requires a positive |J|max, got {j_max!r}; a "
            f"model with no couplings at all has no coupling scale to site "
            f"a critical beta against")
    return math.log(1.0 + math.sqrt(2.0)) / 2.0 / j_max


ONSAGER_ASSUMPTION_NOTE = (
    "Onsager's beta_c is EXACT for the UNIFORM 2-D square-lattice Ising "
    "model with no field. This model is NOT that: couplings vary by rule, "
    "mediator spins are hidden nodes, and the graph is not a plain square "
    "lattice. beta_c below is an ORIENTING estimate only -- not a derived "
    "critical point for this graph.")


@dataclass(frozen=True)
class BetaRegime:
    """beta vs an ORIENTING Onsager beta_c -- see ONSAGER_ASSUMPTION_NOTE,
    which every renderer of this dataclass must show verbatim, not just
    reference. `ratio` > 1 sits past where the orienting estimate would put
    the ordered phase; < 1 sits before it. Neither implies anything exact
    about non-uniform couplings on a mediated, non-square graph."""
    beta: float
    betac: float
    ratio: float
    assumption_note: str = ONSAGER_ASSUMPTION_NOTE


def beta_regime(beta: float, j_max: float) -> BetaRegime:
    betac = onsager_betac(j_max)
    return BetaRegime(beta=beta, betac=betac, ratio=beta / betac)


class Trace:
    """A bounded ring buffer of (x, y) points for a live line plot -- pure
    data, no Tk here (see tests/test_frontier.py). `bounds()` is what a
    renderer MUST use to label its axes: the brief is explicit that an
    unlabelled sparkline is decoration, not instrumentation, so no canvas
    drawing code in this file is allowed to skip calling it.

    I-2/F-R6 + F1/F-R10: `chain_breaks` is a THIRD parallel ring buffer,
    one bool per point, recording whether that point may be honestly drawn
    as a continuation of the point before it. A connected line between two
    points asserts they are successive draws of ONE physical Markov chain
    -- true within a clamped batch's own chain-major run, false at every
    restart and every chain boundary. See `trace_runs` below, which both
    live renderers (_draw_trace, render_line_plot) call to turn this into
    actual line segments rather than each re-deriving the grouping."""

    def __init__(self, maxlen: int = 300):
        self.xs: deque = deque(maxlen=maxlen)
        self.ys: deque = deque(maxlen=maxlen)
        self.chain_breaks: deque = deque(maxlen=maxlen)

    def append(self, x: float, y: float, chain_break: bool = True) -> None:
        """chain_break defaults to True -- the SAFE default. A caller that
        does not explicitly know this point continues the same physical
        chain as the one before it must never silently get a connected
        line; that silent assumption is exactly the bug this fix removes.
        Pass chain_break=False only where continuity is actually true
        (e.g. valid_frac_trace's own running cumulative statistic, which
        is a genuine single well-defined sequence, not a physical
        trajectory sampled from possibly-different chains)."""
        self.xs.append(x)
        self.ys.append(y)
        self.chain_breaks.append(chain_break)

    def __len__(self) -> int:
        return len(self.xs)

    def bounds(self) -> tuple[float, float, float, float] | None:
        """(xmin, xmax, ymin, ymax), or None when empty -- never a
        fabricated (0, 0, 0, 0) range for an empty trace."""
        if not self.xs:
            return None
        return (min(self.xs), max(self.xs), min(self.ys), max(self.ys))


def trace_runs(chain_breaks: Sequence[bool]) -> list[tuple[int, int]]:
    """[(start, end_exclusive), ...] index ranges, one per maximal run of
    points that may be honestly drawn as ONE connected line segment. A run
    boundary starts at every index whose own chain_breaks value is True --
    always true, trivially, for index 0 (there is no earlier point to
    connect it to, regardless of its own flag). This is the ONE place
    either trace renderer decides what counts as "the same physical
    chain" -- both `LatticeApp._draw_trace` and `render_line_plot` call
    this rather than each re-deriving the grouping, which is exactly the
    duplication-without-a-shared-helper that let F-R10 (the identical bug
    in a second renderer) go unnoticed after F-R6 was fixed."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, brk in enumerate(chain_breaks):
        if start is None or brk:
            if start is not None:
                runs.append((start, i))
            start = i
    if start is not None:
        runs.append((start, len(chain_breaks)))
    return runs


def energy_of_draw(im, row) -> float:
    """The physical energy E(x) of one raw {0,1} draw under IsingModel `im`,
    from ITS OWN documented sign convention (lower.py module docstring:
    "sum b s + sum J s s == -E(x)", spins s = 2*occupancy - 1). The same
    formula `tsu.passes.search._from_ising` uses for the compiler's own
    mixing diagnostic (Task B3 reuses it too) -- reimplemented here, in
    pure Python/numpy, as a small local function rather than importing a
    leading-underscore name from another module."""
    s = 2.0 * np.asarray(row, dtype=float) - 1.0
    total = im.offset
    for i in range(len(im.nodes)):
        total -= im.biases[i] * s[i]
    for k, (u, v) in enumerate(im.edges):
        total -= im.weights[k] * s[u] * s[v]
    return float(total)


@dataclass(frozen=True)
class Sampled:
    """F-R11 / R19 F2: marks a float as a SAMPLING-MEASURED value (a
    proportion, or a TV distance measured against a finite sample) so
    fmt_value can round it to the precision its own `uncertainty` supports,
    instead of the blanket 6-significant-figure formatting every other
    float gets. `uncertainty` must be a REAL, already-computed number -- a
    binomial standard error (`_binomial_se`), or a noise floor the receipt
    itself measured (e.g. `execution_noise_floor`) -- NEVER fabricated.

    Verified live at n=6400 (the fixed `_VERIFY_SAMPLE_PARAMS` in
    passes/search.py: n_chains=32 * n_samples=200): task_validity's
    binomial SE is 0.0054 and codeword_violation_rate's is 0.0014, against
    the removed blanket 6-s.f. display -- roughly three orders of
    magnitude more precision than either measurement supports. A DERIVED/
    EXACT float (beta, j_max, onsager_betac, energy_tv -- see
    `_verification_display_value`'s own docstring for why energy_tv is
    excluded) is never wrapped and keeps its existing precision: the fix
    is this category distinction, not a global format change."""
    value: float
    uncertainty: float


def _binomial_se(p: float, n: int) -> float:
    """Standard error of a proportion measured over n i.i.d. draws --
    sqrt(p*(1-p)/n). A real, computable uncertainty (never invented)
    whenever n is a known, positive sample size; NaN otherwise so callers
    can detect "no usable uncertainty" the same way they detect any other
    non-finite value, rather than by a separate sentinel."""
    if not (isinstance(n, int) and n > 0) or not math.isfinite(p):
        return float("nan")
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


def _fmt_sampled(value: float, uncertainty: float) -> str:
    """Round `value` to the decimal place its `uncertainty` supports.
    A non-finite or non-positive uncertainty means no usable sample-size
    information reached this call -- rather than either fabricating a
    precision claim or silently keeping the removed blanket 6
    significant figures, this falls back to a DELIBERATELY conservative,
    explicitly documented fixed precision: 3 significant figures (chosen
    to still read as "a real number", while being conspicuously coarser
    than the removed 6-s.f. default -- there is no receipt-derived
    justification for any more)."""
    if not math.isfinite(uncertainty) or uncertainty <= 0:
        return f"{value:.3g}"
    decimals = max(0, -math.floor(math.log10(uncertainty)))
    return f"{value:.{decimals}f}"


def fmt_value(v: Any) -> str:
    """Render a receipt scalar for display. A string is ALREADY either a
    real value's repr or an 'unavailable: <reason>' message written by the
    compiler itself (see verification.json/regime.json) -- passed through
    verbatim either way, never re-interpreted or replaced.

    F-R11 / R19 F2: a bare float is DERIVED/EXACT and keeps its full
    6-significant-figure precision, unchanged. A `Sampled` value is a
    SAMPLING-MEASURED number and is instead rounded to the precision its
    own `uncertainty` supports -- see `Sampled`/`_fmt_sampled` above."""
    if v is None:
        return "unavailable: field absent from receipt"
    if isinstance(v, Sampled):
        return _fmt_sampled(v.value, v.uncertainty)
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _verification_display_value(key: str, val: Any, verification: dict,
                                cost: dict) -> Any:
    """Which value a VERIFICATION-panel row should hand to `fmt_value` for
    `key` (F-R11 / R19 F2). task_validity/codeword_violation_rate are
    wrapped as `Sampled` against the SAME sampler params (cost.json's
    sampler.params.n_chains * n_samples) the receipt's own compile-time
    verification run actually drew -- never a fabricated sample size, and
    never present for a receipt whose compile never reached that point
    (see demo/receipts/l1_infeasible/cost.json: `"sampler": {}`), in which
    case the value passes through unwrapped. execution_tv reuses its own
    already-measured `execution_noise_floor` (also a real receipt field,
    not invented here). energy_tv is deliberately NEVER wrapped: it is an
    EXACT enumeration agreement (search.py's `_verify` compares the IR's
    energy against the lowered Ising model's over every reachable state,
    not a finite sample of it) -- a different kind of number entirely,
    for which high precision is the correct, honest display (see R3's own
    "agrees to 4.4e-16" framing). cross_check_tv and the diversity_*
    fields have no comparable already-recorded uncertainty and are left
    untouched rather than have one invented for them."""
    if not isinstance(val, float):
        return val
    if key in ("task_validity", "codeword_violation_rate"):
        params = (cost.get("sampler") or {}).get("params") or {}
        n_chains, n_samples = params.get("n_chains"), params.get("n_samples")
        if (isinstance(n_chains, int) and isinstance(n_samples, int)
                and n_chains > 0 and n_samples > 0):
            n = n_chains * n_samples
            return Sampled(val, _binomial_se(val, n))
        return val
    if key == "execution_tv":
        floor = verification.get("execution_noise_floor")
        if isinstance(floor, float):
            return Sampled(val, floor)
        return val
    return val


# --------------------------------------------------------------------------
# receipt loading + the compiler's own decode/validate path
# --------------------------------------------------------------------------
class Receipt:
    """Everything read from demo/receipts/small, once, at startup. No field
    here is recomputed after __init__ -- compilation ran once, when the
    receipt was written; this class only reads what that run produced."""

    def __init__(self, path: Path):
        self.path = path

        def load(name):
            return json.loads((path / name).read_text())

        self.passes = load("passes.json")
        self.metrics = load("metrics.json")
        self.gates = load("gates.json")
        self.target = load("target.json")
        self.program = load("program.json")
        self.verification = load("verification.json")
        self.regime = load("regime.json")
        self.workload = load("workload.json")
        self.environment = load("environment.json")
        # F-R11 / R19 F2: cost.json's sampler.params carries the exact
        # n_chains/n_samples the compile-time verification run drew --
        # the real, receipt-recorded sample size _verification_display_
        # value uses to size task_validity/codeword_violation_rate's
        # display precision, never a fabricated one.
        self.cost = load("cost.json")

        self.spec = load_spec(str(path / "spec.yaml"))
        self.encoding_name = _selected_encoding(path)
        self.enc = encode(self.spec, self.encoding_name)
        self.sampling_program = reconstruct_program(str(path))
        self.im = self.sampling_program.ising

        mediator_set = set(self.program.get("mediator_nodes", ()))
        self.n_spins = len(self.im.nodes)
        self.world_idx = [i for i in range(self.n_spins) if i not in mediator_set]
        self.mediator_idx = sorted(mediator_set)

        # Task 7: `.get("mediation", {})` only supplies {} when the KEY is
        # absent -- an unmediated (bipartite) receipt like elev_band still
        # HAS the key, with value `null` (json -> None), because place()
        # ran the mediation pass and recorded "nothing to mediate", not
        # "mediation didn't run". `.get(...) or {}` catches both absent-key
        # and present-but-null the same honest way: a bipartite overlay's
        # Receipt must load cleanly, not crash on the exact fact (0
        # mediators) this task's temperature control depends on.
        mediation_raw = self.passes.get("mediation")
        med = mediation_raw or {}
        self.mediator_count = med.get("mediator_count", len(self.mediator_idx))
        # Minor #3 (fix-round-2): "field absent from receipt" was wrong for
        # elev_band -- its "mediation" key IS present, with value null (see
        # comment above), not absent. Distinguish the two honestly: a
        # present-but-null mediation pass (bipartite, 0 mediators) gets its
        # own reason string; a genuinely absent key keeps the old one.
        if "partition_method" in med:
            self.partition_method = med["partition_method"]
        elif "mediation" in self.passes and mediation_raw is None:
            self.partition_method = ("unavailable: mediation pass recorded "
                                      "null (bipartite receipt, 0 mediators) "
                                      "-- no partition_method to report")
        else:
            self.partition_method = "unavailable: field absent from receipt"
        self.bipartite_after = med.get("bipartite_after")
        self.beta_used = med.get("beta_used", self.im.beta)


def render_world_image(grid: np.ndarray, up: int = CELL_UP) -> Image.Image:
    """Lifted verbatim (parameterised on `up`) from demo/render_world.py.
    Shading and per-cell decoration are DETERMINISTIC functions of the
    decoded cell values only (spec S1: same cell -> same pixels, always) --
    no world fact is invented here, only a rendering of facts already
    decoded by `enc.decode`."""
    base = PAL[np.kron(grid, np.ones((up, up), int))]
    hgt = np.choose(grid, [0.0, 2.0, 1.0])
    ef = ndimage.gaussian_filter(ndimage.zoom(hgt, up, order=3), up * 0.5)
    gy, gx = [g * up * 1.4 for g in np.gradient(ef)]
    slope, aspect = np.arctan(np.hypot(gy, gx)), np.arctan2(-gx, gy)
    az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope)
                     * np.cos(az - aspect), 0, None) ** 0.85
    shade = 0.58 + 0.92 * shade

    deco = np.zeros(base.shape[:2])
    yy, xx = np.mgrid[0:up, 0:up] / float(up)
    for r in range(H):
        for c in range(W):
            t, rg = grid[r, c], np.random.default_rng(r * 1000 + c)
            tile = np.zeros((up, up))
            if t == GRASS:
                for _ in range(14):
                    cy, cx, s = rg.uniform(.1, .9), rg.uniform(.1, .9), rg.uniform(.03, .055)
                    tile -= 0.5 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
            elif t == ROCK:
                tile += 0.22 * np.sin(15 * (xx * 0.7 - yy * 0.8) + rg.uniform(0, 6.28)) \
                        + 0.10 * rg.normal(0, 1, (up, up))
            else:
                tile += 0.05 * np.sin(9 * yy + rg.uniform(0, 6.28))
            deco[r * up:(r + 1) * up, c * up:(c + 1) * up] = tile
    deco = ndimage.gaussian_filter(deco, 1.1)

    img = base * shade[..., None] * (1 + 0.40 * deco[..., None])
    wet = (np.kron(grid, np.ones((up, up), int)) == WATER)
    shore = np.clip(ndimage.gaussian_filter(wet.astype(float), up * 0.22) - wet, 0, 1)
    img = img * (1 - 0.45 * shore[..., None]) + np.array([222, 214, 180]) * 0.45 * shore[..., None]
    g = np.zeros(base.shape[:2])
    g[::up, :] = g[:, ::up] = 1
    img *= (1 - 0.05 * g[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def render_band_image(grid01: np.ndarray, up: int = CELL_UP) -> Image.Image:
    """Task 7: render one elevation BAND's decoded 0/1 grid ("below"/
    "above" threshold, see demo/elevation.py's own module docstring) --
    deliberately REUSES this file's own terrain palette (PAL[WATER] for
    below, PAL[GRASS] for above) rather than inventing a second colour
    scheme: demo/render_world.py's own pseudo-relief already treats grass
    as the highest terrain class and water as the lowest (hgt = np.choose
    (grid, [0.0, 2.0, 1.0])), so "below=water-blue, above=grass-green"
    reads as the same low/high convention this app already establishes,
    not a new one."""
    below_above = PAL[np.array([WATER, GRASS])]
    img = below_above[np.kron(np.asarray(grid01, dtype=int), np.ones((up, up), int))]
    g = np.zeros(img.shape[:2])
    g[::up, :] = g[:, ::up] = 1
    img = img * (1 - 0.06 * g[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def grid_to_decoded(grid: np.ndarray) -> dict[str, int]:
    """Inverse of classify_draw's own `grid[y, x] = decoded[f"g{x}_{y}"]`
    construction -- Task 7 reconstructs the raw decoded dict
    demo/elevation.band_patch expects (its `base=...` argument) from an
    already-decoded grid, rather than keeping a second copy of the decoded
    assignment around only for this."""
    h, w = grid.shape
    return {f"g{x}_{y}": int(grid[y, x]) for y in range(h) for x in range(w)}


def classify_draw(receipt: Receipt, row: np.ndarray, seed: int) -> dict:
    """One raw physical sample -> a classified draw. Uses ONLY the
    compiler's own enc.is_codeword / enc.decode / spec.contract.validate --
    no decode or contract logic is reimplemented here."""
    bits = dict(zip(receipt.im.nodes, row.tolist()))
    if not receipt.enc.is_codeword(bits):
        return {"kind": "non-codeword", "raw": row.tolist(), "seed": seed}
    decoded = receipt.enc.decode(bits)
    result = receipt.spec.contract.validate(decoded)
    if not result.ok:
        return {"kind": "contract-fail", "raw": row.tolist(), "seed": seed,
                "violations": list(result.violations)}
    grid = np.array([[int(decoded[f"g{x}_{y}"]) for x in range(W)] for y in range(H)])
    image = render_world_image(grid)
    total = float(W * H)
    mix = {TERRAIN_NAMES[k]: 100.0 * float((grid == k).sum()) / total for k in TERRAIN_ORDER}
    return {"kind": "valid", "raw": row.tolist(), "seed": seed, "grid": grid,
            "image": image, "mix": mix}


def _chain_break_at(row_idx: int, samples_per_chain: int) -> bool:
    """I-2/F-R6 + F1/F-R10: True at the FIRST row of every physical chain
    within one SampleWorker call -- the point that must NOT be drawn as a
    continuation of whatever the trace already holds. Both
    `_run_unclamped_tick` and `_run_clamped_batch` share this rather than
    each re-deriving the chain-major row-index arithmetic separately --
    that kind of near-identical-but-separately-maintained logic at two
    call sites is exactly how F-R10 (the same continuity bug, missed in a
    second renderer) came to exist.

    `_run_clamped_batch`'s own `got` is chain-major flattened
    (tsu.simulate.simulate -> sample_chains(...).reshape(-1, ...)): row i
    belongs to chain i // samples_per_chain, so row i starts a new chain
    exactly when i % samples_per_chain == 0. `_run_unclamped_tick` calls
    this with samples_per_chain=N_SAMPLES_PER_CALL, which is always 1 (a
    module constant) -- every row_idx % 1 == 0, so every row is correctly
    its own chain: N_SAMPLES_PER_CALL==1 means no unclamped row is EVER a
    genuine continuation of another, structurally, regardless of session
    length (the same fact C-1's fix already established for why the tau
    claim had to be dropped rather than reshaped)."""
    return samples_per_chain <= 0 or row_idx % samples_per_chain == 0


def should_abort_batch(*, stopping: bool, paused: bool, is_step: bool) -> bool:
    """UI2: the pure decision inside a batch's row-by-row push loop --
    True means "stop pushing further rows from the batch that is ALREADY
    computed and in hand, right now." Checked BEFORE each row is pushed
    (not after, which is what let a few already-computed rows keep
    appearing after Pause was clicked -- see the responsive-pause note in
    ui2-report.md). `stopping` (worker tear-down) always wins. `paused`
    aborts too, UNLESS this call is an explicit Step: Step's entire point
    is to push its one batch even though the worker is sitting paused, so
    a Step-triggered batch is never cut short by the pause flag it was
    called under."""
    if stopping:
        return True
    if paused and not is_step:
        return True
    return False


# --------------------------------------------------------------------------
# worker thread -- the ONLY thing that calls thrml_sample; never touches Tk
# --------------------------------------------------------------------------
class SampleWorker(threading.Thread):
    """Two sampling modes, chosen by whether `clamp` is non-empty:

    UNCLAMPED (clamp is None): the original continuous stream -- small,
    fast per-tick calls straight into `thrml_sample` on the receipt's own
    reconstructed program, unchanged from before this feature.

    CLAMPED (clamp is a non-empty {name: value} dict): repeated calls to
    `tsu.simulate.simulate(..., clamp=clamp)` -- clamping is a WORKLOAD-level
    concept (spec 5.3 / tests/test_clamp.py) that `simulate` already knows
    how to translate through `enc.encode_clamp` and repartition via
    `analyse`/`build_program`; this worker never touches that machinery
    itself, only calls `simulate` and classifies what comes back with the
    same `classify_draw` the unclamped path uses. Larger batch than the
    unclamped per-tick call (below) because a pin change is a deliberate,
    infrequent action, not a per-tick refresh -- the app can afford to
    answer "does this clamp even have a valid world" from a solid batch
    rather than trickling one draw at a time. EXACTLY how large is now the
    active SPEED_LEVELS entry's `clamp_samples`, not a fixed constant --
    see the UI2 note above SPEED_LEVELS.

    UI2 (responsive pause + speed control): `speed_idx` selects a
    SPEED_LEVELS entry read fresh at the start of every batch (a live
    slider change takes effect on the NEXT batch, not the one already in
    flight -- see `set_speed`). `step_evt`, when set while `pause` is also
    set, runs exactly one more batch at the current speed and re-pauses --
    this is the Step button. Neither of those, nor anything else in this
    class, ever changes n_warmup or steps_per_sample: those two alone
    determine what is being sampled (spec 5.3), so they are read from the
    fixed N_WARMUP/CLAMP_N_WARMUP/STEPS_PER_SAMPLE constants everywhere,
    never from a speed level.
    """

    def __init__(self, receipt: Receipt, out_q: "queue.Queue[dict]", seed_base: int,
                 clamp: dict[str, int] | None = None, start_paused: bool = False,
                 speed_idx: int = DEFAULT_SPEED_IDX):
        super().__init__(daemon=True)
        self.receipt = receipt
        self.q = out_q
        self.pause = threading.Event()
        self.stop_evt = threading.Event()
        self.step_evt = threading.Event()  # UI2: Step button, see run()
        if start_paused:
            self.pause.set()
        self.seed_base = seed_base
        self.clamp = dict(clamp) if clamp else None
        self.draw_counter = 0
        self.batch_counter = 0
        # UI2: plain int attribute, mutated live from the UI thread via
        # set_speed(). CPython's GIL makes a single attribute write/read
        # atomic -- the same reasoning this class already relies on for
        # `clamp` being safe to read from the worker thread after
        # construction; `pause`/`stop_evt`/`step_evt` use threading.Event
        # instead because they need blocking-wait semantics, which a plain
        # attribute doesn't give you -- speed_idx only ever needs the
        # latest value, so a plain attribute is the right tool.
        self.speed_idx = speed_idx

    def set_speed(self, idx: int) -> None:
        """UI2: live speed change from the UI thread. Takes effect at the
        start of the next batch (see run()/_pace_delay) -- never
        interrupts a batch already in flight."""
        self.speed_idx = idx

    def request_step(self) -> None:
        """UI2: Step button. Only has an effect while paused (see run());
        harmless no-op otherwise beyond leaving the flag set, which the
        next pause-and-check cycle will consume."""
        self.step_evt.set()

    def _put(self, msg: dict) -> None:
        while not self.stop_evt.is_set():
            try:
                self.q.put(msg, timeout=0.2)
                return
            except queue.Full:
                continue

    def _classify_and_push(self, row, seed, sampler_params: dict,
                           chain_break: bool) -> dict:
        self.draw_counter += 1
        d = classify_draw(self.receipt, row, seed)
        d["draw_idx"] = self.draw_counter
        # UI2: the ACTUAL n_chains/n_samples this particular draw came
        # from, not a fixed constant -- speed control makes those vary
        # batch to batch, and Save World (below) must record what really
        # produced the saved grid, not what Full speed would have used.
        d["sampler_params"] = sampler_params
        # I-2/F-R6 + F1/F-R10: whether THIS draw may be honestly drawn as
        # a continuation of the draw pushed immediately before it -- see
        # _chain_break_at's own docstring for the chain-major arithmetic
        # this is computed from. The live SCOPE trace renderers read this
        # (via Trace.append) instead of assuming every draw continues the
        # last one.
        d["chain_break"] = chain_break
        self._put(d)
        return d

    def run(self) -> None:
        while not self.stop_evt.is_set():
            if self.pause.is_set():
                if self.step_evt.is_set():
                    self.step_evt.clear()
                    self._run_one_batch(is_step=True)
                else:
                    time.sleep(0.05)
                continue
            self._run_one_batch(is_step=False)
            self._pace_delay()

    def _run_one_batch(self, is_step: bool) -> None:
        if self.clamp:
            self._run_clamped_batch(is_step)
        else:
            self._run_unclamped_tick(is_step)

    def _pace_delay(self) -> None:
        """UI2 speed control: sleep out the CURRENT speed level's
        inter-batch delay in short slices, checking pause/stop between each
        slice, so a Pause click (or a mid-delay speed change) lands within
        ~50ms rather than blocking through a bare time.sleep(delay_s) --
        never a busy-wait, never a blind sleep either. Runs strictly
        BETWEEN batches, after a batch's own draws are already pushed, so
        it paces the DISPLAY only -- it cannot alter what gets sampled."""
        remaining = speed_level(self.speed_idx)["delay_s"]
        while remaining > 0 and not self.stop_evt.is_set() and not self.pause.is_set():
            chunk = min(0.05, remaining)
            time.sleep(chunk)
            remaining -= chunk

    def _run_unclamped_tick(self, is_step: bool = False) -> None:
        n_chains = speed_level(self.speed_idx)["chains"]
        sampler_params = {"n_chains": n_chains, "n_samples_per_call": N_SAMPLES_PER_CALL,
                           "n_warmup": N_WARMUP, "steps_per_sample": STEPS_PER_SAMPLE}
        seed = self.seed_base + self.draw_counter
        try:
            rows = thrml_sample(self.receipt.sampling_program,
                                 n_chains=n_chains,
                                 n_samples=N_SAMPLES_PER_CALL,
                                 n_warmup=N_WARMUP,
                                 steps_per_sample=STEPS_PER_SAMPLE,
                                 seed=seed)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self._put({"kind": "error", "message": str(exc)})
            self.stop_evt.set()
            return
        for row_idx, row in enumerate(rows):
            # UI2: checked BEFORE pushing, not after -- see
            # should_abort_batch's docstring for why this is the actual
            # responsive-pause fix, not just the speed control.
            if should_abort_batch(stopping=self.stop_evt.is_set(),
                                   paused=self.pause.is_set(), is_step=is_step):
                return
            self._classify_and_push(
                row, seed, sampler_params,
                chain_break=_chain_break_at(row_idx, N_SAMPLES_PER_CALL))

    def _run_clamped_batch(self, is_step: bool = False) -> None:
        from tsu.simulate import simulate  # local: keeps this app's only
        # entry point into clamping right here, next to the docstring above
        n_samples = speed_level(self.speed_idx)["clamp_samples"]
        sampler_params = {"n_chains": CLAMP_N_CHAINS, "n_samples": n_samples,
                           "n_warmup": CLAMP_N_WARMUP, "steps_per_sample": STEPS_PER_SAMPLE}
        self.batch_counter += 1
        seed = self.seed_base + self.batch_counter
        try:
            _path, got, _im = simulate(str(self.receipt.path),
                                        n_chains=CLAMP_N_CHAINS,
                                        n_samples=n_samples,
                                        n_warmup=CLAMP_N_WARMUP,
                                        steps_per_sample=STEPS_PER_SAMPLE,
                                        seed=seed, clamp=self.clamp,
                                        output_dir=SIM_OUTPUT_DIR)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self._put({"kind": "error", "message": str(exc)})
            self.stop_evt.set()
            return
        draws = []
        for row_idx, row in enumerate(got):
            if should_abort_batch(stopping=self.stop_evt.is_set(),
                                   paused=self.pause.is_set(), is_step=is_step):
                break
            draws.append(self._classify_and_push(
                row, seed, sampler_params,
                chain_break=_chain_break_at(row_idx, n_samples)))
        infeasible, reason = batch_feasibility(draws)
        valid = sum(1 for d in draws if d["kind"] == "valid")
        self._put({"kind": "batch_summary", "infeasible": infeasible, "reason": reason,
                    "valid": valid, "total": len(draws), "seed": seed})


# --------------------------------------------------------------------------
# UI -- Task 0: every colour below is a NAME from demo/theme.py, never a
# hand-copied hex literal (see theme.py's own docstring for the semantic
# rule this must honour: blue means COLD -- mediator spins, a locked
# control -- and NOTHING else; gold is the live/active-data channel; PASS/
# FAIL/WARN stay off that accent channel entirely, per the mockup's own
# "olive PASS / orange warn / red FAIL / ice lock" legend).
#
# WORLD_ON/WORLD_OFF (the LIVE LATTICE panel's live/lit p-bit colour) were,
# before this task, blue ("#6fb3ff") -- a stray DECORATIVE blue, exactly
# backwards from the mockup's own construction note ("Lit gold nodes are
# world p-bits in state 1. Cold-blue nodes are hidden mediators -- frozen
# helpers, not terrain."). MEDIATOR_ON/OFF were a separate purple pair with
# no relation to "cold" at all. Both are corrected here: world p-bits are
# gold (the live/active channel they actually are), mediator spins are
# blue (the cold channel they actually are).
# --------------------------------------------------------------------------
BG = theme.PAGE
PANEL_BG = theme.PANEL
BORDER = theme.BEZEL
FG = theme.CREAM
DIM = theme.CREAM_DIM
GOOD = theme.STATUS_PASS      # PASS -- olive, never gold (gold means "live data", not "this passed")
BAD = theme.STATUS_FAIL       # FAIL -- red, critical only
WARN = theme.STATUS_WARN      # warning -- orange
ACCENT = theme.GOLD           # panel titles / active labels -- the live/useful-window channel
MEDIATOR_ON = theme.BLUE_LIT  # mediator spins are COLD, full stop -- lit is still cold
MEDIATOR_OFF = theme.BLUE_DEEP
WORLD_ON = theme.GOLD         # world p-bits are the LIVE channel, never blue -- see block comment above
WORLD_OFF = theme.GOLD_GHOST
MONO_FAMILY = theme.resolve_mono_family()  # Consolas-fallback pre-Tk-root (headless-safe); re-resolved once a real Tk root exists, see LatticeApp.__init__
MONO = (MONO_FAMILY, 9)
MONO_B = (MONO_FAMILY, 9, "bold")


# Task 6: colours for the SCOPE panel's PIL-rendered plots, derived from
# this file's own palette above (never a second literal set of colours).
# Task 0: PLOT_BG was the hand-picked "#111218", now theme.INSET -- one
# recessed-area colour, matching every other recessed widget below
# (log box, frontier box, energy/valid-frac canvases) instead of a second
# literal only PLOT_BG used.
PLOT_BG = _rgb(theme.INSET)
PLOT_FG = _rgb(FG)
PLOT_DIM = _rgb(DIM)
PLOT_ACCENT = _rgb(ACCENT)
PLOT_WARN = _rgb(WARN)
PLOT_GOOD = _rgb(GOOD)
PLOT_BAD = _rgb(BAD)


# --------------------------------------------------------------------------
# Task 6: SCOPE panel plot renderers. Pure functions -- no Tk here, each
# returns a PIL.Image (the brief requires plots be numpy/PIL images blitted
# to a Tk canvas, not drawn as Tk canvas primitives / widget chrome: a
# semi-log axis, filled histogram bars, and an overlaid scatter+curve are
# not things Tk's own line/rect primitives render well). Axis extremes are
# ALWAYS drawn on the image itself (never skipped -- an unlabelled
# sparkline is decoration, not instrumentation); a longer plain-language
# caption is returned alongside for the caller to show as a Tk label,
# where PIL's small bitmap font would be unreadable.
# --------------------------------------------------------------------------

def render_acf_plot(w: int, h: int, series: Sequence[float]) -> tuple[Image.Image, str]:
    """C-1 fix: this panel's only possible input, self.energy_trace, is
    NEVER one Markov chain's own successive draws -- it cannot be, by
    construction of the two callers that append to it. Every UNCLAMPED
    tick (_run_unclamped_tick) is an independent restart: a fresh seed and
    a fresh n_warmup=300 warmup, contributing exactly one sample per
    chain, every tick. Every CLAMPED batch (_run_clamped_batch ->
    tsu.simulate.simulate -> thrml_backend.sample_chains(...).reshape(-1,
    ...)) flattens n_chains mutually independent parallel chains
    chain-major into one run of consecutive rows. tsu.ess's own module
    contract is explicit that this is not a valid input --
    effective_sample_size's docstring says outright "callers must NOT
    flatten multiple chains into one series before calling this" -- and
    this app's live energy_trace is exactly that flattening, every time,
    with no exception. Previously this function called
    tsu.ess.integrated_autocorrelation_time (via demo.scope.autocorrelation,
    and directly) on that series anyway: audit finding C-1, a
    confident-looking tau/ACF number computed on data the estimator's own
    contract rules out, with tsu.ess.effective_sample_size's reliability
    gate unreachable from this call site by construction (this function
    never called it -- it called integrated_autocorrelation_time
    directly). There is no way to fix that by reshaping THIS data -- a
    live, continuously-streaming, restart-heavy panel structurally has no
    genuine (n_chains, n_samples) buffer of one chain's own draws to
    offer. So this function no longer calls tsu.ess or demo.scope's
    autocorrelation machinery at all, on any input, and makes no tau/ACF
    claim of any kind. A genuine, gated measurement of this receipt's own
    tau exists -- demo/ess_run.py: one sample_chains() call collecting a
    real (n_chains, n_samples) buffer, routed through
    tsu.ess.effective_sample_size so its reliability gate applies -- see
    the VERIFICATION panel above for that receipt-level, compile-time
    result; this live panel has no equivalent to show."""
    img = Image.new("RGB", (w, h), PLOT_BG)
    d = ImageDraw.Draw(img)
    d.text((10, h // 2 - 6), "tau/ACF: unavailable", fill=PLOT_DIM)
    caption = (
        f"unavailable: this LIVE energy trace (ring buffer, "
        f"maxlen={SCOPE_ENERGY_TRACE_MAXLEN}, N={len(series)} draws so "
        f"far) is not a single Markov chain's own successive draws -- "
        f"every unclamped tick restarts fresh (new seed, new warmup, one "
        f"sample per chain) and every clamped batch flattens n_chains "
        f"independent parallel chains chain-major, so tsu.ess's "
        f"autocorrelation/tau estimator (which requires one chain's own "
        f"draws -- see tsu.ess.effective_sample_size's own docstring) "
        f"cannot be validly applied to it, at any sample count. See "
        f"demo/ess_run.py for how a genuine, gated tau measurement looks, "
        f"and the VERIFICATION panel above for this receipt's own frozen "
        f"result -- a SEPARATE, compile-time-only measurement this live "
        f"panel does not and cannot live-update.")
    return img, caption


def render_line_plot(w: int, h: int, xs: Sequence[float], ys: Sequence[float],
                     xlabel: str, ylabel: str,
                     y_range: tuple[float, float] | None = None,
                     chain_breaks: Sequence[bool] | None = None) -> Image.Image:
    """Linear x/y line plot -- used for the magnetization trace, with
    y_range fixed to (-1, 1) (the order parameter's own physical bounds,
    a more honest axis than autoscaling to whatever range happened to be
    observed so far).

    F1/F-R10: `chain_breaks` (one bool per point, see Trace/trace_runs)
    says which points are genuine successive draws of the SAME physical
    chain -- a magnetization trace is chain-major flattened / restart-
    interleaved exactly like the energy trace (I-2/F-R6's own fix), so a
    single polyline through every point regardless of chain_breaks would
    assert continuous single-trajectory dynamics that do not exist. Every
    point still gets a small dot marker (drawn regardless of run
    membership) so an all-singleton-run trace -- the app's default,
    unclamped mode, where every draw is its own one-sample chain -- still
    shows something. `chain_breaks=None` (no provenance given) is treated
    as EVERY point breaking -- the safe default, never assumed-connected."""
    img = Image.new("RGB", (w, h), PLOT_BG)
    d = ImageDraw.Draw(img)
    if len(xs) < 2:
        d.text((10, h // 2 - 6), "(no data yet)", fill=PLOT_DIM)
        return img
    pad_l, pad_r, pad_t, pad_b = 40, 8, 8, 16
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = y_range if y_range is not None else (min(ys), max(ys))
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or (abs(ymax) or 1.0)

    def px(x): return pad_l + (x - xmin) / xspan * pw

    def py(y): return pad_t + (1.0 - (y - ymin) / yspan) * ph

    pts = [(px(x), py(y)) for x, y in zip(xs, ys)]
    breaks = list(chain_breaks) if chain_breaks is not None else [True] * len(pts)
    for start, end in trace_runs(breaks):
        if end - start >= 2:
            d.line(pts[start:end], fill=PLOT_ACCENT, width=1)
    r = 1.5
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], fill=PLOT_ACCENT)
    d.text((pad_l, pad_t), f"{ymax:.3g}", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, pad_t + ph), f"{ymin:.3g}", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, h - 4), f"{xlabel}={xmin:.0f}", fill=PLOT_DIM, anchor="ls")
    d.text((w - pad_r, h - 4), f"{xmax:.0f}", fill=PLOT_DIM, anchor="rs")
    d.text((w - pad_r, pad_t), ylabel, fill=PLOT_DIM, anchor="ra")
    return img


def render_energy_histogram_plot(w: int, h: int, series: Sequence[float],
                                 bins: int = SCOPE_ENERGY_HIST_BINS
                                 ) -> tuple[Image.Image, str]:
    """Filled-bar histogram of `series` (the energy trace's own values) --
    the distribution the trace only samples one point of at a time. Returns
    (image, caption) like every OTHER detachable plot's own render_*
    function (render_sigmoid_plot, render_heatmap_image) -- I2, final
    review: this used to return only the image, with the caption's actual
    substance hand-written TWICE, once at each of its two call sites (the
    inline SCOPE cell in _refresh_scope_panel, and the detached window in
    _render_detached_plot), and the two had already drifted apart -- the
    detached copy dropped the substantive "the trace itself only samples
    one point of this at a time" sentence in favour of implementation
    meta-commentary about which function was called. One caption, computed
    once, used by both hosts -- there is no second copy left to drift."""
    img = Image.new("RGB", (w, h), PLOT_BG)
    d = ImageDraw.Draw(img)
    if len(series) < 4:
        d.text((10, h // 2 - 6), "(no data yet)", fill=PLOT_DIM)
        return img, "waiting for at least 4 energy trace values..."
    edges, counts = energy_histogram(list(series), bins=bins)
    pad_l, pad_r, pad_t, pad_b = 40, 8, 8, 16
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    cmax = max(counts) or 1
    n_bins = len(counts)
    bw = pw / n_bins if n_bins else pw
    for i, c in enumerate(counts):
        x0 = pad_l + i * bw
        x1 = x0 + bw * 0.85
        bar_h = (c / cmax) * ph
        y1 = pad_t + ph
        y0 = y1 - bar_h
        d.rectangle([x0, y0, x1, y1], fill=PLOT_ACCENT)
    d.text((pad_l, pad_t), f"{cmax}", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, h - 4), f"E={edges[0]:.3g}", fill=PLOT_DIM, anchor="ls")
    d.text((w - pad_r, h - 4), f"{edges[-1]:.3g}", fill=PLOT_DIM, anchor="rs")
    d.text((w - pad_r, pad_t), "count", fill=PLOT_DIM, anchor="ra")
    caption = (f"Distribution of the energy trace's own {len(series)} "
              f"value(s) so far this session -- the trace itself only "
              f"samples one point of this at a time.")
    return img, caption


def render_sigmoid_plot(w: int, h: int, draws: Sequence[Sequence[int]], ising,
                        bins: int = SCOPE_LOCAL_FIELD_BINS) -> tuple[Image.Image, str]:
    """Empirical P(s=1) per POST-HOC RECONSTRUCTED local-field bin (blue
    points) plotted alongside the analytic sigmoid(2*gamma) (green curve,
    FOR REFERENCE ONLY) -- the measurable analogue of Extropic's DTM
    paper Figure 4a.

    THIS IS NOT A PLOT OF THE SAMPLER'S CONDITIONAL (fix-round-1 finding,
    see task-5-6-report.md and demo/scope.py's local_field_response
    docstring for the full reasoning): thrml exposes no local field at
    flip time, so gamma is reconstructed after the fact from each draw's
    final joint state, which makes it correlated with the very spin it's
    paired with. Measured points are therefore NOT expected to sit on
    the reference curve near the transition even for a defect-free
    sampler, AND a genuine sampler defect would look the same way -- the
    two are currently indistinguishable from this plot alone. A
    block-split measurement (task-5-6-report.md) ruled out one specific
    alternative (per-substep staleness) but not a sampler defect in
    general. All of this is restated in the returned caption, not just
    here, because the brief requires it be visible ON SCREEN.

    A bin with too few pooled samples (see MIN_LOCAL_FIELD_BIN_COUNT in
    demo/scope.py) is SKIPPED here, not plotted at a fabricated position
    -- the caption reports how many were skipped so that omission is
    visible, not silent."""
    img = Image.new("RGB", (w, h), PLOT_BG)
    d = ImageDraw.Draw(img)
    if len(draws) < 4:
        d.text((10, h // 2 - 6), "(no draws yet)", fill=PLOT_DIM)
        return img, "waiting for draws..."
    centers, probs, counts = local_field_response(draws, ising, bins=bins)
    if not centers:
        d.text((10, h // 2 - 6), "(no field data)", fill=PLOT_DIM)
        return img, "no data"

    pad_l, pad_r, pad_t, pad_b = 40, 8, 8, 16
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    gmin, gmax = min(centers), max(centers)
    gspan = (gmax - gmin) or 1.0

    def px(g): return pad_l + (g - gmin) / gspan * pw

    def py(p): return pad_t + (1.0 - p) * ph

    curve_pts = [(px(gmin + gspan * i / 40.0),
                  py(float(sigmoid(2.0 * (gmin + gspan * i / 40.0)))))
                 for i in range(41)]
    d.line(curve_pts, fill=PLOT_GOOD, width=1)

    n_skipped = 0
    for c, p, n in zip(centers, probs, counts):
        if n == 0:
            continue
        if math.isnan(p):
            n_skipped += 1
            continue
        x, y = px(c), py(p)
        r = 2.5
        d.ellipse([x - r, y - r, x + r, y + r], fill=PLOT_ACCENT)

    d.text((pad_l, pad_t), "1.0", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, pad_t + ph), "0.0", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, h - 4), f"recon.gamma={gmin:.2f}", fill=PLOT_DIM, anchor="ls")
    d.text((w - pad_r, h - 4), f"{gmax:.2f}", fill=PLOT_DIM, anchor="rs")
    d.text((w - pad_r, pad_t), "empirical P(s=1)", fill=PLOT_DIM, anchor="ra")

    caption = (
        f"blue = empirical P(s=1) vs a POST-HOC RECONSTRUCTED local field "
        f"(read from each draw's final state, NOT the sampler's live "
        f"conditional -- thrml exposes no field at flip time). green = "
        f"analytic sigmoid(2*gamma), shown FOR REFERENCE ONLY. Near the "
        f"transition, measured points are NOT expected to sit on the "
        f"reference curve: conditioning on a reconstructed field differs "
        f"from the sampler's own conditional. The gap could ALSO be a "
        f"real sampler defect -- the two explanations cannot currently "
        f"be told apart (see task-5-6-report.md). "
        f"{n_skipped}/{len(centers)} bin(s) skipped -- fewer than "
        f"{MIN_LOCAL_FIELD_BIN_COUNT} pooled samples to report a "
        f"probability (counts are real, just too sparse to plot).")
    return img, caption


# --------------------------------------------------------------------------
# Task 10: the three plots that have NO existing inline host --
# node-and-edge lattice, relaxation strip, per-cell heatmap. All three are
# real, computed data (real topology / real raw draws), never a fabricated
# illustration -- unlike demo/explainer.py's diagrams, which are
# deliberately static concept art, these read this SESSION's own live
# state and are meant to be reopened and watched update.
# --------------------------------------------------------------------------

def render_lattice_graph_image(w: int, h: int, im, world_idx: Sequence[int],
                                mediator_idx: Sequence[int], spins_per_cell: int,
                                grid_w: int) -> tuple[Image.Image, str]:
    """Extropic Fig 3c analogue: nodes coloured by ROLE -- world spins
    gold, mediator spins cold blue -- never by this session's live
    on/off state (that is LIVE LATTICE's own job, the blocks panel next
    to this button). World nodes are positioned by their own cell
    (spin_cell_position -- the SAME single source of truth LIVE LATTICE
    and DECODED WORLD both already key their own layout on), so this
    diagram's grid reads congruently with those two panels; mediator
    nodes sit in their own labelled strip below, laid out row-major, never
    implied to belong to a cell. Edges are real topology, read straight
    from `im.edges` -- at this program's degree, a few hundred thin
    lines, cheap for PIL to draw once per redraw."""
    img = Image.new("RGB", (w, h), _rgb(theme.INSET))
    d = ImageDraw.Draw(img)
    n_world = len(world_idx)
    n_cells = (n_world // spins_per_cell) if spins_per_cell else 0
    grid_h = (n_cells // grid_w) if grid_w else 0
    if n_world == 0 or spins_per_cell <= 0 or grid_w <= 0 or grid_h <= 0:
        d.text((10, h // 2 - 6), "(no topology to draw)", fill=_rgb(DIM))
        return img, "unavailable: receipt carries no world spins to lay out"

    pad = 14
    legend_h = 32
    med_count = len(mediator_idx)
    med_cols = min(16, max(med_count, 1))
    med_rows = -(-med_count // med_cols) if med_count else 0
    strip_gap = 10 if med_count else 0
    strip_h = med_rows * max(6, (w - 2 * pad) // med_cols) if med_count else 0
    avail_w = w - 2 * pad
    avail_h = h - 2 * pad - legend_h - strip_gap - strip_h
    cell_px = max(6, int(min(avail_w / grid_w, max(avail_h, 1) / grid_h)))
    grid_px_w, grid_px_h = grid_w * cell_px, grid_h * cell_px
    ox = pad + max(avail_w - grid_px_w, 0) // 2
    oy = pad

    positions: dict[int, tuple[float, float]] = {}
    for i in world_idx:
        cx, cy, slot = spin_cell_position(i, n_world, spins_per_cell, grid_w)
        slot_w = cell_px / spins_per_cell
        positions[i] = (ox + cx * cell_px + (slot + 0.5) * slot_w,
                        oy + cy * cell_px + cell_px / 2.0)

    med_cw = max(6, grid_px_w // med_cols) if med_count else 0
    strip_y0 = oy + grid_px_h + strip_gap
    for k, i in enumerate(mediator_idx):
        row, col = divmod(k, med_cols)
        positions[i] = (ox + col * med_cw + med_cw / 2.0,
                        strip_y0 + row * med_cw + med_cw / 2.0)

    edge_color = _rgb(theme.RULE)
    for u, v in im.edges:
        if u in positions and v in positions:
            d.line([positions[u], positions[v]], fill=edge_color, width=1)

    gold = _rgb(theme.GOLD)
    blue = _rgb(theme.BLUE)
    r_world = max(2, cell_px // 6)
    for i in world_idx:
        x, y = positions[i]
        d.ellipse([x - r_world, y - r_world, x + r_world, y + r_world], fill=gold)
    r_med = max(2, (med_cw // 6)) if med_count else r_world
    for i in mediator_idx:
        x, y = positions[i]
        d.ellipse([x - r_med, y - r_med, x + r_med, y + r_med], fill=blue)

    ly = h - legend_h + 6
    d.ellipse([pad, ly, pad + 10, ly + 10], fill=gold)
    d.text((pad + 16, ly - 2), f"world spin (role) x{n_world}", fill=_rgb(DIM))
    ly2 = ly + 14
    d.ellipse([pad, ly2, pad + 10, ly2 + 10], fill=blue)
    d.text((pad + 16, ly2 - 2),
           f"mediator spin (cold, frozen helper) x{med_count}", fill=_rgb(DIM))

    caption = (
        f"Structural topology of the compiled program -- {n_world} world "
        f"spins (gold) + {med_count} mediator spins (cold blue, frozen "
        f"helpers, never terrain) and {len(im.edges)} coupling edges. "
        f"Nodes are coloured by ROLE here, NOT by this session's live "
        f"state -- see LIVE LATTICE for the per-sample on/off view of "
        f"these same {n_world + med_count} spins.")
    return img, caption


def render_relaxation_strip_image(w: int, h: int, draws: Sequence[Sequence[int]],
                                   n_frames: int, n_world_spins: int,
                                   spins_per_cell: int, grid_w: int
                                   ) -> tuple[Image.Image, str]:
    """Extropic Fig 5a analogue: `n_frames` evenly-spaced snapshots of
    this session's own raw physical draws (oldest -> newest, left to
    right), each rendered as a small WORLD-spin occupancy thumbnail --
    ONE PIL image (the tokens spec's own buildability rule: "one PNG
    strip, not eight nested Frames of Labels"), never eight separate Tk
    widgets. Each thumbnail's cell colour is gold-intensity = the
    fraction of that cell's own spins_per_cell sub-spins reading 1 IN
    THAT ONE DRAW -- exactly what LIVE LATTICE's own blocks would have
    shown at that instant, RAW and valid-or-not (the same "every draw,
    valid or not" convention the energy/magnetization traces already
    use), never the decoded terrain DECODED WORLD shows and never an
    average across draws (see the per-cell heatmap for that)."""
    img = Image.new("RGB", (w, h), _rgb(theme.INSET))
    d = ImageDraw.Draw(img)
    if len(draws) < 2:
        d.text((10, h // 2 - 6), "(no data yet)", fill=_rgb(DIM))
        return img, "waiting for draws..."
    n_cells = (n_world_spins // spins_per_cell) if spins_per_cell else 0
    grid_h = (n_cells // grid_w) if grid_w else 0
    if n_world_spins == 0 or spins_per_cell <= 0 or grid_w <= 0 or grid_h <= 0:
        d.text((10, h // 2 - 6), "(no topology to draw)", fill=_rgb(DIM))
        return img, "unavailable: receipt carries no world spins to lay out"

    n = len(draws)
    k = min(n_frames, n)
    idxs = sorted(set(int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)))
    pad, gap, label_h = 6, 6, 12
    n_shown = len(idxs)
    frame_w = (w - 2 * pad - gap * max(n_shown - 1, 0)) / n_shown
    cell_px = max(2, int(min(frame_w / grid_w, (h - 2 * pad - label_h) / grid_h)))
    cold = _rgb(theme.GOLD_GHOST)  # 0 -> dim gold, NOT blue: occupancy is not temperature
    hot = _rgb(theme.GOLD)         # 1 -> full gold

    x = float(pad)
    for draw_i in idxs:
        row = np.asarray(draws[draw_i], dtype=float)[:n_world_spins]
        cell_frac = row.reshape(n_cells, spins_per_cell).mean(axis=1)
        for cell in range(n_cells):
            cx, cy = cell % grid_w, cell // grid_w
            frac = float(cell_frac[cell])
            color = tuple(int(round(cold[c] + (hot[c] - cold[c]) * frac)) for c in range(3))
            x0 = int(round(x)) + cx * cell_px
            y0 = pad + cy * cell_px
            d.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=color)
        d.text((int(round(x)), pad + grid_h * cell_px + 1), f"buf#{draw_i}", fill=_rgb(DIM))
        x += frame_w + gap

    caption = (
        f"{n_shown} evenly-spaced raw physical draws from this session's "
        f"own ring buffer ({n} held right now, oldest -> newest left to "
        f"right; buf#N is a POSITION within that buffer, not a global "
        f"sweep counter). Gold intensity = fraction of a cell's own "
        f"{spins_per_cell} world sub-spin(s) reading 1 IN THAT draw, "
        f"valid or not -- NOT the decoded terrain shown in DECODED "
        f"WORLD, and not an average across draws.")
    return img, caption


def _thermal_ramp_rgb(frac: np.ndarray) -> np.ndarray:
    """cold(blue_deep) -> gold(55%) -> hot(orange) -- verbatim the SAME
    3-stop gradient render_temperature_track's ADJUSTABLE branch already
    uses for beta/temperature (see that function's own docstring for why
    those particular stops), factored out here as a reusable point
    function so the per-cell heatmap below can share the identical
    mapping rather than a second, hand-copied gradient that could drift
    from it. `frac` is a 1-D array in [0, 1] (values outside are
    clamped); returns uint8 RGB, shape (len(frac), 3)."""
    frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
    cold = np.array(_rgb(theme.BLUE_DEEP), dtype=float)
    gold = np.array(_rgb(theme.GOLD), dtype=float)
    hot = np.array(_rgb(theme.ORANGE), dtype=float)
    mid = 0.55  # verbatim the tokens spec's CSS gradient stop
    out = np.empty(frac.shape + (3,), dtype=float)
    left = frac <= mid
    right = ~left
    t_left = frac[left] / mid
    out[left] = cold[None, :] * (1 - t_left[:, None]) + gold[None, :] * t_left[:, None]
    t_right = (frac[right] - mid) / (1 - mid)
    out[right] = gold[None, :] * (1 - t_right[:, None]) + hot[None, :] * t_right[:, None]
    return out.clip(0, 255).astype(np.uint8)


def render_heatmap_image(w: int, h: int, cell_values: np.ndarray) -> tuple[Image.Image, str]:
    """Per-cell heatmap with a thermal ramp and legend, per the tokens
    spec's own buildability note. `cell_values` is per_cell_occupancy's
    own (grid_h, grid_w) array of [0, 1] fractions (or NaN -- no draws
    yet, or a degenerate receipt); NaN cells are drawn GHOST-grey, never
    a fabricated colour, and the legend says so whenever any appear.
    Occupancy is not literally a temperature, but the tokens spec names
    this ramp "thermal" and this app's own cold=rare/hot=frequent framing
    is a real, non-decorative reading of the SAME blue/gold/orange
    meaning used everywhere else in this app (a cell that is almost
    never on reads cold; a cell that is almost always on reads hot)."""
    img = Image.new("RGB", (w, h), _rgb(theme.INSET))
    d = ImageDraw.Draw(img)
    grid_h, grid_w = cell_values.shape
    if grid_h == 0 or grid_w == 0 or not np.any(~np.isnan(cell_values)):
        d.text((10, h // 2 - 6), "(no data yet)", fill=_rgb(DIM))
        return img, "unavailable: no draws recorded yet this session"

    pad = 10
    legend_h = 30
    avail_w = w - 2 * pad
    avail_h = h - 2 * pad - legend_h - 8
    cell_px = max(4, int(min(avail_w / grid_w, avail_h / grid_h)))
    grid_px_w, grid_px_h = grid_w * cell_px, grid_h * cell_px
    ox = pad + max(avail_w - grid_px_w, 0) // 2
    oy = pad

    valid = ~np.isnan(cell_values)
    colors = np.zeros(cell_values.shape + (3,), dtype=np.uint8)
    if valid.any():
        colors[valid] = _thermal_ramp_rgb(cell_values[valid])
    ghost = np.array(_rgb(theme.GHOST), dtype=np.uint8)
    colors[~valid] = ghost
    border = _rgb(theme.BEZEL)
    for y in range(grid_h):
        for x in range(grid_w):
            x0, y0 = ox + x * cell_px, oy + y * cell_px
            d.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1],
                       fill=tuple(int(c) for c in colors[y, x]), outline=border, width=1)

    leg_y0 = h - legend_h
    leg_x0, leg_x1 = pad, w - pad
    leg_w = max(leg_x1 - leg_x0, 1)
    ramp = _thermal_ramp_rgb(np.linspace(0.0, 1.0, leg_w))
    for i in range(leg_w):
        d.line([(leg_x0 + i, leg_y0), (leg_x0 + i, leg_y0 + 10)],
              fill=tuple(int(c) for c in ramp[i]))
    d.rectangle([leg_x0, leg_y0, leg_x1 - 1, leg_y0 + 10], outline=border, width=1)
    d.text((leg_x0, leg_y0 + 12), "0.0 cold (rarely on)", fill=_rgb(DIM))
    d.text((leg_x1, leg_y0 + 12), "1.0 hot (almost always on)", fill=_rgb(DIM), anchor="ra")
    n_nodata = int((~valid).sum())
    if n_nodata:
        d.rectangle([leg_x0, leg_y0 - 12, leg_x0 + 10, leg_y0 - 2], fill=tuple(ghost.tolist()))
        d.text((leg_x0 + 14, leg_y0 - 12), "grey = no data", fill=_rgb(DIM))

    caption = (
        f"Per-cell mean WORLD-spin occupancy across this session's own "
        f"raw draw buffer ({grid_h * grid_w} cells), thermal-ramp "
        f"coloured (the same cold/gold/hot mapping the temperature "
        f"control uses, reused here for a continuous [0, 1] fraction, "
        f"NOT an actual temperature). {n_nodata} cell(s) have no data "
        f"yet, drawn grey -- never given a fabricated colour.")
    return img, caption


def _fit_caption_height(label: tk.Label) -> None:
    """I3/I4 (fix-round-2): size `label`'s `height` (Tk's Label height is a
    LINE COUNT, not pixels) to what its CURRENTLY SET text actually needs
    at its configured wraplength/font -- measured via real Tk layout
    (winfo_reqheight), never a guessed line count. The ACF and sigmoid
    captions were both set with a guessed caption_lines (4 and 11) that
    undercounted the real wrapped text (7 and 12 lines respectively) and
    silently clipped the load-bearing sentence in each -- a caption that
    carries one of this app's honesty disclosures must not be sized by
    guess. Call this after every `.config(text=...)` that can change a
    caption's content, not just once at construction, since the safety
    net must hold for text that changes at runtime, not just today's
    wording.

    Jitter fix (see jitter-fix-report.md): the ORIGINAL implementation
    measured by mutating `label` ITSELF -- config(height=0), then
    update_idletasks() to force Tk to lay out the new text at its natural
    (unclamped) size before reading winfo_reqheight(). That update_idletasks()
    call is a REAL, PAINTED relayout of the live, on-screen label, and the
    following config(height=n_lines) that corrects it to the safe,
    clip-proof height was never itself flushed the same way -- so on
    screen the caption visibly shrank to its natural size and then, on
    whatever later idle pass Tk got around to it, grew back. Measured live
    (winfo_height() polled independently every tick): this is not a one-
    time settle, it repeats on every single re-fit, for ANY caption whose
    natural height doesn't happen to land exactly on a whole-line boundary
    -- which is most of them, sooner or later (a caption embedding a
    growing count, e.g. the energy histogram's "N value(s) so far", drifts
    across that boundary constantly while N is still climbing). That
    on-screen shrink/regrow is what the bug report saw as boxes
    'vibrating'.
    Fix: measure on a throwaway PROBE label -- same master, font, wraplength
    and text, but never packed/gridded, so it has no on-screen presence and
    mutating IT never relayouts or repaints anything real. The live `label`
    then gets exactly ONE height mutation, straight from its old correct
    value to its new correct one; nothing observable is ever set to the
    too-short intermediate size. Still real Tk layout (same font object,
    same winfo_reqheight call), never a guessed line count -- only the
    widget being measured changed, not the measurement itself."""
    probe = tk.Label(label.master, text=label.cget("text"),
                      font=label.cget("font"), wraplength=label.cget("wraplength"))
    probe.update_idletasks()  # flush the PROBE's own layout only; it is never mapped, so nothing paints
    req_h = probe.winfo_reqheight()
    probe.destroy()
    line_h = tkfont.Font(font=label.cget("font")).metrics("linespace")
    n_lines = max(1, -(-req_h // max(line_h, 1)))  # ceil division
    label.config(height=n_lines)  # the ONLY height mutation the real, visible label ever sees


class Panel(tk.Frame):
    def __init__(self, master, title: str, **kw):
        super().__init__(master, bg=PANEL_BG, highlightbackground=BORDER,
                          highlightthickness=1, **kw)
        tk.Label(self, text=title, bg=PANEL_BG, fg=ACCENT,
                  font=(MONO_FAMILY, 10, "bold"), anchor="w"
                  ).pack(fill="x", padx=8, pady=(6, 2))
        self.body = tk.Frame(self, bg=PANEL_BG)
        self.body.pack(fill="both", expand=True, padx=8, pady=(0, 8))


class LatticeApp(tk.Tk):
    def __init__(self, receipt: Receipt, overlay_receipt: Receipt):
        super().__init__()
        # Task 0: MONO_FAMILY was resolved once at import time (module
        # scope, before ANY Tk root existed -- lattice_app.py must stay
        # safe to import headlessly, see the "no Tk in pure logic" test
        # convention), so it could only ever see theme.py's own
        # no-root fallback ("Consolas"). Re-resolve now that `self` IS a
        # live Tk root (LatticeApp subclasses tk.Tk) -- this is the first
        # point "Cascadia Mono" can actually be detected as installed.
        # Every widget built below (in _build_layout /
        # _populate_static_panels, both called later in this __init__)
        # reads the MONO_FAMILY global at call time, so rebinding it here,
        # before either runs, is sufficient -- no widget needs rebuilding.
        global MONO_FAMILY, MONO, MONO_B
        MONO_FAMILY = theme.resolve_mono_family()
        MONO = (MONO_FAMILY, 9)
        MONO_B = (MONO_FAMILY, 9, "bold")
        self.receipt = receipt
        self.overlay_receipt = overlay_receipt   # Task 7: demo/receipts/elev_band, compiled once at load, same as `receipt`
        self.title("tsu lattice demo -- live sampling of a compiled receipt")
        # Task 6: extra height for the new SCOPE panel row added below the
        # existing content row -- every other panel's own size/position is
        # unchanged, this only makes room for the addition. Grew again in
        # fix-round 1 (1160 -> 1240) when the sigmoid cell's caption grew
        # from 4 to 11 lines to carry the reconstruction-vs-defect
        # disclosure on screen. Task 7: grew again (1240 -> 1380) for the
        # new LAYERS row. Task 8: grew again (1380 -> 1420) -- the
        # Canvas-drawn temperature control (state label + track + axis
        # labels + SEAL header + reason) needs more vertical room than the
        # single-line tk.Scale it replaced (measured: 208px natural content
        # in a row that previously budgeted 190px, so the reason/seal text
        # was silently clipped off the bottom of the window -- caught only
        # by launching the real app and looking, not by any unit test; see
        # task-0-8-report.md). Kept the temp control itself tight (13pt not
        # 16pt readout, a 20px not 28px track) and grew the window by only
        # 40px rather than more, because 1440px is this screen's own
        # height and a taller request gets silently clamped by the OS --
        # verified 1420 is NOT clamped here, leaving ~20px margin.
        self.geometry("1760x1420")
        self.configure(bg=BG)

        self.in_q: "queue.Queue[dict]" = queue.Queue(maxsize=64)
        self.paused = False
        self.speed_idx = DEFAULT_SPEED_IDX  # UI2: survives worker restarts
        # (pin change / new seed) -- threaded through _start_worker below.
        self.total_draws = 0
        self.valid_count = 0
        self.contract_fail_count = 0
        self.noncodeword_count = 0
        self.last_valid_grid = None
        self.last_valid_mix = None
        self.last_valid_seed = None  # seed of the draw last_valid_grid came from
        self.last_valid_sampler_params = None  # UI2: what actually drew it
        self.world_photo = None  # keep a reference; Tk drops PhotoImages with none
        self.log_lines: deque = deque(maxlen=400)
        self.explainer_window = None  # Task 9: singleton "What is this?" Toplevel
        self.detach_registry = DetachRegistry()  # Task 10: detached plot windows
        self.spins_per_cell = None  # set in _populate_static_panels, before any detach is possible

        # click-to-pin state -- pure ClampState, no Tk in it (see the
        # headless tests). world_stale is True whenever the world currently
        # on screen was NOT drawn under the currently active clamp (e.g.
        # right after a pin change, before a new valid draw has arrived).
        self.clamp = ClampState()
        self.world_stale = False
        self.batch_infeasible = False
        self.batch_reason = ""

        # B2: live traces -- energy over draws (every draw, valid or not:
        # mixing is a property of the raw chain, not the conditional-valid
        # subset) and valid fraction over the session (recomputed at the
        # SAME cadence, from self.valid_count/self.total_draws already
        # tracked above). Bounded ring buffers, see Trace's own docstring.
        # I-2/F-R6: "over draws", not "over sweeps" -- consecutive draws
        # are routinely from different chains or different restarts
        # entirely (chain-major flattening / restart-interleaving, see
        # Trace's own docstring), never one chain's own physical sweep.
        self.energy_trace = Trace(maxlen=SCOPE_ENERGY_TRACE_MAXLEN)
        self.valid_frac_trace = Trace(maxlen=SCOPE_ENERGY_TRACE_MAXLEN)

        # Task 6: SCOPE panel state. magnetization_trace mirrors energy_
        # trace's own convention (every draw, valid or not -- the order
        # parameter is a property of the raw chain too). raw_draws is a
        # SEPARATE, longer ring buffer of full spin rows (not just a
        # scalar per draw): local_field_response needs the actual pooled
        # (gamma, spin) pairs a real draw produced, not a summary of one.
        self.magnetization_trace = Trace(maxlen=SCOPE_ENERGY_TRACE_MAXLEN)
        self.raw_draws: deque = deque(maxlen=SCOPE_RAW_DRAWS_MAXLEN)
        self._scope_redraw_counter = 0
        # keep references so Tk doesn't garbage-collect the blitted images
        self.acf_photo = self.mag_photo = self.hist_photo = self.sigmoid_photo = None
        self.heat_photo = None  # Task 10: per-cell heatmap's own inline SCOPE cell

        # Task 7: LAYERS panel state. `active_layer` selects what DECODED
        # WORLD/PINS show and act on -- "base" reuses every attribute
        # already defined above UNCHANGED (this task adds a NEW mode, it
        # does not touch the base-layer path). Each band gets its own
        # LayerState (own ClampState, own conditioning/beta, own last-valid
        # decode) -- see LayerState's own docstring for why composite has
        # no entry here (it is derived, never stored).
        self.active_layer = "base"
        self.alpha = 1.0   # demo/elevation.band_patch's own `strength` -- see DOSE_RESPONSE_NOTE
        # Task 8: base's own beta-override slot. Base has no LayerState (see
        # LayerState's own docstring), so this is where a base beta
        # override would live IF base ever compiled unmediated and became
        # adjustable -- see get_beta_override/set_beta_override, which read
        # and write this exact attribute for layer_name=="base" instead of
        # indexing self.bands["base"] (which would KeyError -- the fix a
        # prior review flagged).
        self.base_beta_override: float | None = None
        self.bands: dict[str, LayerState] = {
            name: LayerState(name, ClampState(cycle=(0, 1))) for name in BAND_NAMES}
        self.layer_photo = None  # DECODED WORLD's blitted image when a band/composite is active
        # I2 (fix-round-2): the base grid actually captured at the most
        # recent band regenerate -- what the composite's terrain must be
        # rendered from, NOT self.last_valid_grid (which keeps streaming
        # after that regenerate). Set in _on_band_regenerated. See
        # _refresh_layer_view's composite branch.
        self.composite_base_grid: np.ndarray | None = None

        self._build_layout()
        self._populate_static_panels()

        self._start_worker(random.randint(0, 2**31 - 1))

        # Task 7: LAYERS panel's own initial draw -- alpha label, temperature
        # control (base starts selected, so this shows FIXED/disabled), pins
        # list, composite readout (starts "unavailable", honestly, since no
        # band has ever been regenerated this session yet).
        self._on_alpha_change(self.alpha)
        self._refresh_temperature_control()
        self._refresh_pins_panel()
        self._refresh_composite_readout()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_queue)

    # -- layout ------------------------------------------------------
    def _make_scroll_column(self, parent, width=None):
        """A vertically-scrollable column: Canvas + Scrollbar + inner Frame.
        Same pattern demo/explainer.py's WhatIsThisWindow already uses for
        its own scrollable body -- reused here, not reinvented. Mouse-wheel
        is bound to THIS canvas only while the pointer is over it
        (<Enter>/<Leave>-scoped bind_all/unbind_all), the exact fix
        explainer.py's own comment documents: a bare `canvas.bind_all(...)`
        stays application-scoped and outlives the widget, so it must be
        scoped to "pointer is actually over this canvas", never left global.

        Used for the PIPELINE/FRONTIER stack and the VERIFICATION/SAMPLER/
        DECODED MIX/SAMPLE LOG stack: measured (layout-fix-report.md), each
        stack's combined natural height vastly exceeds the fixed row-0
        height available (e.g. PIPELINE 373px + FRONTIER 907px = 1280px
        needed vs 738px available) -- a previous round tried fixing this
        with row-weight ratios and found every ratio that helped one panel
        collapsed the other, because the deficit is a real space shortage,
        not a distribution problem. Every panel below now renders at its
        own full natural height (no more zero-sum weight competition); the
        user scrolls to reach whatever doesn't fit in the visible window.

        Resize-guard (2026-09): ALSO used, with width=None, to wrap the
        whole `content` grid (row 0 + the SCOPE/LAYERS rows) -- see
        _build_layout's own comment at that call site for why: below
        ~1050-1100px of window height, content's three fixed-height rows
        (SCOPE minsize=340 + LAYERS minsize=230, both unconditional) left
        row 0 nothing, and grid silently unmapped it entirely, taking
        PIPELINE/SAMPLER/VERIFICATION/SAMPLE LOG down with it -- the
        fourth instance of this exact failure class on this project. The
        same "stop competing for a fixed budget, let real content take
        its own natural size, and let the user scroll to reach the rest"
        fix already applied to the PIPELINE/FRONTIER and VERIFICATION/.../
        SAMPLE LOG stacks above now applies one level up, to `content`
        itself.

        `width` is the outer footprint, unchanged from the fixed-width grid
        column this replaces, so every OTHER column's sizing (LIVE LATTICE,
        DECODED WORLD, REGIME & TRACES) is untouched by this change. When
        `width` is None (the page-level wrap only), outer instead fills
        whatever horizontal space its own parent gives it -- pack_propagate
        is left at its default (True) in that case, since there is no fixed
        width to freeze; outer is always packed fill="both", expand=True by
        its caller, so its own requested width/height don't matter.

        Width-axis fix (2026-09): the page-level wrap (width=None) also gets
        a horizontal Scrollbar, and `_on_canvas_configure` below never asks
        `inner` to be NARROWER than its own natural (reqwidth) size -- only
        wider, when the window offers more. Before this fix, every resize
        pinned `inner` (and therefore `content`, and therefore row 0's LIVE
        LATTICE / DECODED WORLD weight=1 grid columns) to exactly the
        canvas's own viewport width, so a window narrower than row 0's
        combined natural width squeezed those two weight=1 columns toward
        0px (measured: 8px at 1200x900) while row 0's fixed-width columns
        (PIPELINE/FRONTIER 340 + REGIME & TRACES 300 + the VERIFICATION/
        SAMPLER/DECODED MIX/SAMPLE LOG stack 430 = 1070px) held their size
        unconditionally, absorbing none of the deficit -- the fifth
        instance of "exists in source, occupies zero screen pixels" on this
        project, this time on the WIDTH axis instead of height. Same
        remedy as the height-axis fix one level up: never shrink real
        content below its natural size; let the user scroll (now
        horizontally too) to reach whatever the window is too narrow to
        show all at once.
        Returns (outer_frame, inner_frame) -- caller grids (or packs)
        outer_frame into its parent and packs panels top-to-bottom into
        inner_frame.
        """
        if width is not None:
            outer = tk.Frame(parent, bg=BG, width=width)
            # pack_propagate (NOT grid_propagate) is the one that matters
            # here: canvas+scrollbar below are PACKED into outer, and
            # pack_propagate is what freezes a frame's own reqsize against
            # its PACK-managed children (grid_propagate only governs
            # GRID-managed children, of which outer has none -- measured
            # the difference directly: with grid_propagate(False) alone,
            # outer's reqwidth silently drifted to the canvas's own unset
            # default (~395px) for BOTH the 340px and 430px columns,
            # identically, discarding the `width` argument entirely).
            outer.pack_propagate(False)
        else:
            outer = tk.Frame(parent, bg=BG)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        if width is None:
            # Width-axis fix (2026-09): only the page-level wrap needs a
            # horizontal scrollbar -- every other caller has a fixed pixel
            # `width` (pack_propagate(False) above), so its content already
            # wraps/fits at that width and never needs horizontal reach.
            hsb = tk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
            canvas.configure(xscrollcommand=hsb.set)
            hsb.pack(side="bottom", fill="x")
        else:
            hsb = None
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_evt=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(evt):
            if hsb is not None:
                # Width-axis fix (2026-09): floor at inner's own natural
                # width -- never shrink it to the (possibly narrower)
                # canvas viewport, only grow it to fill a wider one. See
                # this method's own docstring for the full defect.
                target_w = max(evt.width, inner.winfo_reqwidth())
            else:
                target_w = evt.width
            canvas.itemconfigure(inner_id, width=target_w)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(evt):
            canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        return outer, inner

    def _build_layout(self):
        # Controlbar-fix (2026-09): `bottom` (Pause/New seed/Step/Save world
        # + the footer disclosure, built further down this method) is
        # created and PACKED here, before `content`, even though its own
        # children aren't populated until later in this method. Tk's packer
        # hands out space in PACKING ORDER, not code order and not visual
        # order -- same principle already applied to the detached-plot
        # caption in _open_detach_window (packed side="bottom" before its
        # canvas). `content` packs fill="both", expand=True and will claim
        # every leftover pixel; if `bottom` were packed AFTER content (as it
        # used to be), a shrink below the window's natural size gives
        # content first claim on the shortfall and squeezes bottom -- taking
        # the four control buttons and the footer disclosure down toward
        # zero along with it. Packing bottom first, side="bottom", reserves
        # its full requested height up front; content, packed after, only
        # ever expands into whatever remains. (Not pack_propagate(False):
        # that approach froze a disclosure at 1x1px in an earlier round on
        # this branch -- see the LAYERS fix-report -- because it fixes a
        # size BEFORE content exists rather than letting pack order reserve
        # the real, current requested size.)
        bottom = tk.Frame(self, bg=BG)
        bottom.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        # Resize-guard (2026-09): `content` (built below, all three grid
        # rows -- row 0 plus the SCOPE/LAYERS rows) is wrapped in a
        # page-level scroll column, same _make_scroll_column already used
        # for the PIPELINE/FRONTIER and VERIFICATION/.../SAMPLE LOG stacks
        # inside row 0. Root cause this fixes: SCOPE (minsize=340) and
        # LAYERS (minsize=230) reserve 570px UNCONDITIONALLY regardless of
        # window size, and row 0 (the only row with no minsize of its own)
        # absorbed the entire deficit once the window got small enough --
        # measured, below ~1050-1100px window height, that deficit reached
        # 570px+ and grid unmapped row 0 ENTIRELY, taking PIPELINE, LIVE
        # LATTICE, DECODED WORLD, REGIME & TRACES, VERIFICATION, SAMPLER
        # and SAMPLE LOG down with it -- the fourth instance of "content
        # that exists in source but occupies zero pixels" on this branch.
        # A minsize on row 0 alone does not fix this correctly: LIVE
        # LATTICE and DECODED WORLD (row 0, columns 1/2) are plain Panels
        # with real fixed-size canvases (measured natural body height
        # 555px / 486px) and no scroll fallback of their own -- capping
        # row 0's minsize below that would silently CLIP those canvases
        # instead of merely shrinking them, trading one invisible-content
        # bug for another. Wrapping `content` itself removes the fixed
        # window-height budget completely: every row (0, SCOPE, LAYERS)
        # now renders at its own real natural height -- for row 0 that
        # naturally comes out to LIVE LATTICE's own ~596px requirement,
        # comfortably non-zero at every window size -- and the page
        # scrolls (mousewheel while hovered, same <Enter>/<Leave>-scoped
        # binding as every other scroll column here, never a bare
        # `bind_all`) to reach whatever doesn't fit the current window.
        # Same mechanism already documented on _make_scroll_column itself
        # for the two inner stacks, now applied one level up.
        page_outer, page_inner = self._make_scroll_column(self, width=None)
        page_outer.pack(fill="both", expand=True, padx=8, pady=8)

        content = tk.Frame(page_inner, bg=BG)
        content.pack(fill="both", expand=True)
        # 0: PIPELINE+FRONTIER stack, 1: LIVE LATTICE, 2: DECODED WORLD,
        # 3: REGIME & TRACES (B2, fixed width), 4: the VERIFICATION/.../
        # SAMPLE LOG stack (also fixed width).
        for c, weight in enumerate((0, 1, 1, 0, 0)):
            content.grid_columnconfigure(c, weight=weight)
        content.grid_rowconfigure(0, weight=1)
        # Task 6: SCOPE panel -- a new row below the existing content row,
        # spanning every column, fixed height (grid_propagate off) so it
        # never eats into the row-0 panels' own space.
        # Task 6 fix-round 1: minsize grew from 260 -- the sigmoid cell's
        # caption is now 11 lines (was 4) to fit the reconstruction-vs-
        # defect disclosure required on screen, and this row sizes to its
        # tallest cell.
        content.grid_rowconfigure(1, weight=0, minsize=340)

        # Layout-fix (2026-09): PIPELINE + FRONTIER's combined natural
        # height (373 + 907 = 1280px, measured) vastly exceeds the 738px
        # row-0 has to offer -- see _make_scroll_column's own docstring.
        # Scrollable column instead of a fixed 3:2 grid split (which,
        # measured, gave PIPELINE's body only 2px and dropped its nine
        # stage rows entirely -- see layout-fix-report.md).
        left, left_inner = self._make_scroll_column(content, width=340)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.pipeline_panel = Panel(left_inner, "PIPELINE  (compiled once, at load)")
        self.pipeline_panel.pack(fill="x", pady=(0, 4))
        # B1: the capacity frontier -- headroom + next-increment cost, read
        # from demo/frontier.py's own build_frontier_report/render_text so
        # this panel can never drift from what `python demo/frontier.py`
        # prints on the command line.
        self.frontier_panel = Panel(left_inner, "FRONTIER  (headroom + next increment)")
        self.frontier_panel.pack(fill="x", pady=(4, 0))

        self.lattice_panel = Panel(content, "LIVE LATTICE  (raw physical spin state)")
        self.lattice_panel.grid(row=0, column=1, sticky="nsew", padx=6)

        self.world_panel = Panel(content, "DECODED WORLD  (last valid sample)")
        self.world_panel.grid(row=0, column=2, sticky="nsew", padx=6)

        # B2: beta/beta_c siting (Onsager, labelled as an orienting estimate
        # only) plus the energy and valid-fraction traces.
        # Layout-fix (2026-09): same scrollable treatment as left/right --
        # measured, REGIME & TRACES' own natural height (683px) modestly
        # exceeds row-0's available height once LAYERS' real height (fixed
        # above, was silently masked at ~230px, actually needs ~310px) and
        # the two-row footer (fixed above) are honestly accounted for
        # (56px shortfall measured -- see layout-fix-report.md). Same
        # mechanism, not a special case.
        regime_frame, regime_inner = self._make_scroll_column(content, width=300)
        regime_frame.grid(row=0, column=3, sticky="nsew", padx=6)
        self.regime_panel = Panel(regime_inner, "REGIME & TRACES")
        self.regime_panel.pack(fill="both", expand=True)

        # Layout-fix (2026-09): same over-subscription as PIPELINE/FRONTIER
        # -- VERIFICATION + SAMPLER + DECODED MIX + SAMPLE LOG's combined
        # natural height (407+455+143+314 = 1319px, measured) vastly
        # exceeds the 738px row-0 has to offer. The previous weighted grid
        # (5:5:2:5) hid this: VERIFICATION's diversity_* rows, most of
        # SAMPLER's UI2 disclosures, and SAMPLE LOG's own running-log
        # Listbox all collapsed to 1x1 (invisible), not merely clipped --
        # see layout-fix-report.md. Scrollable column instead, same
        # mechanism as left/PIPELINE above.
        right, right_inner = self._make_scroll_column(content, width=430)
        right.grid(row=0, column=4, sticky="nsew", padx=(6, 0))

        self.verif_panel = Panel(right_inner, "VERIFICATION")
        self.verif_panel.pack(fill="x", pady=(0, 4))
        self.sampler_panel = Panel(right_inner, "SAMPLER")
        self.sampler_panel.pack(fill="x", pady=4)
        self.mix_panel = Panel(right_inner, "DECODED MIX")
        self.mix_panel.pack(fill="x", pady=4)
        self.log_panel = Panel(right_inner, "SAMPLE LOG")
        self.log_panel.pack(fill="x", pady=(4, 0))

        # Task 6: SCOPE panel -- autocorrelation (semi-log, tau + the 5000
        # reliability threshold), magnetization trace, energy histogram,
        # and the measured-vs-analytic sigmoid response, four sub-plots in
        # a row. Every plot is a PIL image blitted onto its own Canvas (see
        # render_acf_plot etc. above) -- Tk canvas primitives alone can't
        # do a log axis, filled bars, or an overlaid scatter+curve well.
        # Task 10: a fifth sub-plot (per-cell heatmap) plus two Detach
        # buttons (node-edge lattice graph, relaxation strip) with no
        # inline canvas of their own -- see this method's own comments at
        # each cell below, and DetachRegistry's own module-level docstring
        # for why: "Plots that need their own render area pop out into
        # small Toplevel windows" (task-10-brief.md).
        self.scope_panel = Panel(content, "SCOPE  (autocorrelation / magnetization / "
                                          "energy histogram / local-field response / "
                                          "per-cell heatmap -- ⧉ detaches a plot)")
        self.scope_panel.grid(row=1, column=0, columnspan=5, sticky="nsew", pady=(6, 0))
        scope_row = tk.Frame(self.scope_panel.body, bg=PANEL_BG)
        scope_row.pack(fill="both", expand=True)

        def _scope_cell(title, caption_lines=4, detach_key=None):
            cell = tk.Frame(scope_row, bg=PANEL_BG)
            cell.pack(side="left", fill="both", expand=True, padx=4)
            header = tk.Frame(cell, bg=PANEL_BG)
            header.pack(fill="x")
            tk.Label(header, text=title, bg=PANEL_BG, fg=ACCENT,
                      font=(MONO_FAMILY, 8, "bold"), anchor="w", justify="left"
                      ).pack(side="left", fill="x", expand=True)
            # Task 10: this cell also has a detached, larger-size host --
            # render_energy_histogram_plot/render_sigmoid_plot/
            # render_heatmap_image are called from BOTH this inline canvas
            # (below) AND the detached Toplevel this button opens (see
            # LatticeApp._render_detached_plot) -- one renderer, two hosts,
            # never a second implementation that could drift.
            if detach_key is not None:
                tk.Button(header, text="⧉", command=lambda k=detach_key: self._open_detach(k),
                          bg=PANEL_BG, fg=FG, activebackground=BORDER,
                          activeforeground=FG, relief="flat", padx=5, pady=0,
                          bd=1, font=(MONO_FAMILY, 8)).pack(side="right")
            canvas = tk.Canvas(cell, width=SCOPE_PLOT_W, height=SCOPE_PLOT_H,
                                bg=theme.INSET, highlightthickness=0)
            canvas.pack(pady=(2, 2))
            caption = tk.Label(cell, text="", bg=PANEL_BG, fg=DIM,
                                font=(MONO_FAMILY, 7), anchor="w", justify="left",
                                wraplength=SCOPE_PLOT_W, height=caption_lines)
            caption.pack(fill="x")
            return canvas, caption

        self.acf_canvas, self.acf_caption = _scope_cell(
            "AUTOCORRELATION (semi-log, tau marked)")
        self.mag_canvas, self.mag_caption = _scope_cell(
            "MAGNETIZATION (order parameter, per draw)")
        self.hist_canvas, self.hist_caption = _scope_cell(
            "ENERGY HISTOGRAM (over the session)", detach_key="hist")
        # Task 6 fix-round 1: retitled from "measured P(s=1) vs analytic
        # sigmoid" -- that phrasing claimed the plot showed the sampler's
        # own conditional, which it does not (see render_sigmoid_plot's
        # docstring and demo/scope.py's local_field_response docstring).
        # This cell's caption is long BY REQUIREMENT (the fix-round-1
        # ruling: the reconstruction-vs-defect distinction must be stated
        # ON SCREEN, not only in a docstring) -- caption_lines=11 gives it
        # room without touching any other cell's sizing.
        self.sigmoid_canvas, self.sigmoid_caption = _scope_cell(
            "LOCAL FIELD: EMPIRICAL P(s=1) vs SIGMOID (REFERENCE ONLY)",
            caption_lines=11, detach_key="sigmoid")
        # Task 10: per-cell occupancy heatmap -- new sub-plot, same inline
        # host pattern as hist/sigmoid above.
        self.heat_canvas, self.heat_caption = _scope_cell(
            "PER-CELL HEATMAP (occupancy, thermal ramp)", caption_lines=5,
            detach_key="heatmap")
        for canvas in (self.acf_canvas, self.mag_canvas, self.hist_canvas,
                      self.sigmoid_canvas, self.heat_canvas):
            canvas.create_image(0, 0, anchor="nw", tags="plot")

        # Task 10: the node-edge lattice graph and the relaxation strip
        # have NO inline host at all -- there is no compact spot in this
        # row that does either justice (the graph needs real vertical
        # room for grid + mediator strip + legend; the strip needs real
        # horizontal room for several frames side by side). Both share ONE
        # narrow button-only cell (stacked, not two separate cells) --
        # opening its own Toplevel is the ONLY place either plot is ever
        # rendered, which trivially satisfies "one renderer" (there is no
        # second implementation to drift from). Measured narrow (not just
        # guessed): the row's 5 existing canvas cells request ~1712px
        # against ~1726px actually available -- no deficit, this row has
        # always fit; this cell is kept to the width measured, by
        # screenshot, to land inside what's left over rather than
        # introducing one.
        detach_only_cell = tk.Frame(scope_row, bg=PANEL_BG, width=92)
        detach_only_cell.pack(side="left", fill="y", padx=(4, 0))
        detach_only_cell.pack_propagate(False)

        def _detach_only_row(title, key, note):
            tk.Label(detach_only_cell, text=title, bg=PANEL_BG, fg=ACCENT,
                      font=(MONO_FAMILY, 8, "bold"), anchor="w", justify="left",
                      wraplength=86).pack(fill="x", pady=(2, 0))
            tk.Button(detach_only_cell, text="Open ⧉",
                      command=lambda k=key: self._open_detach(k),
                      bg=PANEL_BG, fg=FG, activebackground=BORDER,
                      activeforeground=FG, relief="flat", padx=4, pady=3,
                      bd=1, font=(MONO_FAMILY, 8)).pack(anchor="w", pady=(4, 4))
            tk.Label(detach_only_cell, text=note, bg=PANEL_BG, fg=DIM,
                      font=(MONO_FAMILY, 7), anchor="w", justify="left",
                      wraplength=86).pack(fill="x", pady=(0, 8))

        _detach_only_row(
            "NODE-EDGE LATTICE (role-coloured)", "lattice_graph",
            "Detached view only. Real topology: world gold, mediator "
            "cold blue.")
        _detach_only_row(
            "RELAXATION STRIP (raw state, sweeps)", "relaxation",
            "Detached view only. Raw physical draws, one PNG filmstrip.")

        # Task 7: LAYERS panel -- a new row below SCOPE, same "new full-
        # width row" pattern Task 6 used for SCOPE itself. Selector / pins-
        # apply-to note / conditioning strength + dose-response reference /
        # per-layer regenerate / temperature (overlay-only) / composite
        # validation readout -- see the brief's own 6 steps, one widget
        # group per step, left to right.
        content.grid_rowconfigure(2, weight=0, minsize=230)  # Task 8: 190 -> 230, see self.geometry's own comment above
        self.layers_panel = Panel(content, "LAYERS  (base / band0.."
                                          f"band{N_BANDS - 1} / composite)")
        self.layers_panel.grid(row=2, column=0, columnspan=5, sticky="nsew", pady=(6, 0))
        layers_row = tk.Frame(self.layers_panel.body, bg=PANEL_BG)
        layers_row.pack(fill="both", expand=True)

        def _layers_cell(title, width=None):
            # Layout-fix (2026-09): pack_propagate(False) used to be called
            # here immediately after creation -- BEFORE any content (radio
            # buttons, disclosure labels) was packed into the cell, INCLUDING
            # everything the caller packs in after this function returns.
            # That froze the cell's HEIGHT at whatever tiny default size a
            # brand-new empty Frame has, forever -- not at its real content
            # height. Measured, real-app consequence: the overlay-pin "can
            # be outvoted" disclosure in LAYER SELECTOR (sel_cell) rendered
            # at literal (1,1)px, completely invisible, regardless of
            # window size -- a content bug hiding as a space bug (see
            # layout-fix-report.md). `width` is kept as an initial sizing
            # hint (every cell's own labels already wrap close to it via
            # their own wraplength), but is no longer frozen -- each cell's
            # true content now determines its own real height.
            cell = tk.Frame(layers_row, bg=PANEL_BG, width=width)
            cell.pack(side="left", fill="both", expand=(width is None), padx=6)
            tk.Label(cell, text=title, bg=PANEL_BG, fg=ACCENT,
                      font=(MONO_FAMILY, 8, "bold"), anchor="w").pack(fill="x")
            return cell

        # Step 1: layer selector.
        sel_cell = _layers_cell("LAYER SELECTOR", width=190)
        self.layer_var = tk.StringVar(value=self.active_layer)
        for name in LAYER_NAMES:
            tk.Radiobutton(sel_cell, text=name, value=name, variable=self.layer_var,
                            command=self._on_layer_selected, bg=PANEL_BG, fg=FG,
                            selectcolor=BORDER, activebackground=PANEL_BG,
                            activeforeground=FG, font=MONO, anchor="w",
                            highlightthickness=0).pack(fill="x")
        tk.Label(sel_cell, text="Pins apply to the SELECTED layer only "
                                 "(see PINS list, tagged by layer).",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 7), anchor="w",
                  justify="left", wraplength=180).pack(fill="x", pady=(4, 0))
        # C1 (fix-round-2): the code comment above OVERLAY_PIN_STRENGTH
        # claims "the UI states this" (that an overlay pin is a strong
        # nudge, not a hard constraint) -- it did not, until this label.
        # Made true here rather than left as an aspirational comment.
        # Resize-guard (2026-09): named attribute handle so a regression
        # test can assert this disclosure stays visible (winfo_ismapped +
        # non-trivial width/height) across window sizes -- this exact label
        # is the one that rendered at literal (1,1)px, invisible, when
        # _layers_cell used to call pack_propagate(False) too early (see
        # that function's own comment above). Never matched by text/index.
        self.overlay_pin_disclosure_label = tk.Label(
            sel_cell, text="Base pins are EXACT clamps. Overlay pins "
                            "(band0/1/2) are a STRONG BIAS NUDGE, not "
                            "a guarantee -- CONDITIONING STRENGTH can "
                            "outvote them and the pinned cell can "
                            "still render the other value.",
            bg=PANEL_BG, fg=WARN, font=(MONO_FAMILY, 7), anchor="w",
            justify="left", wraplength=180)
        self.overlay_pin_disclosure_label.pack(fill="x", pady=(4, 0))

        # Step 3: conditioning strength + dose-response reference table.
        alpha_cell = _layers_cell("CONDITIONING STRENGTH (overlays)", width=330)
        # Minor #4 (fix-round-2): no wraplength in a 330px cell -- the text
        # (incl. the "-- see dose-response note below" pointer) measures
        # ~672px and was getting cut off. Matches the other two labels in
        # this same cell, which already wrap at 320.
        self.alpha_label = tk.Label(alpha_cell, text="", bg=PANEL_BG, fg=FG,
                                     font=MONO, anchor="w", justify="left",
                                     wraplength=320)
        self.alpha_label.pack(fill="x", pady=(2, 0))
        self.alpha_scale = tk.Scale(
            alpha_cell, from_=0.0, to=4.0, resolution=0.05, orient="horizontal",
            showvalue=0, length=300, bg=PANEL_BG, fg=FG, troughcolor=BG,
            highlightthickness=0, bd=0, command=self._on_alpha_change)
        self.alpha_scale.set(self.alpha)
        self.alpha_scale.pack(fill="x")
        table_txt = "  ".join(f"a={a:.2f}->{r:.1f}%" for a, r in DOSE_RESPONSE_TABLE)
        tk.Label(alpha_cell, text=f"dose-response reference: {table_txt}",
                  bg=PANEL_BG, fg=WARN, font=(MONO_FAMILY, 7), anchor="w",
                  justify="left", wraplength=320).pack(fill="x", pady=(2, 0))
        tk.Label(alpha_cell, text=DOSE_RESPONSE_NOTE, bg=PANEL_BG, fg=DIM,
                  font=(MONO_FAMILY, 7), anchor="w", justify="left",
                  wraplength=320).pack(fill="x", pady=(2, 0))

        # Step 4: per-layer regenerate.
        regen_cell = _layers_cell("PER-LAYER REGENERATE", width=190)
        self.regenerate_btn = tk.Button(
            regen_cell, text="Regenerate this layer", command=self._on_regenerate_layer,
            bg=PANEL_BG, fg=FG, activebackground=BORDER, activeforeground=FG,
            relief="flat", padx=8, pady=4, wraplength=170)
        self.regenerate_btn.pack(fill="x", pady=(4, 0))
        self.regenerate_status = tk.Label(regen_cell, text="", bg=PANEL_BG, fg=DIM,
                                           font=(MONO_FAMILY, 7), anchor="w",
                                           justify="left", wraplength=180)
        self.regenerate_status.pack(fill="x", pady=(4, 0))

        # Step 5: temperature control -- Task 8's two explicit states,
        # ADJUSTABLE (overlay layers, a live slider) or FIXED (base today,
        # any layer carrying mediator spins in general -- see
        # temperature_control_state), drawn entirely on a Canvas (see
        # render_temperature_track's own docstring for why: a native Scale
        # cannot be gold, and in the locked state it always looks disabled
        # -- the opposite of "a point of pride, not an apology"). `temp_cell`
        # itself is the state's outer frame -- its 1px border colour is
        # reconfigured per state in _refresh_temperature_control (gold_dim
        # live / blue locked), matching the mockup's own temp-box treatment.
        temp_cell = _layers_cell("TEMPERATURE", width=260)
        self.temp_cell = temp_cell
        temp_cell.config(highlightthickness=1, highlightbackground=theme.GOLD_DIM,
                         highlightcolor=theme.GOLD_DIM)
        # I1 (visual pass): the first cut of this layout measured 236px
        # natural content height for a 190px-tall row -- silently clipping
        # the SEAL/reason text off the bottom of the actual window (caught
        # by launching the real app and looking, not by a unit test: see
        # task-0-8-report.md). Tightened below (13pt not 16pt readout, a
        # 20px not 28px track, tighter pady throughout) instead of growing
        # the window further, since the window height is already
        # screen-height-limited on a 1440px-tall display.
        self.temp_label = tk.Label(temp_cell, text="", bg=PANEL_BG, fg=FG,
                                    font=(MONO_FAMILY, 13, "bold"), anchor="w")
        self.temp_label.pack(fill="x", pady=(2, 0), padx=4)
        self.temp_state_label = tk.Label(temp_cell, text="", bg=PANEL_BG, fg=FG,
                                          font=(MONO_FAMILY, 8), anchor="w")
        self.temp_state_label.pack(fill="x", padx=4)
        self.temp_canvas = tk.Canvas(temp_cell, width=TEMP_TRACK_W, height=TEMP_TRACK_H,
                                     bg=PANEL_BG, highlightthickness=0)
        self.temp_canvas.pack(pady=(4, 1), padx=4)
        self.temp_canvas.create_image(0, 0, anchor="nw", tags="track")
        self.temp_photo = None
        self.temp_canvas.bind("<Button-1>", self._on_temp_canvas_interact)
        self.temp_canvas.bind("<B1-Motion>", self._on_temp_canvas_interact)
        temp_labels_row = tk.Frame(temp_cell, bg=PANEL_BG)
        temp_labels_row.pack(fill="x", padx=4)
        tk.Label(temp_labels_row, text=f"cold {TEMP_T_MIN:.1f}", bg=PANEL_BG, fg=DIM,
                  font=(MONO_FAMILY, 7)).pack(side="left")
        tk.Label(temp_labels_row, text="", bg=PANEL_BG, fg=DIM,
                  font=(MONO_FAMILY, 7)).pack(side="left", expand=True)
        tk.Label(temp_labels_row, text=f"hot {TEMP_T_MAX:.1f}", bg=PANEL_BG, fg=DIM,
                  font=(MONO_FAMILY, 7)).pack(side="right")
        # I1 (visual pass, round 2): a separate "SEAL / SPEC 5.3.5" header
        # Label cost a whole extra line's worth of height for a fact the
        # reason text below can carry as its own opening words just as
        # legibly -- folded together so the disclosure text itself has
        # room to render in full instead of being clipped by the row's
        # fixed height. See _refresh_temperature_control for the merged
        # text this produces.
        self.temp_reason = tk.Label(temp_cell, text="", bg=PANEL_BG, fg=WARN,
                                     font=(MONO_FAMILY, 7), anchor="w",
                                     justify="left", wraplength=250)
        self.temp_reason.pack(fill="x", pady=(3, 2), padx=4)

        # Step 6: composite validation readout.
        comp_cell = _layers_cell("COMPOSITE VALIDATION  (cross-layer, separate "
                                  "from each layer's own contract validity)")
        self.composite_label = tk.Label(comp_cell, text="", bg=PANEL_BG, fg=FG,
                                         font=(MONO_FAMILY, 8), anchor="w",
                                         justify="left", wraplength=340)
        self.composite_label.pack(fill="x", pady=(2, 0))

        # Layout-fix (2026-09): `bottom` used to pack every button, the
        # speed control, AND the footer disclosure into ONE side="left" row
        # -- measured naturally requiring 2417px (the buttons + speed
        # control + the footer text laid out at its wraplength=900), which
        # is what drove the app's OWN reqwidth to ~2433px against a 1760px
        # actual window (see layout-fix-report.md). Since pack gives later
        # children whatever's left over, the footer -- carrying the
        # assumed-cap disclosure ("|J| and |b| caps are assumed project
        # values...") -- was the one squeezed, down to ~203px wide. Split
        # into two stacked rows: controls (unchanged content, now on its
        # own row, comfortably under the window width) and the footer on
        # its own full-width row below, wraplength bound live to that row's
        # actual width so it is never narrower than what's really on screen.
        # Controlbar-fix (2026-09): `bottom` itself is now created and
        # packed at the TOP of _build_layout, before `content` -- see the
        # comment there. It is reused (not recreated) here; only its
        # children are built in this part of the method.
        bottom_controls = tk.Frame(bottom, bg=BG)
        bottom_controls.pack(side="top", fill="x")
        self.pause_btn = tk.Button(bottom_controls, text="Pause", command=self._toggle_pause,
                                    bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                    activeforeground=FG, relief="flat", padx=12, pady=4)
        self.pause_btn.pack(side="left")
        self.seed_btn = tk.Button(bottom_controls, text="New seed", command=self._new_seed,
                                   bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                   activeforeground=FG, relief="flat", padx=12, pady=4)
        self.seed_btn.pack(side="left", padx=(6, 0))
        # A2: Save world -- writes the CURRENTLY DISPLAYED world (last_valid_
        # grid) plus its provenance to demo/worlds/. Disabled (not just
        # ignored) whenever that world is STALE (drawn under a since-changed
        # clamp) or the active clamp is INFEASIBLE -- both states this app
        # already tracks in _refresh_world_status -- so a saved file can
        # never silently be evidence for a clamp it wasn't actually drawn
        # under, or claim a world that never came back valid.
        self.save_btn = tk.Button(bottom_controls, text="Save world", command=self._on_save_world,
                                   bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                   activeforeground=FG, relief="flat", padx=12, pady=4,
                                   state="disabled")
        self.save_btn.pack(side="left", padx=(6, 0))
        # UI2: Step -- runs exactly one batch and re-pauses, for
        # frame-by-frame inspection. Pauses first if currently playing,
        # since "step" only means something from a stopped state.
        self.step_btn = tk.Button(bottom_controls, text="Step", command=self._on_step,
                                   bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                   activeforeground=FG, relief="flat", padx=12, pady=4)
        self.step_btn.pack(side="left", padx=(6, 0))
        # Task 9: "What is this?" -- a newcomer's tour, in its own Toplevel.
        self.explainer_btn = tk.Button(bottom_controls, text="What is this?",
                                        command=self._on_show_explainer,
                                        bg=PANEL_BG, fg=ACCENT, activebackground=BORDER,
                                        activeforeground=ACCENT, relief="flat",
                                        padx=12, pady=4)
        self.explainer_btn.pack(side="left", padx=(6, 0))

        # UI2: speed control -- a stepped selector (Scale in integer,
        # snap-to-level mode) over SPEED_LEVELS, DISPLAY RATE ONLY. Says so
        # right in the UI, not just in a code comment: this control changes
        # n_chains/n_samples per call and the pause between calls, never
        # n_warmup or steps_per_sample -- so it cannot change what is being
        # sampled, only how fast the same distribution scrolls by.
        speed_frame = tk.Frame(bottom_controls, bg=BG)
        speed_frame.pack(side="left", padx=(16, 0))
        tk.Label(speed_frame, text="speed:", bg=BG, fg=DIM,
                  font=(MONO_FAMILY, 8)).pack(side="left")
        # speed_label is created BEFORE the Scale's initial .set() below --
        # tk.Scale.set() can invoke -command synchronously even for a
        # programmatic change, and _on_speed_change/_refresh_speed_label
        # both write to self.speed_label, so it must already exist.
        self.speed_label = tk.Label(speed_frame, text="", bg=BG, fg=DIM,
                                      font=(MONO_FAMILY, 8), justify="left")
        self.speed_scale = tk.Scale(
            speed_frame, from_=0, to=len(SPEED_LEVELS) - 1, orient="horizontal",
            resolution=1, showvalue=0, length=140, bg=BG, fg=FG,
            troughcolor=PANEL_BG, highlightthickness=0, bd=0,
            command=self._on_speed_change)
        self.speed_scale.set(self.speed_idx)
        self.speed_scale.pack(side="left", padx=(4, 4))
        self.speed_label.pack(side="left")
        self._refresh_speed_label()

        # Its own full-width row below the controls (see the `bottom`
        # comment above) -- wraplength bound live to this row's own actual
        # width (via <Configure>) rather than a fixed guess, so the
        # assumed-cap disclosure is never narrower than what the window
        # actually has to give it, at any window size.
        # Resize-guard (2026-09): named attribute (was a bare local) so a
        # regression test can assert this survives resize -- this is the
        # exact label the control-bar bug (bottom packed after content)
        # squeezed toward zero along with the four buttons.
        self.footer_label = tk.Label(bottom, text=FOOTER_TEXT, bg=BG, fg=DIM,
                                  font=(MONO_FAMILY, 8), justify="left", anchor="w")
        self.footer_label.pack(side="top", fill="x", padx=(16, 0), pady=(4, 0))
        footer_label = self.footer_label

        def _on_footer_row_configure(evt):
            footer_label.config(wraplength=max(evt.width - 16, 100))

        bottom.bind("<Configure>", _on_footer_row_configure)

    # -- static (receipt-only) panel content --------------------------
    def _populate_static_panels(self):
        r = self.receipt

        # PIPELINE ------------------------------------------------------
        pf = self.pipeline_panel.body
        verdict = r.passes.get("verdict", "unavailable: field absent from receipt")
        tk.Label(pf, text=f"verdict: {verdict}", bg=PANEL_BG, fg=GOOD if verdict == "COMPILED" else WARN,
                  font=MONO_B, anchor="w").pack(fill="x")
        durations = r.passes.get("pass_durations", {})
        for stage in PASS_ORDER:
            if stage in durations:
                line = f"[x] {stage:<13} {fmt_duration(durations[stage])}"
                fg = FG
            else:
                line = f"[ ] {stage:<13} unavailable: field absent from receipt"
                fg = WARN
            tk.Label(pf, text=line, bg=PANEL_BG, fg=fg, font=MONO, anchor="w").pack(fill="x")
        tk.Label(pf, text="", bg=PANEL_BG).pack()
        tk.Label(pf, text="Compilation (encode..verify) ran ONCE, when this\n"
                           "receipt was written -- the durations above are\n"
                           "that one run's own numbers, read from passes.json.\n"
                           "Only SAMPLING re-runs per world; the pipeline\n"
                           "itself is not re-executing.",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), anchor="w", justify="left"
                  ).pack(fill="x", pady=(4, 0))

        # LIVE LATTICE ----------------------------------------------------
        # `inner` holds the canvas+legend+caption as one block, packed with
        # expand=True (no fill) into the panel body -- that centres the
        # whole block in whatever vertical space the panel grid cell gives
        # it, instead of the block hugging the top and leaving dead PANEL_BG
        # below it.
        #
        # Redraw (was: divmod(i, 16), no cell boundary, mediators tacked on
        # as four more rows -- the two panels read at 2:1 vs 1:1 aspect and
        # mediators inflated the row count, which is why a pinned shape
        # didn't visually match between LIVE LATTICE and DECODED WORLD).
        # Now: an 8x8 grid of bordered cell BLOCKS at the SAME cell_px pitch
        # as the DECODED WORLD panel (WORLD_CELL_PX), each block holding its
        # spins_per_cell sub-spins side by side; mediators get their own
        # labelled strip below, never implying they belong to a cell.
        # spins_per_cell is DERIVED from the receipt (world spin count /
        # cell count), never hardcoded -- this app has no basis for
        # assuming k=3 domain-wall coding stays true if the receipt changes.
        lf = self.lattice_panel.body
        inner = tk.Frame(lf, bg=PANEL_BG)
        inner.pack(expand=True)

        n_world = len(r.world_idx)
        n_cells = W * H
        if n_world % n_cells != 0:
            raise ValueError(
                f"world spin count {n_world} is not evenly divisible by "
                f"{n_cells} cells ({W}x{H}) -- cannot derive spins_per_cell "
                f"without guessing; this receipt's encoding doesn't match "
                f"the assumption this panel is built on")
        spins_per_cell = n_world // n_cells
        self.spins_per_cell = spins_per_cell  # Task 10: cached for the
        # detach-only renderers (render_lattice_graph_image,
        # render_relaxation_strip_image) and per_cell_occupancy, which all
        # need it and have no other place to derive it from without
        # duplicating this same divisibility check.
        cell_px = WORLD_CELL_PX  # same on-screen pitch as DECODED WORLD

        world_w, world_h = W * cell_px, H * cell_px
        med_cols = 16
        med_cw = 16
        med_count = len(r.mediator_idx)
        med_rows = -(-med_count // med_cols) if med_count else 0
        gap, label_h = 10, 16
        canvas_w = max(world_w, med_cols * med_cw)
        canvas_h = world_h + gap + label_h + med_rows * med_cw + 4

        self.lattice_canvas = tk.Canvas(inner, width=canvas_w, height=canvas_h,
                                          bg=PANEL_BG, highlightthickness=0)
        self.lattice_canvas.pack(pady=(2, 6))

        # cell blocks: one bordered outline per cell, drawn once ...
        for cy in range(H):
            for cx in range(W):
                bx0, by0, bx1, by1 = cell_block_bounds(cx, cy, cell_px)
                self.lattice_canvas.create_rectangle(
                    bx0, by0, bx1, by1, outline=BORDER, width=1)

        # ... then each world spin's own fill rectangle inside its block,
        # positioned via spin_cell_position -- the single source of truth
        # for "which cell does this physical spin belong to."
        self.lattice_rects = []
        for i in range(r.n_spins):
            pos = spin_cell_position(i, n_world, spins_per_cell, W)
            if pos is not None:
                cx, cy, slot = pos
                x0, y0, x1, y1 = spin_slot_rect(cx, cy, slot, spins_per_cell, cell_px)
            else:
                m = i - n_world  # mediator's own index within the strip
                mx0, my0, mx1, my1 = mediator_slot_rect(m, med_cols, med_cw)
                strip_y0 = world_h + gap + label_h
                x0, y0, x1, y1 = mx0, my0 + strip_y0, mx1, my1 + strip_y0
            rect = self.lattice_canvas.create_rectangle(
                x0, y0, x1, y1, outline="", fill=WORLD_OFF)
            self.lattice_rects.append(rect)

        self.lattice_canvas.create_text(
            4, world_h + gap + label_h / 2, anchor="w", fill=DIM,
            font=(MONO_FAMILY, 8),
            text=f"MEDIATOR SPINS ({med_count}) -- hidden spins from edge "
                 f"subdivision; belong to no cell")

        legend = tk.Frame(inner, bg=PANEL_BG)
        legend.pack(fill="x")
        self._swatch(legend, WORLD_ON, f"world spin (0-{len(r.world_idx) - 1}), lit = 1")
        self._swatch(legend, MEDIATOR_ON, f"mediator spin ({r.mediator_idx[0]}-{r.mediator_idx[-1]}), lit = 1")
        tk.Label(inner, text="Raw physical spin state of the compiled program --\n"
                           "this is NOT the decoded world; a spin here has no\n"
                           "terrain meaning until enc.decode succeeds. Each\n"
                           "bordered block above is one DECODED WORLD cell, at\n"
                           "the same grid position and pitch as that panel.",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), justify="left", anchor="w"
                  ).pack(fill="x", pady=(6, 0))

        # DECODED WORLD ---------------------------------------------------
        # A Canvas (not a Label) so clicks can be mapped to a grid cell
        # (cell_at) and pin markers can be drawn on top of the rendered
        # world -- the click-to-pin gesture this whole feature is about.
        wf = self.world_panel.body
        self.world_canvas = tk.Canvas(wf, width=WORLD_DISPLAY_PX, height=WORLD_DISPLAY_PX,
                                        bg=PANEL_BG, highlightthickness=0, cursor="hand2")
        self.world_canvas.pack(pady=(4, 6))
        self.world_placeholder_id = self.world_canvas.create_text(
            WORLD_DISPLAY_PX / 2, WORLD_DISPLAY_PX / 2,
            text="(no valid sample drawn yet this session)", fill=DIM, font=MONO)
        self.world_image_item = self.world_canvas.create_image(0, 0, anchor="nw")
        self.world_canvas.bind("<Button-1>", self._on_world_click)
        self.world_status_label = tk.Label(wf, text="", bg=PANEL_BG, fg=DIM,
                                             font=(MONO_FAMILY, 8), justify="left", anchor="w",
                                             wraplength=WORLD_DISPLAY_PX - 8)
        self.world_status_label.pack(fill="x")
        self.infeasible_label = tk.Label(wf, text="", bg=PANEL_BG, fg=BAD,
                                           font=MONO_B, justify="left", anchor="w", wraplength=380)
        self.infeasible_label.pack(fill="x", pady=(2, 0))

        # PINS ------------------------------------------------------------
        pf = tk.Frame(wf, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        pf.pack(fill="x", pady=(8, 0))
        tk.Label(pf, text="PINS  (click a cell above to add/cycle/remove one)",
                  bg=PANEL_BG, fg=ACCENT, font=(MONO_FAMILY, 9, "bold"), anchor="w"
                  ).pack(fill="x", padx=6, pady=(4, 2))
        self.pins_list_label = tk.Label(pf, text="(none)", bg=PANEL_BG, fg=DIM,
                                          font=MONO, justify="left", anchor="w")
        self.pins_list_label.pack(fill="x", padx=6)
        self.clear_pins_btn = tk.Button(pf, text="Clear pins", command=self._on_clear_pins,
                                          bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                          activeforeground=FG, relief="flat", padx=10, pady=3)
        self.clear_pins_btn.pack(anchor="w", padx=6, pady=(4, 6))

        # VERIFICATION ------------------------------------------------
        vf = self.verif_panel.body
        for g in r.gates:
            ok = g.get("passed")
            status = "PASS" if ok else "FAIL"
            color = GOOD if ok else BAD
            extra = []
            if g.get("assumed"):
                extra.append("assumed limit")
            if g.get("downgraded"):
                extra.append("downgraded")
            extra_s = f" ({', '.join(extra)})" if extra else ""
            # A gate can PASS while its measurement was never stored (the
            # colouring gate does this). Rendering that as
            # "PASS ... measured=unavailable" reads as a contradiction, so
            # say plainly that the VERDICT is recorded and the NUMBER is not.
            if "measured" in g and "limit" in g:
                detail = (f"measured={fmt_value(g.get('measured'))} "
                          f"limit={fmt_value(g.get('limit'))}")
            else:
                detail = "verdict recorded; measurement not stored in receipt"
            # Layout-fix: this Label had no wraplength -- "PASS colouring
            # measured=unavailable: field absent from receipt" (492px
            # natural width) silently overran the panel's own ~395px
            # content width with no wrap and no visible warning (measured;
            # see layout-fix-report.md). Matches the wraplength already
            # used on the verification-metrics loop just below.
            tk.Label(vf, text=f"{status:<4} {g['gate']:<13} {detail}{extra_s}",
                      bg=PANEL_BG, fg=color, font=(MONO_FAMILY, 8), anchor="w",
                      justify="left", wraplength=395
                      ).pack(fill="x")
        tk.Label(vf, text="", bg=PANEL_BG).pack()
        v = r.verification
        for key in ("task_validity", "codeword_violation_rate", "ess", "energy_tv",
                    "execution_tv", "cross_check_tv", "diversity_distinct",
                    "diversity_valid_samples", "diversity_reachable"):
            val = v.get(key, None)
            # F-R11 / R19 F2: task_validity/codeword_violation_rate/
            # execution_tv are SAMPLING-MEASURED -- displayed at the
            # precision their own (receipt-derived, never fabricated)
            # uncertainty supports, not a blanket 6 s.f. See
            # _verification_display_value's own docstring for why
            # energy_tv is deliberately excluded.
            text = (fmt_value(_verification_display_value(key, val, v, r.cost))
                    if key in v else "unavailable: field absent from receipt")
            fg = DIM if isinstance(val, str) else FG
            tk.Label(vf, text=f"{key}: {text}", bg=PANEL_BG, fg=fg,
                      font=(MONO_FAMILY, 8), anchor="w", justify="left", wraplength=395
                      ).pack(fill="x")

        # SAMPLER -------------------------------------------------------
        sf = self.sampler_panel.body
        beta = r.im.beta
        rows_txt = [
            f"kernel: {r.sampling_program.schedule}",
            f"beta: {beta:.6g}  (FIXED -- see note below)",
            f"spins: {r.n_spins}  (world={len(r.world_idx)}, mediators={r.mediator_count})",
            f"edges: {len(r.im.edges)}",
            f"colour blocks: {len(r.sampling_program.blocks)}",
            f"encoding: {r.encoding_name}",
            f"coefficient_scale: {fmt_value(r.passes.get('coefficient_scale'))}",
            f"mediation: {r.partition_method}, bipartite_after={r.bipartite_after}, "
            f"beta_used={fmt_value(r.beta_used)}",
        ]
        for t in rows_txt:
            tk.Label(sf, text=t, bg=PANEL_BG, fg=FG, font=(MONO_FAMILY, 8),
                      anchor="w", justify="left", wraplength=395).pack(fill="x")
        tk.Label(sf, text=f"no beta slider: {r.mediator_count} mediator spin(s) were "
                           f"coupled at beta={r.beta_used!r}; tsu.passes.route."
                           f"assert_beta_consistent refuses any other beta for this "
                           f"model (BetaMismatchError, spec 5.3.5) -- shown fixed, "
                           f"not hidden.",
                  bg=PANEL_BG, fg=WARN, font=(MONO_FAMILY, 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(4, 0))
        tk.Label(sf, text=f"live sampler settings (this app, not the receipt) -- "
                           f"UNPINNED at Full speed: n_chains/call={BATCH_CHAINS}, "
                           f"n_warmup={N_WARMUP}, n_samples/call={N_SAMPLES_PER_CALL}",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(4, 0))
        tk.Label(sf, text=f"PINNED at Full speed (via simulate(clamp=...), one batch "
                           f"per pin change): n_chains/call={CLAMP_N_CHAINS}, "
                           f"n_warmup={CLAMP_N_WARMUP}, n_samples/call={CLAMP_N_SAMPLES}",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(2, 0))
        tk.Label(sf, text=f"both: steps_per_sample(thinning)={STEPS_PER_SAMPLE} -- "
                           f"n_warmup and steps_per_sample are FIXED at every speed "
                           f"(see speed control in the bottom bar): only n_chains/call "
                           f"and n_samples/call scale down below Full, which changes "
                           f"batch size/display rate, never the sampled distribution.",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(2, 0))

        # DECODED MIX ---------------------------------------------------
        mf = self.mix_panel.body
        self.mix_label = tk.Label(mf, text="unavailable: no valid sample drawn yet this session",
                                    bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 9), justify="left", anchor="w")
        self.mix_label.pack(fill="x")

        # SAMPLE LOG ------------------------------------------------------
        lgf = self.log_panel.body
        # Layout-fix: no wraplength meant the live text (which grows with
        # every draw -- "valid=N contract-fail=N non-codeword=N total=N")
        # could silently overrun the panel's ~395px content width with no
        # wrap (measured overflowing to 524px natural width once counts
        # grew multi-digit; see layout-fix-report.md).
        self.valid_frac_label = tk.Label(lgf, text="valid fraction: n/a (0 draws)",
                                           bg=PANEL_BG, fg=FG, font=MONO_B, anchor="w",
                                           justify="left", wraplength=395)
        self.valid_frac_label.pack(fill="x")
        tk.Label(lgf, text="Displayed worlds are drawn from p(x | valid),\n"
                            "NOT p(x): only valid draws are ever rendered,\n"
                            "so this mix and the world panel reflect the\n"
                            "conditional distribution, not the raw sampler.",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), justify="left", anchor="w"
                  ).pack(fill="x", pady=(2, 4))
        log_frame = tk.Frame(lgf, bg=PANEL_BG)
        log_frame.pack(fill="both", expand=True)
        sb = tk.Scrollbar(log_frame)
        sb.pack(side="right", fill="y")
        # A Listbox does not wrap, so a long violation message ran off the right
        # edge unreachable. Horizontal scrollbar rather than truncation -- the
        # violation text names the offending edge and is worth reading in full.
        sbx = tk.Scrollbar(log_frame, orient="horizontal")
        sbx.pack(side="bottom", fill="x")
        self.log_box = tk.Listbox(log_frame, bg=theme.INSET, fg=FG, font=(MONO_FAMILY, 8),
                                    highlightthickness=0, relief="flat",
                                    yscrollcommand=sb.set, xscrollcommand=sbx.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        sb.config(command=self.log_box.yview)
        sbx.config(command=self.log_box.xview)

        # FRONTIER --------------------------------------------------------
        # B1: read straight from demo/frontier.py's OWN report/renderer --
        # this panel can never show a number `python demo/frontier.py`
        # itself would not print, because it is the same function call.
        # Computed ONCE at load (same "compiled once" framing as PIPELINE
        # above): the frontier is a property of the compiled receipt, not
        # of anything sampled live.
        ff = self.frontier_panel.body
        report = None
        try:
            report = frontier_mod.build_frontier_report(r.path)
            frontier_text = frontier_mod.render_text(report)
        except Exception as exc:  # never crash the app over a display panel
            frontier_text = f"unavailable: frontier report failed: {exc}"

        GAUGE_W = 300  # panel body's own usable width (left col 340px minus
                        # Panel's border/padding) -- the gauges' fixed drawing
                        # width; re-verified by screenshot, see task report.

        if report is not None:
            # Task 11: load gauges -- one per hardware limit, the binding
            # limit visually dominant, the predicted cost of one more
            # terrain value (k -> k+1) shown against them so a viewer can
            # see which one breaks first. Presentation only: every number
            # below comes straight from `report` (frontier_gauge_specs),
            # never recomputed here.
            tk.Label(ff, text=(f"HEADROOM THIS COMPILED MODEL -- "
                               f"{report.encoding.upper()} -- K={report.shape.k} "
                               f"P={report.shape.worst_partners}"),
                     bg=PANEL_BG, fg=ACCENT, font=(MONO_FAMILY, 8, "bold"),
                     anchor="w", justify="left", wraplength=GAUGE_W
                     ).pack(fill="x", pady=(0, 6))

            gauges_frame = tk.Frame(ff, bg=PANEL_BG)
            gauges_frame.pack(fill="x")
            for spec in frontier_gauge_specs(report):
                self._build_frontier_gauge_row(gauges_frame, spec, GAUGE_W)

            # The bind-first conclusion, called out on its own (verbatim
            # report.binding.headline -- not a paraphrase, so it can never
            # drift from the Text box below or from `python demo/frontier.py`).
            bind_box = tk.Frame(ff, bg=theme.PANEL_2, highlightbackground=theme.RED,
                                highlightthickness=1)
            bind_box.pack(fill="x", pady=(4, 6))
            tk.Label(bind_box, text="FIRST TO BIND", bg=theme.PANEL_2, fg=theme.RED_HOT,
                     font=(MONO_FAMILY, 8, "bold"), anchor="w"
                     ).pack(fill="x", padx=6, pady=(4, 0))
            tk.Label(bind_box, text=report.binding.headline, bg=theme.PANEL_2, fg=FG,
                     font=(MONO_FAMILY, 8), anchor="w", justify="left",
                     wraplength=GAUGE_W).pack(fill="x", padx=6, pady=(0, 4))

        # Full detail (model shape, both next-increment axes, the verified
        # predicted-vs-observed rows, the |J|/|b| assumed-value disclaimer)
        # stays the UNCHANGED frontier_mod.render_text output, in a
        # scrollable Text box below the gauges -- nothing the old panel
        # showed is dropped, only the headroom section is now ALSO a gauge
        # above rather than shown solely as text.
        frontier_frame = tk.Frame(ff, bg=PANEL_BG)
        frontier_frame.pack(fill="both", expand=True)
        fsb = tk.Scrollbar(frontier_frame)
        fsb.pack(side="right", fill="y")
        frontier_box = tk.Text(frontier_frame, bg=theme.INSET, fg=FG, width=38,
                                font=(MONO_FAMILY, 8), wrap="word", relief="flat",
                                highlightthickness=0, yscrollcommand=fsb.set)
        frontier_box.insert("1.0", frontier_text)
        frontier_box.config(state="disabled")
        frontier_box.pack(side="left", fill="both", expand=True)
        fsb.config(command=frontier_box.yview)

        # REGIME & TRACES ---------------------------------------------------
        # B2: beta/beta_c siting (Onsager, ORIENTING estimate only -- the
        # assumption is stated on screen, not just in a comment) plus the
        # two live line plots.
        gf = self.regime_panel.body
        j_max = float(np.abs(r.im.weights).max()) if len(r.im.weights) else 0.0
        try:
            regime = beta_regime(r.im.beta, j_max)
            regime_line = (f"beta={regime.beta:.4g}  beta_c(Onsager)={regime.betac:.4g}"
                           f"  beta/beta_c={regime.ratio:.3g}")
        except ValueError as exc:
            regime = None
            regime_line = f"unavailable: {exc}"
        tk.Label(gf, text=f"|J|max (this program) = {j_max:.4g}", bg=PANEL_BG,
                  fg=FG, font=(MONO_FAMILY, 8), anchor="w").pack(fill="x")
        tk.Label(gf, text=regime_line, bg=PANEL_BG, fg=FG, font=MONO_B,
                  anchor="w", wraplength=260, justify="left").pack(fill="x", pady=(2, 4))
        tk.Label(gf, text=ONSAGER_ASSUMPTION_NOTE, bg=PANEL_BG, fg=WARN,
                  font=(MONO_FAMILY, 8), anchor="w", justify="left",
                  wraplength=260).pack(fill="x", pady=(0, 8))

        # Task 5: temperature alongside beta -- "everything being inverted
        # is what throws me for a loop." beta is inverse temperature only
        # because exp(-beta*E) is tidier to carry through the compiler than
        # exp(-E/T); both belong on screen. beta<=0 is refused by
        # beta_to_temperature itself (see demo/scope.py), never silently
        # displayed as a blank or a guess.
        try:
            temp_line = (f"beta = {r.im.beta:.6g}  (temperature T = 1/beta "
                        f"= {beta_to_temperature(r.im.beta):.3f})")
            temp_fg = FG
        except ValueError as exc:
            temp_line = f"unavailable: {exc}"
            temp_fg = WARN
        tk.Label(gf, text=temp_line, bg=PANEL_BG, fg=temp_fg, font=MONO_B,
                  anchor="w", wraplength=260, justify="left").pack(fill="x", pady=(0, 2))
        tk.Label(gf, text="Higher T = hotter, more disordered. Lower T =\n"
                            "colder, more frozen. The useful window sits\n"
                            "between the two extremes.",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), anchor="w",
                  justify="left", wraplength=260).pack(fill="x", pady=(0, 8))

        tk.Label(gf, text="ENERGY TRACE (over draws)", bg=PANEL_BG, fg=ACCENT,
                  font=(MONO_FAMILY, 8, "bold"), anchor="w").pack(fill="x")
        self.energy_canvas = tk.Canvas(gf, width=260, height=110, bg=theme.INSET,
                                         highlightthickness=0)
        self.energy_canvas.pack(fill="x", pady=(2, 8))

        tk.Label(gf, text="VALID FRACTION TRACE (over the session)", bg=PANEL_BG,
                  fg=ACCENT, font=(MONO_FAMILY, 8, "bold"), anchor="w").pack(fill="x")
        self.valid_frac_canvas = tk.Canvas(gf, width=260, height=110, bg=theme.INSET,
                                             highlightthickness=0)
        self.valid_frac_canvas.pack(fill="x", pady=(2, 4))
        tk.Label(gf, text="Both axes are labelled with their live min/max --\n"
                            "an unlabelled sparkline is decoration, not\n"
                            "instrumentation.",
                  bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8), anchor="w",
                  justify="left").pack(fill="x")
        self._draw_trace(self.energy_canvas, self.energy_trace, "draw", "energy")
        self._draw_trace(self.valid_frac_canvas, self.valid_frac_trace,
                         "draw", "valid %")

        # Task 6: SCOPE panel's own initial draw (all four sub-plots start
        # on their "(no data yet)" / "(no draws yet)" placeholder, same
        # honesty convention as the two traces just above).
        self._refresh_scope_panel()

    def _build_frontier_gauge_row(self, parent: tk.Frame, spec: GaugeSpec,
                                  width: int) -> None:
        """Task 11: one FRONTIER load-gauge row -- CHROME meta text (name,
        tag, fraction, percent) as Labels, IMAGE track (fill + predicted
        hairline / hatched overrun) blitted from render_frontier_gauge_track,
        CHROME foot caption. `spec` already carries every number and label
        this needs (see frontier_gauge_specs) -- this method only lays
        widgets out, it computes nothing."""
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", pady=(0, 8))

        head = tk.Frame(row, bg=PANEL_BG)
        head.pack(fill="x")
        left_head = tk.Frame(head, bg=PANEL_BG)
        left_head.pack(side="left")
        tk.Label(left_head, text=spec.label, bg=PANEL_BG, fg=ACCENT,
                 font=(MONO_FAMILY, 9, "bold"), anchor="w").pack(side="left")
        if spec.tag_text:
            tag_fg = {"bind_first": theme.RED_HOT, "highest_util": WARN, "ok": GOOD}[spec.tag_kind]
            tag_border = {"bind_first": theme.RED, "highest_util": WARN, "ok": GOOD}[spec.tag_kind]
            tk.Label(left_head, text=" " + spec.tag_text + " ", bg=PANEL_BG, fg=tag_fg,
                     font=(MONO_FAMILY, 7, "bold"), highlightbackground=tag_border,
                     highlightthickness=1, bd=0
                     ).pack(side="left", padx=(8, 0))
        right_head = tk.Frame(head, bg=PANEL_BG)
        right_head.pack(side="right")
        frac_text = f"{spec.measured:g} / {spec.limit:g}"
        pct_text = f"{spec.pct_used:.0f}%" if spec.pct_used is not None else "n/a"
        tk.Label(right_head, text=pct_text, bg=PANEL_BG, fg=DIM,
                 font=(MONO_FAMILY, 8), anchor="e").pack(side="right", padx=(6, 0))
        tk.Label(right_head, text=frac_text, bg=PANEL_BG, fg=FG,
                 font=(MONO_FAMILY, 8, "bold"), anchor="e").pack(side="right")

        track_h = 18 if spec.tag_kind == "bind_first" else 14
        overrun_ratio = None
        if spec.predicted_exceeds_cap and spec.predicted_value is not None and spec.limit:
            overrun_ratio = spec.predicted_value / spec.limit
        img = render_frontier_gauge_track(
            width, track_h, spec.frac_current,
            frac_predicted=(spec.frac_predicted if not spec.predicted_exceeds_cap else None),
            overrun_ratio=overrun_ratio)
        photo = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(row, width=width, height=track_h, bg=PANEL_BG,
                           highlightthickness=0)
        canvas.pack(fill="x", pady=(3, 2))
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # keep a reference; Tk drops PhotoImages with none

        if spec.foot_text:
            tk.Label(row, text=spec.foot_text, bg=PANEL_BG, fg=DIM,
                     font=(MONO_FAMILY, 7), anchor="w", justify="left",
                     wraplength=width).pack(fill="x")

    def _draw_trace(self, canvas: tk.Canvas, trace: "Trace", xlabel: str,
                    ylabel: str) -> None:
        """Redraw one line plot from `trace`'s current contents. Axis
        min/max are read from `trace.bounds()` and drawn as text -- never
        skipped, per the brief's own instrumentation-not-decoration rule.

        I-2/F-R6: draws each of `trace_runs(trace.chain_breaks)`'s runs as
        its own `create_line` call -- never one polyline across a chain or
        restart boundary, which would assert continuous single-trajectory
        dynamics between draws that are frequently not even from the same
        chain (see Trace's own docstring). Every point additionally gets a
        small dot marker regardless of run membership, so a trace made
        entirely of one-point runs (unclamped mode, where
        N_SAMPLES_PER_CALL==1 makes EVERY draw its own chain) still shows
        something instead of going blank. This method touches no `self.*`
        attribute -- callable unbound in tests, see
        tests/test_lattice_app_logic.py."""
        canvas.delete("all")
        w = int(canvas["width"]) or 260
        h = int(canvas["height"]) or 110
        pad_l, pad_r, pad_t, pad_b = 44, 8, 8, 16
        bounds = trace.bounds()
        if bounds is None or len(trace) < 2:
            canvas.create_text(w / 2, h / 2, text="(no data yet)", fill=DIM,
                                font=(MONO_FAMILY, 8))
            return
        xmin, xmax, ymin, ymax = bounds
        xspan = (xmax - xmin) or 1.0
        yspan = (ymax - ymin) or (abs(ymax) or 1.0)
        px = lambda x: pad_l + (x - xmin) / xspan * (w - pad_l - pad_r)
        py = lambda y: h - pad_b - (y - ymin) / yspan * (h - pad_t - pad_b)
        xs_list = list(trace.xs)
        ys_list = list(trace.ys)
        pts = [(px(x), py(y)) for x, y in zip(xs_list, ys_list)]
        for start, end in trace_runs(list(trace.chain_breaks)):
            if end - start >= 2:
                seg = []
                for i in range(start, end):
                    seg.extend(pts[i])
                canvas.create_line(*seg, fill=ACCENT, width=1)
        for cx, cy in pts:
            canvas.create_oval(cx - 1, cy - 1, cx + 1, cy + 1, fill=ACCENT, outline="")
        canvas.create_text(pad_l, pad_t, text=f"{ymax:.4g}", fill=DIM,
                            font=(MONO_FAMILY, 7), anchor="nw")
        canvas.create_text(pad_l, h - pad_b, text=f"{ymin:.4g}", fill=DIM,
                            font=(MONO_FAMILY, 7), anchor="sw")
        canvas.create_text(pad_l, h - 4, text=f"{xlabel}={xmin:.0f}", fill=DIM,
                            font=(MONO_FAMILY, 7), anchor="sw")
        canvas.create_text(w - pad_r, h - 4, text=f"{xmax:.0f}", fill=DIM,
                            font=(MONO_FAMILY, 7), anchor="se")
        canvas.create_text(w - pad_r, pad_t, text=ylabel, fill=DIM,
                            font=(MONO_FAMILY, 7), anchor="ne")

    def _refresh_scope_panel(self) -> None:
        """Task 6: rebuild all four SCOPE sub-plots from this session's own
        live history (energy_trace, magnetization_trace, raw_draws) and
        blit them onto their canvases. Each render_* function already
        handles "not enough data yet" honestly; this method just calls
        them, converts to PhotoImage, and keeps a reference (Tk drops a
        PhotoImage with no surviving Python reference -- same pattern
        _update_world already uses for world_photo)."""
        energy_ys = list(self.energy_trace.ys)

        # Jitter fix (see jitter-fix-report.md): the real fix for the
        # reported "boxes vibrating" bug is inside _fit_caption_height
        # itself (it now measures on an unmapped probe widget instead of
        # mutating the live caption, so the caption's own height is never
        # observably set to a wrong intermediate value -- see that
        # function's own docstring). This helper is a secondary, purely
        # cost-saving guard on top of that: skip the measure/resize call
        # entirely when `text` hasn't changed from what the label already
        # shows, since re-measuring identical content is wasted work
        # regardless of whether the underlying resize is now flash-free.
        def _set_caption(label: tk.Label, text: str) -> None:
            if label.cget("text") == text:
                return
            label.config(text=text)
            _fit_caption_height(label)

        # C-1 fix: render_acf_plot no longer calls tsu.ess/demo.scope's
        # autocorrelation machinery at all (see its own docstring for why
        # this LIVE trace can never be a valid single-chain input) -- it
        # always returns an honest "unavailable" caption, never raises.
        # The try/except this call site used to need (a constant series
        # made the old autocorrelation() path raise ValueError) no longer
        # applies; every other honesty path in this panel degrades to an
        # "unavailable: <reason>" caption the same way, without needing a
        # guard at the call site.
        acf_img, acf_caption = render_acf_plot(SCOPE_PLOT_W, SCOPE_PLOT_H, energy_ys)
        self.acf_photo = ImageTk.PhotoImage(acf_img)
        self.acf_canvas.itemconfig("plot", image=self.acf_photo)
        _set_caption(self.acf_caption, acf_caption)  # I3: measured, not guessed

        mag_img = render_line_plot(
            SCOPE_PLOT_W, SCOPE_PLOT_H, list(self.magnetization_trace.xs),
            list(self.magnetization_trace.ys), "draw", "M", y_range=(-1.0, 1.0),
            chain_breaks=list(self.magnetization_trace.chain_breaks))
        self.mag_photo = ImageTk.PhotoImage(mag_img)
        self.mag_canvas.itemconfig("plot", image=self.mag_photo)
        _set_caption(
            self.mag_caption,
            "Mean spin per draw (s=2*occupancy-1), fixed axis [-1, 1] "
            "-- the order parameter's own physical bounds.")

        hist_img, hist_caption = render_energy_histogram_plot(
            SCOPE_PLOT_W, SCOPE_PLOT_H, energy_ys)
        self.hist_photo = ImageTk.PhotoImage(hist_img)
        self.hist_canvas.itemconfig("plot", image=self.hist_photo)
        _set_caption(self.hist_caption, hist_caption)

        sigmoid_img, sigmoid_caption = render_sigmoid_plot(
            SCOPE_PLOT_W, SCOPE_PLOT_H, list(self.raw_draws), self.receipt.im)
        self.sigmoid_photo = ImageTk.PhotoImage(sigmoid_img)
        self.sigmoid_canvas.itemconfig("plot", image=self.sigmoid_photo)
        _set_caption(self.sigmoid_caption, sigmoid_caption)  # I4: measured, not guessed

        # Task 10: per-cell occupancy heatmap -- the SAME
        # render_heatmap_image the detached "heatmap" window's own redraw
        # loop calls (see _render_detached_plot), just at SCOPE_PLOT_W/H
        # instead of its detached size.
        occ = per_cell_occupancy(list(self.raw_draws), len(self.receipt.world_idx),
                                  self.spins_per_cell, W)
        heat_img, heat_caption = render_heatmap_image(SCOPE_PLOT_W, SCOPE_PLOT_H, occ)
        self.heat_photo = ImageTk.PhotoImage(heat_img)
        self.heat_canvas.itemconfig("plot", image=self.heat_photo)
        _set_caption(self.heat_caption, heat_caption)

    def _swatch(self, master, color, text):
        row = tk.Frame(master, bg=PANEL_BG)
        row.pack(fill="x", pady=1)
        tk.Canvas(row, width=12, height=12, bg=color, highlightthickness=0).pack(side="left")
        tk.Label(row, text=" " + text, bg=PANEL_BG, fg=DIM, font=(MONO_FAMILY, 8)).pack(side="left")

    # -- live updates --------------------------------------------------
    def _poll_queue(self):
        drained = 0
        try:
            while drained < 12:
                msg = self.in_q.get_nowait()
                self._handle_msg(msg)
                drained += 1
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _handle_msg(self, msg: dict):
        kind = msg["kind"]
        if kind == "error":
            self._log(f"[worker error] {msg['message']}")
            self.pause_btn.config(text="Resume (worker stopped)")
            self.paused = True
            return

        if kind == "batch_summary":
            # One of these follows every CLAMPED batch (see SampleWorker.
            # _run_clamped_batch) -- it is the ONLY source of the infeasible
            # verdict; a batch with valid=0 is honoured verbatim, never
            # second-guessed or smoothed over. This message is ALWAYS about
            # BASE (bands have no continuous worker -- see
            # BAND_SAMPLE_PARAMS's own note) -- base's own bookkeeping
            # updates regardless of which layer is on screen, but the
            # VISUAL refresh only fires when base is actually the one
            # being shown, so it can never clobber a band/composite view.
            self.batch_infeasible = msg["infeasible"]
            self.batch_reason = msg["reason"]
            if self.active_layer == "base":
                self._refresh_world_status()
            return

        # Task 7: a band's regenerate result -- see _regenerate_band_async.
        # Handled BEFORE the base-only bookkeeping below; a band message
        # never touches total_draws/energy_trace/etc, those are base's own.
        if kind == "band_regenerated":
            self._on_band_regenerated(msg)
            return
        if kind == "band_regenerate_error":
            self.regenerate_status.config(
                text=f"regenerate failed: {msg['message']}", fg=BAD)
            self._log(f"[{msg['band']}] regenerate failed: {msg['message']}")
            return

        self.total_draws += 1
        raw = msg["raw"]
        idx = msg["draw_idx"]
        self._update_lattice(raw)

        # I-2/F-R6 + F1/F-R10: chain_break, as computed by SampleWorker
        # (_chain_break_at) for THIS specific draw -- missing (msg.get's
        # default) is treated the same safe way Trace.append's own default
        # is: never assume continuity with whatever the trace already
        # holds.
        chain_break = msg.get("chain_break", True)

        # B2: energy trace over EVERY draw (valid or not -- mixing is a
        # property of the raw chain, the same convention the compiler's own
        # verify pass uses, see energy_of_draw's own docstring), plotted
        # against this app's own running draw counter.
        self.energy_trace.append(self.total_draws, energy_of_draw(self.receipt.im, raw),
                                 chain_break=chain_break)

        # Task 6: magnetization trace and the raw-draw pool for
        # local_field_response -- SAME "every draw, valid or not" convention
        # as the energy trace just above (mixing/the order parameter/the
        # local field are all properties of the raw chain, not of the
        # conditional-valid subset).
        self.magnetization_trace.append(self.total_draws, magnetization([raw])[0],
                                        chain_break=chain_break)
        self.raw_draws.append(raw)

        if kind == "valid":
            self.valid_count += 1
            self.last_valid_grid = msg["grid"]
            self.last_valid_mix = msg["mix"]
            self.last_valid_seed = msg["seed"]
            # UI2: the ACTUAL per-call sampler params for THIS draw (varies
            # with the speed control) -- Save World below reports this, not
            # a Full-speed constant that may not be what actually ran.
            self.last_valid_sampler_params = msg.get("sampler_params")
            self.world_stale = False
            # A fresh valid draw under the CURRENT clamp is direct proof
            # that clamp is feasible -- don't let a stale "infeasible" verdict
            # from a previous batch keep showing next to a world that just
            # disproved it.
            self.batch_infeasible = False
            self.batch_reason = ""
            # Task 7: base's own bookkeeping (above) always updates so
            # switching back to "base" shows fresh data immediately -- but
            # the visible canvas/labels only redraw when base is actually
            # the layer on screen, so a band/composite view stays undisturbed.
            if self.active_layer == "base":
                self._update_world(msg["image"])
                self._update_mix(msg["mix"])
            # Task 7: base just produced a fresh valid decode -- the
            # composite readout depends on base's CURRENT last_valid_grid
            # (see composite_missing_layers), so it can flip from
            # "unavailable" to a real number on THIS event even when a band
            # is the one currently on screen, not only after a band regenerate.
            self._refresh_composite_readout()
            self._log(f"#{idx:<5} valid")
        elif kind == "contract-fail":
            self.contract_fail_count += 1
            viol = "; ".join(msg.get("violations", ())) or "contract violated"
            self._log(f"#{idx:<5} contract-fail  ({viol})")
        else:  # non-codeword
            self.noncodeword_count += 1
            self._log(f"#{idx:<5} non-codeword")

        frac = 100.0 * self.valid_count / self.total_draws if self.total_draws else 0.0
        self.valid_frac_label.config(
            text=f"valid fraction: {frac:.1f}%  "
                 f"(valid={self.valid_count} contract-fail={self.contract_fail_count} "
                 f"non-codeword={self.noncodeword_count} total={self.total_draws})")
        # chain_break=False (always connect): unlike energy/magnetization,
        # this is one genuine cumulative running statistic -- each point
        # is a well-defined function of the point before it
        # (valid_count/total_draws so far), not a per-draw physical
        # quantity sampled from a possibly-different chain. Connecting it
        # makes no continuity claim about the underlying chains at all.
        self.valid_frac_trace.append(self.total_draws, frac, chain_break=False)
        self._draw_trace(self.energy_canvas, self.energy_trace, "draw", "energy")
        self._draw_trace(self.valid_frac_canvas, self.valid_frac_trace,
                         "draw", "valid %")
        if self.active_layer == "base":
            self._refresh_world_status()

        # Task 6: throttled -- local_field_response pools every spin of
        # every buffered draw on each call, cheap per call but wasted work
        # at the poll cadence (~80ms) if run on literally every draw; every
        # SCOPE_REDRAW_EVERY_N_DRAWS-th draw (plus the first few, so the
        # panel doesn't sit on "(no data yet)" longer than it has to) is
        # plenty for a display, not a measurement.
        self._scope_redraw_counter += 1
        if (self.total_draws <= 5
                or self._scope_redraw_counter % SCOPE_REDRAW_EVERY_N_DRAWS == 0):
            self._refresh_scope_panel()

    def _update_lattice(self, raw):
        med = set(self.receipt.mediator_idx)
        for i, (rect, bit) in enumerate(zip(self.lattice_rects, raw)):
            if i in med:
                color = MEDIATOR_ON if bit else MEDIATOR_OFF
            else:
                color = WORLD_ON if bit else WORLD_OFF
            self.lattice_canvas.itemconfig(rect, fill=color)

    def _update_world(self, image: Image.Image):
        disp = image.resize((WORLD_DISPLAY_PX, WORLD_DISPLAY_PX), Image.LANCZOS)
        self.world_photo = ImageTk.PhotoImage(disp)
        self.world_canvas.itemconfig(self.world_image_item, image=self.world_photo)
        self.world_canvas.itemconfig(self.world_placeholder_id, state="hidden")
        self.world_canvas.tag_raise("pin")

    def _update_mix(self, mix: dict):
        lines = [f"{TERRAIN_NAMES[k]:<6}: {mix[TERRAIN_NAMES[k]]:5.1f}%" for k in TERRAIN_ORDER]
        self.mix_label.config(text="\n".join(lines) +
                               "\n\n(computed live from the current last-valid\n"
                               "8x8 decoded grid; p(x | valid), see note below)",
                               fg=FG)

    def _refresh_world_status(self):
        """The single place that renders world-panel truthfulness: whether
        there IS a valid world yet, whether the one on screen is STALE
        (drawn under a different clamp than the one now active), and
        whether the CURRENT clamp has been shown infeasible. These three
        facts are independent and all shown plainly -- never collapsed into
        a single ambiguous state."""
        if self.last_valid_grid is None:
            self.world_status_label.config(
                text="no valid sample drawn yet this session", fg=DIM)
        elif self.world_stale:
            self.world_status_label.config(
                text=f"STALE -- last valid world shown was drawn BEFORE the "
                     f"current pins; {self.valid_count}/{self.total_draws} "
                     f"draws valid so far under the current pins",
                fg=WARN)
        else:
            self.world_status_label.config(
                text=f"last valid sample -- draw #{self.total_draws} "
                     f"(valid {self.valid_count}/{self.total_draws} draws so far "
                     f"under the current pins)",
                fg=DIM)

        if self.batch_infeasible:
            self.infeasible_label.config(text=f"INFEASIBLE: {self.batch_reason}")
        else:
            self.infeasible_label.config(text="")

        # A2: Save world is only ever enabled for a world that is BOTH
        # present and current -- never stale, never drawn under a clamp
        # since shown infeasible.
        can_save = (self.last_valid_grid is not None
                    and not self.world_stale
                    and not self.batch_infeasible)
        self.save_btn.config(state="normal" if can_save else "disabled")

    # -- click-to-pin ----------------------------------------------------
    def _pin_color(self, value: int) -> str:
        r, g, b = (int(c) for c in PAL[value])
        return f"#{r:02x}{g:02x}{b:02x}"

    def _band_pin_color(self, value: int) -> str:
        return self._pin_color(WATER if value == 0 else GRASS)  # matches render_band_image's own palette reuse

    def _current_layer_clamp(self) -> "ClampState | None":
        """Task 7: which ClampState click-to-pin acts on right now. None
        for composite -- a derived view has no program of its own to pin."""
        if self.active_layer == "base":
            return self.clamp
        if self.active_layer in self.bands:
            return self.bands[self.active_layer].clamp
        return None

    def _redraw_pin_markers(self):
        """Task 7: draws only the ACTIVE layer's own pins -- base and band
        pins live on different ClampState instances (see LayerState), so
        showing both on one canvas would blur exactly the distinction the
        brief requires the UI keep ("pinning in the base and in an overlay
        are different operations")."""
        self.world_canvas.delete("pin")
        clamp = self._current_layer_clamp()
        if clamp is None:
            return
        is_base = self.active_layer == "base"
        color_fn = self._pin_color if is_base else self._band_pin_color
        # C1 (fix-round-2): base pins are exact clamps, overlay pins are a
        # soft nudge (see OVERLAY_PIN_STRENGTH's comment and the LAYER
        # SELECTOR cell's disclosure label) -- a rectangle-plus-dot marker
        # identical for both wore the affordance of a guarantee on the soft
        # case too. Overlay pins now draw with a dashed outline and no
        # solid dot (a hollow ring instead), so "this one can be outvoted"
        # is visible on the canvas itself, not just in a caption.
        rect_kw = {} if is_base else {"dash": (3, 2)}
        for (x, y), v in clamp.items():
            x0, y0 = x * WORLD_CELL_PX, y * WORLD_CELL_PX
            color = color_fn(v)
            self.world_canvas.create_rectangle(
                x0 + 2, y0 + 2, x0 + WORLD_CELL_PX - 2, y0 + WORLD_CELL_PX - 2,
                outline=color, width=3, tags="pin", **rect_kw)
            if is_base:
                self.world_canvas.create_oval(
                    x0 + 3, y0 + 3, x0 + 11, y0 + 11, fill=color, outline=FG, tags="pin")
            else:
                self.world_canvas.create_oval(
                    x0 + 3, y0 + 3, x0 + 11, y0 + 11, fill="", outline=color,
                    width=2, tags="pin")
        self.world_canvas.tag_raise("pin")

    def _refresh_pins_panel(self):
        """Task 7: PINS lists EVERY layer's pins together, each line tagged
        with its own layer name -- base pins never rendered as if they were
        an overlay's, and vice versa (see _redraw_pin_markers's own
        docstring for why the two are kept structurally separate)."""
        lines = [f"base: {name} = {TERRAIN_NAMES[v]}"
                 for name, v in sorted(self.clamp.as_dict().items())]
        for band_name in BAND_NAMES:
            d = self.bands[band_name].clamp.as_dict()
            lines.extend(f"{band_name}: {name} = {BAND_VALUE_NAMES[v]}"
                         for name, v in sorted(d.items()))
        if not lines:
            self.pins_list_label.config(text="(none)", fg=DIM)
        else:
            self.pins_list_label.config(text="\n".join(lines), fg=FG)

    def _on_world_click(self, event):
        if self.active_layer == "composite":
            self.regenerate_status.config(
                text="composite is a read-only derived view -- select "
                     "base or a band to pin a cell", fg=WARN)
            return
        cell = cell_at(event.x, event.y, 0, 0, WORLD_CELL_PX,
                        WORLD_DISPLAY_PX, WORLD_DISPLAY_PX)
        if cell is None:
            return
        x, y = cell
        clamp = self._current_layer_clamp()
        clamp.cycle(x, y)
        self._redraw_pin_markers()
        self._refresh_pins_panel()
        if self.active_layer == "base":
            self._restart_sampling_for_clamp_change()
        # A band's pin does NOT trigger a resample by itself (bands have no
        # continuous worker -- see BAND_SAMPLE_PARAMS's own docstring note):
        # it takes effect on the NEXT "Regenerate this layer" click, folded
        # into that band's conditioning patch via overlay_pin_patch. Marking
        # the band's current world stale here would be dishonest about
        # WHICH world is stale -- there is no live stream to go stale;
        # the status text says so instead (see _refresh_layer_view).
        else:
            self._refresh_layer_view()

    def _on_clear_pins(self):
        """Clears the ACTIVE layer's own pins only -- consistent with
        "pins apply to the selected layer" for every other pin operation
        in this panel."""
        clamp = self._current_layer_clamp()
        if clamp is None or len(clamp) == 0:
            return
        clamp.clear()
        self._redraw_pin_markers()
        self._refresh_pins_panel()
        if self.active_layer == "base":
            self._restart_sampling_for_clamp_change()
        else:
            self._refresh_layer_view()

    # -- Task 7: LAYERS panel ---------------------------------------------
    def _on_layer_selected(self):
        self.active_layer = self.layer_var.get()
        self.regenerate_status.config(text="", fg=DIM)
        self.regenerate_btn.config(
            state="disabled" if self.active_layer == "composite" else "normal")
        self._refresh_layer_view()
        self._refresh_pins_panel()
        self._refresh_temperature_control()
        self._refresh_composite_readout()

    def _on_alpha_change(self, value):
        self.alpha = float(value)
        self.alpha_label.config(text=f"strength (alpha) = {self.alpha:.2f}  "
                                      f"(demo/elevation.band_patch's own scale "
                                      f"-- see dose-response note below)")

    def _on_temp_canvas_interact(self, event):
        """Task 8: click/drag on the Canvas-drawn track. ADJUSTABLE only --
        a drag on the FIXED/locked track (or on composite's n/a track) is
        deliberately a no-op, not merely visually disabled: the tokens
        spec's own trap #2 is that a disabled-LOOKING control reads as
        broken, not as a deliberate refusal, which is exactly why this is
        drawn on a Canvas rather than relying on a native Scale's disabled
        state to communicate anything. Overlay-only in practice today
        (base/composite always fail the `state == "adjustable"` check),
        but keyed on temperature_control_state like everything else here,
        not on `self.active_layer == "base"`.

        Stores the chosen T for the ACTIVE layer via set_beta_override,
        applied on that layer's NEXT regenerate (same 'takes effect next
        batch' idiom the speed control already uses) -- never resamples
        here."""
        ising = self._active_layer_ising()
        if ising is None:
            return
        state, _ = temperature_control_state(ising)
        if state != "adjustable":
            return
        frac = min(1.0, max(0.0, event.x / max(TEMP_TRACK_W - 1, 1)))
        t = TEMP_T_MIN + frac * (TEMP_T_MAX - TEMP_T_MIN)
        beta = 1.0 / t   # T = 1/beta (demo/scope.py's own beta_to_temperature, inverted)
        set_beta_override(self.active_layer, beta, self, self.bands)
        self._refresh_temperature_control()

    def _active_layer_ising(self):
        """The IsingModel the ACTIVE layer's own program is built from --
        base's compiled program for "base", the overlay receipt's compiled
        program for any band (every band starts from the SAME compiled
        demo/receipts/elev_band program; only its biases differ per band,
        never its topology/mediator_nodes -- see demo/layers.bias_patch's
        own guarantee), None for composite (no program of its own)."""
        if self.active_layer == "base":
            return self.receipt.im
        if self.active_layer in self.bands:
            return self.overlay_receipt.im
        return None

    def _refresh_temperature_control(self):
        """Task 8: the temperature control's two explicit states, drawn
        entirely on the Canvas track (render_temperature_track) plus this
        method's own Label/border styling -- ADJUSTABLE (a live slider,
        gold) or FIXED (a locked seal, cold blue -- a point of pride, not
        an apology, never a greyed-out control). Keyed on
        temperature_control_state, which derives the decision from EXACTLY
        the fact `tsu.passes.route.assert_beta_consistent` gates sampling
        on (`ising.mediator_nodes` empty or not) -- never a hardcoded
        'base is locked' flag, and never `self.bands[self.active_layer]`
        directly (see get_beta_override's own docstring for the KeyError a
        prior review flagged in that direct-indexing pattern)."""
        ising = self._active_layer_ising()
        if ising is None:  # composite -- no program, nothing to show at all
            self.temp_cell.config(highlightbackground=BORDER, highlightcolor=BORDER)
            self.temp_label.config(text="n/a", fg=DIM)
            self.temp_state_label.config(text="composite has no program of its own", fg=DIM)
            self.temp_reason.config(text="")
            img = render_temperature_track(TEMP_TRACK_W, TEMP_TRACK_H,
                                           adjustable=False, value_frac=0.5)
            self.temp_photo = ImageTk.PhotoImage(img)
            self.temp_canvas.itemconfig("track", image=self.temp_photo)
            return

        state, reason = temperature_control_state(ising)
        if state == "fixed":
            self.temp_cell.config(highlightbackground=theme.BLUE, highlightcolor=theme.BLUE)
            t = beta_to_temperature(ising.beta)
            self.temp_label.config(text=f"T = {t:.3f}", fg=theme.BLUE_LIT)
            self.temp_state_label.config(text="FIXED · NOT DISABLED", fg=theme.BLUE_LIT)
            # "SEALED" up front carries the same pride-not-apology framing a
            # separate "SEAL / SPEC 5.3.5" header line used to (see I1 fix,
            # round 2, above) without spending a whole extra Label's worth
            # of vertical space on it -- `reason` (from
            # temperature_control_state) already cites "spec 5.3.5" itself.
            self.temp_reason.config(text=f"SEALED -- {reason}", fg=DIM)
            value_frac = 0.5   # frozen -- render_temperature_track's locked branch ignores this too
        else:
            self.temp_cell.config(highlightbackground=theme.GOLD_DIM, highlightcolor=theme.GOLD_DIM)
            beta = get_beta_override(self.active_layer, self.base_beta_override,
                                     self.bands) or ising.beta
            t = beta_to_temperature(beta)
            self.temp_label.config(text=f"T = {t:.3f}", fg=ACCENT)
            self.temp_state_label.config(text=f"ADJUSTABLE · {self.active_layer}", fg=ACCENT)
            # Minor #4 (final review): this used to assert "bipartite" as a
            # FACT. What's actually verified here is `ising.mediator_nodes`
            # being empty (the same fact temperature_control_state keys
            # "adjustable" on) -- zero mediators means none were INSERTED,
            # it does not license concluding the receipt is bipartite,
            # since `self.overlay_receipt.bipartite_after` is itself None
            # (unrecorded) for this receipt, not True. Named the measured
            # fact (mediator count) and stated bipartite_after honestly via
            # fmt_value, rather than asserting a conclusion the receipt
            # never recorded -- the same "unavailable: <reason>" discipline
            # every other unrecorded value in this app follows.
            self.temp_reason.config(
                text=f"beta = {beta:.4g} -- {len(ising.mediator_nodes)} "
                     f"mediator spins in this program (bipartite_after: "
                     f"{fmt_value(self.overlay_receipt.bipartite_after)}), "
                     f"so nothing here is welded to a compile-time beta. "
                     f"Moving this re-samples {self.active_layer} on its "
                     f"NEXT regenerate; it does not recompile.", fg=DIM)
            value_frac = temperature_value_frac(t, TEMP_T_MIN, TEMP_T_MAX)

        img = render_temperature_track(TEMP_TRACK_W, TEMP_TRACK_H,
                                       adjustable=(state == "adjustable"),
                                       value_frac=value_frac)
        self.temp_photo = ImageTk.PhotoImage(img)
        self.temp_canvas.itemconfig("track", image=self.temp_photo)

    def _refresh_composite_readout(self):
        """Task 7 Step 6: cross-layer (monotonicity) validation, SEPARATE
        from each layer's own contract validity (already shown per-layer:
        base's in VERIFICATION, a band's in its own regenerate_status)."""
        band_decoded = [self.bands[n].last_valid_decoded for n in BAND_NAMES]
        missing = composite_missing_layers(self.last_valid_grid, band_decoded)
        if missing:
            self.composite_label.config(
                text=f"unavailable: {', '.join(missing)} has no valid sample "
                     f"yet this session -- composite needs every layer at "
                     f"least once (base streams continuously; regenerate "
                     f"each band at least once)", fg=DIM)
            return
        viol = monotonicity_violations(band_decoded)
        slots = W * H * (N_BANDS - 1)
        rate = len(viol) / slots if slots else float("nan")
        self.composite_label.config(
            text=f"monotonicity violations (band i+1 true where band i "
                 f"false): {len(viol)}/{slots} cell-transitions "
                 f"({100*rate:.1f}%). DETECTED, not repaired -- band_patch's "
                 f"nudge is an encouragement, never a hard constraint, so "
                 f"this can and does exceed 0%; see demo/elevation.py.",
            fg=WARN if rate > 0 else GOOD)

    def _refresh_layer_view(self):
        """Task 7: redraw DECODED WORLD + its status labels + pin markers
        for whichever layer is selected. Never samples anything -- purely a
        redraw of state already held (base's own continuously-updated
        attributes, a band's LayerState, or the composite derivation)."""
        if self.active_layer == "base":
            if self.last_valid_grid is not None:
                self.world_canvas.itemconfig(self.world_image_item, image=self.world_photo)
                self.world_canvas.itemconfig(self.world_placeholder_id, state="hidden")
            else:
                self.world_canvas.itemconfig(self.world_image_item, image="")
                self.world_canvas.itemconfig(
                    self.world_placeholder_id, state="normal",
                    text="(no valid sample drawn yet this session)")
            self._refresh_world_status()
        elif self.active_layer in self.bands:
            layer = self.bands[self.active_layer]
            if layer.last_valid_grid is None:
                self.world_canvas.itemconfig(self.world_image_item, image="")
                self.world_canvas.itemconfig(
                    self.world_placeholder_id, state="normal",
                    text=f"({self.active_layer}: no valid sample yet -- "
                         f"click 'Regenerate this layer')")
                self.world_status_label.config(
                    text="unavailable: no valid sample for this layer yet", fg=DIM)
            else:
                img = render_band_image(layer.last_valid_grid)
                disp = img.resize((WORLD_DISPLAY_PX, WORLD_DISPLAY_PX), Image.LANCZOS)
                self.layer_photo = ImageTk.PhotoImage(disp)
                self.world_canvas.itemconfig(self.world_image_item, image=self.layer_photo)
                self.world_canvas.itemconfig(self.world_placeholder_id, state="hidden")
                # I1 (fix-round-2): this overlay's TaskContract declares
                # ZERO rules (measured live below, not assumed) -- for a
                # binary domain with no chain to violate, contract.validate
                # is vacuously .ok=True for every codeword. "valid" here is
                # therefore ONLY "decoded as a codeword" (a real check);
                # dressing that as "same convention as base" implied a
                # contract was actually checked, which for this overlay it
                # was not. Relabelled, not deleted -- the codeword rate
                # itself is still real.
                n_rules = len(self.overlay_receipt.spec.contract.rules)
                if n_rules == 0:
                    rules_note = ("this overlay's contract declares 0 rules, "
                                   "so \"valid\" means only \"decoded as a "
                                   "codeword\" -- the contract-pass check is "
                                   "vacuously true for every codeword, not "
                                   "evidence any rule was satisfied")
                else:
                    rules_note = (f"{n_rules} contract rule(s) checked, same "
                                   f"convention as base")
                self.world_status_label.config(
                    text=f"{self.active_layer}: valid {layer.valid_count}/"
                         f"{layer.total_draws} draws so far under the current "
                         f"conditioning/pins -- p(x | valid); {rules_note} "
                         f"(see SAMPLE LOG note)", fg=DIM)
            self.infeasible_label.config(
                text=f"INFEASIBLE: {layer.batch_reason}" if layer.batch_infeasible else "")
            self.save_btn.config(state="disabled")  # A2 Save World stays base-only, see report
        else:  # composite
            self.world_canvas.itemconfig(self.world_placeholder_id, state="hidden")
            band_decoded = [self.bands[n].last_valid_decoded for n in BAND_NAMES]
            missing = composite_missing_layers(self.last_valid_grid, band_decoded)
            if missing:
                self.world_canvas.itemconfig(self.world_image_item, image="")
                self.world_canvas.itemconfig(
                    self.world_placeholder_id, state="normal",
                    text=f"(composite unavailable: {', '.join(missing)} has "
                         f"no valid sample yet)")
                self.world_status_label.config(
                    text=f"unavailable: {', '.join(missing)} has no valid "
                         f"sample yet this session", fg=DIM)
            else:
                elevation = np.array(
                    [[thermometer_level(band_decoded, f"g{x}_{y}") for x in range(W)]
                     for y in range(H)])
                # I2 (fix-round-2): terrain now comes from
                # composite_base_grid -- the base decode actually captured
                # at the MOST RECENT band regenerate, whichever band that
                # was (see _regenerate_band_async/_on_band_regenerated) --
                # NOT self.last_valid_grid, which keeps streaming from
                # base's continuous background worker while a band is
                # selected. Rendering self.last_valid_grid here would show
                # elevation conditioned on base-decode-A hillshaded over
                # terrain from a LATER base-decode-B: not a draw from
                # p(base)*p(band|base) at all. composite_base_grid is
                # guaranteed non-None here: `missing` above is empty only
                # once every band has a last_valid_decoded, which is set in
                # the same handler that sets composite_base_grid.
                #
                # NOTE (fix-round-3, re-review residual): composite_base_
                # grid is a SINGLE app-wide value, overwritten unconditionally
                # by WHICHEVER band regenerates most recently -- it is not a
                # base shared by all three bands. Per-layer regenerate is
                # deliberate (bands resample independently), so band0/1/2
                # are the normal case, not the exception, conditioned on
                # DIFFERENT historical base draws. The status text below
                # must therefore name only the most-recently-regenerated
                # band's base, not "the bands" plural -- claiming joint
                # consistency across all three is exactly the overclaim
                # this whole review round was about. Tracking each band's
                # own conditioning snapshot (so the status line could name
                # per-band staleness precisely) is the fuller fix and is
                # deliberately NOT done here -- out of scope for a wording
                # correction, belongs in its own reviewed task.
                terrain_grid = self.composite_base_grid
                img = render_elevation_world_image(terrain_grid, elevation)
                disp = img.resize((WORLD_DISPLAY_PX, WORLD_DISPLAY_PX), Image.LANCZOS)
                self.layer_photo = ImageTk.PhotoImage(disp)
                self.world_canvas.itemconfig(self.world_image_item, image=self.layer_photo)
                base_is_stale = not np.array_equal(terrain_grid, self.last_valid_grid)
                # F-R12/R13: composite_status_text also states the
                # MANDATORY ancestral-factorization caveat (layers.py's own
                # docstring) alongside the pre-existing staleness
                # disclosure -- see that function's own docstring.
                self.world_status_label.config(
                    text=composite_status_text(base_is_stale),
                    fg=(WARN if base_is_stale else DIM))
            self.infeasible_label.config(text="")
            self.save_btn.config(state="disabled")
        self._redraw_pin_markers()

    def _on_save_world(self):
        """Write the currently displayed world to demo/worlds/. Guarded
        twice: the button itself is disabled (see _refresh_world_status)
        whenever the world is STALE or the active clamp is INFEASIBLE, and
        this handler re-checks the same conditions before writing, so a
        stray event (e.g. a queued click landing after a state change)
        can't slip a bad save through."""
        if (self.last_valid_grid is None or self.world_stale
                or self.batch_infeasible):
            return

        grid = self.last_valid_grid
        # grid[y, x] via classify_draw's own construction (see
        # classify_draw above) -- flatten() with numpy's default C order
        # walks x fastest within each y row, i.e. row-major exactly as
        # worldfile.py's format expects (values[y*width + x]).
        values = grid.flatten().tolist()
        clamp = self.clamp.as_dict()
        # UI2: report what ACTUALLY produced this grid -- captured off the
        # worker message when this draw arrived (see _handle_msg), which
        # reflects whatever speed level was active for that batch, not a
        # Full-speed constant that may be wrong if the speed control was
        # touched. Fall back to the Full-speed constants only in the
        # (should-be-impossible-given the save button's own guard)
        # case that a valid grid exists with no recorded sampler_params.
        sampler_params = self.last_valid_sampler_params
        if sampler_params is None:
            sampler_params = (
                {"n_chains": CLAMP_N_CHAINS, "n_samples": CLAMP_N_SAMPLES,
                 "n_warmup": CLAMP_N_WARMUP, "steps_per_sample": STEPS_PER_SAMPLE}
                if clamp else
                {"n_chains": BATCH_CHAINS, "n_samples_per_call": N_SAMPLES_PER_CALL,
                 "n_warmup": N_WARMUP, "steps_per_sample": STEPS_PER_SAMPLE})
        saved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        seed = self.last_valid_seed if self.last_valid_seed is not None else -1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        out_path = WORLDS_DIR / f"world_{stamp}_seed{seed}.json"

        try:
            save_world(out_path, spec_name=self.receipt.spec.name,
                       receipt_dir=str(self.receipt.path), seed=seed,
                       clamp=clamp, width=W, height=H, values=values,
                       value_names=[TERRAIN_NAMES[k] for k in TERRAIN_ORDER],
                       task_valid=True, violations=[],
                       sampler_params=sampler_params, saved_at=saved_at)
        except Exception as exc:  # surfaced in the log, never swallowed
            self._log(f"SAVE FAILED: {exc}")
            return
        self._log(f"saved world -> {out_path.relative_to(REPO_ROOT)}")

    def _log(self, text: str):
        self.log_lines.append(text)
        self.log_box.insert("end", text)
        if self.log_box.size() > 300:
            self.log_box.delete(0)
        self.log_box.see("end")

    # -- controls --------------------------------------------------------
    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.worker.pause.set()
            self.pause_btn.config(text="Resume")
        else:
            self.worker.pause.clear()
            self.pause_btn.config(text="Pause")

    def _on_step(self):
        """UI2: Step -- pause first if currently playing (stepping only
        means something from a stopped state), then ask the worker for
        exactly one batch. The worker consumes step_evt itself (see
        SampleWorker.run) -- this method never runs a batch directly."""
        if not self.paused:
            self._toggle_pause()
        self.worker.request_step()

    # -- Task 9: "What is this?" explainer window -------------------------
    def _on_show_explainer(self):
        """Opens demo/explainer.py's ExplainerWindow -- a real tk.Toplevel
        (draggable/resizable/minimizable via the native window manager,
        never overrideredirect). Singleton: a second click LIFTS the
        existing window (and un-minimizes it via deiconify) rather than
        stacking duplicate windows -- winfo_exists() is checked because the
        user closing the window (WM_DELETE_WINDOW -> destroy, wired in
        ExplainerWindow.__init__) leaves self.explainer_window pointing at
        a destroyed Tk object, which would raise on any method call."""
        if self.explainer_window is not None and self.explainer_window.winfo_exists():
            self.explainer_window.deiconify()
            self.explainer_window.lift()
            self.explainer_window.focus_set()
            return
        self.explainer_window = explainer.open_explainer(self)

    # -- Task 10: detachable plots ----------------------------------------
    def _open_detach(self, key: str) -> None:
        """Opens (or LIFTS) `key`'s own small Toplevel -- the same
        singleton pattern _on_show_explainer above already uses,
        generalised over DETACHABLE_PLOTS via self.detach_registry instead
        of one dedicated attribute per window. Native window manager only
        (resizable, minimizable, no overrideredirect -- the tokens spec's
        own trap #1), CHROME + IMAGE per the tokens spec's own buildability
        rule: a plain Canvas holding one PhotoImage, no drop shadow/blur/
        rounded corners.

        The Toplevel's own redraw loop calls _render_detached_plot, which
        dispatches to THE SAME render_* function this plot's inline host
        calls where one exists (hist/sigmoid/heatmap, see
        _refresh_scope_panel) -- one renderer, two hosts, per the brief's
        own framing; lattice_graph/relaxation have no inline host at all,
        so their only call site IS this one.

        The redraw loop is this window's OWN self.after() timer
        (DETACH_REFRESH_MS), independent of the inline SCOPE panel's
        message-driven cadence -- deliberately, so a detached window still
        redraws even for a plot (lattice_graph, relaxation) that has no
        inline host driving a refresh at all. WM_DELETE_WINDOW routes
        through self.detach_registry.close(key), which is the ONLY place
        the timer is ever cancelled -- see DetachRegistry's own docstring
        for why that single choke point is what makes "closed window still
        receiving updates" structurally impossible rather than merely
        avoided by care."""
        if self.detach_registry.is_open(key):
            win = self.detach_registry.window_for(key)
            if win is not None and win.winfo_exists():
                win.deiconify()
                win.lift()
                win.focus_set()
                return
            self.detach_registry.close(key)  # stale handle -- rebuild below

        spec = DETACH_WINDOW_SPECS[key]
        top = tk.Toplevel(self)
        top.title(spec["title"])
        win_w, win_h = spec["win"]
        top.geometry(f"{win_w}x{win_h}")
        top.resizable(True, True)   # native WM handles drag/resize/minimize
        top.configure(bg=theme.PAGE)

        plot_w, plot_h = spec["plot"]
        # C2 (final review): the caption is packed FIRST, side="bottom" --
        # Tk's pack manager hands out space to widgets in PACKING ORDER
        # (not visual order), so whichever is packed first gets its full
        # requested size before anyone else sees what's left. The canvas
        # used to be packed first at a fixed size and never reflowed, so on
        # a shrink the caption (packed last, fill="x" only) was the one
        # squeezed -- down to fully UNMAPPED at a small enough size, taking
        # a load-bearing on-screen disclosure with it while the plot stayed
        # fully visible and looked authoritative on its own. Packing the
        # caption first, side="bottom", means IT keeps its full requested
        # height first and the canvas (which can tolerate less room, or the
        # window's own minsize below can simply refuse to go that low) is
        # the one that would give way instead.
        caption = tk.Label(top, text="", bg=theme.PAGE, fg=theme.CREAM_DIM,
                            font=(MONO_FAMILY, 8), justify="left", anchor="w",
                            wraplength=win_w - 16)
        caption.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        canvas = tk.Canvas(top, width=plot_w, height=plot_h, bg=theme.INSET,
                            highlightthickness=0)
        canvas.pack(padx=8, pady=(8, 4))
        canvas.create_image(0, 0, anchor="nw", tags="plot")
        photos: list = []  # keep a reference -- Tk drops a PhotoImage with none

        def redraw():
            img, caption_text = self._render_detached_plot(key, plot_w, plot_h)
            photo = ImageTk.PhotoImage(img)
            photos.clear()
            photos.append(photo)
            canvas.itemconfig("plot", image=photo)
            caption.config(text=caption_text)
            _fit_caption_height(caption)

        # C2: `wraplength` (set once above, from the CONSTRUCTION-TIME
        # window width) stayed frozen across a resize, which clipped the
        # caption horizontally on a widen and left dead space on a shrink
        # -- re-set it from the window's OWN current width on every
        # <Configure>, then re-measure the now-rewrapped text's real height
        # (same _fit_caption_height every other caption in this app uses,
        # never a re-guessed line count).
        # Debounced, not run synchronously inside the <Configure> dispatch:
        # `_fit_caption_height` itself changes the caption's own requested
        # size (its whole job), and doing that WHILE still inside the
        # widget's own <Configure> handler re-triggers <Configure> on `top`
        # before the first call has returned -- an unbounded synchronous
        # reflow storm (observed directly: opening one detached window and
        # letting it reach steady state alone produced thousands of
        # "Exception in Tkinter callback" prints before this fix). Coalesce
        # rapid-fire events (a drag-resize fires many) into ONE run after a
        # short quiet period, via top.after -- by the time it runs, Tk has
        # already settled outside the original event's own call stack, so
        # this can no longer recurse into itself.
        cfg_job: dict[str, str | None] = {"id": None}

        def _apply_configure():
            cfg_job["id"] = None
            if not top.winfo_exists():
                return
            new_wrap = max(top.winfo_width() - 16, 40)
            if caption.cget("wraplength") != new_wrap:
                caption.config(wraplength=new_wrap)
            _fit_caption_height(caption)

        def _on_configure(_evt=None):
            if cfg_job["id"] is not None:
                top.after_cancel(cfg_job["id"])
            cfg_job["id"] = top.after(60, _apply_configure)

        top.bind("<Configure>", _on_configure)

        job_id: dict[str, str | None] = {"id": None}

        def tick():
            redraw()
            job_id["id"] = top.after(DETACH_REFRESH_MS, tick)

        def cancel_job():
            if job_id["id"] is not None:
                top.after_cancel(job_id["id"])
                job_id["id"] = None
            if cfg_job["id"] is not None:
                top.after_cancel(cfg_job["id"])
                cfg_job["id"] = None

        def on_close():
            self.detach_registry.close(key)  # invokes cancel_job() exactly once
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_close)
        self.detach_registry.open_window(key, top, cancel_job)
        # C2: minsize used to be a hardcoded (280, 220) guess, unrelated to
        # what this window's OWN plot + caption actually need -- draw once
        # first so the caption holds its real content, then measure both
        # widgets' real requested sizes and set minsize from that (still
        # never smaller than a small hard floor, so the window can't be
        # shrunk to zero).
        redraw()
        top.update_idletasks()
        min_w = max(plot_w + 16, 280)
        min_h = plot_h + caption.winfo_reqheight() + 28
        top.minsize(min_w, max(min_h, 160))
        tick()

    def _render_detached_plot(self, key: str, w: int, h: int) -> tuple[Image.Image, str]:
        """Dispatch table for a detached window's own redraw loop -- see
        _open_detach's own docstring for why this deliberately calls the
        EXACT SAME render_* functions the inline hosts call rather than a
        second, parallel drawing path."""
        if key == "hist":
            # I2 (final review): render_energy_histogram_plot now returns
            # (img, caption) itself -- the SAME caption the inline SCOPE
            # cell shows, not a second hand-written one. This dispatch
            # used to build its own caption text here, and it had already
            # drifted from the inline copy (dropped the substantive "the
            # trace itself only samples one point of this at a time"
            # sentence in favour of naming which function was called).
            return render_energy_histogram_plot(w, h, list(self.energy_trace.ys))
        if key == "sigmoid":
            return render_sigmoid_plot(w, h, list(self.raw_draws), self.receipt.im)
        if key == "heatmap":
            occ = per_cell_occupancy(list(self.raw_draws), len(self.receipt.world_idx),
                                     self.spins_per_cell, W)
            return render_heatmap_image(w, h, occ)
        if key == "lattice_graph":
            return render_lattice_graph_image(
                w, h, self.receipt.im, self.receipt.world_idx,
                self.receipt.mediator_idx, self.spins_per_cell, W)
        if key == "relaxation":
            return render_relaxation_strip_image(
                w, h, list(self.raw_draws), n_frames=8,
                n_world_spins=len(self.receipt.world_idx),
                spins_per_cell=self.spins_per_cell, grid_w=W)
        raise KeyError(key)

    # -- Task 7: per-layer regenerate ------------------------------------
    def _on_regenerate_layer(self):
        """Step 4: re-sample ONE layer without touching any other --
        base's own continuous worker and every OTHER band's held state are
        untouched by this call, demonstrating the amortized-compile point
        the brief names directly: one receipt, patched biases, no
        recompile, and now not even a re-sample of layers that didn't ask
        for one."""
        if self.active_layer == "base":
            # base already has its own "start over" concept -- reuse it
            # rather than inventing a second one.
            self._new_seed()
            return
        if self.active_layer not in self.bands:
            return  # composite: button is disabled, but guard anyway
        self.regenerate_status.config(text="sampling...", fg=DIM)
        self._regenerate_band_async(self.active_layer)

    def _regenerate_band_async(self, band_name: str):
        """Background thread: derive this band's conditioning patch from
        base's and (for band i>0) the band below's CURRENT last-valid
        decode, fold in this band's own pins (overlay_pin_patch) and
        temperature override, sample one batch, decode+validate, and push
        ONE result back through the existing queue/_handle_msg plumbing --
        same responsive-UI pattern every other sampling call in this app
        already uses, never a main-thread blocking call."""
        if self.last_valid_grid is None:
            self.in_q.put({"kind": "band_regenerate_error", "band": band_name,
                           "message": "base has no valid sample yet -- "
                                      "nothing to condition this band on"})
            return
        idx = band_index_from_name(band_name)
        # I2 (fix-round-2): snapshot the grid ALONGSIDE base_decoded, here
        # in the synchronous dispatch (not inside work(), which runs later
        # on a worker thread while base's own background worker keeps
        # mutating self.last_valid_grid) -- this is the exact base decode
        # this band's conditioning patch is computed from, and it is what
        # the composite must later render its terrain from.
        base_grid_snapshot = self.last_valid_grid
        base_decoded = grid_to_decoded(base_grid_snapshot)
        if idx == 0:
            prev_decoded = None
        else:
            prev = self.bands[f"band{idx - 1}"]
            if prev.last_valid_decoded is None:
                self.in_q.put({"kind": "band_regenerate_error", "band": band_name,
                               "message": f"band{idx - 1} has no valid sample "
                                          f"yet -- regenerate it first"})
                return
            prev_decoded = prev.last_valid_decoded

        alpha = self.alpha
        layer = self.bands[band_name]
        pins = list(layer.clamp.items())
        beta_override = layer.beta_override
        overlay = self.overlay_receipt

        def work():
            try:
                patch = dict(band_patch(prev_decoded, base_decoded, alpha))
                for k, v in overlay_pin_patch(pins).items():
                    patch[k] = patch.get(k, 0.0) + v
                patched = bias_patch(overlay.sampling_program, overlay.enc, patch,
                                     field_cap=FIELD_CAP)
                if beta_override is not None:
                    assert_beta_consistent(patched.ising, beta_override)
                    patched = dataclasses.replace(
                        patched, ising=dataclasses.replace(patched.ising, beta=beta_override))
                seed = random.randint(0, 2**31 - 1)
                got = thrml_sample(patched, seed=seed, **BAND_SAMPLE_PARAMS)
            except Exception as exc:  # FieldCapExceeded, BetaMismatchError, or
                # anything else -- surfaced in the UI, never swallowed (same
                # rule every other worker in this app follows).
                self.in_q.put({"kind": "band_regenerate_error", "band": band_name,
                               "message": str(exc)})
                return
            decoded = []
            for row in got:
                bits = dict(zip(patched.ising.nodes, row.tolist()))
                if overlay.enc.is_codeword(bits):
                    d = overlay.enc.decode(bits)
                    if overlay.spec.contract.validate(d).ok:
                        decoded.append(d)
            self.in_q.put({"kind": "band_regenerated", "band": band_name,
                           "valid": decoded, "total": len(got), "patch": patch,
                           "base_grid": base_grid_snapshot})

        threading.Thread(target=work, daemon=True).start()

    def _on_band_regenerated(self, msg: dict):
        band_name = msg["band"]
        layer = self.bands[band_name]
        valid = msg["valid"]
        layer.valid_count += len(valid)
        layer.total_draws += msg["total"]
        layer.conditioning_patch = msg["patch"]
        if not valid:
            layer.batch_infeasible = True
            layer.batch_reason = f"0/{msg['total']} draws valid under this conditioning"
            self.regenerate_status.config(
                text=f"{band_name}: 0/{msg['total']} valid -- infeasible under "
                     f"current conditioning/pins", fg=BAD)
        else:
            layer.batch_infeasible = False
            layer.batch_reason = ""
            rep = valid[0]
            layer.last_valid_decoded = rep
            layer.last_valid_grid = np.array(
                [[int(rep[f"g{x}_{y}"]) for x in range(W)] for y in range(H)])
            # I2 (fix-round-2): record the base decode THIS band was
            # actually conditioned on, so the composite can render terrain
            # consistent with THIS band's contribution rather than base's
            # current (possibly since-advanced) live decode. This
            # unconditionally overwrites whatever the previous regenerate
            # (of this band or any other) had stored -- it names only the
            # MOST RECENT regenerate's base, never a base shared by all
            # three bands (fix-round-3: see the composite status text's
            # own wording for why that distinction matters on screen too).
            self.composite_base_grid = msg["base_grid"]
            self.regenerate_status.config(
                text=f"{band_name}: {len(valid)}/{msg['total']} valid "
                     f"({100*len(valid)/msg['total']:.1f}%)", fg=GOOD)
        self._log(f"[{band_name}] regenerated: {len(valid)}/{msg['total']} valid")
        if self.active_layer == band_name:
            self._refresh_layer_view()
        self._refresh_composite_readout()

    def _on_speed_change(self, value):
        """UI2: speed slider moved. Updates both the app's own remembered
        speed_idx (so a later pin-change/new-seed worker restart carries it
        forward -- see _start_worker) and the LIVE worker in place, which
        picks it up at the start of its next batch (see
        SampleWorker.set_speed's docstring: never interrupts an in-flight
        batch)."""
        self.speed_idx = int(round(float(value)))
        if hasattr(self, "worker"):  # guard: Tk's Scale.set() during layout
            self.worker.set_speed(self.speed_idx)  # construction can fire
        self._refresh_speed_label()  # -command before self.worker exists

    def _refresh_speed_label(self):
        lvl = speed_level(self.speed_idx)
        pace = "no inter-batch delay" if lvl["delay_s"] == 0 else f"+{lvl['delay_s']:.2f}s between batches"
        self.speed_label.config(
            text=f"{lvl['label']}  (n_chains/call={lvl['chains']}, "
                 f"clamp n_samples/call={lvl['clamp_samples']}, {pace} -- "
                 f"display rate only, n_warmup/steps_per_sample unchanged)")

    def _stop_and_drain_worker(self):
        old = self.worker
        old.stop_evt.set()
        # drain whatever is left in the queue so stale draws from the old
        # worker never get attributed to the new run
        try:
            while True:
                self.in_q.get_nowait()
        except queue.Empty:
            pass

    def _start_worker(self, seed_base: int):
        clamp = self.clamp.as_dict() or None
        self.worker = SampleWorker(self.receipt, self.in_q, seed_base=seed_base,
                                    clamp=clamp, start_paused=self.paused,
                                    speed_idx=self.speed_idx)
        self.worker.start()

    def _new_seed(self):
        """Explicit 'start over' -- clears the displayed world too (unlike a
        pin change, which keeps the old world on screen labelled STALE)."""
        self._stop_and_drain_worker()

        self.total_draws = 0
        self.valid_count = 0
        self.contract_fail_count = 0
        self.noncodeword_count = 0
        self.last_valid_grid = None
        self.last_valid_mix = None
        self.last_valid_sampler_params = None
        self.world_photo = None
        self.world_stale = False
        self.batch_infeasible = False
        self.batch_reason = ""
        self.world_canvas.itemconfig(self.world_image_item, image="")
        self.world_canvas.itemconfig(
            self.world_placeholder_id, state="normal",
            text="(re-sampling from a new seed...)")
        self.mix_label.config(text="unavailable: no valid sample drawn yet this session", fg=DIM)
        self.log_box.delete(0, "end")
        self.valid_frac_label.config(text="valid fraction: n/a (0 draws)")
        self.energy_trace = Trace(maxlen=self.energy_trace.xs.maxlen)
        self.valid_frac_trace = Trace(maxlen=self.valid_frac_trace.xs.maxlen)
        # Task 6: a new seed starts a genuinely new chain -- the SCOPE
        # panel's history must not mix pre- and post-seed draws (same
        # reasoning as the energy/valid-fraction resets just above).
        self.magnetization_trace = Trace(maxlen=self.magnetization_trace.xs.maxlen)
        self.raw_draws.clear()
        self._draw_trace(self.energy_canvas, self.energy_trace, "draw", "energy")
        self._draw_trace(self.valid_frac_canvas, self.valid_frac_trace,
                         "draw", "valid %")
        self._refresh_scope_panel()
        self._refresh_world_status()

        new_seed_base = random.randint(0, 2**31 - 1)
        self._start_worker(new_seed_base)
        clamp = self.clamp.as_dict()
        self._log(f"--- new seed base={new_seed_base}  clamp={clamp or '(none)'} ---")

    def _restart_sampling_for_clamp_change(self):
        """A pin was added, cycled, removed, or cleared -- restart sampling
        under the new clamp. Unlike `_new_seed`, the world currently on
        screen is KEPT (never blanked): it is real evidence from a moment
        ago, just not evidence about the clamp that is active now, so it is
        marked STALE (see `_refresh_world_status`) rather than hidden."""
        self._stop_and_drain_worker()

        self.total_draws = 0
        self.valid_count = 0
        self.contract_fail_count = 0
        self.noncodeword_count = 0
        self.world_stale = self.last_valid_grid is not None
        self.batch_infeasible = False
        self.batch_reason = ""
        self.valid_frac_label.config(text="valid fraction: n/a (0 draws)")
        # total_draws resets to 0 above, and the traces are plotted against
        # it (see _handle_msg) -- reset them too, or a post-clamp-change
        # point would be appended at a SMALLER x than points already in the
        # ring buffer, drawing a plot whose x-axis runs backwards.
        self.energy_trace = Trace(maxlen=self.energy_trace.xs.maxlen)
        self.valid_frac_trace = Trace(maxlen=self.valid_frac_trace.xs.maxlen)
        # Task 6: same reasoning -- a clamp change starts a new conditional
        # distribution, so the SCOPE panel's history resets alongside the
        # energy/valid-fraction traces, not just some of them.
        self.magnetization_trace = Trace(maxlen=self.magnetization_trace.xs.maxlen)
        self.raw_draws.clear()
        self._draw_trace(self.energy_canvas, self.energy_trace, "draw", "energy")
        self._draw_trace(self.valid_frac_canvas, self.valid_frac_trace,
                         "draw", "valid %")
        self._refresh_scope_panel()
        self._refresh_world_status()

        new_seed_base = random.randint(0, 2**31 - 1)
        self._start_worker(new_seed_base)
        clamp = self.clamp.as_dict()
        self._log(f"--- pins changed: {clamp or '(none)'}  seed={new_seed_base} ---")

    def _on_close(self):
        self.worker.stop_evt.set()
        self.worker.join(timeout=0.5)
        self.destroy()


def main():
    receipt = Receipt(RECEIPT_DIR)
    overlay_receipt = Receipt(OVERLAY_RECEIPT_DIR)  # Task 7: compiled once, at load -- same as `receipt`
    app = LatticeApp(receipt, overlay_receipt)
    app.mainloop()


if __name__ == "__main__":
    main()
