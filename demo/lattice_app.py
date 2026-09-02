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

import json
import math
import queue
import random
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
DEMO_DIR = REPO_ROOT / "demo"
RECEIPT_DIR = DEMO_DIR / "receipts" / "small"
WORLDS_DIR = DEMO_DIR / "worlds"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DEMO_DIR))

from tsu.spec import load_spec  # noqa: E402
from tsu.passes.encode import encode  # noqa: E402
from tsu.simulate import reconstruct_program, _selected_encoding  # noqa: E402
from tsu.backends.thrml_backend import sample as thrml_sample  # noqa: E402
from worldfile import save_world  # noqa: E402 -- A2: save/load provenance-carrying worlds
import frontier as frontier_mod  # noqa: E402 -- B1: capacity frontier panel
from scope import (beta_to_temperature, autocorrelation, magnetization,  # noqa: E402
                    energy_histogram, local_field_response, sigmoid,
                    MIN_LOCAL_FIELD_BIN_COUNT)  # Task 5/6
from tsu.ess import (integrated_autocorrelation_time,  # noqa: E402
                     RELIABILITY_MIN_N_OVER_TAU)  # SCOPE panel's tau readout

# --------------------------------------------------------------------------
# constants shared by the raw-lattice grid and the decode/render path
# --------------------------------------------------------------------------
W = H = 8                      # decoded world is an 8x8 grid
WATER, ROCK, GRASS = 0, 1, 2
TERRAIN_NAMES = {WATER: "water", ROCK: "rock", GRASS: "grass"}
TERRAIN_ORDER = (WATER, ROCK, GRASS)
PAL = np.array([[46, 92, 132], [124, 116, 106], [126, 158, 84]], float)
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
               "speed or energy claims. |J| and |b| caps are assumed "
               "project values, not sourced Extropic figures.")

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
SCOPE_ENERGY_TRACE_MAXLEN = 400             # same ring-buffer length B2's
# energy_trace already used before this task; named here so the ACF plot's
# own caption can state it rather than hardcoding a second "400" that could
# drift from the Trace(...) construction below.
SCOPE_RAW_DRAWS_MAXLEN = 600                # ring buffer of raw spin rows,
# feeds local_field_response -- 600 draws * up to ~200 spins/draw pools
# comfortably past MIN_LOCAL_FIELD_BIN_COUNT per bin without growing
# unbounded over a long session.
SCOPE_ACF_MAX_LAG = 40                      # lag window drawn on the
# semi-log ACF plot; tau is marked separately even if it falls outside it
SCOPE_ENERGY_HIST_BINS = 24
SCOPE_LOCAL_FIELD_BINS = 16
SCOPE_REDRAW_EVERY_N_DRAWS = 5              # throttle: local_field_response
# pools SCOPE_RAW_DRAWS_MAXLEN draws * every spin each redraw -- cheap per
# call, but recomputing on literally every one of many draws/sec is wasted
# work the display rate (poll cadence, ~80ms) doesn't need.


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

    def __init__(self):
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
        """Advance one cell's pin: unpinned -> water(0) -> rock(1) ->
        grass(2) -> unpinned. Returns the new value (or None if now
        unpinned)."""
        cur = self._pins.get((x, y))
        if cur is None:
            nxt = self._CYCLE[0]
        else:
            i = self._CYCLE.index(cur)
            nxt = self._CYCLE[i + 1] if i + 1 < len(self._CYCLE) else None
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
    drawing code in this file is allowed to skip calling it."""

    def __init__(self, maxlen: int = 300):
        self.xs: deque = deque(maxlen=maxlen)
        self.ys: deque = deque(maxlen=maxlen)

    def append(self, x: float, y: float) -> None:
        self.xs.append(x)
        self.ys.append(y)

    def __len__(self) -> int:
        return len(self.xs)

    def bounds(self) -> tuple[float, float, float, float] | None:
        """(xmin, xmax, ymin, ymax), or None when empty -- never a
        fabricated (0, 0, 0, 0) range for an empty trace."""
        if not self.xs:
            return None
        return (min(self.xs), max(self.xs), min(self.ys), max(self.ys))


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


def fmt_value(v: Any) -> str:
    """Render a receipt scalar for display. A string is ALREADY either a
    real value's repr or an 'unavailable: <reason>' message written by the
    compiler itself (see verification.json/regime.json) -- passed through
    verbatim either way, never re-interpreted or replaced."""
    if v is None:
        return "unavailable: field absent from receipt"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


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

        self.spec = load_spec(str(path / "spec.yaml"))
        self.encoding_name = _selected_encoding(path)
        self.enc = encode(self.spec, self.encoding_name)
        self.sampling_program = reconstruct_program(str(path))
        self.im = self.sampling_program.ising

        mediator_set = set(self.program.get("mediator_nodes", ()))
        self.n_spins = len(self.im.nodes)
        self.world_idx = [i for i in range(self.n_spins) if i not in mediator_set]
        self.mediator_idx = sorted(mediator_set)

        med = self.passes.get("mediation", {})
        self.mediator_count = med.get("mediator_count", len(self.mediator_idx))
        self.partition_method = med.get("partition_method", "unavailable: field absent from receipt")
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

    def _classify_and_push(self, row, seed, sampler_params: dict) -> dict:
        self.draw_counter += 1
        d = classify_draw(self.receipt, row, seed)
        d["draw_idx"] = self.draw_counter
        # UI2: the ACTUAL n_chains/n_samples this particular draw came
        # from, not a fixed constant -- speed control makes those vary
        # batch to batch, and Save World (below) must record what really
        # produced the saved grid, not what Full speed would have used.
        d["sampler_params"] = sampler_params
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
        for row in rows:
            # UI2: checked BEFORE pushing, not after -- see
            # should_abort_batch's docstring for why this is the actual
            # responsive-pause fix, not just the speed control.
            if should_abort_batch(stopping=self.stop_evt.is_set(),
                                   paused=self.pause.is_set(), is_step=is_step):
                return
            self._classify_and_push(row, seed, sampler_params)

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
                                        seed=seed, clamp=self.clamp)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self._put({"kind": "error", "message": str(exc)})
            self.stop_evt.set()
            return
        draws = []
        for row in got:
            if should_abort_batch(stopping=self.stop_evt.is_set(),
                                   paused=self.pause.is_set(), is_step=is_step):
                break
            draws.append(self._classify_and_push(row, seed, sampler_params))
        infeasible, reason = batch_feasibility(draws)
        valid = sum(1 for d in draws if d["kind"] == "valid")
        self._put({"kind": "batch_summary", "infeasible": infeasible, "reason": reason,
                    "valid": valid, "total": len(draws), "seed": seed})


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
BG = "#1a1b22"
PANEL_BG = "#22242e"
BORDER = "#3a3d4d"
FG = "#e7e7ee"
DIM = "#9497a8"
GOOD = "#5ed38a"
BAD = "#e0667a"
WARN = "#e0b155"
ACCENT = "#6fb3ff"
MEDIATOR_ON = "#c98cff"
MEDIATOR_OFF = "#3c2f4d"
WORLD_ON = "#6fb3ff"
WORLD_OFF = "#243149"
MONO = ("Consolas", 9)
MONO_B = ("Consolas", 9, "bold")


def _rgb(hex_str: str) -> tuple[int, int, int]:
    """'#rrggbb' -> (r, g, b) ints -- so the SCOPE panel's PIL-rendered
    plots (below) can share this file's own colour palette instead of a
    second, hand-copied one that could drift from it."""
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# Task 6: colours for the SCOPE panel's PIL-rendered plots, derived from
# this file's own palette above (never a second literal set of colours).
PLOT_BG = _rgb("#111218")      # same canvas background _draw_trace uses
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

def render_acf_plot(w: int, h: int, series: Sequence[float],
                    max_lag: int = SCOPE_ACF_MAX_LAG, floor: float = 1e-3
                    ) -> tuple[Image.Image, str]:
    """Semi-log (log-y) autocorrelation plot: x=lag (linear), y=|rho(lag)|
    on a log scale -- matches Extropic's DTM paper (arXiv 2510.23972)
    Figure 4b's own semi-log-with-decorrelation-time-marked convention
    (see demo/scope.py's module docstring). rho values <= `floor` are
    FLOORED to `floor` so the log axis has something to plot (a log scale
    cannot show zero or negative values) -- floored points are marked with
    a small warn-coloured dot so the flooring is visible, not hidden.
    tau (Sokal's windowed estimate) is marked as a vertical line when it
    falls within the plotted lag range."""
    img = Image.new("RGB", (w, h), PLOT_BG)
    d = ImageDraw.Draw(img)
    if len(series) < 12:
        d.text((10, h // 2 - 6), "(not enough draws yet)", fill=PLOT_DIM)
        return img, "waiting for more draws before an ACF can be estimated..."

    lag_cap = min(max_lag, len(series) - 1)
    acf = autocorrelation(list(series), max_lag=lag_cap)
    iat = integrated_autocorrelation_time(np.asarray(series, dtype=float))
    n = len(series)
    n_over_tau = (n / iat.tau) if iat.tau > 0 else float("inf")

    pad_l, pad_r, pad_t, pad_b = 40, 8, 8, 16
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    ylo, yhi = math.log10(floor), 0.0  # rho[0] == 1.0 always -> log10(1)=0

    def px(lag): return pad_l + (lag / lag_cap) * pw if lag_cap else pad_l

    def py(v):
        vv = max(v, floor)
        return pad_t + (1.0 - (math.log10(vv) - ylo) / (yhi - ylo)) * ph

    pts = [(px(k), py(v)) for k, v in enumerate(acf)]
    if len(pts) >= 2:
        d.line(pts, fill=PLOT_ACCENT, width=1)
    for k, v in enumerate(acf):
        if v <= floor:
            x, y = px(k), py(v)
            d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=PLOT_WARN)

    tau_in_range = 0 < iat.tau <= lag_cap
    if tau_in_range:
        xp = px(iat.tau)
        d.line([(xp, pad_t), (xp, pad_t + ph)], fill=PLOT_WARN, width=1)
        d.text((min(xp + 2, w - 40), pad_t), f"tau~{iat.tau:.1f}", fill=PLOT_WARN)

    d.text((pad_l, pad_t), "1.0", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, pad_t + ph), f"{floor:g}", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, h - 4), "lag=0", fill=PLOT_DIM, anchor="ls")
    d.text((w - pad_r, h - 4), f"{lag_cap}", fill=PLOT_DIM, anchor="rs")
    d.text((w - pad_r, pad_t), "rho (log)", fill=PLOT_DIM, anchor="ra")

    reliable = n_over_tau >= RELIABILITY_MIN_N_OVER_TAU
    caption = (
        f"tau~={iat.tau:.2f} (Sokal window={iat.window}"
        f"{'*, saturated -- see tsu.ess' if iat.window_saturated else ''})"
        f"{'  (beyond the plotted window)' if not tau_in_range else ''}  "
        f"N={n}  N/tau={n_over_tau:.3g}  reliability threshold (tsu.ess."
        f"RELIABILITY_MIN_N_OVER_TAU)=5000: {'MET' if reliable else 'NOT MET'}. "
        f"This is this LIVE session's own energy trace (ring buffer, "
        f"maxlen={SCOPE_ENERGY_TRACE_MAXLEN}) -- a SEPARATE measurement from "
        f"the receipt's own precomputed ess in the VERIFICATION panel above, "
        f"not a live update of it.")
    return img, caption


def render_line_plot(w: int, h: int, xs: Sequence[float], ys: Sequence[float],
                     xlabel: str, ylabel: str,
                     y_range: tuple[float, float] | None = None) -> Image.Image:
    """Linear x/y line plot -- used for the magnetization trace, with
    y_range fixed to (-1, 1) (the order parameter's own physical bounds,
    a more honest axis than autoscaling to whatever range happened to be
    observed so far)."""
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
    d.line(pts, fill=PLOT_ACCENT, width=1)
    d.text((pad_l, pad_t), f"{ymax:.3g}", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, pad_t + ph), f"{ymin:.3g}", fill=PLOT_DIM, anchor="la")
    d.text((pad_l, h - 4), f"{xlabel}={xmin:.0f}", fill=PLOT_DIM, anchor="ls")
    d.text((w - pad_r, h - 4), f"{xmax:.0f}", fill=PLOT_DIM, anchor="rs")
    d.text((w - pad_r, pad_t), ylabel, fill=PLOT_DIM, anchor="ra")
    return img


def render_energy_histogram_plot(w: int, h: int, series: Sequence[float],
                                 bins: int = SCOPE_ENERGY_HIST_BINS) -> Image.Image:
    """Filled-bar histogram of `series` (the energy trace's own values) --
    the distribution the trace only samples one point of at a time."""
    img = Image.new("RGB", (w, h), PLOT_BG)
    d = ImageDraw.Draw(img)
    if len(series) < 4:
        d.text((10, h // 2 - 6), "(no data yet)", fill=PLOT_DIM)
        return img
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
    return img


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


class Panel(tk.Frame):
    def __init__(self, master, title: str, **kw):
        super().__init__(master, bg=PANEL_BG, highlightbackground=BORDER,
                          highlightthickness=1, **kw)
        tk.Label(self, text=title, bg=PANEL_BG, fg=ACCENT,
                  font=("Consolas", 10, "bold"), anchor="w"
                  ).pack(fill="x", padx=8, pady=(6, 2))
        self.body = tk.Frame(self, bg=PANEL_BG)
        self.body.pack(fill="both", expand=True, padx=8, pady=(0, 8))


class LatticeApp(tk.Tk):
    def __init__(self, receipt: Receipt):
        super().__init__()
        self.receipt = receipt
        self.title("tsu lattice demo -- live sampling of a compiled receipt")
        # Task 6: extra height for the new SCOPE panel row added below the
        # existing content row -- every other panel's own size/position is
        # unchanged, this only makes room for the addition. Grew again in
        # fix-round 1 (1160 -> 1240) when the sigmoid cell's caption grew
        # from 4 to 11 lines to carry the reconstruction-vs-defect
        # disclosure on screen.
        self.geometry("1760x1240")
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

        # click-to-pin state -- pure ClampState, no Tk in it (see the
        # headless tests). world_stale is True whenever the world currently
        # on screen was NOT drawn under the currently active clamp (e.g.
        # right after a pin change, before a new valid draw has arrived).
        self.clamp = ClampState()
        self.world_stale = False
        self.batch_infeasible = False
        self.batch_reason = ""

        # B2: live traces -- energy over sweeps (every draw, valid or not:
        # mixing is a property of the raw chain, not the conditional-valid
        # subset) and valid fraction over the session (recomputed at the
        # SAME cadence, from self.valid_count/self.total_draws already
        # tracked above). Bounded ring buffers, see Trace's own docstring.
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

        self._build_layout()
        self._populate_static_panels()

        self._start_worker(random.randint(0, 2**31 - 1))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_queue)

    # -- layout ------------------------------------------------------
    def _build_layout(self):
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=8, pady=8)
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

        left = tk.Frame(content, bg=BG, width=340)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_propagate(False)
        left.grid_rowconfigure(0, weight=3)
        left.grid_rowconfigure(1, weight=2)
        left.grid_columnconfigure(0, weight=1)
        self.pipeline_panel = Panel(left, "PIPELINE  (compiled once, at load)")
        self.pipeline_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        # B1: the capacity frontier -- headroom + next-increment cost, read
        # from demo/frontier.py's own build_frontier_report/render_text so
        # this panel can never drift from what `python demo/frontier.py`
        # prints on the command line.
        self.frontier_panel = Panel(left, "FRONTIER  (headroom + next increment)")
        self.frontier_panel.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        self.lattice_panel = Panel(content, "LIVE LATTICE  (raw physical spin state)")
        self.lattice_panel.grid(row=0, column=1, sticky="nsew", padx=6)

        self.world_panel = Panel(content, "DECODED WORLD  (last valid sample)")
        self.world_panel.grid(row=0, column=2, sticky="nsew", padx=6)

        # B2: beta/beta_c siting (Onsager, labelled as an orienting estimate
        # only) plus the energy and valid-fraction traces.
        regime_frame = tk.Frame(content, bg=BG, width=300)
        regime_frame.grid(row=0, column=3, sticky="nsew", padx=6)
        regime_frame.grid_propagate(False)
        regime_frame.grid_rowconfigure(0, weight=1)
        regime_frame.grid_columnconfigure(0, weight=1)
        self.regime_panel = Panel(regime_frame, "REGIME & TRACES")
        self.regime_panel.grid(row=0, column=0, sticky="nsew")

        right = tk.Frame(content, bg=BG, width=430)
        right.grid(row=0, column=4, sticky="nsew", padx=(6, 0))
        right.grid_propagate(False)
        # VERIFICATION (~15 rows) and SAMPLE LOG (scrolling) need room;
        # DECODED MIX holds at most k lines. An even 4-way split collapsed
        # MIX to zero height, so weight them by actual content.
        for r, w in ((0, 5), (1, 5), (2, 2), (3, 5)):
            right.grid_rowconfigure(r, weight=w)
        right.grid_rowconfigure(2, minsize=150)
        right.grid_columnconfigure(0, weight=1)

        self.verif_panel = Panel(right, "VERIFICATION")
        self.verif_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self.sampler_panel = Panel(right, "SAMPLER")
        self.sampler_panel.grid(row=1, column=0, sticky="nsew", pady=4)
        self.mix_panel = Panel(right, "DECODED MIX")
        self.mix_panel.grid(row=2, column=0, sticky="nsew", pady=4)
        self.log_panel = Panel(right, "SAMPLE LOG")
        self.log_panel.grid(row=3, column=0, sticky="nsew", pady=(4, 0))

        # Task 6: SCOPE panel -- autocorrelation (semi-log, tau + the 5000
        # reliability threshold), magnetization trace, energy histogram,
        # and the measured-vs-analytic sigmoid response, four sub-plots in
        # a row. Every plot is a PIL image blitted onto its own Canvas (see
        # render_acf_plot etc. above) -- Tk canvas primitives alone can't
        # do a log axis, filled bars, or an overlaid scatter+curve well.
        self.scope_panel = Panel(content, "SCOPE  (autocorrelation / magnetization / "
                                          "energy histogram / local-field response)")
        self.scope_panel.grid(row=1, column=0, columnspan=5, sticky="nsew", pady=(6, 0))
        scope_row = tk.Frame(self.scope_panel.body, bg=PANEL_BG)
        scope_row.pack(fill="both", expand=True)

        def _scope_cell(title, caption_lines=4):
            cell = tk.Frame(scope_row, bg=PANEL_BG)
            cell.pack(side="left", fill="both", expand=True, padx=4)
            tk.Label(cell, text=title, bg=PANEL_BG, fg=ACCENT,
                      font=("Consolas", 8, "bold"), anchor="w").pack(fill="x")
            canvas = tk.Canvas(cell, width=SCOPE_PLOT_W, height=SCOPE_PLOT_H,
                                bg="#111218", highlightthickness=0)
            canvas.pack(pady=(2, 2))
            caption = tk.Label(cell, text="", bg=PANEL_BG, fg=DIM,
                                font=("Consolas", 7), anchor="w", justify="left",
                                wraplength=SCOPE_PLOT_W, height=caption_lines)
            caption.pack(fill="x")
            return canvas, caption

        self.acf_canvas, self.acf_caption = _scope_cell(
            "AUTOCORRELATION (semi-log, tau marked)")
        self.mag_canvas, self.mag_caption = _scope_cell(
            "MAGNETIZATION (order parameter, per draw)")
        self.hist_canvas, self.hist_caption = _scope_cell(
            "ENERGY HISTOGRAM (over the session)")
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
            caption_lines=11)
        for canvas in (self.acf_canvas, self.mag_canvas, self.hist_canvas,
                      self.sigmoid_canvas):
            canvas.create_image(0, 0, anchor="nw", tags="plot")

        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.pause_btn = tk.Button(bottom, text="Pause", command=self._toggle_pause,
                                    bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                    activeforeground=FG, relief="flat", padx=12, pady=4)
        self.pause_btn.pack(side="left")
        self.seed_btn = tk.Button(bottom, text="New seed", command=self._new_seed,
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
        self.save_btn = tk.Button(bottom, text="Save world", command=self._on_save_world,
                                   bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                   activeforeground=FG, relief="flat", padx=12, pady=4,
                                   state="disabled")
        self.save_btn.pack(side="left", padx=(6, 0))
        # UI2: Step -- runs exactly one batch and re-pauses, for
        # frame-by-frame inspection. Pauses first if currently playing,
        # since "step" only means something from a stopped state.
        self.step_btn = tk.Button(bottom, text="Step", command=self._on_step,
                                   bg=PANEL_BG, fg=FG, activebackground=BORDER,
                                   activeforeground=FG, relief="flat", padx=12, pady=4)
        self.step_btn.pack(side="left", padx=(6, 0))

        # UI2: speed control -- a stepped selector (Scale in integer,
        # snap-to-level mode) over SPEED_LEVELS, DISPLAY RATE ONLY. Says so
        # right in the UI, not just in a code comment: this control changes
        # n_chains/n_samples per call and the pause between calls, never
        # n_warmup or steps_per_sample -- so it cannot change what is being
        # sampled, only how fast the same distribution scrolls by.
        speed_frame = tk.Frame(bottom, bg=BG)
        speed_frame.pack(side="left", padx=(16, 0))
        tk.Label(speed_frame, text="speed:", bg=BG, fg=DIM,
                  font=("Consolas", 8)).pack(side="left")
        # speed_label is created BEFORE the Scale's initial .set() below --
        # tk.Scale.set() can invoke -command synchronously even for a
        # programmatic change, and _on_speed_change/_refresh_speed_label
        # both write to self.speed_label, so it must already exist.
        self.speed_label = tk.Label(speed_frame, text="", bg=BG, fg=DIM,
                                      font=("Consolas", 8), justify="left")
        self.speed_scale = tk.Scale(
            speed_frame, from_=0, to=len(SPEED_LEVELS) - 1, orient="horizontal",
            resolution=1, showvalue=0, length=140, bg=BG, fg=FG,
            troughcolor=PANEL_BG, highlightthickness=0, bd=0,
            command=self._on_speed_change)
        self.speed_scale.set(self.speed_idx)
        self.speed_scale.pack(side="left", padx=(4, 4))
        self.speed_label.pack(side="left")
        self._refresh_speed_label()

        tk.Label(bottom, text=FOOTER_TEXT, bg=BG, fg=DIM,
                  font=("Consolas", 8), wraplength=900, justify="left"
                  ).pack(side="left", padx=(16, 0))

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
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w", justify="left"
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
            font=("Consolas", 8),
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
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), justify="left", anchor="w"
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
                                             font=("Consolas", 8), justify="left", anchor="w",
                                             wraplength=WORLD_DISPLAY_PX - 8)
        self.world_status_label.pack(fill="x")
        self.infeasible_label = tk.Label(wf, text="", bg=PANEL_BG, fg=BAD,
                                           font=MONO_B, justify="left", anchor="w", wraplength=380)
        self.infeasible_label.pack(fill="x", pady=(2, 0))

        # PINS ------------------------------------------------------------
        pf = tk.Frame(wf, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        pf.pack(fill="x", pady=(8, 0))
        tk.Label(pf, text="PINS  (click a cell above to add/cycle/remove one)",
                  bg=PANEL_BG, fg=ACCENT, font=("Consolas", 9, "bold"), anchor="w"
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
            tk.Label(vf, text=f"{status:<4} {g['gate']:<13} {detail}{extra_s}",
                      bg=PANEL_BG, fg=color, font=("Consolas", 8), anchor="w"
                      ).pack(fill="x")
        tk.Label(vf, text="", bg=PANEL_BG).pack()
        v = r.verification
        for key in ("task_validity", "codeword_violation_rate", "ess", "energy_tv",
                    "execution_tv", "cross_check_tv", "diversity_distinct",
                    "diversity_valid_samples", "diversity_reachable"):
            val = v.get(key, None)
            text = fmt_value(val) if key in v else "unavailable: field absent from receipt"
            fg = DIM if isinstance(val, str) else FG
            tk.Label(vf, text=f"{key}: {text}", bg=PANEL_BG, fg=fg,
                      font=("Consolas", 8), anchor="w", justify="left", wraplength=395
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
            tk.Label(sf, text=t, bg=PANEL_BG, fg=FG, font=("Consolas", 8),
                      anchor="w", justify="left", wraplength=395).pack(fill="x")
        tk.Label(sf, text=f"no beta slider: {r.mediator_count} mediator spin(s) were "
                           f"coupled at beta={r.beta_used!r}; tsu.passes.route."
                           f"assert_beta_consistent refuses any other beta for this "
                           f"model (BetaMismatchError, spec 5.3.5) -- shown fixed, "
                           f"not hidden.",
                  bg=PANEL_BG, fg=WARN, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(4, 0))
        tk.Label(sf, text=f"live sampler settings (this app, not the receipt) -- "
                           f"UNPINNED at Full speed: n_chains/call={BATCH_CHAINS}, "
                           f"n_warmup={N_WARMUP}, n_samples/call={N_SAMPLES_PER_CALL}",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(4, 0))
        tk.Label(sf, text=f"PINNED at Full speed (via simulate(clamp=...), one batch "
                           f"per pin change): n_chains/call={CLAMP_N_CHAINS}, "
                           f"n_warmup={CLAMP_N_WARMUP}, n_samples/call={CLAMP_N_SAMPLES}",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(2, 0))
        tk.Label(sf, text=f"both: steps_per_sample(thinning)={STEPS_PER_SAMPLE} -- "
                           f"n_warmup and steps_per_sample are FIXED at every speed "
                           f"(see speed control in the bottom bar): only n_chains/call "
                           f"and n_samples/call scale down below Full, which changes "
                           f"batch size/display rate, never the sampled distribution.",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(2, 0))

        # DECODED MIX ---------------------------------------------------
        mf = self.mix_panel.body
        self.mix_label = tk.Label(mf, text="unavailable: no valid sample drawn yet this session",
                                    bg=PANEL_BG, fg=DIM, font=("Consolas", 9), justify="left", anchor="w")
        self.mix_label.pack(fill="x")

        # SAMPLE LOG ------------------------------------------------------
        lgf = self.log_panel.body
        self.valid_frac_label = tk.Label(lgf, text="valid fraction: n/a (0 draws)",
                                           bg=PANEL_BG, fg=FG, font=MONO_B, anchor="w")
        self.valid_frac_label.pack(fill="x")
        tk.Label(lgf, text="Displayed worlds are drawn from p(x | valid),\n"
                            "NOT p(x): only valid draws are ever rendered,\n"
                            "so this mix and the world panel reflect the\n"
                            "conditional distribution, not the raw sampler.",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), justify="left", anchor="w"
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
        self.log_box = tk.Listbox(log_frame, bg="#111218", fg=FG, font=("Consolas", 8),
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
        try:
            report = frontier_mod.build_frontier_report(r.path)
            frontier_text = frontier_mod.render_text(report)
        except Exception as exc:  # never crash the app over a display panel
            frontier_text = f"unavailable: frontier report failed: {exc}"
        frontier_frame = tk.Frame(ff, bg=PANEL_BG)
        frontier_frame.pack(fill="both", expand=True)
        fsb = tk.Scrollbar(frontier_frame)
        fsb.pack(side="right", fill="y")
        frontier_box = tk.Text(frontier_frame, bg="#111218", fg=FG, width=38,
                                font=("Consolas", 8), wrap="word", relief="flat",
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
                  fg=FG, font=("Consolas", 8), anchor="w").pack(fill="x")
        tk.Label(gf, text=regime_line, bg=PANEL_BG, fg=FG, font=MONO_B,
                  anchor="w", wraplength=260, justify="left").pack(fill="x", pady=(2, 4))
        tk.Label(gf, text=ONSAGER_ASSUMPTION_NOTE, bg=PANEL_BG, fg=WARN,
                  font=("Consolas", 8), anchor="w", justify="left",
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
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=260).pack(fill="x", pady=(0, 8))

        tk.Label(gf, text="ENERGY TRACE (over sweeps)", bg=PANEL_BG, fg=ACCENT,
                  font=("Consolas", 8, "bold"), anchor="w").pack(fill="x")
        self.energy_canvas = tk.Canvas(gf, width=260, height=110, bg="#111218",
                                         highlightthickness=0)
        self.energy_canvas.pack(fill="x", pady=(2, 8))

        tk.Label(gf, text="VALID FRACTION TRACE (over the session)", bg=PANEL_BG,
                  fg=ACCENT, font=("Consolas", 8, "bold"), anchor="w").pack(fill="x")
        self.valid_frac_canvas = tk.Canvas(gf, width=260, height=110, bg="#111218",
                                             highlightthickness=0)
        self.valid_frac_canvas.pack(fill="x", pady=(2, 4))
        tk.Label(gf, text="Both axes are labelled with their live min/max --\n"
                            "an unlabelled sparkline is decoration, not\n"
                            "instrumentation.",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left").pack(fill="x")
        self._draw_trace(self.energy_canvas, self.energy_trace, "sweep", "energy")
        self._draw_trace(self.valid_frac_canvas, self.valid_frac_trace,
                         "draw", "valid %")

        # Task 6: SCOPE panel's own initial draw (all four sub-plots start
        # on their "(no data yet)" / "(no draws yet)" placeholder, same
        # honesty convention as the two traces just above).
        self._refresh_scope_panel()

    def _draw_trace(self, canvas: tk.Canvas, trace: "Trace", xlabel: str,
                    ylabel: str) -> None:
        """Redraw one line plot from `trace`'s current contents. Axis
        min/max are read from `trace.bounds()` and drawn as text -- never
        skipped, per the brief's own instrumentation-not-decoration rule."""
        canvas.delete("all")
        w = int(canvas["width"]) or 260
        h = int(canvas["height"]) or 110
        pad_l, pad_r, pad_t, pad_b = 44, 8, 8, 16
        bounds = trace.bounds()
        if bounds is None or len(trace) < 2:
            canvas.create_text(w / 2, h / 2, text="(no data yet)", fill=DIM,
                                font=("Consolas", 8))
            return
        xmin, xmax, ymin, ymax = bounds
        xspan = (xmax - xmin) or 1.0
        yspan = (ymax - ymin) or (abs(ymax) or 1.0)
        px = lambda x: pad_l + (x - xmin) / xspan * (w - pad_l - pad_r)
        py = lambda y: h - pad_b - (y - ymin) / yspan * (h - pad_t - pad_b)
        pts = []
        for x, y in zip(trace.xs, trace.ys):
            pts.extend((px(x), py(y)))
        canvas.create_line(*pts, fill=ACCENT, width=1)
        canvas.create_text(pad_l, pad_t, text=f"{ymax:.4g}", fill=DIM,
                            font=("Consolas", 7), anchor="nw")
        canvas.create_text(pad_l, h - pad_b, text=f"{ymin:.4g}", fill=DIM,
                            font=("Consolas", 7), anchor="sw")
        canvas.create_text(pad_l, h - 4, text=f"{xlabel}={xmin:.0f}", fill=DIM,
                            font=("Consolas", 7), anchor="sw")
        canvas.create_text(w - pad_r, h - 4, text=f"{xmax:.0f}", fill=DIM,
                            font=("Consolas", 7), anchor="se")
        canvas.create_text(w - pad_r, pad_t, text=ylabel, fill=DIM,
                            font=("Consolas", 7), anchor="ne")

    def _refresh_scope_panel(self) -> None:
        """Task 6: rebuild all four SCOPE sub-plots from this session's own
        live history (energy_trace, magnetization_trace, raw_draws) and
        blit them onto their canvases. Each render_* function already
        handles "not enough data yet" honestly; this method just calls
        them, converts to PhotoImage, and keeps a reference (Tk drops a
        PhotoImage with no surviving Python reference -- same pattern
        _update_world already uses for world_photo)."""
        energy_ys = list(self.energy_trace.ys)
        acf_img, acf_caption = render_acf_plot(SCOPE_PLOT_W, SCOPE_PLOT_H, energy_ys)
        self.acf_photo = ImageTk.PhotoImage(acf_img)
        self.acf_canvas.itemconfig("plot", image=self.acf_photo)
        self.acf_caption.config(text=acf_caption)

        mag_img = render_line_plot(
            SCOPE_PLOT_W, SCOPE_PLOT_H, list(self.magnetization_trace.xs),
            list(self.magnetization_trace.ys), "draw", "M", y_range=(-1.0, 1.0))
        self.mag_photo = ImageTk.PhotoImage(mag_img)
        self.mag_canvas.itemconfig("plot", image=self.mag_photo)
        self.mag_caption.config(
            text="Mean spin per draw (s=2*occupancy-1), fixed axis [-1, 1] "
                 "-- the order parameter's own physical bounds.")

        hist_img = render_energy_histogram_plot(SCOPE_PLOT_W, SCOPE_PLOT_H, energy_ys)
        self.hist_photo = ImageTk.PhotoImage(hist_img)
        self.hist_canvas.itemconfig("plot", image=self.hist_photo)
        self.hist_caption.config(
            text=f"Distribution of the energy trace's own {len(energy_ys)} "
                 f"value(s) so far this session -- the trace itself only "
                 f"samples one point of this at a time.")

        sigmoid_img, sigmoid_caption = render_sigmoid_plot(
            SCOPE_PLOT_W, SCOPE_PLOT_H, list(self.raw_draws), self.receipt.im)
        self.sigmoid_photo = ImageTk.PhotoImage(sigmoid_img)
        self.sigmoid_canvas.itemconfig("plot", image=self.sigmoid_photo)
        self.sigmoid_caption.config(text=sigmoid_caption)

    def _swatch(self, master, color, text):
        row = tk.Frame(master, bg=PANEL_BG)
        row.pack(fill="x", pady=1)
        tk.Canvas(row, width=12, height=12, bg=color, highlightthickness=0).pack(side="left")
        tk.Label(row, text=" " + text, bg=PANEL_BG, fg=DIM, font=("Consolas", 8)).pack(side="left")

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
            # second-guessed or smoothed over.
            self.batch_infeasible = msg["infeasible"]
            self.batch_reason = msg["reason"]
            self._refresh_world_status()
            return

        self.total_draws += 1
        raw = msg["raw"]
        idx = msg["draw_idx"]
        self._update_lattice(raw)

        # B2: energy trace over EVERY draw (valid or not -- mixing is a
        # property of the raw chain, the same convention the compiler's own
        # verify pass uses, see energy_of_draw's own docstring), plotted
        # against this app's own running draw counter as "sweep".
        self.energy_trace.append(self.total_draws, energy_of_draw(self.receipt.im, raw))

        # Task 6: magnetization trace and the raw-draw pool for
        # local_field_response -- SAME "every draw, valid or not" convention
        # as the energy trace just above (mixing/the order parameter/the
        # local field are all properties of the raw chain, not of the
        # conditional-valid subset).
        self.magnetization_trace.append(self.total_draws, magnetization([raw])[0])
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
            self._update_world(msg["image"])
            self._update_mix(msg["mix"])
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
        self.valid_frac_trace.append(self.total_draws, frac)
        self._draw_trace(self.energy_canvas, self.energy_trace, "sweep", "energy")
        self._draw_trace(self.valid_frac_canvas, self.valid_frac_trace,
                         "draw", "valid %")
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

    def _redraw_pin_markers(self):
        self.world_canvas.delete("pin")
        for (x, y), v in self.clamp.items():
            x0, y0 = x * WORLD_CELL_PX, y * WORLD_CELL_PX
            color = self._pin_color(v)
            self.world_canvas.create_rectangle(
                x0 + 2, y0 + 2, x0 + WORLD_CELL_PX - 2, y0 + WORLD_CELL_PX - 2,
                outline=color, width=3, tags="pin")
            self.world_canvas.create_oval(
                x0 + 3, y0 + 3, x0 + 11, y0 + 11, fill=color, outline=FG, tags="pin")
        self.world_canvas.tag_raise("pin")

    def _refresh_pins_panel(self):
        d = self.clamp.as_dict()
        if not d:
            self.pins_list_label.config(text="(none)", fg=DIM)
        else:
            lines = [f"{name} = {TERRAIN_NAMES[v]}" for name, v in sorted(d.items())]
            self.pins_list_label.config(text="\n".join(lines), fg=FG)

    def _on_world_click(self, event):
        cell = cell_at(event.x, event.y, 0, 0, WORLD_CELL_PX,
                        WORLD_DISPLAY_PX, WORLD_DISPLAY_PX)
        if cell is None:
            return
        x, y = cell
        self.clamp.cycle(x, y)
        self._redraw_pin_markers()
        self._refresh_pins_panel()
        self._restart_sampling_for_clamp_change()

    def _on_clear_pins(self):
        if len(self.clamp) == 0:
            return
        self.clamp.clear()
        self._redraw_pin_markers()
        self._refresh_pins_panel()
        self._restart_sampling_for_clamp_change()

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
        self._draw_trace(self.energy_canvas, self.energy_trace, "sweep", "energy")
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
        self._draw_trace(self.energy_canvas, self.energy_trace, "sweep", "energy")
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
    app = LatticeApp(receipt)
    app.mainloop()


if __name__ == "__main__":
    main()
