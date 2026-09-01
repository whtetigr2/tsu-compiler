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
import queue
import random
import sys
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageTk
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
    same `classify_draw` the unclamped path uses. Larger batch (verified:
    n_chains=6, n_samples=30, n_warmup=600 -> ~1.4s) because a pin change is
    a deliberate, infrequent action, not a per-tick refresh -- the app can
    afford to answer "does this clamp even have a valid world" from a solid
    batch rather than trickling one draw at a time.
    """

    def __init__(self, receipt: Receipt, out_q: "queue.Queue[dict]", seed_base: int,
                 clamp: dict[str, int] | None = None, start_paused: bool = False):
        super().__init__(daemon=True)
        self.receipt = receipt
        self.q = out_q
        self.pause = threading.Event()
        self.stop_evt = threading.Event()
        if start_paused:
            self.pause.set()
        self.seed_base = seed_base
        self.clamp = dict(clamp) if clamp else None
        self.draw_counter = 0
        self.batch_counter = 0

    def _put(self, msg: dict) -> None:
        while not self.stop_evt.is_set():
            try:
                self.q.put(msg, timeout=0.2)
                return
            except queue.Full:
                continue

    def _classify_and_push(self, row, seed) -> dict:
        self.draw_counter += 1
        d = classify_draw(self.receipt, row, seed)
        d["draw_idx"] = self.draw_counter
        self._put(d)
        return d

    def run(self) -> None:
        while not self.stop_evt.is_set():
            if self.pause.is_set():
                time.sleep(0.08)
                continue
            if self.clamp:
                self._run_clamped_batch()
            else:
                self._run_unclamped_tick()

    def _run_unclamped_tick(self) -> None:
        seed = self.seed_base + self.draw_counter
        try:
            rows = thrml_sample(self.receipt.sampling_program,
                                 n_chains=BATCH_CHAINS,
                                 n_samples=N_SAMPLES_PER_CALL,
                                 n_warmup=N_WARMUP,
                                 steps_per_sample=STEPS_PER_SAMPLE,
                                 seed=seed)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self._put({"kind": "error", "message": str(exc)})
            self.stop_evt.set()
            return
        for row in rows:
            if self.stop_evt.is_set():
                return
            self._classify_and_push(row, seed)
            if self.pause.is_set() or self.stop_evt.is_set():
                break

    def _run_clamped_batch(self) -> None:
        from tsu.simulate import simulate  # local: keeps this app's only
        # entry point into clamping right here, next to the docstring above
        self.batch_counter += 1
        seed = self.seed_base + self.batch_counter
        try:
            _path, got, _im = simulate(str(self.receipt.path),
                                        n_chains=CLAMP_N_CHAINS,
                                        n_samples=CLAMP_N_SAMPLES,
                                        n_warmup=CLAMP_N_WARMUP,
                                        steps_per_sample=STEPS_PER_SAMPLE,
                                        seed=seed, clamp=self.clamp)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self._put({"kind": "error", "message": str(exc)})
            self.stop_evt.set()
            return
        draws = []
        for row in got:
            if self.stop_evt.is_set():
                return
            draws.append(self._classify_and_push(row, seed))
            if self.pause.is_set() or self.stop_evt.is_set():
                break
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
        self.geometry("1400x900")
        self.configure(bg=BG)

        self.in_q: "queue.Queue[dict]" = queue.Queue(maxsize=64)
        self.paused = False
        self.total_draws = 0
        self.valid_count = 0
        self.contract_fail_count = 0
        self.noncodeword_count = 0
        self.last_valid_grid = None
        self.last_valid_mix = None
        self.last_valid_seed = None  # seed of the draw last_valid_grid came from
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

        self._build_layout()
        self._populate_static_panels()

        self._start_worker(random.randint(0, 2**31 - 1))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll_queue)

    # -- layout ------------------------------------------------------
    def _build_layout(self):
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=8, pady=8)
        for c, weight in enumerate((0, 1, 1, 0)):
            content.grid_columnconfigure(c, weight=weight)
        content.grid_rowconfigure(0, weight=1)

        self.pipeline_panel = Panel(content, "PIPELINE  (compiled once, at load)")
        self.pipeline_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.lattice_panel = Panel(content, "LIVE LATTICE  (raw physical spin state)")
        self.lattice_panel.grid(row=0, column=1, sticky="nsew", padx=6)

        self.world_panel = Panel(content, "DECODED WORLD  (last valid sample)")
        self.world_panel.grid(row=0, column=2, sticky="nsew", padx=6)

        right = tk.Frame(content, bg=BG, width=430)
        right.grid(row=0, column=3, sticky="nsew", padx=(6, 0))
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
        tk.Label(bottom, text=FOOTER_TEXT, bg=BG, fg=DIM,
                  font=("Consolas", 8), wraplength=1000, justify="left"
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
        # below it. `cw` (px/spin) is also bumped up from 18 so the grid
        # itself reads less like a postage stamp.
        lf = self.lattice_panel.body
        inner = tk.Frame(lf, bg=PANEL_BG)
        inner.pack(expand=True)
        cols = 16
        rows = -(-r.n_spins // cols)
        cw = 24
        self.lattice_canvas = tk.Canvas(inner, width=cols * cw, height=rows * cw,
                                          bg=PANEL_BG, highlightthickness=0)
        self.lattice_canvas.pack(pady=(2, 6))
        self.lattice_rects = []
        for i in range(r.n_spins):
            row, col = divmod(i, cols)
            x0, y0 = col * cw + 1, row * cw + 1
            rect = self.lattice_canvas.create_rectangle(
                x0, y0, x0 + cw - 2, y0 + cw - 2, outline="", fill=WORLD_OFF)
            self.lattice_rects.append(rect)
        legend = tk.Frame(inner, bg=PANEL_BG)
        legend.pack(fill="x")
        self._swatch(legend, WORLD_ON, f"world spin (0-{len(r.world_idx) - 1}), lit = 1")
        self._swatch(legend, MEDIATOR_ON, f"mediator spin ({r.mediator_idx[0]}-{r.mediator_idx[-1]}), lit = 1")
        tk.Label(inner, text="Raw physical spin state of the compiled program --\n"
                           "this is NOT the decoded world; a spin here has no\n"
                           "terrain meaning until enc.decode succeeds.",
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
                           f"UNPINNED: n_chains/call={BATCH_CHAINS}, n_warmup={N_WARMUP}, "
                           f"n_samples/call={N_SAMPLES_PER_CALL}",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(4, 0))
        tk.Label(sf, text=f"PINNED (via simulate(clamp=...), one batch per pin "
                           f"change): n_chains/call={CLAMP_N_CHAINS}, "
                           f"n_warmup={CLAMP_N_WARMUP}, n_samples/call={CLAMP_N_SAMPLES}",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(2, 0))
        tk.Label(sf, text=f"both: steps_per_sample(thinning)={STEPS_PER_SAMPLE}",
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

        if kind == "valid":
            self.valid_count += 1
            self.last_valid_grid = msg["grid"]
            self.last_valid_mix = msg["mix"]
            self.last_valid_seed = msg["seed"]
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
        self._refresh_world_status()

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
        if clamp:
            sampler_params = {
                "n_chains": CLAMP_N_CHAINS, "n_samples": CLAMP_N_SAMPLES,
                "n_warmup": CLAMP_N_WARMUP, "steps_per_sample": STEPS_PER_SAMPLE,
            }
        else:
            sampler_params = {
                "n_chains": BATCH_CHAINS, "n_samples_per_call": N_SAMPLES_PER_CALL,
                "n_warmup": N_WARMUP, "steps_per_sample": STEPS_PER_SAMPLE,
            }
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
                                    clamp=clamp, start_paused=self.paused)
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
