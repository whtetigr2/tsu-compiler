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
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
RECEIPT_DIR = REPO_ROOT / "demo" / "receipts" / "small"
sys.path.insert(0, str(SRC))

from tsu.spec import load_spec  # noqa: E402
from tsu.passes.encode import encode  # noqa: E402
from tsu.simulate import reconstruct_program, _selected_encoding  # noqa: E402
from tsu.backends.thrml_backend import sample as thrml_sample  # noqa: E402

# --------------------------------------------------------------------------
# constants shared by the raw-lattice grid and the decode/render path
# --------------------------------------------------------------------------
W = H = 8                      # decoded world is an 8x8 grid
WATER, ROCK, GRASS = 0, 1, 2
TERRAIN_NAMES = {WATER: "water", ROCK: "rock", GRASS: "grass"}
TERRAIN_ORDER = (WATER, ROCK, GRASS)
PAL = np.array([[46, 92, 132], [124, 116, 106], [126, 158, 84]], float)
CELL_UP = 32                    # px per grid cell in the rendered world image (render_world.py uses 64; halved here so a redraw fits inside one animation tick)

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
    def __init__(self, receipt: Receipt, out_q: "queue.Queue[dict]", seed_base: int,
                 start_paused: bool = False):
        super().__init__(daemon=True)
        self.receipt = receipt
        self.q = out_q
        self.pause = threading.Event()
        self.stop_evt = threading.Event()
        if start_paused:
            self.pause.set()
        self.seed_base = seed_base
        self.draw_counter = 0

    def _put(self, msg: dict) -> None:
        while not self.stop_evt.is_set():
            try:
                self.q.put(msg, timeout=0.2)
                return
            except queue.Full:
                continue

    def run(self) -> None:
        while not self.stop_evt.is_set():
            if self.pause.is_set():
                time.sleep(0.08)
                continue
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
                return
            for row in rows:
                if self.stop_evt.is_set():
                    return
                self.draw_counter += 1
                self._put(classify_draw(self.receipt, row, seed))
                if self.pause.is_set() or self.stop_evt.is_set():
                    break


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
        self.world_photo = None  # keep a reference; Tk drops PhotoImages with none
        self.log_lines: deque = deque(maxlen=400)

        self._build_layout()
        self._populate_static_panels()

        self.worker = SampleWorker(receipt, self.in_q, seed_base=random.randint(0, 2**31 - 1))
        self.worker.start()

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
        right.grid_rowconfigure(2, minsize=110)
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
        lf = self.lattice_panel.body
        cols = 16
        rows = -(-r.n_spins // cols)
        cw = 18
        self.lattice_canvas = tk.Canvas(lf, width=cols * cw, height=rows * cw,
                                          bg=PANEL_BG, highlightthickness=0)
        self.lattice_canvas.pack(pady=(2, 6))
        self.lattice_rects = []
        for i in range(r.n_spins):
            row, col = divmod(i, cols)
            x0, y0 = col * cw + 1, row * cw + 1
            rect = self.lattice_canvas.create_rectangle(
                x0, y0, x0 + cw - 2, y0 + cw - 2, outline="", fill=WORLD_OFF)
            self.lattice_rects.append(rect)
        legend = tk.Frame(lf, bg=PANEL_BG)
        legend.pack(fill="x")
        self._swatch(legend, WORLD_ON, f"world spin (0-{len(r.world_idx) - 1}), lit = 1")
        self._swatch(legend, MEDIATOR_ON, f"mediator spin ({r.mediator_idx[0]}-{r.mediator_idx[-1]}), lit = 1")
        tk.Label(lf, text="Raw physical spin state of the compiled program --\n"
                           "this is NOT the decoded world; a spin here has no\n"
                           "terrain meaning until enc.decode succeeds.",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), justify="left", anchor="w"
                  ).pack(fill="x", pady=(6, 0))

        # DECODED WORLD ---------------------------------------------------
        wf = self.world_panel.body
        self.world_image_label = tk.Label(wf, bg=PANEL_BG, text="(no valid sample drawn yet this session)",
                                            fg=DIM, font=MONO)
        self.world_image_label.pack(pady=(4, 6))
        self.world_status_label = tk.Label(wf, text="", bg=PANEL_BG, fg=DIM,
                                             font=("Consolas", 8), justify="left", anchor="w")
        self.world_status_label.pack(fill="x")

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
        tk.Label(sf, text=f"live sampler settings (this app, not the receipt): "
                           f"n_chains/call={BATCH_CHAINS}, n_warmup={N_WARMUP}, "
                           f"steps_per_sample(thinning)={STEPS_PER_SAMPLE}, "
                           f"n_samples/call={N_SAMPLES_PER_CALL}",
                  bg=PANEL_BG, fg=DIM, font=("Consolas", 8), anchor="w",
                  justify="left", wraplength=395).pack(fill="x", pady=(4, 0))

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
        self.log_box = tk.Listbox(log_frame, bg="#111218", fg=FG, font=("Consolas", 8),
                                    highlightthickness=0, relief="flat",
                                    yscrollcommand=sb.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        sb.config(command=self.log_box.yview)

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

        self.total_draws += 1
        raw = msg["raw"]
        self._update_lattice(raw)

        if kind == "valid":
            self.valid_count += 1
            self.last_valid_grid = msg["grid"]
            self.last_valid_mix = msg["mix"]
            self._update_world(msg["image"])
            self._update_mix(msg["mix"])
            self._log(f"seed={msg['seed']:<10} valid")
        elif kind == "contract-fail":
            self.contract_fail_count += 1
            viol = "; ".join(msg.get("violations", ())) or "contract violated"
            self._log(f"seed={msg['seed']:<10} contract-fail  ({viol})")
        else:  # non-codeword
            self.noncodeword_count += 1
            self._log(f"seed={msg['seed']:<10} non-codeword")

        frac = 100.0 * self.valid_count / self.total_draws if self.total_draws else 0.0
        self.valid_frac_label.config(
            text=f"valid fraction: {frac:.1f}%  "
                 f"(valid={self.valid_count} contract-fail={self.contract_fail_count} "
                 f"non-codeword={self.noncodeword_count} total={self.total_draws})")

    def _update_lattice(self, raw):
        med = set(self.receipt.mediator_idx)
        for i, (rect, bit) in enumerate(zip(self.lattice_rects, raw)):
            if i in med:
                color = MEDIATOR_ON if bit else MEDIATOR_OFF
            else:
                color = WORLD_ON if bit else WORLD_OFF
            self.lattice_canvas.itemconfig(rect, fill=color)

    def _update_world(self, image: Image.Image):
        disp = image.resize((320, 320), Image.LANCZOS)
        self.world_photo = ImageTk.PhotoImage(disp)
        self.world_image_label.config(image=self.world_photo, text="")
        self.world_status_label.config(
            text=f"last valid sample -- draw #{self.total_draws} "
                 f"(valid {self.valid_count}/{self.total_draws} draws so far this session)")

    def _update_mix(self, mix: dict):
        lines = [f"{TERRAIN_NAMES[k]:<6}: {mix[TERRAIN_NAMES[k]]:5.1f}%" for k in TERRAIN_ORDER]
        self.mix_label.config(text="\n".join(lines) +
                               "\n\n(computed live from the current last-valid\n"
                               "8x8 decoded grid; p(x | valid), see note below)",
                               fg=FG)

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

    def _new_seed(self):
        old = self.worker
        old.stop_evt.set()
        # drain whatever is left in the queue so stale draws from the old
        # seed never get attributed to the new run
        try:
            while True:
                self.in_q.get_nowait()
        except queue.Empty:
            pass

        self.total_draws = 0
        self.valid_count = 0
        self.contract_fail_count = 0
        self.noncodeword_count = 0
        self.last_valid_grid = None
        self.last_valid_mix = None
        self.world_photo = None
        self.world_image_label.config(image="", text="(re-sampling from a new seed...)")
        self.world_status_label.config(text="")
        self.mix_label.config(text="unavailable: no valid sample drawn yet this session", fg=DIM)
        self.log_box.delete(0, "end")
        self.valid_frac_label.config(text="valid fraction: n/a (0 draws)")

        new_seed_base = random.randint(0, 2**31 - 1)
        self.worker = SampleWorker(self.receipt, self.in_q, seed_base=new_seed_base,
                                    start_paused=self.paused)
        self.worker.start()
        self._log(f"--- new seed base={new_seed_base} ---")

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
