"""Task 9: the "What is this?" explainer window -- a newcomer's tour of
LATTICE, in a native `tk.Toplevel`.

Split the same way demo/scope.py splits from lattice_app.py: the CONTENT
(explainer_sections/clipboard_text/explainer_diagrams) is pure and headlessly
tested (tests/test_explainer.py, no Tk import required), so a reader can
verify every section body is real prose and Copy-to-clipboard can never
silently omit one, without ever constructing a window. ExplainerWindow, at
the bottom of this file, is the only Tk-dependent thing here -- it lays the
already-computed sections/diagrams out, it does not invent any of its own.

Six topics, the brief's own list, in the brief's own order: what a p-bit is;
what sampling from an energy landscape means and how it differs from
computing an answer; why temperature matters; what mediator spins are and
why they exist; what the hardware gates check; and why some readouts
honestly say `unavailable`. That LAST one is flagged in the task as the one
that matters most -- a visitor is more likely to read "unavailable" as a bug
than as the tool declining to guess, so its own section says so explicitly,
not just defines the word.

Diagrams are STATIC PNGs (numpy/PIL), never a live render of this session's
own data -- the tokens spec's own buildability note ("explainer diagrams
(static, not live)"). They illustrate the CONCEPT, not a measurement of
anything: no diagram here reads a receipt, a sample, or any other live
state, so none of it can go stale or drift from what's actually on screen
elsewhere in the app.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import theme  # noqa: E402 -- Task 0: theme foundation, see theme.py's own docstring

DIAGRAM_SIZE = (200, 130)


def _hexrgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _font(size: int = 11) -> "ImageFont.ImageFont":
    """PIL's own built-in bitmap default font -- never a web font, never a
    guess at a system .ttf path that might not exist on this machine. Small
    diagram labels don't need the app's own Cascadia/Consolas stack; they
    only need to render reliably headless (tests import this module with no
    Tk root and no display)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: load_default() takes no size arg
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Diagrams -- one static PIL image per section, same order as
# explainer_sections() below. Each is a small, abstract illustration of the
# CONCEPT (never live data): p-bit flicker, a sampled energy landscape vs a
# single computed answer, a cold->gold->hot ramp, a mediator bridging two
# world spins, four small gate gauges, and an honest "unavailable" readout
# next to a crossed-out fabricated one.
# ---------------------------------------------------------------------------

def _new_canvas() -> tuple[Image.Image, "ImageDraw.ImageDraw"]:
    img = Image.new("RGB", DIAGRAM_SIZE, _hexrgb(theme.INSET))
    return img, ImageDraw.Draw(img)


def _diagram_pbit() -> Image.Image:
    img, d = _new_canvas()
    w, h = DIAGRAM_SIZE
    cx, cy, r = w // 2, 46, 26
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_hexrgb(theme.GHOST), width=1)
    d.ellipse([cx - r + 6, cy - r + 6, cx + r - 6, cy + r - 6],
              fill=_hexrgb(theme.GOLD_GHOST), outline=_hexrgb(theme.GOLD), width=1)
    d.text((cx - 4, cy - 7), "s", fill=_hexrgb(theme.CREAM), font=_font(12))
    # a fluctuating trace beneath, alternating 0/1 -- the p-bit's own state
    # over time, unlike a fixed classical bit.
    y0, y1 = 92, 112
    xs = np.linspace(16, w - 16, 24)
    rng = np.random.default_rng(7)
    bits = (rng.random(len(xs)) < 0.65).astype(int)
    prev = None
    for x, b in zip(xs, bits):
        y = y0 if b else y1
        # I1 (final review): a world p-bit in state 0 is GOLD_GHOST
        # everywhere else in this app (lattice_app.py's own WORLD_OFF,
        # asserted != theme.BLUE by test) -- BLUE is reserved for COLD
        # mediator spins, never a decorative "off" colour. This diagram
        # used to draw state 0 in BLUE, teaching a newcomer the exact
        # on/off-colour inversion Task 0 existed to fix, directly
        # contradicting the "dim" wording two lines below.
        color = _hexrgb(theme.GOLD) if b else _hexrgb(theme.GOLD_GHOST)
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=color)
        if prev is not None:
            d.line([prev, (x, y)], fill=_hexrgb(theme.GHOST), width=1)
        prev = (x, y)
    d.text((10, 8), "P-BIT: FLUCTUATES 0/1", fill=_hexrgb(theme.GOLD), font=_font(10))
    return img


def _diagram_sampling() -> Image.Image:
    img, d = _new_canvas()
    w, h = DIAGRAM_SIZE
    # an energy landscape (a valley) ...
    xs = np.linspace(10, w - 10, 80)
    base = 90
    ys = base - 40 * np.exp(-((xs - w * 0.42) ** 2) / (2 * 22 ** 2)) \
             - 22 * np.exp(-((xs - w * 0.78) ** 2) / (2 * 14 ** 2))
    pts = list(zip(xs.tolist(), ys.tolist()))
    d.line(pts, fill=_hexrgb(theme.GHOST), width=2)
    # ... with drawn SAMPLES clustered near the low points, not one answer.
    rng = np.random.default_rng(3)
    centers = [(w * 0.42, 0.7), (w * 0.78, 0.3)]
    for cx_frac, weight in centers:
        n = int(14 * weight) + 3
        for _ in range(n):
            x = cx_frac + rng.normal(0, 9)
            idx = int(np.clip((x - 10) / (w - 20) * (len(xs) - 1), 0, len(xs) - 1))
            y = ys[idx] - rng.uniform(2, 9)
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=_hexrgb(theme.GOLD))
    d.text((8, 8), "SAMPLE: MANY DRAWS ~ e^-bE", fill=_hexrgb(theme.GOLD), font=_font(10))
    d.text((8, h - 18), "(not: compute ONE answer)", fill=_hexrgb(theme.CREAM_DIM), font=_font(9))
    return img


def _diagram_temperature() -> Image.Image:
    img, d = _new_canvas()
    w, h = DIAGRAM_SIZE
    top, bot = 30, 54
    cold = np.array(_hexrgb(theme.BLUE_DEEP), dtype=float)
    gold = np.array(_hexrgb(theme.GOLD), dtype=float)
    hot = np.array(_hexrgb(theme.ORANGE), dtype=float)
    mid = 0.55
    for x in range(10, w - 10):
        frac = (x - 10) / (w - 20)
        if frac <= mid:
            t = frac / mid
            c = cold * (1 - t) + gold * t
        else:
            t = (frac - mid) / (1 - mid)
            c = gold * (1 - t) + hot * t
        d.line([(x, top), (x, bot)], fill=tuple(c.astype(int)))
    d.rectangle([10, top, w - 10, bot], outline=_hexrgb(theme.GOLD_DIM), width=1)
    d.text((8, 8), "COLD -> USEFUL WINDOW -> HOT", fill=_hexrgb(theme.GOLD), font=_font(9))
    d.text((8, bot + 8), "low T: frozen   high T: disordered",
           fill=_hexrgb(theme.CREAM_DIM), font=_font(9))
    d.text((8, bot + 24), "beta = 1/T", fill=_hexrgb(theme.CREAM_DIM), font=_font(9))
    return img


def _diagram_mediators() -> Image.Image:
    img, d = _new_canvas()
    w, h = DIAGRAM_SIZE
    a, b, m = (34, 70), (w - 34, 70), (w // 2, 30)
    for p, q in ((a, m), (m, b)):
        d.line([p, q], fill=_hexrgb(theme.RULE), width=1)
    d.line([a, b], fill=_hexrgb(theme.RULE), width=1, joint=None)
    for p, label, color in ((a, "world", theme.GOLD), (b, "world", theme.GOLD)):
        d.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10],
                  fill=_hexrgb(color), outline=_hexrgb(theme.CREAM), width=1)
    d.ellipse([m[0] - 10, m[1] - 10, m[0] + 10, m[1] + 10],
              fill=_hexrgb(theme.BLUE), outline=_hexrgb(theme.BLUE_LIT), width=1)
    d.text((a[0] - 16, a[1] + 14), "gold", fill=_hexrgb(theme.CREAM_DIM), font=_font(9))
    d.text((b[0] - 16, b[1] + 14), "gold", fill=_hexrgb(theme.CREAM_DIM), font=_font(9))
    d.text((m[0] - 20, m[1] - 26), "mediator (cold)", fill=_hexrgb(theme.BLUE_LIT), font=_font(9))
    d.text((8, h - 20), "hidden helper, not terrain", fill=_hexrgb(theme.CREAM_DIM), font=_font(9))
    return img


def _diagram_gates() -> Image.Image:
    img, d = _new_canvas()
    w, h = DIAGRAM_SIZE
    labels = ("DEGREE", "COUPLING", "FIELD", "NODES")
    fracs = (0.56, 0.42, 0.27, 0.01)
    row_h = 22
    for i, (label, frac) in enumerate(zip(labels, fracs)):
        y0 = 12 + i * row_h
        y1 = y0 + 10
        d.rectangle([70, y0, w - 12, y1], outline=_hexrgb(theme.GOLD_GHOST), width=1)
        fw = int((w - 12 - 70) * frac)
        if fw > 0:
            d.rectangle([70, y0, 70 + fw, y1], fill=_hexrgb(theme.GOLD_DIM))
        d.text((6, y0 - 1), label, fill=_hexrgb(theme.CREAM_DIM), font=_font(8))
    d.text((8, h - 14), "gate checks pass -> receipt", fill=_hexrgb(theme.GOLD), font=_font(9))
    return img


def _diagram_unavailable() -> Image.Image:
    img, d = _new_canvas()
    w, h = DIAGRAM_SIZE
    # a fabricated number, crossed out -- what this app refuses to do ...
    d.text((16, 20), "42", fill=_hexrgb(theme.RED_HOT), font=_font(22))
    d.line([(14, 26), (48, 40)], fill=_hexrgb(theme.RED), width=2)
    d.line([(14, 40), (48, 26)], fill=_hexrgb(theme.RED), width=2)
    d.text((16, 46), "(fabricated)", fill=_hexrgb(theme.CREAM_DIM), font=_font(8))
    # ... versus the honest label this app prints instead.
    box_y = 78
    d.rectangle([12, box_y, w - 12, box_y + 34], outline=_hexrgb(theme.GOLD_GHOST), width=1)
    d.text((18, box_y + 6), "unavailable:", fill=_hexrgb(theme.STATUS_WARN), font=_font(10))
    d.text((18, box_y + 20), "reason stated here", fill=_hexrgb(theme.CREAM_DIM), font=_font(9))
    d.text((8, 8), "DECLINE TO GUESS", fill=_hexrgb(theme.GOLD), font=_font(10))
    return img


_DIAGRAM_FUNCS = (
    _diagram_pbit,
    _diagram_sampling,
    _diagram_temperature,
    _diagram_mediators,
    _diagram_gates,
    _diagram_unavailable,
)


def explainer_diagrams() -> list[Image.Image]:
    """One static PIL image per explainer_sections() entry, same order,
    same size (DIAGRAM_SIZE) -- built fresh each call (cheap: a handful of
    PIL primitives, no numpy heavy lifting) rather than cached at import
    time, so importing this module has no side effect beyond defining
    functions (same "safe to import headlessly" convention lattice_app.py
    itself documents)."""
    return [f() for f in _DIAGRAM_FUNCS]


# ---------------------------------------------------------------------------
# Content -- the six sections, the brief's own topics in the brief's own
# order. Every body is real prose (no placeholder), and every body is
# checked non-empty by tests/test_explainer.py before this module is
# trusted for anything else.
# ---------------------------------------------------------------------------

def explainer_sections() -> list[tuple[str, str]]:
    return [
        ("01  WHAT IS A P-BIT",
         "A p-bit (probabilistic bit) is a physical bit that does not sit "
         "still at 0 or 1 the way a classical bit does. Left alone, it "
         "flips back and forth, spending a FRACTION of its time in each "
         "state -- a fraction set by its own local energy, not by a "
         "program counter. LATTICE's compiled program is 192 p-bits: 128 "
         "GOLD world spins (the map you see decoded) and 64 COLD-BLUE "
         "mediator spins (hidden helpers -- see section 04). Every p-bit "
         "on screen is either lit gold (state 1) or dim (state 0) at the "
         "instant it was last sampled, not a fixed value."),

        ("02  SAMPLING, NOT COMPUTING",
         "LATTICE does not calculate a single correct map the way a normal "
         "program calculates a single correct sum. It compiles your "
         "workload into an ENERGY LANDSCAPE E(x) over every possible "
         "world x, then draws worlds from the probability distribution "
         "p(x) proportional to exp(-beta * E(x)) -- low-energy worlds are "
         "more likely, but never certain. Sampling means every world you "
         "see in DECODED WORLD is one draw from that distribution, not "
         "'the answer'. Run it again and you get a DIFFERENT valid draw. "
         "This is also why roughly THREE QUARTERS of raw draws are "
         "discarded (this receipt's own task_validity is ~0.25 -- see "
         "verification.json). The dominant reason is the workload's own "
         "TASK CONTRACT: most discarded draws ARE legal codewords that "
         "simply violate one of the workload's rules. An outright illegal, "
         "non-codeword draw is a small minority of what's discarded (this "
         "receipt's own codeword_violation_rate is ~1%). What's shown is "
         "drawn from p(x | valid), never the raw, unfiltered chain -- the "
         "SAMPLE LOG panel says this in plain words too."),

        ("03  WHY TEMPERATURE MATTERS",
         "beta (the compiled program's own inverse temperature) controls "
         "how sharply the distribution favours low-energy worlds. High "
         "temperature (low beta): the landscape barely matters, draws are "
         "close to random and disordered. Low temperature (high beta): "
         "the sampler freezes onto the very lowest-energy worlds and "
         "stops exploring alternatives. The USEFUL WINDOW sits between "
         "the two extremes -- structured enough to be meaningful, loose "
         "enough to still be sampling rather than stuck. This receipt's "
         "192 p-bits, including its 64 mediator spins, were compiled with "
         "their couplings baked in at ONE specific beta -- so the "
         "temperature control you see is FIXED, not merely defaulted: "
         "sampling this exact program at a different beta would silently "
         "invalidate the energy it encodes (tsu_compiler.passes.route's own "
         "beta-consistency gate refuses it), so the app shows FIXED and "
         "states why rather than offering a control that would just error."),

        ("04  MEDIATOR SPINS",
         "The hardware graph LATTICE targets only allows each p-bit a "
         "limited number of direct couplings (the DEGREE gate -- see "
         "section 05). A real workload often needs more connections than "
         "that between its own world spins. The compiler's own routing "
         "pass inserts MEDIATOR spins -- extra, hidden p-bits that carry "
         "a coupling through an intermediate hop the hardware graph can "
         "support, the way a relay carries a signal one node can't reach "
         "directly. Mediators are COLD-BLUE everywhere in this app "
         "(never decorative blue) because that is literally what they "
         "are: frozen helpers, not terrain. They are never decoded into "
         "the world you see, and they carry no meaning of their own -- "
         "only the couplings they were inserted to route."),

        ("05  WHAT THE HARDWARE GATES CHECK",
         "Before a compiled program is accepted, four gates check it "
         "against hardware limits: DEGREE (how many direct couplings any "
         "one p-bit has), COUPLING CAP |J| and FIELD CAP |b| (how strong "
         "any single coupling or bias may be), and NODE BUDGET (how many "
         "physical p-bits the whole program occupies). All four are shown "
         "as load gauges in the FRONTIER panel, with the gate PREDICTED TO "
         "BIND FIRST as the next terrain value is added made visually "
         "dominant -- not necessarily the gate with the highest CURRENT "
         "utilisation (that one is tagged separately, and the two need not "
         "be the same gate: this receipt's own DEGREE gate currently runs "
         "hotter, at 56%, than the FIELD CAP gate that is actually "
         "predicted to bind first, at 27%). |J| <= 6.0 is an "
         "Extropic-documented Z1 hardware cap (Thermalizers paper, Fig. 12 "
         "cap-sweep, annotated \"6 (Z1)\"); |b| <= 6.0 remains an ASSUMED "
         "project value, not a sourced Extropic figure -- every place this "
         "app displays either cap says which is which."),

        ("06  WHY SOME READOUTS SAY 'unavailable'",
         "This is the one most worth reading carefully: an 'unavailable: "
         "<reason>' readout is NOT a bug, and it is not this app failing "
         "to compute something it should have. It means the receipt this "
         "app reads from genuinely does not contain that field, or a real "
         "value cannot honestly be produced yet (no valid sample drawn "
         "this session, a layer that has never been regenerated, a "
         "composite view missing one of its layers). Rather than guess, "
         "fabricate a plausible-looking number, or silently show a blank, "
         "this app states the reason in plain words every time. A blank "
         "or an invented value would be indistinguishable from a real "
         "measurement -- 'unavailable' is what honesty looks like on "
         "screen; treat every one you see as the tool deliberately "
         "declining to guess, not as something broken."),
    ]


def clipboard_text() -> str:
    """Every section's title AND body, concatenated -- so Copy-to-clipboard
    can never silently drop a section (Step 5 of the brief). Plain text,
    not markdown/HTML, since the destination is an arbitrary clipboard."""
    parts = ["LATTICE -- WHAT IS THIS?", ""]
    for title, body in explainer_sections():
        parts.append(title)
        parts.append(body)
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The Toplevel itself -- Tk-dependent, built from the already-computed
# sections/diagrams above. Native window manager only: no overrideredirect
# anywhere in this class (the tokens spec's own explicit trap #1) -- drag,
# resize and minimize all come from the OS's own title bar for free.
# ---------------------------------------------------------------------------

class ExplainerWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.title("LATTICE -- What is this?")
        self.geometry("760x640")
        self.minsize(480, 360)
        self.resizable(True, True)   # native WM handles drag/resize/minimize
        self.configure(bg=theme.PAGE)

        mono = theme.resolve_mono_family()

        header = tk.Frame(self, bg=theme.PANEL, highlightbackground=theme.BEZEL,
                          highlightthickness=1)
        header.pack(fill="x")
        tk.Label(header, text="WHAT IS THIS", bg=theme.PANEL, fg=theme.GOLD,
                 font=(mono, 12, "bold")).pack(side="left", padx=10, pady=8)
        self.copy_btn = tk.Button(header, text="Copy to clipboard",
                                  command=self._on_copy, bg=theme.PANEL_2,
                                  fg=theme.CREAM, activebackground=theme.BEZEL,
                                  activeforeground=theme.CREAM, relief="flat",
                                  padx=10, pady=3, bd=1)
        self.copy_btn.pack(side="right", padx=10, pady=6)
        self.copy_status = tk.Label(header, text="", bg=theme.PANEL,
                                    fg=theme.GOLD_DIM, font=(mono, 8))
        self.copy_status.pack(side="right", padx=(0, 8))

        # Scrollable body -- six sections, each diagram beside its prose,
        # can exceed the default window height on a small display; a
        # Canvas+Scrollbar (mouse-wheel bound) lets the user reach every
        # section regardless of how the window is resized, rather than
        # relying on the user to grow the window to fit everything.
        outer = tk.Frame(self, bg=theme.PAGE)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=theme.PAGE, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(canvas, bg=theme.PAGE)
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def _on_body_configure(_evt=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(evt):
            canvas.itemconfigure(body_id, width=evt.width)

        body.bind("<Configure>", _on_body_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(evt):
            canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")

        # I3 (final review): `bind_all` is APPLICATION-scoped, not
        # window-scoped -- a bare `canvas.bind_all(...)` here stayed live
        # after this window closed, so the very next wheel event anywhere
        # in the app raised TclError against this destroyed canvas for the
        # rest of the session. Scope it to "pointer is actually over this
        # canvas" (bind/unbind on Enter/Leave, the standard Tk idiom for a
        # global wheel binding that must not outlive one widget) AND
        # unbind on close as a belt-and-suspenders fallback for the case
        # where the window is closed (e.g. via WM close, not by moving the
        # mouse off the canvas first) while the pointer is still over it.
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        self._photos = []  # keep references; Tk drops PhotoImages with none
        sections = explainer_sections()
        diagrams = explainer_diagrams()
        for i, ((title, prose), diagram) in enumerate(zip(sections, diagrams)):
            self._build_section(body, mono, title, prose, diagram)

        def _on_close():
            canvas.unbind_all("<MouseWheel>")  # I3: belt-and-suspenders, see above
            self.destroy()

        self.protocol("WM_DELETE_WINDOW", _on_close)

    def _build_section(self, parent: tk.Frame, mono: str, title: str,
                       prose: str, diagram: Image.Image) -> None:
        from PIL import ImageTk
        row = tk.Frame(parent, bg=theme.PANEL, highlightbackground=theme.BEZEL,
                       highlightthickness=1)
        row.pack(fill="x", padx=10, pady=6)

        left = tk.Frame(row, bg=theme.PANEL)
        left.pack(side="left", padx=8, pady=8)
        photo = ImageTk.PhotoImage(diagram)
        self._photos.append(photo)
        tk.Label(left, image=photo, bg=theme.PANEL).pack()

        right = tk.Frame(row, bg=theme.PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=8)
        tk.Label(right, text=title, bg=theme.PANEL, fg=theme.GOLD,
                 font=(mono, 10, "bold"), anchor="w", justify="left"
                 ).pack(fill="x")
        tk.Label(right, text=prose, bg=theme.PANEL, fg=theme.CREAM_DIM,
                 font=(mono, 9), anchor="w", justify="left", wraplength=440
                 ).pack(fill="x", pady=(4, 0))

    def _on_copy(self) -> None:
        text = clipboard_text()
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()  # some platforms drop the clipboard on window close
                        # without this -- keep it live immediately.
        self.copy_status.config(text="copied")
        self.after(1500, lambda: self.copy_status.config(text=""))


def open_explainer(master: tk.Misc) -> ExplainerWindow:
    """Always opens a NEW Toplevel -- callers that want a singleton (only
    ever one explainer window open) are responsible for tracking and
    lifting an existing one; this function itself makes no such claim."""
    return ExplainerWindow(master)
