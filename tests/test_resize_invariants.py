"""Regression guard: critical LATTICE widgets must stay VISIBLE across a
range of window sizes.

Five separate bugs of this exact class shipped on this branch before this
test existed (or, for #4-5, before this test's own coverage was wide
enough to catch them), each caught only by a human launching the real app
and looking, never by a test:

  1. Panels starved to 1x1px (fixed in d71cb9e / lattice-layout) -- a fixed
     3:2 / 5:5:2:5 weighted grid split, once row-0's combined content
     height exceeded what was available, collapsed the losing panels'
     BODIES to 1x1px (invisible), not a uniform shrink. PIPELINE's nine
     stage rows, SAMPLE LOG's Listbox, and (via a second, independent bug
     in the same commit) the LAYERS overlay-pin "can be outvoted by
     conditioning" disclosure -- a Critical fix a prior review round had
     added -- were all literally 1x1px regardless of window size.
  2. The LAYERS overlay-pin disclosure specifically: _layers_cell used to
     call `cell.pack_propagate(False)` immediately after creating the
     cell, BEFORE any of its content (including everything the caller
     packs in after the function returns) existed -- freezing the cell's
     height at an empty Frame's default forever. Also fixed in d71cb9e.
  3. The control bar vanished below 1200x900 (fixed in a225b5f /
     lattice-controlbar) -- `bottom` (the four buttons + footer
     disclosure) was packed AFTER `content` (fill="both", expand=True), so
     a shrink below the window's natural size gave `content` first claim
     on the shortfall and squeezed `bottom` toward zero, unmapping all
     four buttons and the footer disclosure.
  4. Row 0 unmapped entirely below ~1050-1100px window HEIGHT (fixed in
     951cea0) -- SCOPE (minsize=340) and LAYERS (minsize=230) reserved
     570px unconditionally, and row 0 (no minsize of its own) absorbed the
     whole deficit once the window got short enough that grid silently
     unmapped it, taking PIPELINE/LIVE LATTICE/DECODED WORLD/REGIME &
     TRACES/VERIFICATION/SAMPLER/SAMPLE LOG down with it. Fixed by
     wrapping `content` itself in a page-level scroll column (see
     _make_scroll_column / _build_layout in lattice_app.py) so every row
     renders at its own natural height and the page scrolls vertically to
     reach the rest.
  5. The same class on the WIDTH axis (fixed on lattice-width-axis): LIVE
     LATTICE and DECODED WORLD (row 0's two weight=1 grid columns)
     squeezed toward zero WIDTH whenever the window was narrower than row
     0's combined natural width, while row 0's three fixed-width columns
     (PIPELINE/FRONTIER 340 + REGIME & TRACES 300 + VERIFICATION/SAMPLER/
     DECODED MIX/SAMPLE LOG 430 = 1070px) held their size unconditionally
     and absorbed none of the deficit -- measured at 8px for one of them
     at 1200x900 on unmodified master. The page-level scroll column added
     for bug #4 above made this marginally worse (it cost ~17px of width)
     but did not cause it -- confirmed present on unmodified master.
     Fixed the same way as #4, one axis over: `_on_canvas_configure` in
     the page-level `_make_scroll_column` call now floors `inner`'s width
     at its own natural (reqwidth) size instead of always pinning it to
     the canvas viewport, and a horizontal Scrollbar reaches whatever the
     window is too narrow to show. This test's own coverage was
     effectively height-oriented until this fix (no width-axis widgets in
     the guarded set, no narrow-but-tall sizes in SIZES) -- extended
     below, along with MIN_WIDTH_AXIS, specifically because plain MIN_DIM
     (8) sat exactly at the collapse floor for width the way #1-4's
     collapse floor was for height.

The common pattern: content that exists in the source and passes every
source grep, occupying zero screen pixels at runtime. This test builds the
REAL LatticeApp, drives it through a range of window sizes, and asserts
that a named set of critical widgets stay mapped and hold more than a
sliver of screen space at every size.

WHAT THIS DOES NOT COVER: this guards VISIBILITY ONLY -- winfo_ismapped()
plus a non-trivial winfo_width()/winfo_height(). It says nothing about
whether a widget's CONTENT is correct, whether text wraps sensibly, scroll
position, color, or any other aspect of appearance. A widget can pass this
guard and still look wrong. A passing run of this test must never be read
as "the layout is correct" -- only "these widgets were not collapsed to
nothing."

Skips cleanly (no failure) if no Tk display is available.
"""
import sys
import time
import tkinter as tk
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

# NOTE on headless detection: this module deliberately does NOT do a
# separate throwaway `tk.Tk(); .destroy()` probe before the real build.
# Measured on this machine: a first Tk() immediately destroyed, followed by
# a second Tk() moments later in the same process, is intermittently flaky
# under this repo's pytest plugin set (jaxtyping's pytest11 entry point
# specifically -- reproduced with `-p no:jaxtyping` clearing it, isolated
# with a minimal two-Tk() repro script outside this repo) -- Tcl's own
# init.tcl lookup would spuriously raise TclError ("couldn't read file
# ...init.tcl: No error") on the SECOND Tk() even though a real, working
# display was present throughout (confirmed: the identical second Tk() call
# succeeds 100% of the time when it is the ONLY Tk() built in the process).
# A single, real Tk() -- the actual LatticeApp construction below, wrapped
# in its own TclError guard -- does not carry this risk. Skip on TclError
# there instead: still "skip cleanly, never fail spuriously" for a genuinely
# headless box, without introducing a redundant Tk() that this environment
# has shown can itself trigger a spurious failure.
from lattice_app import LatticeApp, Receipt, RECEIPT_DIR, OVERLAY_RECEIPT_DIR  # noqa: E402

# Task requirement: at minimum these four sizes, largest (the app's own
# default geometry) to smallest. 1100x900 and 900x800 (width-axis fix,
# 2026-09) add two sizes that are narrow relative to row 0's ~1070px+ of
# fixed-width columns while still tall enough not to also trip the
# height-axis row-0-unmapping bug (#4 above) -- they exercise the WIDTH
# axis specifically, which none of the original four sizes isolated (each
# pairs a width squeeze with a height squeeze).
SIZES = [
    (1760, 1420),
    (1200, 900),
    (1100, 900),
    (1000, 700),
    (900, 800),
    (800, 600),
]

# A widget collapsed to the historical failure mode measures 1x1px. This
# floor is well above that and well below any real guarded widget's actual
# size -- it exists only to catch "collapsed to nothing", not to assert a
# minimum design size.
MIN_DIM = 8

# Width-axis fix (2026-09): LIVE LATTICE / DECODED WORLD's measured
# collapse was 8px -- exactly MIN_DIM, the same floor every other guarded
# widget uses. A `width <= MIN_DIM` check technically still catches an
# 8px reading (8 <= 8), but that is a razor's edge, not a meaningful
# margin, for widgets whose real natural width is ~320-340px
# (WORLD_DISPLAY_PX=320 plus panel padding -- see lattice_app.py). 150
# sits with a comfortable margin on both sides: far above any collapse
# reading, far below the real size.
MIN_WIDTH_AXIS = 150

# Widgets that get the stricter MIN_WIDTH_AXIS floor above, IN ADDITION TO
# (not instead of) the general MIN_DIM floor every guarded widget's height
# still gets.
WIDTH_AXIS_WIDGETS = {"lattice_panel.body", "world_panel.body"}


def _build_app() -> LatticeApp:
    receipt = Receipt(RECEIPT_DIR)
    overlay_receipt = Receipt(OVERLAY_RECEIPT_DIR)
    app = LatticeApp(receipt, overlay_receipt)
    app.update_idletasks()
    app.update()
    return app


def _pump(app: LatticeApp, ticks: int = 8, delay: float = 0.02) -> None:
    for _ in range(ticks):
        app.update()
        time.sleep(delay)
    app.update_idletasks()


def _teardown_app(app: LatticeApp) -> None:
    try:
        app.worker.stop_evt.set()
        app.worker.join(timeout=1.0)
    except Exception:
        pass
    try:
        app.destroy()
    except tk.TclError:
        pass


def _guarded_widgets(app: LatticeApp):
    """Named-attribute handles only (never positional/text matching) --
    see the module docstring's bug #1/#2 for why a widget with no handle
    is exactly the kind that silently drops out of coverage."""
    return [
        ("pause_btn", app.pause_btn),
        ("seed_btn", app.seed_btn),
        ("step_btn", app.step_btn),
        ("save_btn", app.save_btn),
        ("footer_label", app.footer_label),
        ("pipeline_panel.body", app.pipeline_panel.body),
        ("log_box", app.log_box),
        ("overlay_pin_disclosure_label", app.overlay_pin_disclosure_label),
        ("sampler_panel.body", app.sampler_panel.body),
        ("verif_panel.body", app.verif_panel.body),
        # Width-axis fix (2026-09): row 0's two weight=1 grid columns -- the
        # ones with no fixed pixel width of their own, and therefore the
        # ones that squeezed toward zero WIDTH (bug #5, module docstring).
        ("lattice_panel.body", app.lattice_panel.body),
        ("world_panel.body", app.world_panel.body),
    ]


def test_critical_widgets_stay_visible_across_resize():
    try:
        app = _build_app()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display available in this environment: {exc}")
        return
    try:
        # Let the worker produce a few real draws so the app is in steady
        # state (matches verify_controlbar_resize.py's own approach) before
        # the resize sweep starts.
        _pump(app, ticks=15, delay=0.05)

        widgets = _guarded_widgets(app)
        failures = []
        for w_px, h_px in SIZES:
            app.geometry(f"{w_px}x{h_px}")
            _pump(app)
            for name, w in widgets:
                mapped = w.winfo_ismapped()
                width = w.winfo_width()
                height = w.winfo_height()
                width_floor = MIN_WIDTH_AXIS if name in WIDTH_AXIS_WIDGETS else MIN_DIM
                if not mapped or width <= width_floor or height <= MIN_DIM:
                    failures.append(
                        f"{name} at window {w_px}x{h_px}: "
                        f"mapped={mapped} width={width} height={height} "
                        f"(width_floor={width_floor})")
        assert not failures, "widget(s) collapsed/unmapped on resize:\n" + "\n".join(failures)
    finally:
        _teardown_app(app)
