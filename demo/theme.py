"""LATTICE theme foundation: every colour token and the font stack, as named
constants, extracted VERBATIM from the design authority --
`SPR/docs/superpowers/specs/2026-09-02-lattice-theme-tokens.md` (itself
extracted from `SPR/reference/LATTICE_theme_mockup_v2_Grok.html`, the
approved visual target). Do not hand-copy a hex value from this file into
another one -- import the name instead, so a future palette change has one
place to happen.

THE PALETTE IS SEMANTIC, not decorative -- this is the fact that makes the
theme cohere (see the tokens spec's own "rule that makes it cohere"):

    GOLD    the useful window -- live data, active labels, curves.
    ORANGE  hot and agitated -- a second data series, high energy.
    BLUE    COLD, full stop -- frozen, locked, mediator spins, a locked
            temperature control. NEVER decorative. A viewer should be able
            to learn this scheme without being told; a stray decorative
            blue destroys that (see demo/lattice_app.py's Task 0 fix, where
            WORLD_ON/WORLD_OFF -- the LIVE LATTICE panel's own p-bit
            on/off colour -- was wrongly blue before this pass; the
            mockup's own construction note says it plainly: "Lit gold
            nodes are world p-bits in state 1. Cold-blue nodes are hidden
            mediators -- frozen helpers, not terrain.").
    RED     critical / FAIL only.

Semantic STATUS colour (PASS/FAIL/warning) is kept distinct from the
decorative accent channel per the mockup's own legend ("olive PASS / orange
warn / red FAIL / ice lock" -- see `SPR/reference/LATTICE_theme_mockup_v2_
Grok.html` line ~1303): PASS uses the (otherwise terrain-only) `olive`
token, never gold -- gold stays reserved for "this is live/active data",
not "this succeeded". See the STATUS_* aliases at the bottom of this file.
"""
from __future__ import annotations

import tkinter.font as _tkfont

# --------------------------------------------------------------------------
# Palette -- verbatim from the tokens spec's table. Grounds.
# --------------------------------------------------------------------------
PAGE = "#150a08"          # window ground, warm near-black -- never pure #000
PAGE_2 = "#1a0d0d"        # alternate ground band
PANEL = "#241414"         # panel ground, barely lifted
PANEL_2 = "#2c1816"       # secondary panel
INSET = "#1c0e0c"         # recessed area (plots, logs)
BEZEL = "#3d2420"         # panel bezel
RULE = "#5a3830"          # hairlines, dividers
GHOST = "#5c4a42"         # faintest structure

# Gold -- the useful window where structure forms.
GOLD = "#e8c04a"          # primary accent -- live data, active labels, curves
GOLD_HOT = "#f4d56a"      # highlight / hover
GOLD_DIM = "#7a6a3a"      # inactive tabs, disabled labels
GOLD_GHOST = "#3d3420"    # gold at lowest emphasis

# Orange -- hot and agitated.
ORANGE = "#d2691e"        # second data series
ORANGE_HOT = "#e8893a"    # hot state, high energy

# Blue -- COLD. Frozen, locked, mediator spins. Never decorative.
BLUE = "#3f7fa8"
BLUE_LIT = "#6aa3c4"      # active cold element
BLUE_DEEP = "#1e3a50"     # cold ground

# Red -- critical / FAIL only.
RED = "#b8352a"
RED_HOT = "#d44538"       # critical, emphasised
RED_DEEP = "#6a2018"      # dark red ground -- e.g. the FRONTIER gauge's
                           # hatched-overrun background (I5, final review:
                           # this hex lived as a hand-copied literal in
                           # demo/lattice_app.py, the one global constraint
                           # this branch otherwise holds to everywhere else
                           # -- import the name, don't hand-copy the hex).

# Text.
CREAM = "#f5f0e8"         # body text -- warm white, not #fff
CREAM_DIM = "#c8bdb0"     # secondary text
MUTE = "#8a7a6c"          # tertiary text

# Terrain (decoded world).
WATER = "#2e5d78"
ROCK = "#5a3a30"
GRASS = "#c4a03a"
OLIVE = "#8a9a54"

# --------------------------------------------------------------------------
# Semantic status colour -- kept OFF the decorative accent channel (gold/
# orange/blue above mean "live"/"hot"/"cold data", never "this passed").
# Per the mockup's own legend: olive PASS / orange warn / red FAIL / ice
# (blue) lock.
# --------------------------------------------------------------------------
STATUS_PASS = OLIVE
STATUS_WARN = ORANGE
STATUS_FAIL = RED
STATUS_LOCK = BLUE

# --------------------------------------------------------------------------
# Font stack -- verbatim from the tokens spec. Tk has no font-family
# fallback list of its own (an unavailable family silently substitutes a
# platform default, not the next candidate), so `resolve_mono_family` walks
# this list itself via `tkinter.font.families()`.
# --------------------------------------------------------------------------
FONT_STACK = ("Cascadia Mono", "Consolas", "IBM Plex Mono",
              "DejaVu Sans Mono", "Menlo", "monospace")


def resolve_mono_family(candidates: tuple[str, ...] = FONT_STACK,
                         fallback: str = "Consolas") -> str:
    """First installed family from `candidates`, per Tk's own installed-font
    list. Requires a live Tk default root -- `tkinter.font.families()`
    raises RuntimeError before one exists ("Too early to use font.families()
    ..."), which happens routinely here: demo/lattice_app.py is safe to
    import with no Tk instance at all (see its own module docstring / the
    "no Tk in pure logic" test convention), so this must not crash at
    import time. Callers with no root yet get `fallback` (still a real,
    near-universally-installed monospace font, matching this file's own
    Consolas-fallback token); a caller with a root gets the actual first
    match, e.g. "Cascadia Mono" where it's installed."""
    try:
        available = set(_tkfont.families())
    except RuntimeError:
        return fallback
    for name in candidates:
        if name in available:
            return name
    return fallback
