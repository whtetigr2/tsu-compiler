"""Task 0: demo/theme.py -- every LATTICE colour token as a named constant
(verbatim from the design authority, SPR/docs/superpowers/specs/2026-09-02-
lattice-theme-tokens.md) plus the font stack. Headless: no Tk instance is
created by importing theme.py itself (tkfont.families() is only called
inside resolve_mono_family, and only actually queries Tk when a root
exists -- see that function's own docstring for the no-root fallback path).
"""
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

HEX6 = re.compile(r"^#[0-9a-fA-F]{6}$")

TOKEN_NAMES = [
    "PAGE", "PAGE_2", "PANEL", "PANEL_2", "INSET", "BEZEL", "RULE", "GHOST",
    "GOLD", "GOLD_HOT", "GOLD_DIM", "GOLD_GHOST",
    "ORANGE", "ORANGE_HOT",
    "BLUE", "BLUE_LIT", "BLUE_DEEP",
    "RED", "RED_HOT",
    "CREAM", "CREAM_DIM", "MUTE",
    "WATER", "ROCK", "GRASS", "OLIVE",
]

# Spot-check values verbatim from the tokens spec's table -- not every
# token (that would just re-type the whole spec), but enough that a typo'd
# hex digit is caught, not just a missing name.
SPEC_VALUES = {
    "PAGE": "#150a08",
    "GOLD": "#e8c04a",
    "BLUE": "#3f7fa8",
    "RED": "#b8352a",
    "CREAM": "#f5f0e8",
    "WATER": "#2e5d78",
    "ROCK": "#5a3a30",
    "GRASS": "#c4a03a",
    "OLIVE": "#8a9a54",
}


def test_every_documented_token_exists_as_a_valid_hex_colour():
    import theme
    for name in TOKEN_NAMES:
        value = getattr(theme, name)
        assert HEX6.match(value), f"{name} = {value!r} is not '#rrggbb'"


def test_spot_checked_token_values_match_the_tokens_spec_verbatim():
    import theme
    for name, expected in SPEC_VALUES.items():
        assert getattr(theme, name) == expected


def test_page_is_never_pure_black():
    """The spec is explicit: page is warm near-black, never pure #000."""
    import theme
    assert theme.PAGE != "#000000"


# ---------------------------------------------------------------------------
# Semantic status colour stays OFF the gold/orange/blue accent channel's
# "live/hot/cold data" meaning -- olive PASS / orange warn / red FAIL / blue
# lock, per the mockup's own legend.
# ---------------------------------------------------------------------------

def test_status_pass_is_olive_not_gold():
    import theme
    assert theme.STATUS_PASS == theme.OLIVE
    assert theme.STATUS_PASS != theme.GOLD


def test_status_fail_is_red():
    import theme
    assert theme.STATUS_FAIL == theme.RED


def test_status_lock_is_blue():
    import theme
    assert theme.STATUS_LOCK == theme.BLUE


# ---------------------------------------------------------------------------
# Font stack.
# ---------------------------------------------------------------------------

def test_font_stack_starts_with_cascadia_mono_and_falls_back_to_consolas():
    import theme
    assert theme.FONT_STACK[0] == "Cascadia Mono"
    assert "Consolas" in theme.FONT_STACK


def test_resolve_mono_family_is_headless_safe_with_no_tk_root():
    """No Tk root exists in this test process -- tkinter.font.families()
    would raise RuntimeError ('Too early to use font.families()'); this
    must not propagate, it must fall back."""
    import theme
    assert theme.resolve_mono_family() == "Consolas"


def test_resolve_mono_family_honours_a_custom_fallback():
    import theme
    assert theme.resolve_mono_family(candidates=(), fallback="Menlo") == "Menlo"
