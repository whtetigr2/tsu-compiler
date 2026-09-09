"""The deterministic readout: sampled parameter fields -> height -> terrain
-> movement cost.

THIS MODULE CONTRIBUTES NO RANDOMNESS. Every value it produces is a pure
function of the parameters handed in, all of which were sampled on the TSU.
That is the architectural boundary this project locked long ago -- TSU-sampled
versus deterministically-derived -- and it is what lets a claim about world
structure remain a claim about what the chip sampled. The spline interprets;
it does not invent.

Shape borrowed from Minecraft's world generation (minecraft.wiki/w/World_
generation): continentalness sets average height, erosion sets how much
relief survives, peaks-and-valleys sets local relief. Erosion enters
MULTIPLICATIVELY so that a heavily eroded region is flat regardless of its
peaks-and-valleys value -- "the higher the erosion, the lower and flatter the
terrain".

EVERY TERRAIN IS ENTERABLE. Cost varies; permission never does. This is what
makes every generated world traversable by construction, and it is why the
generator needs no spanning validator, no rejection sampling and no repair --
connectivity is a global property and an Ising energy is strictly pairwise, so
it could never have been written as a coupling in the first place.

Glyphs are deliberately pure ASCII rather than the spec's `≈`/`▲`: this project
runs on Windows consoles where a codepage mismatch turns a rendering bug into a
crash, and the low-to-high ramp reads fine without them.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

LEVELS = 3
"""Binary layers per parameter field, so a field holds integer levels 0..LEVELS."""


@dataclass(frozen=True)
class Terrain:
    name: str
    glyph: str
    cost: int


TERRAINS: tuple[Terrain, ...] = (
    Terrain("deep_water", "~", 4),
    Terrain("shallow_water", "-", 3),
    Terrain("sand", ".", 1),
    Terrain("grass", ",", 1),
    Terrain("scree", "^", 2),
    Terrain("mountain", "#", 3),
)

_UPPER_BOUNDS = (-0.60, -0.20, 0.15, 0.60, 1.00)
"""Height at or below which each terrain applies; the last terrain takes the
rest. len(_UPPER_BOUNDS) == len(TERRAINS) - 1 by construction."""

_BASE = (-1.00, -0.20, 0.30, 0.70)
_AMP = (1.00, 0.70, 0.40, 0.15)
_RELIEF = (-0.50, -0.10, 0.30, 0.80)


def _spline(knots: tuple[float, ...], v: float) -> float:
    """Piecewise-linear interpolation of `knots` over v in [0, 1].

    Interpolates rather than indexes because bilinear upsampling produces
    FRACTIONAL parameter values -- an integer-indexed lookup would silently
    truncate them and throw away the smoothing that upsampling just paid for.
    """
    if v <= 0.0:
        return knots[0]
    if v >= 1.0:
        return knots[-1]
    t = v * (len(knots) - 1)
    i = min(int(t), len(knots) - 2)
    f = t - i
    return knots[i] * (1.0 - f) + knots[i + 1] * f


def height(c: float, e: float, pv: float) -> float:
    """Terrain height from normalised continentalness, erosion and
    peaks-and-valleys, each in [0, 1]. Range [-1.50, 1.50]."""
    return _spline(_BASE, c) + _spline(_AMP, e) * _spline(_RELIEF, pv)


def terrain_index(h: float) -> int:
    """Index into TERRAINS for a height."""
    for i, upper in enumerate(_UPPER_BOUNDS):
        if h <= upper:
            return i
    return len(TERRAINS) - 1


def cost_of(h: float) -> int:
    """Movement cost of entering a cell at this height. Always >= 1."""
    return TERRAINS[terrain_index(h)].cost


# --- parameterised entry points -------------------------------------------
# Added for the World Studio tab. The scalar `height`/`terrain_index` above are
# the REFERENCE implementation and are unchanged; everything below is an array
# path fast enough for a live slider, plus the band-shifting the sliders need.
#
# Nothing here mutates the module constants. A slider that reached in and
# rewrote _UPPER_BOUNDS would make every consumer's result depend on whatever
# the UI last did -- untestable, and wrong the moment two things read it.

DEFAULT_BOUNDS: tuple[float, ...] = _UPPER_BOUNDS
BOUND_EPS = 1e-6
"""Minimum gap enforced between adjacent band bounds. Bounds are CLAMPED to
stay strictly increasing rather than sorted: sorting would silently reassign
which terrain a height falls into, so a band table that would go out of order
is pushed apart instead."""


def shifted_bounds(sea_level: float = 0.0,
                   mountain_line: float = 0.0) -> tuple[float, ...]:
    """Band bounds with the readout sliders applied.

    `sea_level` moves the two water bounds up (flooding) or down (exposing
    land). `mountain_line` moves the scree/mountain bound DOWN as it rises,
    because mountain is everything above that bound -- so a higher slider must
    yield more mountain.

    Bounds are clamped to remain strictly increasing by at least BOUND_EPS.
    """
    b = list(DEFAULT_BOUNDS)
    b[0] += float(sea_level)
    b[1] += float(sea_level)
    b[4] -= float(mountain_line)
    for i in range(1, len(b)):
        if b[i] < b[i - 1] + BOUND_EPS:
            b[i] = b[i - 1] + BOUND_EPS
    return tuple(b)


def _spline_array(knots: tuple[float, ...], v: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of `_spline`. `np.interp` is piecewise-linear over
    evenly spaced sample points and clamps outside the range -- the same three
    behaviours `_spline` implements one value at a time. Tested against it
    directly rather than assumed equivalent."""
    xp = np.linspace(0.0, 1.0, len(knots))
    return np.interp(np.asarray(v, dtype=float), xp, np.asarray(knots, dtype=float))


def height_array(c, e, pv, relief: float = 1.0) -> np.ndarray:
    """Vectorised `height`. `relief` multiplies the amplitude term only, so it
    changes how much the terrain swings without moving its base elevation."""
    return (_spline_array(_BASE, c)
            + float(relief) * _spline_array(_AMP, e) * _spline_array(_RELIEF, pv))


def terrain_array(h, bounds: tuple[float, ...]) -> np.ndarray:
    """Vectorised `terrain_index` against an arbitrary band table. `np.searchsorted`
    with side='left' places a height exactly equal to a bound INTO that band,
    matching the scalar version's `h <= upper`."""
    return np.searchsorted(np.asarray(bounds, dtype=float),
                           np.asarray(h, dtype=float), side="left").astype(int)
