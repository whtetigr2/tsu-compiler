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
