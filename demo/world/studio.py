"""The readout state machine behind the World Studio tab.

WHY THIS EXISTS. Terrain is a READOUT: height is sampled once, and turning
height into water/sand/grass/mountain is a deterministic lookup. So the
controls split in two -- sampled parameters (seed, beta*J, size) that cost a
full ~3.3s resample, and readout parameters (sea level, mountain line, relief,
cost table) that cost milliseconds. That split IS this project's
sampled-versus-derived boundary, and this module is where it lives.

THE CACHE BOUNDARY is the normalised upsampled parameter fields -- c, e, pv
after upsampling and division by LEVELS. Everything downstream of them (spline
knots -> height -> bands -> cost) recomputes from cache. Caching the HEIGHT
instead would have been the obvious choice and would have been wrong: relief is
a spline knot, so it changes height, and a height cache would force a resample
for a slider that needs none.

No Tk here, and no I/O. The tab is a thin renderer over this.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from world.generate import FIELD_PLAN, World
from world.scale import upsample
from world.spline import (LEVELS, TERRAINS, shifted_bounds, height_array,
                          terrain_array)

COST_TABLES: dict[str, tuple[int, ...]] = {
    "shipped": (4, 3, 1, 1, 2, 3),
    "steep": (8, 4, 1, 2, 4, 8),
    "wide": (12, 5, 1, 2, 5, 12),
}
"""Measured against the shuffled-null permutation test: the shipped table put
57% of cells at the cheapest cost and gave a real-vs-shuffled detour gap of
+0.0483; `steep` gives +0.0999 and `wide` +0.1192, both at a 22% cheapest-cell
share. `wide` routes around water entirely, which is a game decision rather
than a physics one -- hence a choice, not a constant."""


@dataclass(frozen=True)
class Readout:
    sea_level: float = 0.0
    mountain_line: float = 0.0
    relief: float = 1.0
    cost_table: str = "steep"


@dataclass(frozen=True)
class View:
    height: np.ndarray
    terrain: np.ndarray
    cost: np.ndarray
    bounds: tuple[float, ...]


class Studio:
    """Holds one sampled world and applies readout parameters to it."""

    def __init__(self) -> None:
        self._world: World | None = None
        self._norm: dict[str, np.ndarray] = {}
        self._readout = Readout()

    @property
    def world(self) -> World:
        if self._world is None:
            raise RuntimeError("no world loaded: call Studio.load(world) first")
        return self._world

    @property
    def readout(self) -> Readout:
        return self._readout

    def load(self, world: World) -> None:
        """Cache the normalised upsampled fields for `world`. This is the only
        expensive step, and it happens once per sampled world."""
        self._world = world
        self._norm = {}
        for name, base_src in FIELD_PLAN:
            raw = world.fields[name]
            src = raw.shape[0]
            self._norm[name] = upsample(raw.astype(float), world.size // src,
                                        world.mode) / LEVELS

    def set_readout(self, readout: Readout) -> View:
        if readout.cost_table not in COST_TABLES:
            raise ValueError(
                f"unknown cost table {readout.cost_table!r}; "
                f"expected one of {sorted(COST_TABLES)}")
        self._readout = readout
        return self.view()

    def view(self) -> View:
        """Recompute height, terrain and cost from the cached fields. Never
        samples -- see this module's docstring."""
        w = self.world
        r = self._readout
        c, e, pv = (self._norm[n] for n, _ in FIELD_PLAN)
        h = height_array(c, e, pv, relief=r.relief)
        bounds = shifted_bounds(r.sea_level, r.mountain_line)
        t = terrain_array(h, bounds)
        cost = np.asarray(COST_TABLES[r.cost_table], dtype=int)[t]
        return View(height=h, terrain=t, cost=cost, bounds=bounds)
