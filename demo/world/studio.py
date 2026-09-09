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

import time
from dataclasses import dataclass, replace

import numpy as np

from world.generate import FIELD_PLAN, World
from world.generate import generate as _generate
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


BAND: tuple[float, float] = (0.38, 0.45)
"""The measured usable coupling band. Below ~0.85x Kc the world is noise; above
~1.05x Kc one value swallows the map. The UI marks this on the beta*J track so a
world sampled outside it is obvious at a glance rather than buried in a number."""


@dataclass(frozen=True)
class Sampled:
    seed: int = 0
    beta_j: float = 0.42
    size: int = 64
    mode: str = "bilinear"
    warmup: int = 4000
    """Fixed and REPORTED, not a control. Measured to make no difference to
    structure in this workload (+0.0483 at 4,000 against +0.0546 at 100,000), so
    offering a slider would imply a knob that tunes nothing. It is not claimed
    irrelevant in general -- another graph or operating point could make settling
    time matter, and that is the day it becomes a control."""


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
        self._sampled = Sampled()
        self._timings: dict[str, float] = {}

    @property
    def world(self) -> World:
        if self._world is None:
            raise RuntimeError("no world loaded: call Studio.load(world) first")
        return self._world

    @property
    def readout(self) -> Readout:
        return self._readout

    @property
    def sampled(self) -> Sampled:
        return self._sampled

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)

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

    def generate(self, sampled: Sampled) -> View:
        """Sample a world and load it. This is the ~3.3s path; everything in
        `set_readout` is the millisecond path."""
        t0 = time.time()
        world = _generate(size=sampled.size, seed=sampled.seed,
                          beta_j=sampled.beta_j, mode=sampled.mode,
                          warmup=sampled.warmup)
        total = time.time() - t0
        self._sampled = sampled
        self.load(world)
        per = {name: total * (world.fields[name].size /
                              sum(world.fields[n].size for n, _ in FIELD_PLAN))
               for name, _src in FIELD_PLAN}
        self._timings = dict(per, total=total)
        return self.view()

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
