"""Assembling a world: three sampled parameter fields, upsampled, read through
the spline.

THE BOUNDARY. Everything random and everything structural comes from
`world.fields.sample_field` -- the TSU. Everything after it (upsampling, the
spline, terrain banding, cost) is a deterministic readout that contributes no
entropy. That is why a claim about this world's structure remains a claim about
what the chip sampled.

WHY THREE FIELDS AT THREE SIZES. Borrowed from Minecraft's pipeline: an
independent low-frequency field decides continents, a mid-frequency one decides
how eroded and flat a region is, and a full-resolution one decides local relief.
Sampling each at its own grid size is what gives the scale separation that
coupling alone cannot (correlation length spans only ~0.75-2.66 cells).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.fields import sample_field
from world.scale import upsample
from world.spline import LEVELS, TERRAINS, height, terrain_index

FIELD_PLAN: tuple[tuple[str, int], ...] = (
    ("continentalness", 8),
    ("erosion", 16),
    ("peaks_valleys", 64),
)
"""(name, sampling grid size) at output size 64. Scaled proportionally for
other output sizes so the RATIO of feature scales is preserved."""

BASE_SIZE = 64


@dataclass(frozen=True)
class World:
    size: int
    seed: int
    beta_j: float
    mode: str
    warmup: int
    fields: dict[str, np.ndarray]  # name -> raw sampled field, BEFORE upsampling
    height: np.ndarray      # (size, size) float
    terrain: np.ndarray     # (size, size) int, index into TERRAINS
    cost: np.ndarray        # (size, size) int


def generate(size: int = 64, seed: int = 0, beta_j: float = 0.42,
             mode: str = "bilinear", warmup: int = 4000) -> World:
    """Sample and assemble one world.

    Each parameter field gets its own seed derived from `seed`, so the three
    fields are independent draws rather than three views of one sample -- which
    they must be, or continentalness and erosion would be perfectly correlated
    and two of the three parameters would carry no information.
    """
    if size < BASE_SIZE or size % BASE_SIZE != 0:
        raise ValueError(
            f"size must be a positive multiple of {BASE_SIZE}, got {size}; a "
            f"non-multiple gives a non-integer upsample factor, and sizes at "
            f"or below zero pass a bare modulo check while failing much later "
            f"inside upsample -- after a full TSU sample has already run")
    est_spins = sum(max(4, s * (size // BASE_SIZE)) ** 2 for _, s in FIELD_PLAN) * LEVELS
    if est_spins > 250_000:
        raise ValueError(
            f"size {size} needs about {est_spins:,} spins across the three "
            f"field stacks, over the Z1 node budget of 250,000; place() would "
            f"raise, but only while compiling the THIRD field, after two are "
            f"already sampled")
    scale = size // BASE_SIZE

    raw: dict[str, np.ndarray] = {}
    norm: dict[str, np.ndarray] = {}
    for i, (name, base_src) in enumerate(FIELD_PLAN):
        src = max(4, base_src * scale)
        f = sample_field(src, LEVELS, beta_j, seed * 1000 + i, warmup=warmup)
        raw[name] = f
        up = upsample(f.astype(float), size // src, mode)
        norm[name] = up / LEVELS

    c, e, pv = (norm[n] for n, _ in FIELD_PLAN)
    h = np.vectorize(height)(c, e, pv)
    t = np.vectorize(terrain_index)(h).astype(int)
    cost = np.array([[TERRAINS[i].cost for i in row] for row in t], dtype=int)
    return World(size=size, seed=seed, beta_j=beta_j, mode=mode,
                 warmup=warmup, fields=raw, height=h, terrain=t, cost=cost)
