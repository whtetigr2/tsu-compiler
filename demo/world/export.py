"""Serialising a world to disk.

A world you cannot regenerate is an anecdote, so the JSON carries its own
provenance -- seed, coupling, upsample mode, size -- alongside the grids.

The PNG is written with numpy and zlib only. This project has no image
dependency and this is not the place to add one for the sake of six colours.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from world.spline import TERRAINS

PALETTE: tuple[tuple[int, int, int], ...] = (
    (24, 54, 112),     # deep_water
    (52, 104, 168),    # shallow_water
    (214, 200, 148),   # sand
    (96, 140, 76),     # grass
    (140, 132, 120),   # scree
    (232, 232, 236),   # mountain
)
assert len(PALETTE) == len(TERRAINS)


def to_text(world) -> str:
    """Glyph grid, one line per row."""
    return "\n".join("".join(TERRAINS[i].glyph for i in row)
                     for row in world.terrain)


def to_json(world) -> dict:
    return dict(size=world.size, seed=world.seed, beta_j=world.beta_j,
                mode=world.mode,
                terrain_names=[t.name for t in TERRAINS],
                terrain_glyphs=[t.glyph for t in TERRAINS],
                terrain=world.terrain.tolist(),
                cost=world.cost.tolist(),
                height=[[round(float(v), 6) for v in row]
                        for row in world.height],
                fields={k: v.tolist() for k, v in world.fields.items()})


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Minimal 8-bit RGB PNG. Each scanline is prefixed with filter type 0."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def write(world, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = out / "world.json"
    j.write_text(json.dumps(to_json(world), indent=2), encoding="utf-8")
    t = out / "world.txt"
    t.write_text(to_text(world) + "\n", encoding="utf-8")
    pal = np.array(PALETTE, dtype=np.uint8)
    p = out / "world.png"
    p.write_bytes(_png_bytes(pal[world.terrain]))
    return [j, t, p]
