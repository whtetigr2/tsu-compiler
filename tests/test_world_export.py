"""Task 6: export. Uses one small generated world, reused across tests."""
import sys
import json
import struct
import zlib

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate
from world.export import to_text, to_json, write, PALETTE
from world.spline import TERRAINS


@pytest.fixture(scope="module")
def world():
    return generate(size=64, seed=0, warmup=200)


def _decode_png(raw: bytes) -> dict:
    """A real PNG reader for the writer this project ships: walks the chunk
    stream, VERIFIES EVERY CRC, and reconstructs pixels. Filter type 0 only,
    which is all `_png_bytes` emits. Written out rather than using an image
    library because the point is to check our bytes against the format, not to
    add an image dependency this project deliberately does not have."""
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG signature"
    pos, chunks, order = 8, {}, []
    while pos < len(raw):
        ln = int.from_bytes(raw[pos:pos + 4], "big")
        tag = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + ln]
        crc = int.from_bytes(raw[pos + 8 + ln:pos + 12 + ln], "big")
        assert crc == zlib.crc32(tag + data) & 0xFFFFFFFF, f"bad CRC on {tag!r}"
        chunks[tag] = chunks.get(tag, b"") + data
        order.append(tag)
        pos += 12 + ln
    w, h, depth, ctype, comp, filt, inter = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    flat = zlib.decompress(chunks[b"IDAT"])
    stride = w * 3
    rows = []
    for y in range(h):
        off = y * (stride + 1)
        assert flat[off] == 0, f"row {y}: expected filter type 0, got {flat[off]}"
        rows.append(np.frombuffer(flat[off + 1:off + 1 + stride],
                                  dtype=np.uint8).reshape(w, 3))
    return dict(w=w, h=h, depth=depth, ctype=ctype, comp=comp, filt=filt,
                inter=inter, order=order, px=np.array(rows))


def test_to_text_has_one_line_per_row_and_one_glyph_per_cell(world):
    lines = to_text(world).splitlines()
    assert len(lines) == world.size
    assert all(len(ln) == world.size for ln in lines)


def test_to_text_uses_only_known_glyphs(world):
    glyphs = {t.glyph for t in TERRAINS}
    assert set(to_text(world).replace("\n", "")) <= glyphs


def test_to_json_round_trips_the_terrain_grid(world):
    d = to_json(world)
    assert json.loads(json.dumps(d))["terrain"] == world.terrain.tolist()


def test_to_json_records_the_provenance_of_the_world(world):
    """A world you cannot regenerate is an anecdote. Seed, coupling and mode
    must travel with it."""
    d = to_json(world)
    assert d["seed"] == world.seed
    assert d["beta_j"] == world.beta_j
    assert d["mode"] == world.mode
    assert d["size"] == world.size
    assert d["terrain_glyphs"] == [t.glyph for t in TERRAINS]


def test_to_json_carries_terrain_names_as_well_as_glyphs(world):
    d = to_json(world)
    assert d["terrain_names"] == [t.name for t in TERRAINS]
    assert d["terrain_glyphs"] == [t.glyph for t in TERRAINS]


def test_warmup_is_recorded_so_a_world_can_actually_be_regenerated(world):
    """seed + beta_j + mode + size is NOT sufficient: the same seed at warmup
    200 versus 201 yields worlds differing in ~1.9% of cells. Without warmup the
    JSON's provenance is incomplete and this module's own stated principle -- a
    world you cannot regenerate is an anecdote -- would be false."""
    assert to_json(world)["warmup"] == world.warmup


def test_write_produces_all_three_files(tmp_path, world):
    paths = write(world, tmp_path)
    names = {p.name for p in paths}
    assert names == {"world.json", "world.txt", "world.png"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_png_is_structurally_valid_and_every_crc_checks(tmp_path, world):
    """The previous version read only the first 24 bytes, so a corrupt CRC, a
    dropped IEND, or a colour-type regression all passed. `_decode_png` verifies
    every chunk CRC as it walks the stream."""
    p = [q for q in write(world, tmp_path) if q.suffix == ".png"][0]
    d = _decode_png(p.read_bytes())
    assert (d["w"], d["h"]) == (world.size, world.size)
    assert (d["depth"], d["ctype"]) == (8, 2)          # 8-bit truecolour RGB
    assert (d["comp"], d["filt"], d["inter"]) == (0, 0, 0)
    assert d["order"][0] == b"IHDR"
    assert d["order"][-1] == b"IEND"


def test_png_pixels_match_the_palette_for_every_cell(tmp_path, world):
    """Catches a swapped colour channel AND a reordered palette -- neither of
    which changes the file length, the header, or any CRC, so nothing else in
    this file would notice."""
    p = [q for q in write(world, tmp_path) if q.suffix == ".png"][0]
    px = _decode_png(p.read_bytes())["px"]
    assert np.array_equal(px, np.array(PALETTE, dtype=np.uint8)[world.terrain])
