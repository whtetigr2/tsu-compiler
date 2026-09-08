"""Task 6: export. Uses one small generated world, reused across tests."""
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate
from world.export import to_text, to_json, write
from world.spline import TERRAINS


@pytest.fixture(scope="module")
def world():
    return generate(size=64, seed=0, warmup=200)


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


def test_write_produces_all_three_files(tmp_path, world):
    paths = write(world, tmp_path)
    names = {p.name for p in paths}
    assert names == {"world.json", "world.txt", "world.png"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_written_png_has_a_valid_signature_and_ihdr(tmp_path, world):
    """Asserting the file is non-empty would pass on garbage. Check the PNG
    magic bytes and that the IHDR declares the right dimensions."""
    p = [q for q in write(world, tmp_path) if q.suffix == ".png"][0]
    raw = p.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert raw[12:16] == b"IHDR"
    w = int.from_bytes(raw[16:20], "big")
    h = int.from_bytes(raw[20:24], "big")
    assert (w, h) == (world.size, world.size)
