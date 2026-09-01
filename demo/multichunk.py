"""A3: a multi-chunk world, stitched together in RASTER order.

A1 (`demo/seam_check.py`) measured the stitching hazard on this receipt:
single-edge clamping is honoured exactly (240/240) and still finds valid
worlds at a usable rate (31.25% under clamp vs ~25% unclamped -- clamping
HELPS), but when two chunks are generated INDEPENDENTLY as diagonal
neighbours and a third chunk is clamped against both of them at once, their
shared corner cell disagrees 52% of the time -- a clamp that is
self-contradictory before sampling even starts. Free/on-demand (any-order)
generation is therefore not viable and is out of scope here; see the A1
report for the numbers.

RASTER ORDER removes that hazard structurally, not statistically: chunk
(i,j)'s north neighbour is (i,j-1) and its west neighbour is (i-1,j); in
raster order those two neighbours were THEMSELVES generated with a shared
ancestor -- (i,j-1)'s own west neighbour and (i-1,j)'s own north neighbour
are the SAME chunk, (i-1,j-1). Both neighbours' contribution to (i,j)'s
corner cell g0_0 therefore traces back to that one chunk's own g{w-1}_{h-1}
value -- not two independently-sampled numbers that happen to coincide 48%
of the time, but literally the same clamped-through value read twice. See
`neighbour_clamp` below and `tests/test_multichunk.py`'s interior-chunk test
for this traced out concretely, with no sampling involved.

Boundary-column choice (the note A1 raised, carried forward here): each
chunk's own west column / north row is CLAMPED EQUAL to its neighbour's
east column / south row -- so every chunk keeps its own full 8x8 grid
(nothing is torn out or shared storage), and the assembled COLS*8 x ROWS*8
world DUPLICATES each interior boundary column/row (24x24 for a 3x3 grid of
8x8 chunks, not 22x22). Under that choice the seam pair AT the clamped
column itself (chunk A's x=7 vs chunk B's x=0) is checked here too, but
labelled TRIVIAL and excluded from the reported violation count: B's x=0
was clamped equal to A's x=7, so every such pair is (v, v) and can never be
the forbidden {water, rock} set -- checking it again would be checking that
a clamp we already verified honoured (A1a: 240/240) is honoured, not
checking a seam. The MEANINGFUL/HONEST seam check (A1a's own term) compares
chunk A's boundary column/row against chunk B's SECOND column/row (x=1 /
y=1) -- the first cell whose value the clamp did NOT dictate, i.e. the pair
that is actually free to disagree and is reported as this task's
seam-violation count.

Everything here rides on the compiler's OWN `enc.is_codeword` / `enc.decode`
/ `spec.contract.validate`, exactly the pattern `demo/render_world.py` and
`demo/seam_check.py` use -- no decode or contract logic is reimplemented.
`demo/lattice_app.py`'s OWN `render_world_image` is reused verbatim (called
once per chunk, the 9 results pasted into one canvas) rather than
reimplementing rendering math for a bigger grid.

Run: PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" demo/multichunk.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Mapping

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

from PIL import Image

from tsu.simulate import simulate
from tsu.spec import load_spec
from tsu.passes.encode import encode

R = "demo/receipts/small"
CHUNK_W = CHUNK_H = 8
WATER, ROCK, GRASS = 0, 1, 2


def _selected_encoding(receipt_dir) -> str:
    passes = json.loads((Path(receipt_dir) / "passes.json").read_text())
    for c in passes["candidates"]:
        if c["state"] == "SELECTED":
            return c["encoding"]
    raise ValueError(f"receipt at {receipt_dir} has no SELECTED candidate")


# ---------------------------------------------------------------------------
# Step 1: pure chunk-grid bookkeeping -- no sampling, no receipt access.
# ---------------------------------------------------------------------------

def raster_order(cols: int, rows: int):
    """Chunk coordinates (i, j) in raster order: left-to-right within a row,
    then top-to-bottom across rows. i is the COLUMN index (x-ish), j is the
    ROW index (y-ish) -- matching the g{x}_{y} naming a single chunk already
    uses, just one level up."""
    for j in range(rows):
        for i in range(cols):
            yield (i, j)


def neighbour_clamp(i: int, j: int, chunks: Mapping[tuple[int, int], Mapping[str, int]],
                     width: int, height: int) -> dict[str, int]:
    """The workload-level clamp for chunk (i, j), given a mapping of
    ALREADY-GENERATED chunk coordinates to their decoded {g{x}_{y}: value}
    grids. Only two neighbours are ever consulted -- north (i, j-1) and
    west (i-1, j) -- because raster order guarantees those are the only
    ones that can already exist:

      - north's SOUTH row (g{x}_{height-1}) becomes (i,j)'s NORTH row
        (g{x}_0), for every x.
      - west's EAST column (g{width-1}_{y}) becomes (i,j)'s WEST column
        (g0_{y}), for every y.

    A neighbour absent from `chunks` (edge/corner chunks, or a caller
    testing a case where it simply hasn't been generated yet) contributes
    no clamp entries at all -- never a fabricated default.

    The shared corner cell g0_0 is written by BOTH edges when both
    neighbours exist. Raster order makes them provably consistent (see the
    module docstring), but this still asserts it explicitly rather than
    silently letting a dict-literal update overwrite one value with the
    other -- an assertion failure here would mean generation order was
    NOT actually raster (e.g. a caller passed a diagonal neighbour instead
    of north/west), which is exactly the corner hazard A1 measured.
    """
    clamp: dict[str, int] = {}

    north = chunks.get((i, j - 1))
    north_clamp = ({f"g{x}_0": north[f"g{x}_{height - 1}"] for x in range(width)}
                    if north is not None else {})

    west = chunks.get((i - 1, j))
    west_clamp = ({f"g0_{y}": west[f"g{width - 1}_{y}"] for y in range(height)}
                   if west is not None else {})

    if north_clamp and west_clamp:
        a, b = north_clamp["g0_0"], west_clamp["g0_0"]
        if a != b:
            raise ValueError(
                f"corner disagreement for chunk ({i},{j}): north neighbour "
                f"(i={i}, j={j - 1}) says g0_0={a}, west neighbour "
                f"(i={i - 1}, j={j}) says g0_0={b} -- this should be "
                f"structurally impossible in raster order (see module "
                f"docstring); it would mean the caller is not actually "
                f"generating in raster order")

    clamp.update(north_clamp)
    clamp.update(west_clamp)
    return clamp


# ---------------------------------------------------------------------------
# Step 2: generate a chunk (sampling) -- reuses the verified simulate() +
# is_codeword/decode/contract.validate path, nothing hand-rolled.
# ---------------------------------------------------------------------------

def _classify(got, im, enc, spec):
    valid, noncodeword, invalid = [], 0, 0
    for row in got:
        bits = dict(zip(im.nodes, row.tolist()))
        if not enc.is_codeword(bits):
            noncodeword += 1
            continue
        dec = enc.decode(bits)
        if spec.contract.validate(dec).ok:
            valid.append(dec)
        else:
            invalid += 1
    return valid, noncodeword, invalid


def generate_chunk(receipt_dir, i, j, chunks, enc, spec, *, width=CHUNK_W,
                    height=CHUNK_H, seed_base=0, n_chains=6, n_samples=30,
                    n_warmup=600, max_tries=4):
    """Generate chunk (i, j): derive its clamp from already-generated raster
    neighbours (Step 1's `neighbour_clamp`), sample, and return the first
    valid decoded world found. Retries (RE-SAMPLES, not re-derives the
    clamp) up to `max_tries` times with a different seed if the first batch
    yields zero valid draws -- this is the "did this chunk need
    re-sampling" count the report tracks.

    Returns (decoded_dict, n_attempts, wall_time_s). Raises RuntimeError if
    no valid draw turns up in max_tries attempts -- a measured failure to
    report, not something to paper over with more attempts than declared.
    """
    clamp = neighbour_clamp(i, j, chunks, width, height)
    t0 = time.perf_counter()
    for attempt in range(max_tries):
        seed = seed_base + (j * 1000 + i * 37) + attempt * 97
        _path, got, im = simulate(receipt_dir, n_chains=n_chains,
                                  n_samples=n_samples, n_warmup=n_warmup,
                                  seed=seed, clamp=clamp)
        valid, _, _ = _classify(got, im, enc, spec)
        if valid:
            wall = time.perf_counter() - t0
            return valid[0], attempt + 1, wall
    wall = time.perf_counter() - t0
    raise RuntimeError(
        f"chunk ({i},{j}): no valid draw in {max_tries} attempts "
        f"(clamp={clamp}) -- this is itself a finding, not a bug to hide")


def generate_world(receipt_dir, cols, rows, enc, spec, *, seed_base=0,
                    **gen_kwargs):
    """Generate every chunk of a cols x rows grid in raster order. Returns
    (chunks dict, list of per-chunk (coord, n_attempts, wall_time_s))."""
    chunks: dict[tuple[int, int], dict[str, int]] = {}
    log = []
    for (i, j) in raster_order(cols, rows):
        decoded, n_attempts, wall = generate_chunk(
            receipt_dir, i, j, chunks, enc, spec, seed_base=seed_base,
            **gen_kwargs)
        chunks[(i, j)] = decoded
        log.append(((i, j), n_attempts, wall))
    return chunks, log


# ---------------------------------------------------------------------------
# Step 3: seam validation -- every adjacent chunk pair, both the TRIVIAL
# (clamped/duplicated) column and the HONEST (first free column) pair. See
# the module docstring for why only the honest count is reported as
# "violations".
# ---------------------------------------------------------------------------

def _pairs_illegal(cells):
    """cells: iterable of (value_a, value_b) pairs. Returns the ones that
    are the contract's forbidden set: water directly adjacent to rock."""
    return [(a, b) for (a, b) in cells if {a, b} == {WATER, ROCK}]


def check_seams(chunks: Mapping[tuple[int, int], Mapping[str, int]], cols: int,
                 rows: int, width=CHUNK_W, height=CHUNK_H):
    """For every horizontally- and vertically-adjacent chunk pair, check
    both the trivial boundary (expected legal by construction -- the clamp
    forced it) and the honest boundary (the first actually-free cells).
    Returns a dict with both counts broken out, plus the honest violations
    themselves (never hidden)."""
    trivial_bad = []
    honest_bad = []
    trivial_pairs_checked = 0
    honest_pairs_checked = 0

    for j in range(rows):
        for i in range(cols):
            # horizontal seam: (i,j) west of (i+1,j)
            if i + 1 < cols:
                L, Rt = chunks[(i, j)], chunks[(i + 1, j)]
                trivial = [(L[f"g{width-1}_{y}"], Rt[f"g0_{y}"]) for y in range(height)]
                honest = [(L[f"g{width-1}_{y}"], Rt[f"g1_{y}"]) for y in range(height)]
                trivial_pairs_checked += len(trivial)
                honest_pairs_checked += len(honest)
                for p in _pairs_illegal(trivial):
                    trivial_bad.append({"seam": "h", "left": (i, j), "right": (i + 1, j), "pair": p})
                for p in _pairs_illegal(honest):
                    honest_bad.append({"seam": "h", "left": (i, j), "right": (i + 1, j), "pair": p})
            # vertical seam: (i,j) north of (i,j+1)
            if j + 1 < rows:
                Tp, Bt = chunks[(i, j)], chunks[(i, j + 1)]
                trivial = [(Tp[f"g{x}_{height-1}"], Bt[f"g{x}_0"]) for x in range(width)]
                honest = [(Tp[f"g{x}_{height-1}"], Bt[f"g{x}_1"]) for x in range(width)]
                trivial_pairs_checked += len(trivial)
                honest_pairs_checked += len(honest)
                for p in _pairs_illegal(trivial):
                    trivial_bad.append({"seam": "v", "top": (i, j), "bottom": (i, j + 1), "pair": p})
                for p in _pairs_illegal(honest):
                    honest_bad.append({"seam": "v", "top": (i, j), "bottom": (i, j + 1), "pair": p})

    return {
        "trivial_pairs_checked": trivial_pairs_checked,
        "trivial_violations": trivial_bad,
        "honest_pairs_checked": honest_pairs_checked,
        "honest_violations": honest_bad,
    }


# ---------------------------------------------------------------------------
# Step 4: render the stitched world with the EXISTING renderer
# (lattice_app.render_world_image), called once per chunk and pasted.
# ---------------------------------------------------------------------------

def render_multichunk(chunks: Mapping[tuple[int, int], "any"], cols: int, rows: int,
                       out_path: str, up: int = 32, width=CHUNK_W, height=CHUNK_H):
    import numpy as np
    from lattice_app import render_world_image  # local import: keeps this
    # module importable (by tests) without pulling in tkinter at import time

    canvas = Image.new("RGB", (cols * width * up, rows * height * up))
    for (i, j), decoded in chunks.items():
        grid = np.array([[int(decoded[f"g{x}_{y}"]) for x in range(width)]
                          for y in range(height)])
        tile = render_world_image(grid, up=up)
        canvas.paste(tile, (i * width * up, j * height * up))
    canvas.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Script entry point -- Steps 2-5: generate a 3x3 (24x24) world, validate
# every seam, render it, report timing and re-sample counts.
# ---------------------------------------------------------------------------

def main():
    spec = load_spec(str(Path(R) / "spec.yaml"))
    enc = encode(spec, _selected_encoding(R))

    cols = rows = 3
    print("=" * 70)
    print(f"A3 -- {cols}x{rows} chunk world ({cols*CHUNK_W}x{rows*CHUNK_H} cells), "
          f"RASTER order (decision carried over from A1 -- see module docstring)")
    print("=" * 70)

    t_total0 = time.perf_counter()
    chunks, log = generate_world(R, cols, rows, enc, spec, seed_base=2_000_000)
    total_wall = time.perf_counter() - t_total0

    resampled = 0
    for (i, j), n_attempts, wall in log:
        tag = "" if n_attempts == 1 else f"  <-- needed {n_attempts} attempts (RESAMPLED)"
        print(f"  chunk ({i},{j}): {n_attempts} attempt(s), {wall:.3f}s{tag}")
        if n_attempts > 1:
            resampled += 1

    print()
    print(f"per-chunk wall time: min={min(w for _,_,w in log):.3f}s "
          f"max={max(w for _,_,w in log):.3f}s "
          f"mean={sum(w for _,_,w in log)/len(log):.3f}s")
    print(f"total wall time ({cols*rows} chunks): {total_wall:.3f}s")
    print(f"chunks needing re-sampling (first attempt yielded no valid world): "
          f"{resampled}/{len(log)}")

    print()
    print("-" * 70)
    print("seam validation")
    print("-" * 70)
    seams = check_seams(chunks, cols, rows)
    print(f"TRIVIAL boundary (clamped/duplicated column, expected legal by "
          f"construction): {len(seams['trivial_violations'])} violations "
          f"across {seams['trivial_pairs_checked']} pairs checked")
    print(f"HONEST boundary (first FREE column past the clamp -- the pair "
          f"that actually tests whether stitching produced an illegal "
          f"world): {len(seams['honest_violations'])} violations across "
          f"{seams['honest_pairs_checked']} pairs checked")
    if seams["honest_violations"]:
        print("  HONEST VIOLATIONS (reported, not hidden):")
        for v in seams["honest_violations"]:
            print(f"    {v}")
    else:
        print("  no honest-seam violations found in this world")

    out_path = "demo/world_multi.png"
    render_multichunk(chunks, cols, rows, out_path)
    print()
    print(f"rendered -> {out_path}")

    return {
        "cols": cols, "rows": rows,
        "total_wall_s": total_wall,
        "per_chunk_wall_s": [w for _, _, w in log],
        "resampled_chunks": resampled,
        "n_chunks": len(log),
        "seams": {
            "trivial_pairs_checked": seams["trivial_pairs_checked"],
            "trivial_violations": len(seams["trivial_violations"]),
            "honest_pairs_checked": seams["honest_pairs_checked"],
            "honest_violations": len(seams["honest_violations"]),
        },
    }


if __name__ == "__main__":
    result = main()
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))
