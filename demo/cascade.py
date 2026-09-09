"""Task 1 of the coarse-to-fine cascade plan
(SPR/docs/superpowers/plans/2026-09-07-coarse-to-fine-cascade.md): upsampling
a coarser level's decode, and building the `demo/layers.bias_patch`-compatible
conditioning patch that rewards a finer level for keeping what it inherited.

THE HYPOTHESIS THIS SERVES (see the plan's own "why the last diagnosis was
probably wrong"): a local pairwise rule has a correlation length -- at 8x8
"water clumps with water" is a continental-scale rule, at 64x64 the same
rule is a puddle rule. This module is the mechanism for deciding structure
once, at the top (where the rule spans a meaningful fraction of the grid),
and carrying it down rather than re-deciding it independently, and
identically-blindly, at every scale.

Both functions here are pure -- no sampling, no randomness, no I/O -- the
same design demo/elevation.py's `thermometer_level`/`monotonicity_violations`
and demo/binary_world.py's `compose_state` use for their own composition
logic: tested directly as ordinary Python (tests/test_cascade.py), with
sampling-dependent measurement left to demo/cascade_world.py.

VALUES ARE CARRIED, NEVER INTERPOLATED. A coarse cell's value (e.g. 0/1/2
for water/rock/grass on lattice_small_8x8_k3.yaml's domain) is a categorical
state, not a number on a continuum -- averaging two categorical states (or
worse, four of them) would invent a state nobody asked for and that may not
even exist in the domain (e.g. "1.5" is not a value of a 3-way categorical
variable). `upsample` therefore REPLICATES each coarse cell's value across
its entire fine-resolution block; it never blends, rounds, or interpolates.

CELL NAMING CONVENTION: every grid cell in this project is named
`g<x>_<y>` (see specs/*.yaml's `generate: {kind: grid, ...}` and every
demo/*.py module's own `f"g{x}_{y}"` grid-indexing convention) -- `x` the
column, `y` the row, both 0-based. `upsample` parses that convention on the
way in and re-emits it (at the finer grid's own `g<x>_<y>` coordinates) on
the way out; it is not a new naming scheme.
"""
from __future__ import annotations

import re
from typing import Mapping

_CELL_RE = re.compile(r"^g(\d+)_(\d+)$")


def _parse_cell(name: str) -> tuple[int, int]:
    m = _CELL_RE.match(name)
    if not m:
        raise ValueError(
            f"cell name {name!r} does not match this project's g<x>_<y> "
            f"grid-naming convention (see specs/*.yaml's grid generator)")
    return int(m.group(1)), int(m.group(2))


def upsample(decoded: Mapping[str, int], src_w: int, factor: int) -> dict[str, int]:
    """Expand a coarse decoded grid into one `factor` times larger in each
    dimension, replicating each coarse cell's value across its own
    `factor` x `factor` block of fine cells.

    `decoded`: a decoded assignment at the coarse resolution, e.g. exactly
    what `Encoded.decode` returns for a compiled receipt -- `{cell_name:
    value}`, one entry per coarse cell.
    `src_w`: the coarse grid's width. Needed to split each `g<x>_<y>` name's
    `x` from `y` unambiguously (the height is inferred as
    `len(decoded) // src_w`, since every receipt this project compiles is a
    complete rectangular grid -- see `generate: {kind: grid, width, height}`
    in every specs/*.yaml this project uses).
    `factor`: the integer upsampling factor (e.g. 2 for 8x8 -> 16x16).

    Returns a `{cell_name: value}` dict at the fine resolution
    (`src_w * factor` wide, `len(decoded)//src_w * factor` tall), of size
    `len(decoded) * factor**2`.

    VALUES ARE CARRIED, NEVER INTERPOLATED -- see this module's own
    docstring. Every fine cell in a given coarse cell's block gets that
    coarse cell's EXACT value, unchanged.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    src_w = int(src_w)
    if src_w < 1:
        raise ValueError(f"src_w must be >= 1, got {src_w}")
    if len(decoded) % src_w != 0:
        raise ValueError(
            f"len(decoded)={len(decoded)} is not a multiple of src_w="
            f"{src_w}; cannot infer a rectangular grid height from it")
    src_h = len(decoded) // src_w

    fine: dict[str, int] = {}
    for name, value in decoded.items():
        cx, cy = _parse_cell(name)
        if cx >= src_w or cy >= src_h:
            raise ValueError(
                f"cell {name!r} (parsed x={cx}, y={cy}) falls outside the "
                f"{src_w}x{src_h} grid implied by src_w={src_w} and "
                f"len(decoded)={len(decoded)}")
        base_x, base_y = cx * factor, cy * factor
        for dy in range(factor):
            for dx in range(factor):
                fine[f"g{base_x + dx}_{base_y + dy}"] = value
    return fine


def cascade_patch(coarse_up: Mapping[str, int],
                   strength: float) -> dict[tuple[str, int], float]:
    """Build the `demo/layers.bias_patch`-compatible `{(cell, value):
    weight}` patch that rewards each fine cell for keeping the value it
    inherited from `upsample()` -- and touches nothing else.

    `coarse_up`: an ALREADY-UPSAMPLED coarse decode (e.g. straight out of
    `upsample()`) -- one inherited value per fine cell.
    `strength`: the per-cell reward magnitude handed straight to
    `bias_patch` as that (cell, value) pair's weight (matching
    demo/elevation.py's `band_patch` and demo/binary_world.py's
    `build_relocation_patch`, which both hand their own `strength` argument
    to `bias_patch` the same unscaled way). Per `bias_patch`'s own sign
    convention, a positive `strength` ENCOURAGES the inherited value; this
    function does not clamp or validate its sign, since a caller sweeping
    `strength` down through zero or negative (to see the reward turn off or
    invert) is a legitimate use, not a misuse.

    Only the inherited `(cell, value)` pair gets an entry. No entry is
    written for any OTHER value a fine cell's domain might hold:
    `cascade_patch` is never told how many values a domain has (it only
    ever sees the one inherited value per cell), so it cannot enumerate
    "every other value" without guessing a domain size nobody gave it --
    the same one-sided-reward reasoning `demo/elevation.py`'s `band_patch`
    and `demo/layers.py`'s own module docstring already use. A patch that
    also rewarded every OTHER value equally would reward nothing in
    particular -- exactly the failure mode this function's own test
    (`tests/test_cascade.py::test_cascade_patch_favours_the_inherited_value`)
    exists to catch.
    """
    strength = float(strength)
    return {(cell, value): strength for cell, value in coarse_up.items()}
