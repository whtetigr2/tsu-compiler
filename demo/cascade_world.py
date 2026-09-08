"""Task 2 of the coarse-to-fine cascade plan
(SPR/docs/superpowers/plans/2026-09-07-coarse-to-fine-cascade.md): two
levels, and the first real test of the hypothesis.

BEFORE BUILDING FOUR LEVELS, PROVE TWO. If 8x8 -> 16x16 does not preserve
coarse structure, the cascade is dead and Task 3/4 never run. This module:

1. Samples the existing 8x8 base receipt (demo/receipts/small -- hard
   rules, already compiled) via `tsu.simulate.simulate`. NEVER recompiles
   it (`simulate()` reuses the receipt's own compiled program verbatim --
   see that function's own module docstring). `output_dir` is always
   outside the receipt directory (tempfile.gettempdir(), matching
   demo/lattice_app.py's own SIM_OUTPUT_DIR precedent) so an ordinary
   sampling run never mutates git-tracked compile-time evidence (RP-1).

2. Upsamples that decode to 16x16 (`cascade.upsample`) and builds the
   conditioning patch (`cascade.cascade_patch`).

3. Compiles a FRESH 16x16 bipartite layer and samples it under the patch
   (`demo/layers.bias_patch`).

THE FINE LAYER'S DESIGN, stated up front, not discovered mid-task: k=3
categorical SELF-rules-only was already measured NON-bipartite even at 8x8
(this repo's own commit d3c6298, "Route B is dead" -- part of the prior
all-bipartite-world plan's history on this same branch). Any
product_over_edges term on a k=3 categorical domain therefore risks the
exact 6-32 minute place()-on-non-bipartite failure the plan's Global
Constraints warn about by name. This module's fine layer instead carries
ZERO product_over_edges terms -- no self-rule, no cross-rule -- so the ONLY
structure in its Ising graph is domain_wall's own per-cell representation-
penalty chain (2 physical spins per k=3 cell, one intra-cell edge, zero
inter-cell edges). A disjoint union of 2-node chains has no cycles at all,
so it is bipartite BY CONSTRUCTION -- confirmed, not assumed, via
`analyse(...).bipartite is True` before every `place()` call (THE SAFETY
RULE, same as `demo/binary_world.py::_compile_layer` and
`audit/bipartite_routes.py::bipartite_at`). This also isolates exactly what
Task 2 needs to measure: with no clumping term of its own, every bit of
structure the fine layer's decode exhibits must come from the conditioning
patch -- there is no confound from a fine-level self-rule also pulling
cells toward their neighbours' values.

MANDATORY CAVEAT (same one demo/layers.py, demo/elevation.py, and
demo/binary_world.py already carry): conditioning the fine layer on the
coarse layer's decode is DIRECTED / ANCESTRAL sampling -- p(coarse) *
p(fine | coarse) -- never the joint Boltzmann distribution over both
levels at once. The coarse level can never be influenced by the fine
level's draw.

POOLING, not a single backdrop: the inheritance measurement pools over
MULTIPLE INDEPENDENT coarse parents (distinct seeds, each a genuinely
different sampled 8x8 world) AND every valid fine-layer draw per parent --
never one draw of one parent. `demo/binary_world.py`'s own module docstring
explains why a single backdrop is a degenerate instance for a measurement
like this one (rare eligible configurations can be entirely absent from
one draw, producing a spurious extreme rate that looks like evidence but
is a sampling artifact) -- the same caution applies here, and this module
follows the same fix.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

import numpy as np

from tsu.spec import load_spec
from tsu.passes.encode import encode
from tsu.passes.lower import lower
from tsu.passes.analyse import analyse
from tsu.passes.place import place
from tsu.passes.route import route
from tsu.passes.program import build_program
from tsu.target import PROFILES
from tsu.backends.thrml_backend import sample as thrml_sample
from tsu.simulate import simulate

from layers import bias_patch, FieldCapExceeded, FIELD_CAP
from cascade import upsample, cascade_patch

BASE_RECEIPT = "demo/receipts/small"
BASE_W = BASE_H = 8

# Matches demo/render_world.py / demo/layers.py's own test scale for this
# receipt (~0.4-1.4s per call, measured there) -- reused rather than
# re-derived.
BASE_SAMPLE_PARAMS = dict(n_chains=8, n_samples=60, n_warmup=1200, steps_per_sample=4)
FINE_SAMPLE_PARAMS = dict(n_chains=8, n_samples=60, n_warmup=1200, steps_per_sample=4)

# simulate() writes simulation.json to `output_dir`; this must NEVER be the
# receipt directory itself (RP-1 -- demo/receipts/small is git-tracked
# compile-time evidence). Matches demo/lattice_app.py's SIM_OUTPUT_DIR
# precedent exactly (Path(tempfile.gettempdir()) / "tsu_..._sim_output").
SIM_OUTPUT_DIR = Path(tempfile.gettempdir()) / "tsu_cascade_sim_output"

_FINE_LAYER_BODY = """name: cascade_fine
generate: {{kind: grid, width: {n}, height: {n}, variable_domain: {{domain: categorical, k: 3}}}}
terms: []
"""


def _compile_fine_layer(n: int):
    """Compile a fresh n x n, k=3 categorical layer with ZERO
    product_over_edges terms (see this module's docstring for why). Returns
    (spec, enc, prog). Refuses to place() unless analyse() first confirms
    bipartite -- THE SAFETY RULE, non-negotiable per the plan's Global
    Constraints."""
    f = os.path.join(tempfile.gettempdir(), f"cascade_fine_{n}.yaml")
    with open(f, "w") as fh:
        fh.write(_FINE_LAYER_BODY.format(n=n))
    spec = load_spec(f)
    enc = encode(spec, "domain_wall")
    ising = lower(enc.model)
    report = analyse(ising)
    assert report.bipartite is True, (
        f"{n}x{n} fine layer: analyse() reports bipartite=False -- "
        f"refusing place() per THE SAFETY RULE (place() above 8x8 on a "
        f"non-bipartite graph costs 6-32 minutes to FAIL); this IS the "
        f"finding, not a bug to route around (plan Global Constraints)")
    target = PROFILES["z1"]
    placement = place(ising, report, target, restarts=12, iters=200_000)
    assert placement.mediation is None, (
        f"{n}x{n} fine layer: place() mediated a graph analyse() reported "
        f"bipartite -- should be unreachable, see place()'s own docstring")
    ising = route(ising, report, target)
    prog = build_program(ising, report)
    return spec, enc, prog, report


def _decode_codewords(rows, im, enc) -> list[dict]:
    """Every DECODED codeword row (workload variable names -> values).
    Non-codeword rows (domain_wall chains that didn't land on a monotone
    codeword) are dropped, never repaired -- matches
    demo/elevation_world.py's and demo/binary_world.py's own
    `_decode_codewords`/`_decode_draws` helpers exactly."""
    out = []
    for row in rows:
        bits = dict(zip(im.nodes, row.tolist()))
        if enc.is_codeword(bits):
            out.append(enc.decode(bits))
    return out


def _base_decode(seed: int) -> dict:
    """Sample the existing, already-compiled 8x8 base receipt via
    `tsu.simulate.simulate` (NEVER recompiled) and return its ONE
    representative valid decode (`decoded_example.decoded`). Raises if this
    run drew no valid world -- not fabricating a fallback (matches
    demo/elevation_world.py's own "not fabricating a fallback world" refusal)."""
    SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _got, _im = simulate(
        BASE_RECEIPT, seed=seed, output_dir=SIM_OUTPUT_DIR, **BASE_SAMPLE_PARAMS)
    doc = json.loads(Path(out_path).read_text())
    ex = doc["decoded_example"]
    if not (ex["is_codeword"] and ex["task_valid"] and ex["decoded"] is not None):
        raise SystemExit(
            f"base layer (seed={seed}): no valid world sampled -- refusing "
            f"to fabricate a fallback (task_validity={doc['task_validity']}, "
            f"decoded_example={ex!r})")
    return ex["decoded"]


# ---------------------------------------------------------------------------
# Interface: run_cascade(levels, strength, seed) -> list[dict], coarsest first.
# ---------------------------------------------------------------------------

def run_cascade(levels: list[int], strength: float, seed: int) -> list[dict]:
    """Run the cascade across `levels` (e.g. [8, 16]), coarsest first.
    `levels[0]` MUST be 8 -- that is the only size with an existing compiled
    receipt (demo/receipts/small) to sample from without recompiling
    anything, which is the one hard constraint this plan will not cross.

    Each level i>0 is: upsample level i-1's representative decode to level
    i's resolution, build a `cascade_patch` at `strength`, compile a FRESH
    bipartite layer at that size (`_compile_fine_layer`), sample it under
    the patch (`demo/layers.bias_patch`), and keep ONE representative valid
    decode (the first valid codeword drawn) to carry forward -- this is the
    ANCESTRAL/DIRECTED sampling this module's docstring describes: each
    level conditions on ONE realized draw of the level above it, not on a
    distribution over draws.

    Returns one decoded `{cell_name: value}` dict per level, coarsest
    first -- `len(result) == len(levels)`.
    """
    if not levels or levels[0] != BASE_W:
        raise ValueError(
            f"levels[0] must be {BASE_W} (the only size with an existing "
            f"compiled receipt, demo/receipts/small); got {levels!r}")

    results: list[dict] = [_base_decode(seed)]
    prev_w = levels[0]
    for level_i, n in enumerate(levels[1:], start=1):
        if n % prev_w != 0:
            raise ValueError(
                f"level {n} is not an integer multiple of the previous "
                f"level {prev_w}; upsample() requires an integer factor")
        factor = n // prev_w
        coarse_up = upsample(results[-1], src_w=prev_w, factor=factor)
        patch = cascade_patch(coarse_up, strength=strength)

        _spec, enc, prog, _report = _compile_fine_layer(n)
        patched = bias_patch(prog, enc, patch, field_cap=FIELD_CAP)
        rows = thrml_sample(patched, seed=seed + level_i, **FINE_SAMPLE_PARAMS)
        decoded_list = _decode_codewords(rows, patched.ising, enc)
        if not decoded_list:
            raise SystemExit(
                f"level {n} (seed={seed}, strength={strength}): no valid "
                f"codeword drawn -- refusing to fabricate a fallback world")
        results.append(decoded_list[0])
        prev_w = n
    return results


# ---------------------------------------------------------------------------
# The measurement that decides the plan: pooled inheritance fraction.
# ---------------------------------------------------------------------------

def measure_inheritance(strength: float, parent_seeds: list[int],
                         src_w: int = BASE_W, factor: int = 2,
                         fine_layer=None, out_path: str | None = None,
                         coarse_decode_fn=None) -> dict:
    """For each of `parent_seeds` (INDEPENDENT 8x8 base draws -- never one
    backdrop, see this module's docstring), sample the fine layer under the
    strength-`strength` cascade patch and pool, over EVERY valid fine draw
    of EVERY parent, what fraction of (coarse cell, child) pairs kept the
    parent's inherited value.

    `fine_layer`: an already-compiled (spec, enc, prog, report) tuple for
    the fine grid size (`_compile_fine_layer(src_w * factor)`), so a caller
    sweeping `strength` (Step 5) compiles the bipartite layer ONCE and only
    re-patches/re-samples per strength -- place() is a few milliseconds for
    this graph, but there is no reason to pay it n_sweep times when bias_patch
    alone changes between strengths (topology is untouched, see
    demo/layers.py's own module docstring).

    Returns a dict with the per-parent breakdown (never just the pooled
    total -- so a reader can see whether the pooled number is being carried
    by one unusually cooperative parent) and the pooled fraction. Writes
    incrementally to `out_path` (json.dump after every parent) if given, so
    a killed run keeps every completed parent's contribution.

    `coarse_decode_fn`: how to obtain each parent seed's "coarse" backdrop
    to upsample from. None (the default) reproduces Task 2's own behaviour
    exactly -- `_base_decode(seed)`, the 8x8 base receipt. Task 3 passes
    `_repr_decode_upto(seed, src_w, strength)` here to measure a DEEPER
    transition (16->32, 32->64), where the correct "coarse" backdrop is the
    real ancestrally-sampled decode a live cascade would have produced at
    that level for that seed -- never a fresh, independently-drawn sample
    at that resolution, which would not be this seed's own cascade lineage.
    """
    n_fine = src_w * factor
    if fine_layer is None:
        fine_layer = _compile_fine_layer(n_fine)
    _spec, enc, prog, _report = fine_layer
    if coarse_decode_fn is None:
        coarse_decode_fn = _base_decode

    per_parent = []
    total_matches = 0
    total_children = 0
    for seed in parent_seeds:
        base_decoded = coarse_decode_fn(seed)
        coarse_up = upsample(base_decoded, src_w=src_w, factor=factor)
        patch = cascade_patch(coarse_up, strength=strength)
        try:
            patched = bias_patch(prog, enc, patch, field_cap=FIELD_CAP)
        except FieldCapExceeded as e:
            per_parent.append(dict(seed=seed, error=f"FieldCapExceeded: {e}"))
            if out_path:
                _write_incremental(out_path, strength, parent_seeds, per_parent,
                                    total_matches, total_children)
            continue

        rows = thrml_sample(patched, seed=seed + 1000, **FINE_SAMPLE_PARAMS)
        decoded_list = _decode_codewords(rows, patched.ising, enc)
        if not decoded_list:
            per_parent.append(dict(seed=seed, error="no valid codeword drawn"))
            if out_path:
                _write_incremental(out_path, strength, parent_seeds, per_parent,
                                    total_matches, total_children)
            continue

        parent_matches = 0
        parent_children = 0
        for decoded in decoded_list:
            for cell, parent_val in coarse_up.items():
                parent_children += 1
                if decoded[cell] == parent_val:
                    parent_matches += 1
        per_parent.append(dict(
            seed=seed, n_valid_draws=len(decoded_list), n_total_draws=len(rows),
            matches=parent_matches, children=parent_children,
            fraction=parent_matches / parent_children if parent_children else None))
        total_matches += parent_matches
        total_children += parent_children
        if out_path:
            _write_incremental(out_path, strength, parent_seeds, per_parent,
                                total_matches, total_children)

    result = dict(
        strength=strength, src_w=src_w, factor=factor, n_fine=n_fine,
        parent_seeds=parent_seeds, per_parent=per_parent,
        total_matches=total_matches, total_children=total_children,
        pooled_fraction=(total_matches / total_children
                        if total_children else None))
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def _write_incremental(out_path, strength, parent_seeds, per_parent,
                       total_matches, total_children):
    partial = dict(strength=strength, parent_seeds=parent_seeds,
                   per_parent=per_parent, total_matches=total_matches,
                   total_children=total_children,
                   pooled_fraction=(total_matches / total_children
                                   if total_children else None),
                   status="in_progress")
    Path(out_path).write_text(json.dumps(partial, indent=2))


# Independent backdrops -- five distinct seeds for the coarse 8x8 base, not
# one repeated draw. Reused across every strength in the Step 5 sweep so the
# comparison is apples-to-apples (same five worlds, only the fine-layer
# conditioning strength changes).
PARENT_SEEDS = [7, 71, 137, 211, 389]

# Step 5's sweep. Upper end chosen empirically at run time (see main()):
# swept until bias_patch starts raising FieldCapExceeded, which is recorded
# as a per-parent error rather than crashing the sweep (verification never
# fabricates; a refused patch IS a data point, not a bug to hide).
STRENGTH_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)


def main_task2():
    """Task 2's own entry point, UNCHANGED from before Task 3/4 existed --
    kept runnable verbatim (still reproduces the committed 0.5764 headline
    and the full sweep) via `python demo/cascade_world.py task2`."""
    out_dir = Path("demo/cascade_runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Steps 1-3: one concrete two-level cascade, for inspection. ---
    print("=== Steps 1-3: one concrete 8x8 -> 16x16 cascade (seed=7, strength=1.0) ===")
    t0 = time.perf_counter()
    levels_out = run_cascade([8, 16], strength=1.0, seed=7)
    wall = time.perf_counter() - t0
    print(f"run_cascade([8,16], strength=1.0, seed=7) wall={wall:.3f}s")
    print(f"level 0 (8x8): {len(levels_out[0])} cells")
    print(f"level 1 (16x16): {len(levels_out[1])} cells")

    # --- Step 4/5: the pooled inheritance measurement, swept over strength. ---
    print("\n=== Step 4/5: pooled inheritance vs strength "
          f"({len(PARENT_SEEDS)} independent 8x8 parents per strength) ===")
    fine_layer = _compile_fine_layer(16)
    print(f"fine layer (16x16): bipartite={fine_layer[3].bipartite}, "
          f"n_nodes={fine_layer[3].n_nodes}, n_edges={fine_layer[3].n_edges}, "
          f"max_abs_J={fine_layer[3].max_abs_J}, max_abs_b={fine_layer[3].max_abs_b}")

    sweep_results = []
    sweep_path = out_dir / "inheritance_sweep.json"
    for strength in STRENGTH_SWEEP:
        t0 = time.perf_counter()
        res = measure_inheritance(
            strength, PARENT_SEEDS, src_w=8, factor=2, fine_layer=fine_layer,
            out_path=str(out_dir / f"inheritance_strength_{strength}.json"))
        wall = time.perf_counter() - t0
        res["wall_s"] = wall
        sweep_results.append(res)
        frac = res["pooled_fraction"]
        frac_str = f"{frac:.4f}" if frac is not None else "unavailable"
        print(f"strength={strength:>4}: pooled_fraction={frac_str}  "
              f"({res['total_matches']}/{res['total_children']})  "
              f"wall={wall:.2f}s", flush=True)
        Path(sweep_path).write_text(json.dumps(sweep_results, indent=2))

    print(f"\nwrote {sweep_path}")


# ===========================================================================
# Task 3 of the coarse-to-fine cascade plan: the full cascade, 8->16->32->64.
#
# `run_cascade` (Task 2, above) already generalises over `levels` and is not
# touched here -- its signature and behaviour stay exactly what Task 2
# measured and committed. Task 3 needs strictly MORE return data per level
# (wall time, valid fraction, state mix) than run_cascade's own "one
# representative decode per level" contract promises, so this is a NEW
# function (`run_full_cascade`) rather than an in-place change to Task 2's.
# ===========================================================================

from tsu.simulate import _selected_encoding  # noqa: E402  (see module's own
                                              # sys.path.insert(0, "src") above)

# The renderer, adapted from demo/render_world.py / demo/lattice_app.py's own
# render_world_image -- see render_grid_image's own docstring for exactly
# why this module carries its own parameterised copy instead of importing
# lattice_app's version directly.
from scipy import ndimage  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

WATER, ROCK, GRASS = 0, 1, 2
PAL = np.array([[46, 92, 132], [124, 116, 106], [126, 158, 84]], float)


def _grid_of_decoded(decoded: dict, n: int) -> np.ndarray:
    """A decoded `{cell_name: value}` dict at resolution n x n, as a
    (row=y, col=x) numpy array -- matches every other _grid_of helper in
    this project (demo/binary_world.py, demo/elevation_world.py,
    demo/render_world.py's own inline version)."""
    return np.array([[int(decoded[f"g{x}_{y}"]) for x in range(n)] for y in range(n)])


def _state_mix(decoded_list: list[dict]) -> dict:
    """Fraction of cells at each of the domain's 3 values (0=water, 1=rock,
    2=grass), POOLED over every decoded grid in `decoded_list` -- matches
    demo/render_world.py's own single-draw 'cell counts' printout,
    generalised to pool over many draws rather than reading just one, same
    non-degeneracy discipline as everywhere else in this module."""
    counts = {0: 0, 1: 0, 2: 0}
    total = 0
    for decoded in decoded_list:
        for v in decoded.values():
            counts[v] = counts.get(v, 0) + 1
            total += 1
    return {str(k): (counts[k] / total if total else None) for k in (0, 1, 2)}


def render_grid_image(grid: np.ndarray, up: int) -> "Image.Image":
    """Render a decoded n x n categorical grid to a coloured image.
    ADAPTED, verbatim in its math, from demo/render_world.py's own
    module-level rendering code (also lifted, parameterised only on `up`,
    into demo/lattice_app.py::render_world_image) -- same palette, same
    pseudo-relief shading, same per-cell decoration, same shoreline
    softening. Shading and decoration remain DETERMINISTIC functions of the
    decoded cell values only (spec S1): same cell -> same pixels, always.

    NOT calling demo.lattice_app.render_world_image directly: that
    function's own per-cell decoration loop iterates `range(H)` / `range(W)`
    from lattice_app.py's OWN module-level globals (`W = H = 8`, set for
    that module's own 8x8 dashboard use), never `grid.shape` -- calling it
    on any other resolution would silently decorate only the top-left 8x8
    block of a larger canvas, leaving the remainder with base colour and
    shading but no per-cell decoration. This is a genuine latent bug in
    that function for a non-8x8 caller, discovered while building this
    renderer for the cascade's 16/32/64 levels; it is recorded here rather
    than fixed there (lattice_app.py is a large, separately-scoped
    dashboard module and fixing another module's bug is outside this
    plan's own file list). This function is this module's own
    correctly-parameterised copy, following the same per-module-copy
    precedent demo/binary_world.py, demo/stacked_world.py, and
    demo/elevation_world.py already use rather than importing a renderer
    tied to another module's fixed grid size.
    """
    h, w = grid.shape
    base = PAL[np.kron(grid, np.ones((up, up), int))]
    hgt = np.choose(grid, [0.0, 2.0, 1.0])
    ef = ndimage.gaussian_filter(ndimage.zoom(hgt, up, order=3), up * 0.5)
    gy, gx = [g * up * 1.4 for g in np.gradient(ef)]
    slope, aspect = np.arctan(np.hypot(gy, gx)), np.arctan2(-gx, gy)
    az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope)
                     * np.cos(az - aspect), 0, None) ** 0.85
    shade = 0.58 + 0.92 * shade

    deco = np.zeros(base.shape[:2])
    yy, xx = np.mgrid[0:up, 0:up] / float(up)
    for r in range(h):
        for c in range(w):
            t, rg = grid[r, c], np.random.default_rng(r * 1000 + c)
            tile = np.zeros((up, up))
            if t == GRASS:
                for _ in range(14):
                    cy, cx, s = rg.uniform(.1, .9), rg.uniform(.1, .9), rg.uniform(.03, .055)
                    tile -= 0.5 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
            elif t == ROCK:
                tile += 0.22 * np.sin(15 * (xx * 0.7 - yy * 0.8) + rg.uniform(0, 6.28)) \
                        + 0.10 * rg.normal(0, 1, (up, up))
            else:
                tile += 0.05 * np.sin(9 * yy + rg.uniform(0, 6.28))
            deco[r * up:(r + 1) * up, c * up:(c + 1) * up] = tile
    deco = ndimage.gaussian_filter(deco, 1.1)

    img = base * shade[..., None] * (1 + 0.40 * deco[..., None])
    wet = (np.kron(grid, np.ones((up, up), int)) == WATER)
    shore = np.clip(ndimage.gaussian_filter(wet.astype(float), up * 0.22) - wet, 0, 1)
    img = img * (1 - 0.45 * shore[..., None]) + np.array([222, 214, 180]) * 0.45 * shore[..., None]
    g = np.zeros(base.shape[:2])
    g[::up, :] = g[:, ::up] = 1
    img *= (1 - 0.05 * g[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def render_levels_side_by_side(grids: list[np.ndarray], levels: list[int],
                                out_path: str, panel_px: int = 512) -> str:
    """Task 3, Step 2: all four cascade levels rendered at the SAME panel
    pixel width (`up = panel_px // n`, so 8x8 gets up=64 -- identical to
    demo/render_world.py's own UP=64 -- and 64x64 gets up=8), pasted side
    by side with a small label under each, so refinement is VISIBLE, not
    asserted."""
    panels = []
    for grid, n in zip(grids, levels):
        up = max(1, panel_px // n)
        img = render_grid_image(grid, up=up)
        if img.size != (panel_px, panel_px):
            img = img.resize((panel_px, panel_px), Image.NEAREST)
        panels.append(img)

    label_h = 28
    canvas = Image.new("RGB", (panel_px * len(panels), panel_px + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for i, (img, n) in enumerate(zip(panels, levels)):
        canvas.paste(img, (i * panel_px, 0))
        draw.text((i * panel_px + 10, panel_px + 6), f"{n}x{n}", fill=(230, 220, 200))
    canvas.save(out_path)
    return out_path


def _cascade_step(decoded_repr: dict, prev_w: int, n: int, strength: float,
                   seed_for_level: int, fine_layer=None):
    """One cascade step (upsample -> patch -> compile-or-reuse -> sample ->
    decode), factored out so Task 3's `run_full_cascade` and Task 4's
    pooling functions share it instead of each re-deriving the step logic.
    `run_cascade` (Task 2's own committed function, above) is NOT rebuilt
    on top of this -- it stays exactly as measured. Returns
    (decoded_list, wall_s, fine_layer_used)."""
    t0 = time.perf_counter()
    factor = n // prev_w
    coarse_up = upsample(decoded_repr, src_w=prev_w, factor=factor)
    patch = cascade_patch(coarse_up, strength=strength)
    if fine_layer is None:
        fine_layer = _compile_fine_layer(n)
    _spec, enc, prog, _report = fine_layer
    patched = bias_patch(prog, enc, patch, field_cap=FIELD_CAP)
    rows = thrml_sample(patched, seed=seed_for_level, **FINE_SAMPLE_PARAMS)
    decoded_list = _decode_codewords(rows, patched.ising, enc)
    wall = time.perf_counter() - t0
    return decoded_list, wall, fine_layer, len(rows)


def run_full_cascade(levels: list[int], strength: float, seed: int) -> dict:
    """Task 3, Step 1: run `levels` (must start at 8) coarsest-first,
    recording PER LEVEL: wall time, n_total_draws, n_valid_draws,
    valid_fraction, and state_mix pooled over EVERY valid draw at that
    level (not just the one representative decode ancestrally carried
    forward -- that data is already drawn; pooling it costs nothing extra
    and gives a much less degenerate state-mix estimate than reading one
    grid). The representative decode carried forward to the next level
    (decoded_list[0]) is the SAME selection rule `run_cascade` uses.
    """
    if not levels or levels[0] != BASE_W:
        raise ValueError(f"levels[0] must be {BASE_W}; got {levels!r}")

    per_level = []
    decoded_per_level: list[dict] = []

    # --- level 0: the 8x8 base, via simulate() on the existing receipt ---
    t0 = time.perf_counter()
    SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path, got, im = simulate(BASE_RECEIPT, seed=seed, output_dir=SIM_OUTPUT_DIR,
                                  **BASE_SAMPLE_PARAMS)
    wall = time.perf_counter() - t0
    doc = json.loads(Path(out_path).read_text())
    ex = doc["decoded_example"]
    if not (ex["is_codeword"] and ex["task_valid"] and ex["decoded"] is not None):
        raise SystemExit(
            f"base layer (seed={seed}): no valid world sampled -- refusing "
            f"to fabricate a fallback")
    base_spec = load_spec(str(Path(BASE_RECEIPT) / "spec.yaml"))
    base_enc = encode(base_spec, _selected_encoding(Path(BASE_RECEIPT)))
    valid_decoded = []
    for row in got:
        bits = dict(zip(im.nodes, row.tolist()))
        if base_enc.is_codeword(bits):
            dec = base_enc.decode(bits)
            if base_spec.contract.validate(dec).ok:
                valid_decoded.append(dec)
    decoded_per_level.append(ex["decoded"])
    per_level.append(dict(
        n=BASE_W, wall_s=wall, n_total_draws=len(got),
        n_valid_draws=len(valid_decoded),
        valid_fraction=len(valid_decoded) / len(got) if len(got) else None,
        task_validity_from_simulate=doc["task_validity"],
        state_mix=_state_mix(valid_decoded)))

    prev_w = levels[0]
    for level_i, n in enumerate(levels[1:], start=1):
        if n % prev_w != 0:
            raise ValueError(f"level {n} is not an integer multiple of {prev_w}")
        decoded_list, wall, _fine_layer, n_total = _cascade_step(
            decoded_per_level[-1], prev_w, n, strength, seed_for_level=seed + level_i)
        if not decoded_list:
            raise SystemExit(
                f"level {n} (seed={seed}, strength={strength}): no valid "
                f"codeword drawn -- refusing to fabricate a fallback")
        decoded_per_level.append(decoded_list[0])
        per_level.append(dict(
            n=n, wall_s=wall, n_total_draws=n_total,
            n_valid_draws=len(decoded_list),
            valid_fraction=len(decoded_list) / n_total if n_total else None,
            state_mix=_state_mix(decoded_list)))
        prev_w = n

    return dict(levels=levels, strength=strength, seed=seed,
               decoded_per_level=decoded_per_level, per_level=per_level,
               total_wall_s=sum(p["wall_s"] for p in per_level))


def _repr_decode_upto(seed: int, target_w: int, strength: float) -> dict:
    """The ancestrally-sampled representative decode AT `target_w`, obtained
    by running the real cascade from the 8x8 base up through target_w --
    the actual backdrop a transition deeper than 8->16 sees in a real
    cascade run. A fresh, independently-drawn sample at target_w would NOT
    be this: it would be an unconditioned draw, not this seed's own cascade
    lineage. Built on `run_cascade` (Task 2's own committed function) --
    not a new stepping mechanism."""
    levels = [8]
    w = 8
    while w < target_w:
        w *= 2
        levels.append(w)
    if levels[-1] != target_w:
        raise ValueError(f"{target_w} is not reachable from 8 by doubling: {levels!r}")
    return run_cascade(levels, strength=strength, seed=seed)[-1]


def main_task3():
    """Task 3: the full cascade, 8->16->32->64."""
    out_dir = Path("demo/cascade_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    LEVELS = [8, 16, 32, 64]
    STRENGTH = 1.0  # inside the pre-registered [0.40, 0.90] working band
                    # (Task 2 measured 0.5764 here) -- the audit report's
                    # own recommended default, not re-optimised by this task.
    SEED = 7

    print(f"=== Task 3, Step 1: run_full_cascade({LEVELS}, strength={STRENGTH}, seed={SEED}) ===",
          flush=True)
    result = run_full_cascade(LEVELS, strength=STRENGTH, seed=SEED)
    for p in result["per_level"]:
        vf = p["valid_fraction"]
        print(f"level {p['n']:>3}x{p['n']:<3}: wall={p['wall_s']:.4f}s  "
              f"valid={p['n_valid_draws']}/{p['n_total_draws']} "
              f"({vf*100:.1f}%)  state_mix={p['state_mix']}", flush=True)
    print(f"TOTAL wall = {result['total_wall_s']:.4f}s", flush=True)
    Path(out_dir / "task3_full_cascade.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "decoded_per_level"} |
        {"decoded_per_level_cell_counts": [len(d) for d in result["decoded_per_level"]]},
        indent=2))

    grids = [_grid_of_decoded(d, n) for d, n in zip(result["decoded_per_level"], LEVELS)]

    print("\n=== Task 3, Step 2: render all four levels side by side ===", flush=True)
    p1 = render_levels_side_by_side(grids, LEVELS, "demo/cascade_levels.png")
    print(f"wrote {p1}", flush=True)

    print("\n=== Task 3, Step 3: render the final 64x64 with the existing renderer ===",
          flush=True)
    final_img = render_grid_image(grids[-1], up=64)
    final_img.save("demo/cascade_world.png")
    print("wrote demo/cascade_world.png", final_img.size, flush=True)

    # --- Step: inheritance at each of the three transitions ---
    print(f"\n=== Task 3: inheritance at each transition "
          f"({len(PARENT_SEEDS)} independent 8x8 parents, strength={STRENGTH}) ===",
          flush=True)
    transitions = [
        ("8->16", 8, 16, None),
        ("16->32", 16, 32, lambda seed: _repr_decode_upto(seed, 16, STRENGTH)),
        ("32->64", 32, 64, lambda seed: _repr_decode_upto(seed, 32, STRENGTH)),
    ]
    transition_results = {}
    for label, src_w, n, coarse_fn in transitions:
        t0 = time.perf_counter()
        fine_layer = _compile_fine_layer(n)
        res = measure_inheritance(
            STRENGTH, PARENT_SEEDS, src_w=src_w, factor=n // src_w,
            fine_layer=fine_layer, coarse_decode_fn=coarse_fn,
            out_path=str(out_dir / f"task3_inheritance_{label.replace('->', 'to')}.json"))
        wall = time.perf_counter() - t0
        frac = res["pooled_fraction"]
        frac_str = f"{frac:.4f}" if frac is not None else "unavailable"
        print(f"{label}: pooled_fraction={frac_str} "
              f"({res['total_matches']}/{res['total_children']})  wall={wall:.2f}s",
              flush=True)
        transition_results[label] = res | {"wall_s": wall}
    Path(out_dir / "task3_transitions.json").write_text(
        json.dumps(transition_results, indent=2))
    print(f"\nwrote {out_dir / 'task3_transitions.json'}")


# ===========================================================================
# Task 4 of the coarse-to-fine cascade plan: does structure emerge, and
# what does it cost? Pre-registered in audit/cascade_report.md's own
# "Task 4, Step 1" section, committed BEFORE any function below was ever
# run against real data (see that commit's own message for the ordering
# proof) -- this module and that commit are deliberately kept separate.
# ===========================================================================

def same_value_fraction_at_distance(grids: list[np.ndarray], r: int) -> tuple[int, int]:
    """Pooled (matches, total) counts of same-value cell pairs at grid
    (Manhattan, axis-aligned) distance `r` -- horizontal and vertical pairs
    both counted, pooled over EVERY grid in `grids` (many independent
    draws, never one -- same non-degeneracy discipline as `measure_
    inheritance` and demo/binary_world.py's own violation pooling)."""
    matches = 0
    total = 0
    for g in grids:
        h, w = g.shape
        if w - r > 0:
            a, b = g[:, :w - r], g[:, r:]
            matches += int((a == b).sum())
            total += a.size
        if h - r > 0:
            a, b = g[:h - r, :], g[r:, :]
            matches += int((a == b).sum())
            total += a.size
    return matches, total


def correlation_length(grids: list[np.ndarray], max_r: int) -> dict:
    """Task 4, Step 2: the same-value spatial autocorrelation length, pooled
    over `grids`. g(r) = P(same value at distance r) - chance, where
    chance = sum_v p_v^2 over the POOLED empirical marginal (generalises
    Task 2's own chance-floor reasoning to whatever marginal these grids
    actually realize, rather than assuming a uniform 1/3). xi is fit from
    ln(g(r)) ~ ln(g(1)) - (r-1)/xi via ordinary least squares over every r
    where g(r) > 0 (positive, above-chance correlation) -- fewer than 2
    such r, or a non-negative fitted slope, means xi is UNAVAILABLE and is
    reported as such (verification never fabricates a decay constant from a
    curve that shows no decay)."""
    if not grids:
        raise ValueError("correlation_length: grids is empty")
    counts = {0: 0, 1: 0, 2: 0}
    ncells = 0
    for g in grids:
        for v in (0, 1, 2):
            counts[v] += int((g == v).sum())
        ncells += g.size
    marginal = {v: counts[v] / ncells for v in (0, 1, 2)}
    chance = sum(p * p for p in marginal.values())

    curve = {}
    for r in range(1, max_r + 1):
        m, t = same_value_fraction_at_distance(grids, r)
        curve[r] = dict(matches=m, n_pairs=t, same_fraction=(m / t if t else None))

    excess = {r: curve[r]["same_fraction"] - chance for r in curve
              if curve[r]["same_fraction"] is not None}
    positive = {r: v for r, v in excess.items() if v > 0}

    if len(positive) < 2:
        xi = None
        fit_note = (f"only {len(positive)} of {max_r} distances show a "
                    f"positive chance-excess g(r) -- fewer than 2 points to "
                    f"fit a decay; xi UNAVAILABLE, not invented")
    else:
        rs = np.array(sorted(positive))
        ys = np.log(np.array([positive[r] for r in rs]))
        A = np.vstack([rs, np.ones_like(rs, dtype=float)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, ys, rcond=None)
        if slope >= 0:
            xi = None
            fit_note = (f"fit over r in {sorted(positive)}: slope={slope:.5f} "
                        f"is non-negative (g(r) not decaying with distance) "
                        f"-- xi UNAVAILABLE, not invented")
        else:
            xi = float(-1.0 / slope)
            fit_note = (f"fit over r in {sorted(positive)}: slope={slope:.5f}, "
                        f"xi={xi:.3f} cells")

    return dict(n_grids=len(grids), n_cells_pooled=ncells, marginal=marginal,
               chance_same_value=chance, curve=curve, xi=xi, fit_note=fit_note)


def _all_adjacent_pairs_n(n: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Every unordered 4-neighbour adjacent cell-pair position in an n x n
    grid, matching demo/binary_world.py's own `_all_adjacent_pairs`
    convention (W*(H-1) vertical + H*(W-1) horizontal), generalised to any
    n rather than the fixed 8x8 that module hardcodes."""
    pairs = []
    for y in range(n):
        for x in range(n):
            if x + 1 < n:
                pairs.append(((y, x), (y, x + 1)))
            if y + 1 < n:
                pairs.append(((y, x), (y + 1, x)))
    return pairs


def count_water_rock_violations(grids: list[np.ndarray]) -> tuple[int, int]:
    """Task 4, Step 3: pooled (bad, total_pairs) count of 4-adjacent cell
    pairs reading {water, rock} = {0, 1} -- the base spec's own hard rule
    (lattice_small_8x8_k3.yaml's forbid_value_pair_over_edges), pooled over
    EVERY grid in `grids`. Unlike demo/binary_world.py's composed-state
    measurement, EVERY cell here is eligible: this is one categorical
    variable per cell (water/rock/grass), not three independently-decoded
    binary layers composed together, so there is no separate 'eligible'
    subset to restrict to -- every adjacent pair in every grid can, in
    principle, read {0, 1}."""
    bad = 0
    total = 0
    for g in grids:
        n = g.shape[0]
        pairs = _all_adjacent_pairs_n(n)
        total += len(pairs)
        for (p1, p2) in pairs:
            if {int(g[p1]), int(g[p2])} == {0, 1}:
                bad += 1
    return bad, total


def pool_cascade_final_grids(parent_seeds: list[int], strength: float,
                             fine_layer_64=None, seed_offset: int = 3000
                             ) -> tuple[list[np.ndarray], dict]:
    """For each of `parent_seeds`, run the REAL cascade 8->16->32 (one
    ancestrally-sampled representative decode per level, via `run_cascade`
    -- Task 2's own committed function, not re-derived here) to get a
    genuine 32x32 backdrop, then sample the 64x64 fine layer ONE more time
    under that backdrop's conditioning patch and keep EVERY valid decoded
    grid from that batch (not just one) -- this is the population Task 4's
    correlation-length and violation-rate measurements pool over: many
    independent PARENTS (distinct top-level 8x8 draws) times many
    independent FINAL-LEVEL draws per parent, never one cascade's one
    representative world."""
    if fine_layer_64 is None:
        fine_layer_64 = _compile_fine_layer(64)
    _spec, enc, prog, _report = fine_layer_64

    all_grids: list[np.ndarray] = []
    per_parent = []
    for seed in parent_seeds:
        backdrop32 = _repr_decode_upto(seed, 32, strength)
        coarse_up = upsample(backdrop32, src_w=32, factor=2)
        patch = cascade_patch(coarse_up, strength=strength)
        patched = bias_patch(prog, enc, patch, field_cap=FIELD_CAP)
        rows = thrml_sample(patched, seed=seed + seed_offset, **FINE_SAMPLE_PARAMS)
        decoded_list = _decode_codewords(rows, patched.ising, enc)
        grids = [_grid_of_decoded(d, 64) for d in decoded_list]
        all_grids.extend(grids)
        per_parent.append(dict(seed=seed, n_valid=len(grids), n_total=len(rows)))
    return all_grids, dict(condition="cascaded", strength=strength,
                           per_parent=per_parent, n_grids=len(all_grids))


def pool_flat_grids(seeds: list[int], n: int = 64, fine_layer=None
                    ) -> tuple[list[np.ndarray], dict]:
    """The FLAT (no cascade) comparison, per Task 4's own pre-registration:
    the IDENTICAL `_compile_fine_layer(n)` architecture used at every level
    of the cascade -- zero product_over_edges terms, domain_wall k=3,
    bipartite by construction -- sampled with NO conditioning patch applied
    at all (not even an empty-dict bias_patch call; `prog` is sampled
    exactly as compiled). 'Same rules, same everything else' means this
    literally: the only variable that differs from the cascaded condition
    is the presence of the cascade_patch conditioning."""
    if fine_layer is None:
        fine_layer = _compile_fine_layer(n)
    _spec, enc, prog, _report = fine_layer

    all_grids: list[np.ndarray] = []
    per_seed = []
    for seed in seeds:
        rows = thrml_sample(prog, seed=seed, **FINE_SAMPLE_PARAMS)
        decoded_list = _decode_codewords(rows, prog.ising, enc)
        grids = [_grid_of_decoded(d, n) for d in decoded_list]
        all_grids.extend(grids)
        per_seed.append(dict(seed=seed, n_valid=len(grids), n_total=len(rows)))
    return all_grids, dict(condition="flat", per_seed=per_seed, n_grids=len(all_grids))


# Same five independent-backdrop seeds Task 2/3 already use, reused here so
# the cascaded population is built from the SAME five top-level worlds
# rather than a fresh, incomparable set.
FLAT_SEEDS = [1007, 1071, 1137, 1211, 1389]  # distinct from PARENT_SEEDS --
# the flat condition samples a completely different, unconditioned model
# (no coarse backdrop at all), so there is no reason -- and no shared
# meaning -- to reuse the coarse-seed numbers here; kept as a visibly
# different, but equally arbitrary-and-fixed, list of five.

MAX_R = 16  # half the 32-cell Nyquist-ish span for a 64-wide grid; large
            # enough to see the correlation curve decay to (or past) chance.


def main_task4():
    """Task 4: measure correlation length and violation rate, cascaded vs.
    flat, and judge both against audit/cascade_report.md's own
    pre-registration (committed before this function was ever run)."""
    out_dir = Path("demo/cascade_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    STRENGTH = 1.0

    print("=== Task 4, Step 2: pooling grids (cascaded vs. flat) ===", flush=True)
    t0 = time.perf_counter()
    cascaded_grids, cascaded_meta = pool_cascade_final_grids(PARENT_SEEDS, STRENGTH)
    wall_cascaded = time.perf_counter() - t0
    print(f"cascaded: {cascaded_meta['n_grids']} grids pooled from "
          f"{len(PARENT_SEEDS)} independent parents, wall={wall_cascaded:.2f}s",
          flush=True)
    Path(out_dir / "task4_cascaded_pool_meta.json").write_text(
        json.dumps(cascaded_meta | {"wall_s": wall_cascaded}, indent=2))

    t0 = time.perf_counter()
    flat_grids, flat_meta = pool_flat_grids(FLAT_SEEDS, n=64)
    wall_flat = time.perf_counter() - t0
    print(f"flat: {flat_meta['n_grids']} grids pooled from "
          f"{len(FLAT_SEEDS)} independent seeds, wall={wall_flat:.2f}s", flush=True)
    Path(out_dir / "task4_flat_pool_meta.json").write_text(
        json.dumps(flat_meta | {"wall_s": wall_flat}, indent=2))

    print(f"\n=== Task 4, Step 2: correlation length (max_r={MAX_R}) ===", flush=True)
    cascaded_corr = correlation_length(cascaded_grids, MAX_R)
    flat_corr = correlation_length(flat_grids, MAX_R)
    print(f"CASCADED: marginal={cascaded_corr['marginal']}, "
          f"chance={cascaded_corr['chance_same_value']:.4f}, "
          f"xi={cascaded_corr['xi']}, {cascaded_corr['fit_note']}", flush=True)
    print(f"FLAT:     marginal={flat_corr['marginal']}, "
          f"chance={flat_corr['chance_same_value']:.4f}, "
          f"xi={flat_corr['xi']}, {flat_corr['fit_note']}", flush=True)
    for r in range(1, MAX_R + 1):
        cf = cascaded_corr["curve"][r]["same_fraction"]
        ff = flat_corr["curve"][r]["same_fraction"]
        print(f"  r={r:>2}: cascaded same_fraction={cf:.4f}  flat same_fraction={ff:.4f}",
              flush=True)
    Path(out_dir / "task4_correlation_length.json").write_text(json.dumps(
        dict(cascaded=cascaded_corr, flat=flat_corr, max_r=MAX_R), indent=2))

    print(f"\n=== Task 4, Step 3: water/rock violation rate ===", flush=True)
    cascaded_bad, cascaded_total = count_water_rock_violations(cascaded_grids)
    flat_bad, flat_total = count_water_rock_violations(flat_grids)
    cascaded_rate = cascaded_bad / cascaded_total if cascaded_total else float("nan")
    flat_rate = flat_bad / flat_total if flat_total else float("nan")
    print(f"CASCADED: {cascaded_bad}/{cascaded_total} pairs "
          f"({cascaded_rate*100:.4f}%)", flush=True)
    print(f"FLAT (this study's own zero-terms flat, NOT the prior plan's "
          f"binary-stack relocation): {flat_bad}/{flat_total} pairs "
          f"({flat_rate*100:.4f}%)", flush=True)
    print(f"Comparison baselines (prior plan, different mechanism): "
          f"18.15% zero-nudge, 9.69% best-strength", flush=True)
    Path(out_dir / "task4_violation_rate.json").write_text(json.dumps(dict(
        cascaded=dict(bad=cascaded_bad, total=cascaded_total, rate=cascaded_rate),
        flat_zero_terms=dict(bad=flat_bad, total=flat_total, rate=flat_rate),
        prior_plan_relocation_baseline=dict(zero_nudge=0.1815, best_strength=0.0969),
    ), indent=2))

    print("\nwrote demo/cascade_runs/task4_correlation_length.json and "
          "task4_violation_rate.json", flush=True)


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else "task2"
    if _mode == "task2":
        main_task2()
    elif _mode == "task3":
        main_task3()
    elif _mode == "task4":
        main_task4()
    else:
        raise SystemExit(f"unknown mode {_mode!r}; use 'task2', 'task3', or 'task4'")
