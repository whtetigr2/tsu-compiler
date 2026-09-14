"""Task 4: stacked elevation world, measured.

The first COMPOSITE this project produces: a base terrain layer (water/
rock/grass, demo/receipts/small) plus three thermometer elevation bands
(demo/receipts/elev_band, see demo/elevation.py) stacked on top of it, each
band conditioned on the base and on the band immediately below it via
demo/layers.bias_patch -- no recompile, three receipts' worth of layers off
two compiled programs.

Each layer's own contract is checked individually (spec.contract.validate
per layer, same as every other layer in this project). Nothing today
validates the CROSS-layer rule ("band i+1 true only where band i is true")
except as a soft encouragement at sample time (demo/elevation.py's
band_patch) -- so THIS script is what measures how often that soft rule is
actually honoured. Spec section 3.1's own words: this is the plan's most
important measurement, and it must be reported as a rate, never silently
repaired to zero (see demo/elevation.py's own module docstring and
`monotonicity_violations`'s docstring for why sorting/clamping the bands
would destroy the very number this script exists to produce).

MANDATORY CAVEAT (same one demo/layers.py and demo/stacked_world.py already
carry): stacking samples p(base) * p(band0 | base) * p(band1 | band0, base)
* p(band2 | band1, base), a DIRECTED / ANCESTRAL factorization -- NOT the
joint Boltzmann distribution over every layer at once. Every individual
factor is a genuine Boltzmann sample of its own (possibly patched)
IsingModel; the STACK across layers is not itself one Boltzmann sample of a
combined model, and a layer can never be influenced by a layer above it.

FIELD_CAP (|b| <= 6.0) is an ASSUMED project working value (see
demo/layers.py's own module docstring and demo/receipts/*/target.json's
"assumed" tag on max_abs_bias -- F-R2: not max_abs_coupling, the |J| cap,
which is now Extropic-documented; see P-3/F-A5 + I-9a/F-R7), not a sourced
Extropic figure -- every value this module prints against it says so.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

import numpy as np
from PIL import Image
from scipy import ndimage

from tsu_compiler.spec import load_spec
from tsu_compiler.passes.encode import encode
from tsu_compiler.simulate import reconstruct_program, _selected_encoding
from tsu_compiler.backends.thrml_backend import sample as thrml_sample

from layers import FIELD_CAP, FieldCapExceeded, bias_patch
from elevation import band_patch, thermometer_level, monotonicity_violations
from worldfile import save_stack

BASE_R = "demo/receipts/small"
BAND_R = "demo/receipts/elev_band"
W = H = 8
WATER, ROCK, GRASS = 0, 1, 2
TERRAIN_NAMES = {WATER: "water", ROCK: "rock", GRASS: "grass"}
PAL = np.array([[46, 92, 132], [124, 116, 106], [126, 158, 84]], float)
N_BANDS = 3

# Same scale demo/render_world.py, demo/layers.py's tests, and
# demo/stacked_world.py already use for these two receipts (~0.4-1.4s per
# call, measured there) -- reused here rather than re-measured, per Step 7's
# own expectation of ~0.4-1s per layer.
SAMPLE_PARAMS = dict(n_chains=8, n_samples=60, n_warmup=1200, steps_per_sample=4)

# CALIBRATION NOTE (measured 2026-09-01 on these two receipts, same spirit
# as demo/stacked_world.py's own ALPHA0 sweep note -- a real dose-response
# knob, not a free parameter, values are receipt- and seed-specific).
# Pooled monotonicity violation rate (every valid draw of band i checked
# against the SAME fixed representative band i-1, over 3 bands' worth of
# transitions -- see main()'s own pooled measurement below for the method):
#     strength   pooled violation rate
#     0.0 (none)   8.32%   (5109/61440 -- no monotonicity nudge at all)
#     1.0          4.64%   (2853/61440)
#     2.0          2.50%   (1537/61440)
# Monotone-decreasing and clearly non-degenerate at every strength tried --
# never collapses to ~0% (band_patch is a soft encourage, never a hard
# constraint, so it CANNOT and must not). STRENGTH=1.0 is used below: the
# same order of magnitude as this receipt's own biases (|b| in [0.4, 0.8]
# before patching, see target.json), giving a real, clearly measured
# reduction from the unconditioned baseline without pretending the nudge
# eliminates the violations it only discourages.
STRENGTH = 1.0


def _decode_codewords(rows, im, enc):
    out = []
    for row in rows:
        bits = dict(zip(im.nodes, row.tolist()))
        if enc.is_codeword(bits):
            out.append(enc.decode(bits))
    return out


def _grid_of(d):
    return np.array([[int(d[f"g{x}_{y}"]) for x in range(W)] for y in range(H)])


def render_elevation_world_image(base_grid: np.ndarray, elevation: np.ndarray,
                                 up: int = 48) -> Image.Image:
    """Terrain colour from `base_grid` (water/rock/grass), shading from a
    hillshade over the REAL sampled `elevation` field (0..N_BANDS thermometer
    level per cell) -- not the derived/fake pseudo-relief height map
    demo/render_world.py falls back to (that file's own docstring: "there is
    no elevation layer in this spec" -- this script is what supplies one).

    CELL-UNIT SLOPE, not per-pixel: `np.gradient` on a field already
    upscaled by `up` (via `ndimage.zoom`) returns the per-PIXEL slope, which
    is 1/up of the true per-CELL-unit slope -- multiplying the gradient by
    `up` (below) is what converts it back. Getting this wrong was already
    one full debugging round on demo/render_world.py's own hillshade (see
    that file and this task's brief); done right here from the start.
    """
    base = PAL[np.kron(base_grid, np.ones((up, up), int))]

    ef = ndimage.gaussian_filter(ndimage.zoom(elevation.astype(float), up, order=3), up * 0.5)
    gy, gx = [g * up * 1.4 for g in np.gradient(ef)]  # per-pixel -> per-cell-unit
    slope, aspect = np.arctan(np.hypot(gy, gx)), np.arctan2(-gx, gy)
    az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope)
                     * np.cos(az - aspect), 0, None) ** 0.85
    shade = 0.55 + 0.95 * shade

    img = base * shade[..., None]
    wet = (np.kron(base_grid, np.ones((up, up), int)) == WATER)
    shore = np.clip(ndimage.gaussian_filter(wet.astype(float), up * 0.22) - wet, 0, 1)
    img = img * (1 - 0.45 * shore[..., None]) + np.array([222, 214, 180]) * 0.45 * shore[..., None]
    g = np.zeros(base.shape[:2])
    g[::up, :] = g[:, ::up] = 1
    img *= (1 - 0.05 * g[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main():
    base_spec = load_spec(str(Path(BASE_R) / "spec.yaml"))
    base_enc = encode(base_spec, _selected_encoding(Path(BASE_R)))
    base_prog = reconstruct_program(BASE_R)

    band_spec = load_spec(str(Path(BAND_R) / "spec.yaml"))
    band_enc = encode(band_spec, _selected_encoding(Path(BAND_R)))
    band_prog = reconstruct_program(BAND_R)

    layer_report = []  # one dict per layer, for the printed report below

    # ---------------------------------------------------------------
    # Step 1: base layer -- sample, decode, validate.
    # ---------------------------------------------------------------
    bmax_base_before = float(np.abs(base_prog.ising.biases).max())
    t0 = time.perf_counter()
    got_base = thrml_sample(base_prog, seed=7, **SAMPLE_PARAMS)
    wall_base = time.perf_counter() - t0
    decoded_base = _decode_codewords(got_base, base_prog.ising, base_enc)
    valid_base = [d for d in decoded_base if base_spec.contract.validate(d).ok]
    if not valid_base:
        raise SystemExit("base layer: no valid world sampled -- cannot "
                          "condition anything on nothing; not fabricating "
                          "a fallback world")
    valid_frac_base = len(valid_base) / len(got_base)
    base_repr = valid_base[0]
    base_grid = _grid_of(base_repr)
    layer_report.append(dict(
        name="base", bmax_before=bmax_base_before, bmax_after=bmax_base_before,
        valid_frac=valid_frac_base, n_valid=len(valid_base), n_total=len(got_base),
        wall=wall_base))

    # ---------------------------------------------------------------
    # Steps 2/3: bands 0..N_BANDS-1, each conditioned on the base and on
    # the band immediately below (band_patch(prev_band, base, STRENGTH),
    # None for band 0's "below").
    # ---------------------------------------------------------------
    band_reprs: list[dict] = []       # one decoded assignment per band -- THE composite
    band_valid_lists: list[list[dict]] = []   # every valid draw per band, for pooling
    bmax_band_before = float(np.abs(band_prog.ising.biases).max())

    prev_repr = None
    for i in range(N_BANDS):
        patch = band_patch(prev_repr, base_repr, STRENGTH)
        try:
            patched = bias_patch(band_prog, band_enc, patch, field_cap=FIELD_CAP)
        except FieldCapExceeded as e:
            raise SystemExit(f"band {i}: {e}")
        bmax_after = float(np.abs(patched.ising.biases).max())

        t0 = time.perf_counter()
        got = thrml_sample(patched, seed=100 + i, **SAMPLE_PARAMS)
        wall = time.perf_counter() - t0

        decoded = _decode_codewords(got, patched.ising, band_enc)
        valid = [d for d in decoded if band_spec.contract.validate(d).ok]
        if not valid:
            raise SystemExit(f"band {i}: no valid world sampled under "
                              f"conditioning -- not fabricating a fallback world")
        valid_frac = len(valid) / len(got)
        rep = valid[0]

        band_reprs.append(rep)
        band_valid_lists.append(valid)
        layer_report.append(dict(
            name=f"band{i}", bmax_before=bmax_band_before, bmax_after=bmax_after,
            valid_frac=valid_frac, n_valid=len(valid), n_total=len(got), wall=wall,
            conditioning_patch=patch))
        prev_repr = rep

    # ---------------------------------------------------------------
    # Step 4: elevation field from the COMPOSITE (the one representative
    # decode per band), and the monotonicity violation rate -- the number
    # this whole task exists to produce. DETECTED, NEVER REPAIRED: no
    # sorting, clamping, or otherwise fixing up band_reprs happens anywhere
    # in this script (see demo/elevation.py's own docstring for why that
    # would destroy the measurement).
    # ---------------------------------------------------------------
    elevation = np.array([[thermometer_level(band_reprs, f"g{x}_{y}")
                           for x in range(W)] for y in range(H)])
    viol_composite = monotonicity_violations(band_reprs)
    composite_slots = W * H * (N_BANDS - 1)   # (cell, i) for i in [1, N_BANDS-1]
    rate_composite = len(viol_composite) / composite_slots

    # Pooled measurement: EVERY valid draw of band i checked against the
    # SAME fixed representative band i-1 (the one band_patch actually
    # conditioned on) -- a much larger sample (thousands of (cell, draw)
    # pairs vs. this composite's 128), and the statistic reported as
    # STRENGTH's own calibration note above. Both rates measure the same
    # thing (how often the encouraged rung breaks); the pooled one is the
    # stabler estimate, the composite one is what the RENDERED/SAVED world
    # actually exhibits -- both are reported, neither replaces the other.
    total_viol_pooled, total_slots_pooled = 0, 0
    for i in range(1, N_BANDS):
        prev_rep = band_reprs[i - 1]
        for d in band_valid_lists[i]:
            v = monotonicity_violations([prev_rep, d])
            total_viol_pooled += len(v)
            total_slots_pooled += W * H
    rate_pooled = total_viol_pooled / total_slots_pooled

    # ---------------------------------------------------------------
    # Step 5: render, elevation-driven hillshade.
    # ---------------------------------------------------------------
    img = render_elevation_world_image(base_grid, elevation)
    img.save("demo/world_elevation.png")

    # ---------------------------------------------------------------
    # Step 6: save the stack (base + N_BANDS bands), one JSON file, every
    # layer's own receipt/seed/clamp/conditioning-patch/task_valid.
    # ---------------------------------------------------------------
    stack_layers = [dict(
        name="base", spec_name=base_spec.name, receipt_dir=BASE_R, seed=7,
        clamp={}, conditioning_patch={}, width=W, height=H,
        values=base_grid.flatten().tolist(),
        value_names=[TERRAIN_NAMES[k] for k in (WATER, ROCK, GRASS)],
        task_valid=True, violations=[],
        sampler_params=dict(SAMPLE_PARAMS, seed=7))]
    for i, rep in enumerate(band_reprs):
        grid_i = np.array([[int(rep[f"g{x}_{y}"]) for x in range(W)] for y in range(H)])
        lr = layer_report[i + 1]
        stack_layers.append(dict(
            name=f"band{i}", spec_name=band_spec.name, receipt_dir=BAND_R,
            seed=100 + i, clamp={}, conditioning_patch=lr["conditioning_patch"],
            width=W, height=H, values=grid_i.flatten().tolist(),
            value_names=["below", "above"], task_valid=True, violations=[],
            sampler_params=dict(SAMPLE_PARAMS, seed=100 + i)))
    stack_path = save_stack("demo/world_elevation_stack.json", layers=stack_layers,
                             saved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ---------------------------------------------------------------
    # Step 7: report.
    # ---------------------------------------------------------------
    print("=== elevation world: measured results (no fabricated values) ===")
    print(f"STRENGTH used: {STRENGTH}  (see this module's own calibration note)")
    print(f"assumed field cap |b|max <= {FIELD_CAP} (project working value, "
          f"NOT a sourced Extropic figure)")
    for lr in layer_report:
        extra = "" if lr["name"] == "base" else \
            f"  |b|max before={lr['bmax_before']:.4f} after={lr['bmax_after']:.4f}"
        print(f"layer {lr['name']:<6}: valid {lr['n_valid']}/{lr['n_total']} "
              f"({100*lr['valid_frac']:.1f}%)  wall={lr['wall']:.3f}s{extra}")
    print(f"base cell counts: "
          f"{ {TERRAIN_NAMES[k]: int((base_grid == k).sum()) for k in (0, 1, 2)} }")
    print(f"elevation level counts (0..{N_BANDS}): "
          f"{ {lvl: int((elevation == lvl).sum()) for lvl in range(N_BANDS + 1)} }")
    print(f"MONOTONICITY VIOLATION RATE (composite, the rendered/saved world): "
          f"{rate_composite:.4f}  ({len(viol_composite)}/{composite_slots} "
          f"(cell, band) slots)")
    print(f"MONOTONICITY VIOLATION RATE (pooled over every valid draw of each "
          f"band vs. its fixed conditioning band): {rate_pooled:.4f}  "
          f"({total_viol_pooled}/{total_slots_pooled} slots)")
    print(f"wrote demo/world_elevation.png {np.array(img).shape}")
    print(f"wrote {stack_path}")


if __name__ == "__main__":
    main()
