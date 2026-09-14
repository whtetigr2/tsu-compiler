"""Two-layer conditioned world stack on demo/receipts/small.

Layer 1 is sampled normally from the compiled receipt and decoded -- the
base world, exactly what demo/render_world.py already produces. Layer 2 is
NOT a fresh independent draw: a bias patch is derived from layer 1's own
decoded grid (bias each cell toward ROCK in proportion to its distance from
the nearest WATER cell in layer 1) and folded into the SAME compiled
program's biases via demo/layers.bias_patch -- no recompile, one receipt
for both layers.

MANDATORY CAVEAT (see demo/layers.py's module docstring for the full
derivation): this stack samples p(layer1) * p(layer2 | layer1), a DIRECTED
/ ANCESTRAL factorization, NOT the joint Boltzmann distribution over both
layers together. Each factor -- layer 1's draw, layer 2's draw -- is a
genuine Boltzmann sample of its own (possibly patched) IsingModel, but the
two-layer STACK is not itself one Boltzmann sample of a combined model, and
layer 1 can never be influenced by layer 2 (conditioning flows forward
only). This is a deliberate modelling choice, not an approximation error.

FIELD_CAP (|b| <= 6.0) is an assumed project working value, not a sourced
Extropic figure -- see demo/layers.py and demo/receipts/small/target.json's
own "assumed" tag on max_abs_bias (F-R2: not max_abs_coupling, the |J| cap,
which is now Extropic-documented -- see P-3/F-A5 + I-9a/F-R7).
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

R = "demo/receipts/small"
W = H = 8
WATER, ROCK, GRASS = 0, 1, 2
NAMES = {WATER: "water", ROCK: "rock", GRASS: "grass"}
PAL = np.array([[46, 92, 132], [124, 116, 106], [126, 158, 84]], float)

# Same scale demo/render_world.py already uses for this receipt (~0.4-1.4s
# per call, measured there and in tests/test_layers.py).
SAMPLE_PARAMS = dict(n_chains=8, n_samples=60, n_warmup=1200, steps_per_sample=4)

# Starting nudge-per-unit-distance; backed off automatically (and reported,
# never silently) if the patched |b|max would exceed FIELD_CAP. Chosen by a
# swept measurement (reported in layers-report.md) that showed a clear,
# stable ROCK-frequency gradient (far-from-water cells more often ROCK than
# near-water cells) without saturating every cell to ROCK the way alpha>=0.4
# did in that sweep.
# CALIBRATION NOTE (measured 2026-09-01, controller sweep on this receipt).
# ALPHA0 is the conditioning STRENGTH and it is a real dose-response knob, not a
# free parameter -- swept with the sign convention correct (patch weight > 0
# ENCOURAGES), rock frequency moves monotonically:
#     alpha  |b|max   water    rock   grass
#     0.00   1.600   26.5%   15.3%   58.2%
#     0.05   1.675   18.1%   23.6%   58.3%
#     0.10   1.750   17.3%   33.1%   49.5%
#     0.20   1.900    3.3%   52.8%   43.9%
#     0.35   2.125    3.0%   66.6%   30.4%
# Above ~0.2 the conditioning OVERWHELMS the base rules and the world degenerates
# into near-total rock with almost no water -- the first version of this demo shipped
# at that end and produced a 73-87% rock world, which is not a second layer so much
# as an erased first one. Values are receipt- and seed-specific; re-sweep for any
# other model rather than carrying this number across.
ALPHA0 = 0.03


def _decode_codewords(rows, im, enc):
    out = []
    for row in rows:
        bits = dict(zip(im.nodes, row.tolist()))
        if enc.is_codeword(bits):
            out.append(enc.decode(bits))
    return out


def _grid_of(d):
    return np.array([[int(d[f"g{x}_{y}"]) for x in range(W)] for y in range(H)])


def _render(grid, path):
    UP = 48
    img = PAL[np.kron(grid, np.ones((UP, UP), int))]
    gridlines = np.zeros(img.shape[:2])
    gridlines[::UP, :] = gridlines[:, ::UP] = 1
    img = img * (1 - 0.10 * gridlines[..., None])
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    print(f"wrote {path} {img.shape}")


def main():
    spec = load_spec(str(Path(R) / "spec.yaml"))
    enc = encode(spec, _selected_encoding(Path(R)))
    prog = reconstruct_program(R)
    im = prog.ising

    bmax_before = float(np.abs(im.biases).max())

    # ---------------------------------------------------------------
    # Layer 1: sample normally, decode. This is the base world.
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    got1 = thrml_sample(prog, seed=7, **SAMPLE_PARAMS)
    wall1 = time.perf_counter() - t0

    worlds1 = _decode_codewords(got1, im, enc)
    valid1 = [d for d in worlds1 if spec.contract.validate(d).ok]
    valid_frac1 = len(valid1) / len(got1)
    if not valid1:
        raise SystemExit(
            "layer 1: no valid world sampled -- cannot condition layer 2 on "
            "nothing; not fabricating a fallback world")
    grids1 = [_grid_of(d) for d in valid1]
    grid1 = grids1[0]  # THE base world layer 2's patch is derived from

    # distance from every cell to the nearest WATER cell in layer 1's grid.
    # ndimage.distance_transform_edt(mask) gives, for every True cell in
    # `mask`, the Euclidean distance to the nearest False cell -- so run it
    # on "is not water" to get "distance to nearest water" (water cells
    # themselves are False -> distance 0, correctly).
    dist1 = ndimage.distance_transform_edt(grid1 != WATER)

    # ---------------------------------------------------------------
    # Layer 2: bias patch derived from layer 1, re-sample the SAME
    # program. weight > 0 (see layers.bias_patch) encourages ROCK.
    # ---------------------------------------------------------------
    base_patch = {(f"g{x}_{y}", ROCK): float(dist1[y, x])
                  for x in range(W) for y in range(H)}

    scale = ALPHA0
    patched_prog = None
    backoffs = []
    while patched_prog is None:
        scaled = {k: v * scale for k, v in base_patch.items()}
        try:
            patched_prog = bias_patch(prog, enc, scaled, field_cap=FIELD_CAP)
        except FieldCapExceeded as e:
            backoffs.append((scale, str(e)))
            scale *= 0.5
            if scale < ALPHA0 / 64:
                raise RuntimeError(
                    f"could not find a nudge scale under {ALPHA0/64} that "
                    f"satisfies the field cap; last refusals: {backoffs}")
    bmax_after = float(np.abs(patched_prog.ising.biases).max())

    t0 = time.perf_counter()
    got2 = thrml_sample(patched_prog, seed=8, **SAMPLE_PARAMS)
    wall2 = time.perf_counter() - t0

    worlds2 = _decode_codewords(got2, patched_prog.ising, enc)
    valid2 = [d for d in worlds2 if spec.contract.validate(d).ok]
    valid_frac2 = len(valid2) / len(got2)
    if not valid2:
        raise SystemExit(
            "layer 2: no valid world sampled under conditioning -- not "
            "fabricating a fallback world")
    grids2 = [_grid_of(d) for d in valid2]
    grid2 = grids2[0]  # the picture drawn to demo/world_stacked.png

    # ---------------------------------------------------------------
    # Measure the conditioning effect -- a number, not an impression.
    # Pooled over EVERY valid draw in each layer (n_valid1 / n_valid2 below),
    # not just the single pictured grid -- a single-draw comparison was
    # measured to be highly seed-sensitive (see layers-report.md's ALPHA
    # sweep) because the model is multimodal; pooling is the stable
    # statistic. Both layers are scored against dist1, layer 1's OWN
    # distance-to-water field (fixed, from the single base world the patch
    # was actually derived from), so the two layers are compared on the
    # same yardstick.
    # ---------------------------------------------------------------
    rock_stack1 = np.stack([g == ROCK for g in grids1])
    rock_stack2 = np.stack([g == ROCK for g in grids2])
    rock_dist_list1 = np.concatenate([dist1[g == ROCK] for g in grids1]) \
        if any((g == ROCK).any() for g in grids1) else np.array([])
    rock_dist_list2 = np.concatenate([dist1[g == ROCK] for g in grids2]) \
        if any((g == ROCK).any() for g in grids2) else np.array([])
    mean_dist_rock1 = float(rock_dist_list1.mean()) if rock_dist_list1.size else float("nan")
    mean_dist_rock2 = float(rock_dist_list2.mean()) if rock_dist_list2.size else float("nan")

    freq_map1 = rock_stack1.mean(axis=0)  # per-cell P(cell == ROCK), layer 1
    freq_map2 = rock_stack2.mean(axis=0)  # per-cell P(cell == ROCK), layer 2
    near = dist1 <= np.median(dist1)
    far = ~near
    rock_freq_near1 = float(freq_map1[near].mean())
    rock_freq_far1 = float(freq_map1[far].mean())
    rock_freq_near2 = float(freq_map2[near].mean())
    rock_freq_far2 = float(freq_map2[far].mean())

    _render(grid1, "demo/world_layer1.png")
    _render(grid2, "demo/world_stacked.png")

    print("=== layer conditioning: measured results (no fabricated values) ===")
    print(f"nudge scale used (ALPHA): {scale}"
          + (f"  [backed off {len(backoffs)}x from {ALPHA0}]" if backoffs else ""))
    print(f"|b|max before patch: {bmax_before:.4f}")
    print(f"|b|max after  patch: {bmax_after:.4f}  (assumed field cap: {FIELD_CAP})")
    print(f"layer 1 cell counts: "
          f"{ {NAMES[k]: int((grid1 == k).sum()) for k in (0, 1, 2)} }")
    print(f"layer 2 cell counts: "
          f"{ {NAMES[k]: int((grid2 == k).sum()) for k in (0, 1, 2)} }")
    print(f"mean distance-to-water of ROCK cells: "
          f"layer1={mean_dist_rock1:.3f}  layer2={mean_dist_rock2:.3f}  "
          f"(delta={mean_dist_rock2 - mean_dist_rock1:+.3f})")
    print(f"ROCK frequency, near-water half of grid: "
          f"layer1={rock_freq_near1:.3f}  layer2={rock_freq_near2:.3f}")
    print(f"ROCK frequency, far-from-water half of grid: "
          f"layer1={rock_freq_far1:.3f}  layer2={rock_freq_far2:.3f}")
    print(f"valid fraction (contract-passing draws / total draws): "
          f"unconditioned layer1={valid_frac1:.3f} ({len(valid1)}/{len(got1)})  "
          f"conditioned layer2={valid_frac2:.3f} ({len(valid2)}/{len(got2)})")
    print(f"wall time: layer1={wall1:.3f}s  layer2={wall2:.3f}s  "
          f"(no recompile -- same receipt, patched biases only)")


if __name__ == "__main__":
    main()
