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
                         fine_layer=None, out_path: str | None = None) -> dict:
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
    """
    n_fine = src_w * factor
    if fine_layer is None:
        fine_layer = _compile_fine_layer(n_fine)
    _spec, enc, prog, _report = fine_layer

    per_parent = []
    total_matches = 0
    total_children = 0
    for seed in parent_seeds:
        base_decoded = _base_decode(seed)
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


def main():
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


if __name__ == "__main__":
    main()
