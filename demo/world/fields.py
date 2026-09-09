"""Sampling one parameter field on the TSU.

A parameter field is a stack of `layers` binary Ising layers at one grid size
and one coupling, summed per cell to give an integer level in 0..layers. It is
NOT terrain -- it is a parameter that demo/world/spline.py later interprets.

WHY ONE COMPILED LAYER AND SEVERAL CHAINS. The design calls for `layers`
INDEPENDENT binary layers at the same size and coupling. Layers with identical
energy are identical distributions, so `layers` separate layers and `layers`
independent draws from one compiled layer are the same object statistically --
and the second compiles once instead of `layers` times. Each thrml chain starts
from its own init under a split key, so the final sample of each chain is an
independent draw.

THE SYMMETRIC RULE IS NOT OPTIONAL. The one-sided clumping rule used everywhere
in this project until now (`a_value: 1, b_value: 1` alone) expands under
x = (s+1)/2 to a coupling PLUS a uniform field of w*deg/4 -- four times the
coupling at degree 4, measured exactly as |b|/|J| = 4.000. That field pins every
cell toward one value rather than clumping them, and it is why every world this
project sampled before now was structureless. Adding the mirrored (0, 0) term
cancels it to exactly zero, doubles the coupling, and leaves the graph bipartite
so placement stays instant.
"""
from __future__ import annotations

import os
import tempfile

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

FIELD_YAML = """name: world_field
generate:
  kind: grid
  width: {size}
  height: {size}
  variable_domain: {{domain: binary}}
terms:
  - {{kind: product_over_edges, a_value: 1, b_value: 1, weight: {w}}}
  - {{kind: product_over_edges, a_value: 0, b_value: 0, weight: {w}}}
"""


def compile_layer(size: int, beta_j: float):
    """Compile one binary symmetric-rule layer. Returns (spec, enc, ising, prog).

    Refuses to place() unless analyse() first confirms the graph is bipartite --
    THE SAFETY RULE. Placing a non-bipartite graph above 8x8 costs 6-32 minutes
    to FAIL, so this assertion is a time bound, not a style preference.
    """
    path = os.path.join(tempfile.gettempdir(), f"world_field_{size}_{beta_j}.yaml")
    with open(path, "w") as fh:
        fh.write(FIELD_YAML.format(size=size, w=-abs(float(beta_j))))
    spec = load_spec(path)
    enc = encode(spec, "domain_wall")
    ising = lower(enc.model)
    report = analyse(ising)
    assert report.bipartite is True, (
        f"{size}x{size} field: analyse() reports bipartite=False -- refusing "
        f"place() per THE SAFETY RULE")
    placement = place(ising, report, PROFILES["z1"], restarts=12, iters=200_000)
    assert placement.mediation is None, (
        f"{size}x{size} field: place() mediated a graph analyse() reported "
        f"bipartite -- should be unreachable")
    prog = build_program(route(ising, report, PROFILES["z1"]), report)
    return spec, enc, ising, prog


def _grid_of(decoded: dict, size: int) -> np.ndarray:
    """Decoded {cell_name: value} -> (size, size) array, using this project's
    g<x>_<y> naming convention (x is the column, y the row, both 0-based)."""
    a = np.zeros((size, size), dtype=int)
    for k, v in decoded.items():
        x, y = k[1:].split("_")
        a[int(y), int(x)] = v
    return a


def sample_field_layers(size: int, layers: int, beta_j: float, seed: int,
                        warmup: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Sample one parameter field, returning BOTH the summed levels and the
    individual binary draws that make them up, as `(levels, stack)` where
    `stack` has shape `(layers, size, size)` and holds 0/1 values.

    The stack is kept because a field's levels are its layers SUMMED: saving
    only the sum makes the decode replayable while leaving the Ising sampling
    behind it unverifiable, since a reader cannot check the draws themselves.
    The corresponding spins are `2 * stack - 1`.

    Sample one parameter field: (size, size) integer levels in 0..layers.

    Raises rather than returning a short stack if the sampler produced fewer
    valid codewords than `layers` -- a quietly thinner field would change the
    level range without saying so, and every downstream threshold assumes the
    full 0..layers range.
    """
    _spec, enc, ising, prog = compile_layer(size, beta_j)
    draws = thrml_sample(prog, seed=seed, n_chains=layers, n_samples=4,
                         n_warmup=warmup, steps_per_sample=25)
    rows = np.asarray(draws).reshape(layers, 4, -1)
    grids = []
    for c in range(layers):
        bits = dict(zip(ising.nodes, rows[c, -1].tolist()))
        if enc.is_codeword(bits):
            grids.append(_grid_of(enc.decode(bits), size))
    if len(grids) != layers:
        raise RuntimeError(
            f"{size}x{size} field at seed {seed}: {len(grids)} of {layers} "
            f"chains produced a valid codeword; refusing to return a field "
            f"whose level range is not 0..{layers}")
    stack = np.asarray(grids, dtype=int)
    return np.sum(stack, axis=0).astype(int), stack


def sample_field(size: int, layers: int, beta_j: float, seed: int,
                 warmup: int = 4000) -> np.ndarray:
    """The summed level field only. See `sample_field_layers` when the
    individual draws are needed."""
    return sample_field_layers(size, layers, beta_j, seed, warmup=warmup)[0]
