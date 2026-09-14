"""Cross-layer conditioning via per-cell bias patches.

The idea: sample layer 1 from a compiled receipt, decode it, derive a bias
patch from that decoded world, and re-sample the SAME compiled program with
those biases patched in as layer 2. Layer 2's distribution is conditioned on
layer 1's decoded output. Six terrain types come from STACKING two 3-value
layers, never from widening one layer past the field gate's k=4 bound.

Why this is legal without a recompile: a term "value v is encouraged or
discouraged at cell c" is LINEAR in that cell's own physical spins (see
tsu_compiler.passes.encode's domain-wall/one-hot rewrite tables -- a categorical
indicator VarRef(c, v) always rewrites to a LinearForm over ONLY that cell's
chain). A linear term only ever changes an IsingModel's `biases` array --
nodes, edges, weights (couplings), and a SamplingProgram's colour `blocks`
are governed by TOPOLOGY, which this never touches. One receipt, patched
biases, no recompile, every layer.

Conditioning CANNOT be a clamp: a clamp fixes a spec's OWN variables, and
layer 2's variables are disjoint from layer 1's (they are literally the same
cells, resampled) -- there is nothing to clamp layer 2 TO from layer 1
except through the energy itself, which is exactly what a bias patch does
and a clamp cannot.

MANDATORY CAVEAT -- read before treating a stack as "the model": stacking
samples p(layer1) * p(layer2 | layer1), a DIRECTED / ANCESTRAL
factorization, NOT the joint Boltzmann distribution over both layers at
once. Every individual factor (layer 1's draw, layer 2's draw) is a genuine
Boltzmann sample of its own (possibly patched) IsingModel -- but the STACK
across layers is not itself one Boltzmann sample of a combined model, and
layer 1 can never be influenced by layer 2 (information flows forward only).
This is a deliberate modelling choice, the same one Extropic's Denoising
Thermodynamic Models make (chained simple EBM layers instead of one
monolithic model), and it must be stated plainly, never glossed as "the
joint distribution."

`FIELD_CAP` (|b| <= 6.0) reads `tsu_compiler.target.Z1.max_abs_bias` (F-R2; see
demo/frontier.py's module docstring for this same read-from-the-target-
profile pattern applied to its own |J| cap). `max_abs_bias` is the |b| cap
specifically -- P-3/F-A5 + I-9a/F-R7 split it from `max_abs_coupling` (the
|J| cap) into two independent `Sourced` fields (tsu/target.py) because they
carry different provenance even though both currently hold 6.0:
`max_abs_coupling` is Extropic-documented (Thermalizers 2608.01615v1.pdf
p.22, Fig. 12 cap-sweep axis: "6 (Z1)"), while `max_abs_bias` remains a
genuine, unsourced project assumption (hmax is named symbolically in the
same paper but no numeric value for it appears anywhere). `FIELD_CAP` gates
a bias (`bias_patch` below bounds |b|max, never a coupling -- see its own
docstring), so it must read `max_abs_bias`, never `max_abs_coupling`, even
though the two agree numerically today. Every value this module prints
against it says which cap it is.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping

import numpy as np

from tsu_compiler.ir import LinearForm, VarRef
from tsu_compiler.passes.encode import Encoded, _REWRITE
from tsu_compiler.passes.lower import _affine
from tsu_compiler.passes.program import SamplingProgram
from tsu_compiler.target import Z1

# The |b| cap specifically (F-R2) -- see module docstring. NOT
# Z1.max_abs_coupling (the |J| cap): the two are separate Sourced fields
# that happen to share a value today, never to be conflated again.
FIELD_CAP = Z1.max_abs_bias.value


class FieldCapExceeded(ValueError):
    """A bias patch would push |b|max past the target's assumed field cap."""


def bias_patch(prog: SamplingProgram, enc: Encoded,
               patch: Mapping[tuple[str, int], float],
               field_cap: float = FIELD_CAP) -> SamplingProgram:
    """Return a new SamplingProgram with `patch` folded into `prog`'s biases.

    `patch`: {(cell_name, value): weight}. weight > 0 ENCOURAGES (rewards)
    that (cell, value) pair -- it makes the compiled model MORE likely to
    decode that cell to that value; weight < 0 discourages it. `cell_name`
    must be one of `enc`'s own variables -- either a categorical (e.g.
    "g3_5" on the 8x8 k=3 lattice spec) or a BINARY one (e.g. "g3_5" on
    the elev_band overlay spec, where the two-value domain means `enc`
    carries it in `enc.binary_names` rather than `enc.categorical`) --
    `value` one of that variable's domain values.

    Mechanism, and why it never hand-rolls the spin mapping: for a
    categorical cell this builds the SAME categorical-indicator LinearForm
    `encode`'s own rewrite tables consume for any ordinary spec term --
    `LinearForm({VarRef(cell, value): 1.0})` -- and rewrites it through
    `tsu_compiler.passes.encode._REWRITE[enc.encoding]` (the exact per-encoding
    table `encode()` itself uses; domain-wall or one-hot) to get a
    LinearForm over that cell's OWN physical chain spins, never any other
    cell's. For a BINARY cell there is no chain to rewrite through -- the
    IR's own convention (`tsu_compiler.ir.VarRef`'s docstring: "VarRef('a') -> the
    occupancy of binary variable a") is that the cell's physical spin IS
    its own value-1 indicator, so this builds the identical LinearForm
    `tsu_compiler.spec._value_indicator` itself builds for a Binary domain
    (`VarRef(cell): 1.0` for value 1, `VarRef(cell): -1.0, const: 1.0` for
    value 0 -- the complement) rather than inventing a second convention.
    Either physical form is then converted to a spin-space
    (const, {index: coeff}) pair via `tsu_compiler.passes.lower._affine` -- the
    identical occupancy-to-spin affine map (`n = (s+1)/2`) `lower()` itself
    uses for every Linear term in a spec. Accumulating `weight * coeff`
    into `biases[index]` is then exactly what `lower()` would do for a
    fresh `Linear(form, weight=-weight)` IR term (ir.py's own convention:
    "a PENALTY carries a POSITIVE weight", so a positive PATCH weight
    -- a reward -- is a NEGATIVE IR term weight; that negation and the ONE
    global sign flip `lower()` applies -- "sum b s + sum J s s == -E" --
    cancel, leaving the `+= weight * coeff` below). The constant part of
    the indicator's rewritten form is folded into `offset` the same way, so
    E(x) stays reconstructible.

    Only `biases` (and `offset`, an additive constant that does not affect
    sampling probabilities) ever change. `nodes`, `edges`, `weights`
    (couplings), and `blocks` are returned bit-for-bit identical to
    `prog`'s -- a categorical indicator's rewritten form is a LinearForm,
    never a Product, so it can never introduce a new coupling edge.

    Refuses (raises FieldCapExceeded) rather than returning a program whose
    patched |b|max exceeds `field_cap` -- see FIELD_CAP's docstring note:
    silently exceeding an assumed hardware field cap would produce a model
    that could not run on the hardware this project claims to target, so
    the violation is reported as an error instead of laundered through.
    """
    im = prog.ising
    idx = {n: i for i, n in enumerate(im.nodes)}
    biases = np.array(im.biases, dtype=float, copy=True)
    offset = float(im.offset)
    rewrite = _REWRITE[enc.encoding]

    for (cell, value), weight in patch.items():
        w = float(weight)
        if cell in enc.categorical:
            physical = rewrite(LinearForm({VarRef(cell, value): 1.0}),
                               enc.categorical)
        elif cell in enc.binary_names:
            # Same construction as tsu_compiler.spec._value_indicator for a Binary
            # domain: the cell's own occupancy IS its value-1 indicator, and
            # value-0 is its linear complement. No rewrite table involved --
            # there is no chain to rewrite through for a two-value domain.
            if value == 1:
                physical = LinearForm({VarRef(cell): 1.0})
            elif value == 0:
                physical = LinearForm({VarRef(cell): -1.0}, const=1.0)
            else:
                raise ValueError(
                    f"binary cell {cell!r} has no value {value!r}; binary "
                    f"domains are {{0, 1}}")
        else:
            raise KeyError(
                f"{cell!r} is not a variable of this encoding (known cells: "
                f"{sorted(set(enc.categorical) | set(enc.binary_names))})")
        const, lin = _affine(physical, idx)
        for i, coeff in lin.items():
            biases[i] += w * coeff
        offset += w * const

    bmax = float(np.abs(biases).max()) if len(biases) else 0.0
    if bmax > field_cap:
        raise FieldCapExceeded(
            f"patched |b|max = {bmax:.4f} exceeds the target's assumed "
            f"field cap {field_cap} (tsu_compiler.target.Z1.max_abs_bias -- a "
            f"project working value, NOT a sourced Extropic figure, and a "
            f"separate field from max_abs_coupling even though both "
            f"currently hold 6.0); refusing this patch rather than "
            f"returning a model that could not run on the hardware we "
            f"claim to target")

    new_im = dataclasses.replace(im, biases=biases, offset=offset)
    return dataclasses.replace(prog, ising=new_im)
