"""demo/layers.py -- bias-patch cross-layer conditioning.

Conditioning a later layer on an earlier layer's decoded output is
implemented as a per-cell BIAS PATCH on an already-compiled program's
`biases` array -- never a clamp (layer 2's variables are disjoint from
layer 1's; there is nothing to clamp TO except through the energy), and
never a recompile (topology is untouched; see demo/layers.py's module
docstring for the derivation).

These tests exercise demo/layers.py directly against the real compiled
receipt at demo/receipts/small -- no synthetic/toy model, no fabricated
data, per project epistemic discipline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

from tsu.spec import load_spec
from tsu.passes.encode import encode
from tsu.simulate import reconstruct_program, _selected_encoding
from tsu.backends.thrml_backend import sample as thrml_sample

from layers import bias_patch, FieldCapExceeded, FIELD_CAP

R = "demo/receipts/small"


def _load():
    spec = load_spec(str(Path(R) / "spec.yaml"))
    enc = encode(spec, _selected_encoding(Path(R)))
    prog = reconstruct_program(R)
    return spec, enc, prog


# --------------------------------------------------------------------------
# 1. A patch changes `biases` and leaves `nodes`, `edges`, `blocks` identical.
# --------------------------------------------------------------------------

def test_patch_changes_only_biases_topology_untouched():
    _, enc, prog = _load()
    patched = bias_patch(prog, enc, {("g0_0", 1): 4.0})

    assert patched.ising.nodes == prog.ising.nodes
    assert patched.ising.edges == prog.ising.edges
    np.testing.assert_array_equal(patched.ising.weights, prog.ising.weights)
    assert patched.blocks == prog.blocks
    assert not np.array_equal(patched.ising.biases, prog.ising.biases)


# --------------------------------------------------------------------------
# 2. A patch that would push |b|max past the cap is refused with a clear error.
# --------------------------------------------------------------------------

def test_patch_refused_when_it_would_exceed_the_field_cap():
    _, enc, prog = _load()
    base_bmax = float(np.abs(prog.ising.biases).max())
    assert base_bmax < FIELD_CAP  # sanity: headroom exists before the patch

    with pytest.raises(FieldCapExceeded):
        bias_patch(prog, enc, {("g0_0", 2): 1000.0})


# --------------------------------------------------------------------------
# 3. An empty patch is a no-op producing an equal model.
# --------------------------------------------------------------------------

def test_empty_patch_is_a_noop():
    _, enc, prog = _load()
    patched = bias_patch(prog, enc, {})

    assert patched.ising.nodes == prog.ising.nodes
    assert patched.ising.edges == prog.ising.edges
    np.testing.assert_array_equal(patched.ising.weights, prog.ising.weights)
    np.testing.assert_array_equal(patched.ising.biases, prog.ising.biases)
    assert patched.ising.offset == prog.ising.offset
    assert patched.ising.beta == prog.ising.beta
    assert patched.blocks == prog.blocks
    assert patched.clamped == prog.clamped
    assert patched.clamp_values == prog.clamp_values


# --------------------------------------------------------------------------
# 4. A patch on (cell, value) moves that cell's decoded value toward `value`
#    MORE OFTEN than an unpatched run -- statistical, over enough samples to
#    be meaningful, sample count stated explicitly.
# --------------------------------------------------------------------------

def test_patch_statistically_shifts_the_targeted_cells_decoded_value():
    """8 chains x 80 samples = 640 draws per run (matches the scale
    demo/render_world.py already uses for this same receipt, ~0.4-1.4s per
    call). Domain-wall decode of a single chain is a PROJECTION
    (sum of the chain's bits) that always returns a value, monotone or not
    (encode.py's own docstring), so every one of the 640 draws contributes a
    reading -- no non-codeword draws need to be discarded to compare the two
    runs' frequencies at cell g0_0."""
    spec, enc, prog = _load()
    cell, target_value = "g0_0", 2  # grass
    chain = enc.categorical[cell]

    n_chains, n_samples = 8, 80
    common = dict(n_chains=n_chains, n_samples=n_samples, n_warmup=1200,
                  steps_per_sample=4)

    baseline = thrml_sample(prog, seed=11, **common)
    patched_prog = bias_patch(prog, enc, {(cell, target_value): 6.0})
    patched = thrml_sample(patched_prog, seed=11, **common)

    def freq_at_target(rows):
        idx = [prog.ising.nodes.index(s) for s in chain]
        hits = 0
        for row in rows:
            bits = {s: int(row[i]) for s, i in zip(chain, idx)}
            hits += int(sum(bits.values()) == target_value)
        return hits / len(rows)

    n = n_chains * n_samples
    base_freq = freq_at_target(baseline)
    patched_freq = freq_at_target(patched)
    print(f"n={n} draws/run: baseline freq(g0_0==2)={base_freq:.3f}, "
          f"patched freq(g0_0==2)={patched_freq:.3f}")

    assert patched_freq > base_freq
