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
import dataclasses
import importlib
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
import tsu.target as target_mod

import layers
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

# --------------------------------------------------------------------------
# 5. bias_patch also works on a BINARY-domain cell (Task 4 finding): the
#    elev_band overlay spec (specs/lattice_elev_band_8x8.yaml) declares
#    every cell as Binary, so `enc.categorical` is EMPTY for it and every
#    cell lives in `enc.binary_names` instead. The original bias_patch
#    guarded ONLY on `cell not in enc.categorical`, which raised KeyError
#    for every binary cell -- this would have made demo/elevation.py's own
#    band_patch (which produces exactly {(cell, 1): weight} patches, see
#    that module's docstring) unusable against the one receipt it exists
#    to condition. Fixed to build the SAME value-1/value-0 LinearForm
#    tsu.spec._value_indicator itself builds for a Binary domain.
# --------------------------------------------------------------------------

def _load_elev_band():
    R = "demo/receipts/elev_band"
    spec = load_spec(str(Path(R) / "spec.yaml"))
    enc = encode(spec, _selected_encoding(Path(R)))
    prog = reconstruct_program(R)
    return spec, enc, prog


def test_patch_works_on_a_binary_cell_not_just_categorical():
    _, enc, prog = _load_elev_band()
    assert enc.categorical == {}          # sanity: this receipt has none
    assert "g0_0" in enc.binary_names     # sanity: it's a binary cell

    idx = prog.ising.nodes.index("g0_0")
    before = float(prog.ising.biases[idx])

    encouraged = bias_patch(prog, enc, {("g0_0", 1): 0.5})
    assert float(encouraged.ising.biases[idx]) == pytest.approx(before + 0.25)

    discouraged = bias_patch(prog, enc, {("g0_0", 0): 0.5})
    assert float(discouraged.ising.biases[idx]) == pytest.approx(before - 0.25)

    # topology untouched, same guarantee as the categorical path
    assert encouraged.ising.nodes == prog.ising.nodes
    assert encouraged.ising.edges == prog.ising.edges
    np.testing.assert_array_equal(encouraged.ising.weights, prog.ising.weights)


def test_patch_on_unknown_binary_value_raises():
    _, enc, prog = _load_elev_band()
    with pytest.raises(ValueError, match="has no value"):
        bias_patch(prog, enc, {("g0_0", 2): 0.5})


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


# --------------------------------------------------------------------------
# 6. F-R2: FIELD_CAP must read TargetProfile.max_abs_bias (the |b| cap) --
#    not max_abs_coupling (the |J| cap), and not a bare hardcoded literal.
#    Both real-profile fields currently hold the identical value 6.0 (see
#    tsu/target.py's Z1), so a test against the real profile alone cannot
#    tell a correct wiring from a wrong one -- FIELD_CAP would read 6.0
#    either way. This test builds a synthetic TargetProfile where the two
#    fields DIVERGE and reloads demo/layers.py against it, so the
#    assertion can only pass if the module genuinely reads max_abs_bias.
# --------------------------------------------------------------------------

def test_field_cap_reads_max_abs_bias_not_max_abs_coupling(monkeypatch):
    diverging = dataclasses.replace(
        target_mod.Z1,
        max_abs_bias=target_mod.Sourced(9.0, "test", "synthetic, this test only"),
        max_abs_coupling=target_mod.Sourced(3.0, "test", "synthetic, this test only"),
    )
    monkeypatch.setattr(target_mod, "Z1", diverging)

    try:
        importlib.reload(layers)
        assert layers.FIELD_CAP == pytest.approx(9.0), (
            "FIELD_CAP must track TargetProfile.max_abs_bias (9.0 in this "
            f"synthetic profile), got {layers.FIELD_CAP!r}")
        assert layers.FIELD_CAP != pytest.approx(3.0), (
            "FIELD_CAP must NOT track TargetProfile.max_abs_coupling "
            "(3.0 in this synthetic profile)")
    finally:
        importlib.reload(layers)  # restore the real Z1-backed FIELD_CAP
