"""The four gaps that stood between a saved run and a citable one.

An external reader (Grok) reviewed a saved run and identified what it does and
does not prove. Three of four points held; the fourth was wrong on the facts (the
raw fields ARE saved) but right in weaker form (they are summed LEVELS, not the
per-layer binary draws, so the decode is replayable while the Ising sampling
behind it is not re-verifiable).

The load-bearing one: `mode` is an interpolation choice applied on the
measure-to-decode path, so for any single world you cannot tell how much of a
shoreline came from thrml and how much from the interpolator. A statistical
answer already exists -- shuffling each field at its OWN sampling resolution and
re-running the identical pipeline leaves real ahead of shuffled in both modes
(+0.0092 nearest, +0.0161 bilinear) -- but that is aggregate over 8 seeds and
labelled indicative. It says nothing about THIS world's coastline.

The per-world answer is a diff: decode the same spins both ways. A feature
present in both was sampled; a feature present only under bilinear was
introduced by the interpolator.
"""
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, "demo")
sys.path.insert(0, "src")

from world.generate import generate, FIELD_PLAN
from world.spline import LEVELS
from world.studio import Studio, Sampled, Readout, KC
from world.run_log import save_run


@pytest.fixture(scope="module")
def studio():
    s = Studio()
    s.generate(Sampled(seed=31, size=64, warmup=200))
    s.set_readout(Readout(sea_level=0.1, mountain_line=0.05, relief=1.2))
    return s


# --- the per-layer draws travel with the world -----------------------------

def test_world_carries_the_per_layer_draws_not_only_their_sum(studio):
    """A field's LEVELS are its layers summed. Saving only the sum makes the
    decode replayable but the Ising sampling behind it unverifiable -- a reader
    cannot check the draws against thrml. The layers are the evidence."""
    w = studio.world
    for name, _src in FIELD_PLAN:
        raw = w.fields[name]
        stack = w.layers[name]
        assert stack.shape == (LEVELS,) + raw.shape
        assert set(np.unique(stack)) <= {0, 1}, "layers are binary draws"


def test_the_layers_actually_sum_to_the_field(studio):
    """The strong check: layers are not a decorative extra, they are what the
    field IS. If they did not reconstruct it, one of the two is not what it
    claims to be."""
    w = studio.world
    for name, _src in FIELD_PLAN:
        assert np.array_equal(w.layers[name].sum(axis=0), w.fields[name])


def test_the_export_carries_the_layers(tmp_path, studio):
    save_run(studio, tmp_path)
    d = json.loads((tmp_path / "world.json").read_text(encoding="utf-8"))
    assert set(d["layers"]) == {name for name, _ in FIELD_PLAN}
    for name, _src in FIELD_PLAN:
        stack = np.array(d["layers"][name])
        assert stack.shape[0] == LEVELS
        assert np.array_equal(stack.sum(axis=0), np.array(d["fields"][name]))


# --- both decodes of the same spins ----------------------------------------

def test_terrain_for_mode_returns_a_different_decode_of_the_same_spins(studio):
    """Mode is an interpolation choice, not a sampling one, so both decodes come
    from identical draws. That is exactly what makes their diff meaningful."""
    near = studio.terrain_for_mode("nearest")
    bil = studio.terrain_for_mode("bilinear")
    assert near.shape == bil.shape == (studio.world.size,) * 2
    assert not np.array_equal(near, bil), (
        "the two interpolations must actually differ, or the diff proves nothing")


def test_terrain_for_mode_matches_the_live_view_for_the_worlds_own_mode(studio):
    """The alternate-decode path must agree with the main one where they
    overlap. If it did not, the diff would be measuring the two code paths
    against each other rather than the two interpolations."""
    assert np.array_equal(
        studio.terrain_for_mode(studio.world.mode), studio.view().terrain)


def test_terrain_for_mode_honours_the_current_readout(studio):
    """The diff must be taken at the sliders the user is actually looking at,
    not at defaults."""
    dry = studio.set_readout(Readout(sea_level=-0.4))
    near_dry = studio.terrain_for_mode("nearest")
    studio.set_readout(Readout(sea_level=+0.4))
    near_wet = studio.terrain_for_mode("nearest")
    assert not np.array_equal(near_dry, near_wet)
    studio.set_readout(Readout(sea_level=0.1, mountain_line=0.05, relief=1.2))


def test_terrain_for_mode_never_resamples(studio, monkeypatch):
    """Same property the readout path has: the alternate decode reinterprets
    cached fields. Patching where the name is actually BOUND, and without
    raising=False, so a missing attribute fails loudly."""
    import world.generate as gen_mod

    def _forbidden(*a, **k):
        raise AssertionError("terrain_for_mode reached the sampler")

    monkeypatch.setattr(gen_mod, "sample_field", _forbidden)
    studio.terrain_for_mode("nearest")
    studio.terrain_for_mode("bilinear")


def test_the_run_saves_both_decodes_and_their_disagreement(tmp_path, studio):
    """The artifact a reader needs: same spins, two interpolations, and the
    cells where they disagree. Structure in both is the lattice; structure only
    under bilinear is the interpolator."""
    save_run(studio, tmp_path)
    d = json.loads((tmp_path / "decodes.json").read_text(encoding="utf-8"))
    near = np.array(d["terrain_nearest"])
    bil = np.array(d["terrain_bilinear"])
    assert np.array_equal(near, studio.terrain_for_mode("nearest"))
    assert np.array_equal(bil, studio.terrain_for_mode("bilinear"))
    assert d["cells_differing"] == int((near != bil).sum())
    assert d["fraction_differing"] == pytest.approx(
        float((near != bil).mean()))


# --- the coupling is recorded against Onsager ------------------------------

def test_kc_is_computed_not_pasted():
    """sinh(2Kc) = 1, so Kc = arcsinh(1)/2. Computed every time; a pasted
    decimal is exactly the kind of unverified figure this project refuses."""
    import math
    assert KC == math.asinh(1.0) / 2.0
    assert abs(KC - 0.4406867935) < 1e-9


def test_studio_json_records_the_coupling_against_kc(tmp_path, studio):
    """beta*J alone does not say where a run sits. 0.39 is not "0.39 of
    critical" -- it is 0.885 x Kc, slightly subcritical, which is a real
    operating choice and should be legible without the reader doing the
    division."""
    save_run(studio, tmp_path)
    d = json.loads((tmp_path / "studio.json").read_text(encoding="utf-8"))
    assert d["kc"] == pytest.approx(KC)
    assert d["beta_j_over_kc"] == pytest.approx(
        studio.sampled.beta_j / KC)


# --- the README must not imply a mixing claim ------------------------------

def test_readme_distinguishes_structure_stability_from_mixing(tmp_path, studio):
    """Structure-stable is NOT mixed. The warmup finding is that 4,000 and
    100,000 sweeps give the same STRUCTURE; whether the chain has converged is
    a different question this project has never measured. The README must not
    let the first be read as the second."""
    save_run(studio, tmp_path)
    txt = (tmp_path / "README.md").read_text(encoding="utf-8").lower()
    assert "structure" in txt
    assert "mixing" in txt or "converg" in txt
    assert "no claim" in txt
