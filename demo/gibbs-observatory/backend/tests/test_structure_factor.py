"""The structure factor must land its peak where physics says it does.

S(k) is only worth showing if it is right, and it has a known answer available:
an antiferromagnet orders into a checkerboard, whose Bragg peak sits at the
corner of the Brillouin zone, k = (pi, pi). A ferromagnet peaks at k = 0.

The shipped `alloy_ordering_8x8` program prefers unlike neighbours, which is a
textbook antiferromagnet, so it is a test with an answer nobody here chose.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.structure_factor import (  # noqa: E402
    recover_lattice, structure_factor,
)


def _lattice(w: int, h: int):
    return recover_lattice([f"g{x}_{y}" for y in range(h) for x in range(w)])


# --------------------------------------------------------------------------
# Lattice recovery, and its refusals.
# --------------------------------------------------------------------------

def test_a_grid_is_recovered_from_the_site_names():
    lat = _lattice(8, 8)
    assert lat is not None
    assert (lat.width, lat.height) == (8, 8)
    assert lat.index_to_xy[0] == (0, 0)
    assert lat.index_to_xy[1] == (1, 0)
    assert lat.index_to_xy[8] == (0, 1)


def test_names_that_are_not_a_grid_produce_no_lattice():
    """A model off a lattice has no structure factor, and must say so."""
    assert recover_lattice(["p0", "p1", "p2"]) is None
    assert recover_lattice(None) is None
    assert recover_lattice([]) is None


def test_a_partially_covered_grid_is_refused():
    """Holes transform into edge artefacts that look exactly like structure."""
    names = [f"g{x}_{y}" for y in range(4) for x in range(4)]
    assert recover_lattice(names[:-1]) is None


# --------------------------------------------------------------------------
# The physics. Known answers.
# --------------------------------------------------------------------------

def test_a_perfect_checkerboard_peaks_at_the_zone_corner():
    """An antiferromagnet's Bragg peak is at k = (pi, pi). Constructed, exact."""
    lat = _lattice(8, 8)
    state = [1.0 if (xy[0] + xy[1]) % 2 == 0 else -1.0
             for _, xy in sorted(lat.index_to_xy.items())]
    out = structure_factor([state], lat)
    assert out["available"]
    kx, ky = out["peak_k_over_pi"]
    assert abs(abs(kx) - 1.0) < 1e-9 and abs(abs(ky) - 1.0) < 1e-9, (
        f"a checkerboard must peak at (pi, pi), got ({kx}, {ky})")
    assert "checkerboard" in out["peak_label"]


def test_a_uniform_field_puts_everything_at_k_zero():
    """A ferromagnet's weight is all at the origin, which is why it is dropped."""
    lat = _lattice(8, 8)
    out = structure_factor([[1.0] * 64], lat)
    assert out["k0_intensity"] == pytest.approx(64 ** 2)
    assert out["max"] == pytest.approx(0.0, abs=1e-9), (
        "with k=0 excluded a perfectly uniform field has no other weight")


def test_stripes_peak_on_a_zone_edge():
    lat = _lattice(8, 8)
    state = [1.0 if xy[0] % 2 == 0 else -1.0
             for _, xy in sorted(lat.index_to_xy.items())]
    out = structure_factor([state], lat)
    kx, ky = out["peak_k_over_pi"]
    assert abs(abs(kx) - 1.0) < 1e-9 and abs(ky) < 1e-9, (
        f"vertical stripes must peak at (pi, 0), got ({kx}, {ky})")
    assert "stripe" in out["peak_label"]


def test_noise_does_not_produce_a_high_symmetry_peak():
    """A control. If random spins landed on (pi, pi) the test above proves
    nothing, because everything would."""
    lat = _lattice(16, 16)
    rng = np.random.default_rng(0)
    corner = 0
    for _ in range(20):
        state = rng.choice([-1.0, 1.0], size=256).tolist()
        kx, ky = structure_factor([state], lat)["peak_k_over_pi"]
        if abs(abs(kx) - 1.0) < 0.05 and abs(abs(ky) - 1.0) < 0.05:
            corner += 1
    assert corner <= 3, (
        f"{corner} of 20 random fields peaked at the zone corner; the corner "
        f"test is then not evidence of order")


def test_no_draws_is_unavailable_not_an_empty_picture():
    lat = _lattice(4, 4)
    out = structure_factor([], lat)
    assert out["available"] is False
    assert "unavailable" in out["reason"]


# --------------------------------------------------------------------------
# Against the shipped program, sampled for real.
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_shipped_alloy_orders_into_a_checkerboard_when_sampled():
    """End to end: THRML samples the real receipt and the peak lands correctly."""
    from backend.app.sampler_engine import SamplerConfig, SamplerEngine

    engine = SamplerEngine(SamplerConfig(
        receipt_id="prog_alloy_ordering_8x8", beta=0.9, batch_size=32, seed=0))
    lat = recover_lattice(engine.state.graph.node_names)
    if lat is None:
        pytest.skip("alloy receipt carries no grid names here")
    batch = engine.sample_batch(n_samples=64, warmup=400)
    out = structure_factor(batch["states"], lat)
    kx, ky = out["peak_k_over_pi"]
    assert abs(abs(kx) - 1.0) < 0.2 and abs(abs(ky) - 1.0) < 0.2, (
        f"the ordering alloy prefers unlike neighbours, so it must order into "
        f"a checkerboard peaking at (pi, pi). Got ({kx}, {ky}).")


@pytest.mark.slow
def test_the_accumulator_resets_when_the_model_changes():
    """Mixing one lattice's spectrum into another's would be silent and wrong.

    S(k) accumulates across batches so the Bragg peak sharpens out of the haze
    while the sampler runs. That is only safe if switching programs starts over.
    """
    from backend.app.sampler_engine import SamplerConfig, SamplerEngine

    engine = SamplerEngine(SamplerConfig(
        receipt_id="prog_alloy_ordering_8x8", beta=0.9, batch_size=8, seed=0))
    first = engine.sample_batch(n_samples=8, warmup=200)["structure_factor"]
    second = engine.sample_batch(n_samples=8, warmup=0)["structure_factor"]
    assert second["n_draws"] > first["n_draws"], (
        "draws must accumulate across batches within one model")

    engine.reset(SamplerConfig(receipt_id="prog_mrf_denoise_8x8",
                               beta=0.6, batch_size=8, seed=0))
    after = engine.sample_batch(n_samples=8, warmup=200)["structure_factor"]
    assert after["n_draws"] == 8, (
        f"after switching programs the accumulator carried {after['n_draws']} "
        f"draws forward; the new model's spectrum is contaminated by the old "
        f"model's")


# --------------------------------------------------------------------------
# What S(k) throws away. The view states these, so they must stay true.
# --------------------------------------------------------------------------

def _field_to_state(field, lat):
    return [field[y][x] for _, (x, y) in sorted(lat.index_to_xy.items())]


def _spectrum(field, lat):
    return np.array(structure_factor([_field_to_state(field, lat)], lat)["values"])


@pytest.mark.parametrize("pattern", ["checkerboard", "random"])
def test_every_translation_gives_an_identical_spectrum(pattern):
    """S(k) is translation invariant, so it cannot locate a pattern.

    This is the precise content of calling it a shadow: the transform maps
    2^(W*H) configurations onto W*H numbers, and every shift lands on the same
    point. The view says so, and this keeps that honest.
    """
    W = H = 6
    lat = _lattice(W, H)
    if pattern == "checkerboard":
        base = np.array([[1.0 if (x + y) % 2 == 0 else -1.0 for x in range(W)]
                         for y in range(H)])
    else:
        base = np.random.default_rng(0).choice([-1.0, 1.0], size=(H, W))

    reference = _spectrum(base, lat)
    identical = sum(
        1 for dy in range(H) for dx in range(W)
        if np.allclose(_spectrum(np.roll(np.roll(base, dy, 0), dx, 1), lat),
                       reference, atol=1e-9))
    assert identical == W * H, (
        f"only {identical} of {W * H} translations matched; the view claims "
        f"all of them do")


def test_a_global_spin_flip_is_invisible_to_it():
    lat = _lattice(6, 6)
    base = np.array([[1.0 if (x + y) % 2 == 0 else -1.0 for x in range(6)]
                     for y in range(6)])
    assert np.allclose(_spectrum(-base, lat), _spectrum(base, lat), atol=1e-9)


def test_a_mirror_reflects_the_spectrum_rather_than_leaving_it_unchanged():
    """An earlier version of this claimed a mirror was invisible to S(k).

    That was generalised from a checkerboard, which is already symmetric, so
    the spectrum happened not to move. It is false for a general field: for a
    real input |FFT|^2 obeys Friedel's law, S(-k) = S(k), and mirroring x maps
    S(kx, ky) to S(-kx, ky). The spectrum is reflected, not preserved.

    The claim reached the interface before this test caught it.
    """
    lat = _lattice(6, 6)
    base = np.random.default_rng(1).choice([-1.0, 1.0], size=(6, 6))
    S = _spectrum(base, lat)
    mirrored = _spectrum(base[:, ::-1], lat)
    assert not np.allclose(mirrored, S, atol=1e-9), (
        "a mirror must move the spectrum of an asymmetric field")
    assert np.allclose(mirrored, np.roll(S[:, ::-1], 1, axis=1), atol=1e-9), (
        "mirroring the field must reflect the spectrum in kx")


def test_the_spectrum_is_centrosymmetric():
    """Friedel's law, S(-k) = S(k), which holds for any real field."""
    lat = _lattice(6, 6)
    base = np.random.default_rng(2).choice([-1.0, 1.0], size=(6, 6))
    S = _spectrum(base, lat)
    assert np.allclose(np.roll(np.roll(S[::-1, ::-1], 1, 0), 1, 1), S,
                       atol=1e-9)
