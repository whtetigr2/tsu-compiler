"""Turn a term off and see whether the answer moves.

The comparison baseline is NOT zero. Two runs of the SAME model under different
seeds differ by some amount, and a change smaller than that is indistinguishable
from noise. R23 is in this repository because a lattice's couplings could be
deleted with no effect at all while the prose credited them with the result.

The threshold is 3x the noise floor, matching audit/game_on_thrml.py, so a claim
made here and a claim made there mean the same thing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.ablation import RATIO_THRESHOLD, run_ablation  # noqa: E402

# A ferromagnetic hypercube: every edge pulls its endpoints together, so if
# zeroing the couplings does NOT change the marginals, something is broken.
MODEL = "tesseract_16"


def test_the_noise_floor_is_measured_not_assumed():
    out = run_ablation(MODEL, target="none", n_samples=64, seed=0)
    assert out["noise_floor"] > 0, (
        "a noise floor of exactly zero means two runs under different seeds "
        "agreed perfectly, which does not happen; it was not measured")


def test_ablating_nothing_is_indistinguishable_from_noise():
    """The control. If this fails, the noise floor is wrong and every other
    verdict this function produces is worthless."""
    out = run_ablation(MODEL, target="none", n_samples=64, seed=0)
    assert out["ratio"] < RATIO_THRESHOLD, (
        f"ablating NOTHING moved the marginals {out['ratio']:.1f}x the noise "
        f"floor. The noise floor is being measured wrongly.")
    assert out["verdict"] == "not measurable"


def test_zeroing_couplings_moves_a_ferromagnetic_model():
    out = run_ablation(MODEL, target="couplings", n_samples=64, seed=0)
    assert out["ratio"] > RATIO_THRESHOLD, (
        f"zeroing every coupling on a ferromagnetic hypercube moved the "
        f"marginals only {out['ratio']:.1f}x the noise floor. Either the "
        f"couplings are inert or the ablation is not zeroing them. "
        f"{out['explanation']}")
    assert out["verdict"] == "load-bearing"


def test_the_explanation_states_both_numbers_and_the_threshold():
    """A verdict without its numbers is an opinion."""
    out = run_ablation(MODEL, target="couplings", n_samples=64, seed=0)
    text = out["explanation"]
    assert f"{out['shift']:.4f}" in text
    assert f"{out['noise_floor']:.4f}" in text
    assert str(RATIO_THRESHOLD) in text or f"{RATIO_THRESHOLD:.1f}" in text


def test_it_refuses_an_unknown_target_rather_than_ablating_nothing():
    """Silently ablating nothing would report 'not measurable' and look fine."""
    with pytest.raises(ValueError) as exc:
        run_ablation(MODEL, target="spin_colour", n_samples=16, seed=0)
    assert "spin_colour" in str(exc.value)
