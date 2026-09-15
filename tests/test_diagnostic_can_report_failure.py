"""Can our sampling diagnostics report FAILURE, and which ones go blind?

Every workload this project has measured came back "sampleable". A verdict
function that has only ever returned PASS has not been shown capable of
returning FAIL, and quoting its passes as evidence is circular. This file is the
missing negative, run every suite.

The construction has a KNOWN answer rather than a measured one: a square-lattice
Ising model in zero field is exactly solved, and Onsager's Kc = ln(1+sqrt(2))/2
= 0.440687 says where the ordered phase begins. Deep inside it, sixteen chains
started independently cannot each have explored both symmetry-related modes, so
any diagnostic reporting R-hat ~ 1.000 there is reporting health it never
earned. That is not an opinion about mixing; it is the point of the model.

The full sweep across both coupling signs lives in `audit/diagnostic_control.py`
and is the receipt. This is the fast version that guards the property.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu_compiler.backends.thrml_backend import sample_chains
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.program import build_program
from tsu_compiler.preflight.diagnostics import RHAT_THRESHOLD, r_hat

L = 8
N = L * L
KC = float(np.log(1 + np.sqrt(2)) / 2)
ORDERED = 0.80      # comfortably above Kc
DISORDERED = 0.15   # comfortably below
CHAINS, SAMPLES, WARMUP, STEPS = 16, 1500, 1000, 4


def lattice(beta_j: float, weight: float) -> IsingModel:
    idx = lambda i, j: (i % L) * L + (j % L)
    e = sorted({(min(idx(i, j), idx(i + di, j + dj)),
                 max(idx(i, j), idx(i + di, j + dj)))
                for i in range(L) for j in range(L)
                for di, dj in ((1, 0), (0, 1))
                if idx(i, j) != idx(i + di, j + dj)})
    return IsingModel(nodes=tuple(f"n{i}" for i in range(N)), edges=tuple(e),
                      weights=np.full(len(e), weight), biases=np.zeros(N),
                      beta=beta_j, offset=0.0)


def diagnostics(beta_j: float, weight: float):
    """R-hat three ways on the SAME draws: folded scalar, signed scalar, and the
    maximum over individual spins."""
    model = lattice(beta_j, weight)
    draws = np.asarray(sample_chains(
        build_program(model, analyse(model)), n_chains=CHAINS,
        n_samples=SAMPLES, n_warmup=WARMUP, steps_per_sample=STEPS, seed=0))
    mag = (2 * draws.astype(np.int8) - 1).mean(axis=2)
    per_spin = [float(r_hat((2.0 * draws[:, :, j] - 1.0).astype(np.float64)))
                for j in range(N)
                if (2.0 * draws[:, :, j] - 1.0).std() > 0]
    return (float(r_hat(np.abs(mag))), float(r_hat(mag)),
            max(per_spin) if per_spin else float("nan"))


def test_the_per_spin_diagnostic_fires_on_chains_that_are_provably_stuck():
    """WHAT THIS PINS: the diagnostic backing every "sampleable" verdict in this
    project is capable of returning FAIL. Without this, the codon results are a
    function that has never been observed to reject anything.

    HOW IT FAILS: weaken the per-spin pass in `audit/codon_sampleability.py` --
    average the spins, drop the R-hat check, raise the threshold -- and this
    stops firing on chains that are stuck by construction.

    PROVENANCE: Onsager's exact Kc = 0.440687 for this lattice; beta*J = 0.80 is
    deep in the ordered phase, where 16 independently started chains cannot each
    have visited both ground states."""
    for weight in (1.0, -1.0):
        _folded, _signed, per_spin = diagnostics(ORDERED, weight)
        assert per_spin > RHAT_THRESHOLD, (
            f"weight {weight:+.1f} at beta*J={ORDERED} (ordered phase): "
            f"max per-spin R-hat {per_spin:.4f} did not exceed "
            f"{RHAT_THRESHOLD}, so the diagnostic cannot detect a stuck chain")


def test_the_diagnostic_stays_quiet_when_the_chains_are_genuinely_fine():
    """WHAT THIS PINS: specificity. A check that fires on everything is as
    useless as one that fires on nothing, and the test above would pass for a
    diagnostic hardwired to complain.

    HOW IT FAILS: make the per-spin check over-sensitive and this catches it.

    PROVENANCE: beta*J = 0.15 is far below Kc; the model is disordered and
    block Gibbs mixes it easily."""
    for weight in (1.0, -1.0):
        folded, signed, per_spin = diagnostics(DISORDERED, weight)
        assert per_spin <= RHAT_THRESHOLD, (
            f"weight {weight:+.1f} at beta*J={DISORDERED} (disordered): max "
            f"per-spin R-hat {per_spin:.4f} flags a model that mixes fine")
        assert folded <= RHAT_THRESHOLD and signed <= RHAT_THRESHOLD


def test_folding_by_abs_hides_a_stuck_ferromagnet():
    """WHAT THIS PINS: the R20 defect, as a live demonstration rather than an
    argument. Below Tc a ferromagnet's two ground states are all-up and
    all-down; chains split between them. Signed m separates them and R-hat
    explodes. Folding by abs() maps them onto ONE value and reports perfect
    health on chains that never mixed.

    HOW IT FAILS: reintroduce `np.abs(...)` around the order parameter in
    `audit/codon_sampleability.py` and this is the evidence of what it costs.

    PROVENANCE: measured, not argued -- at beta*J = 0.80 the folded R-hat is
    1.0000 while the signed R-hat is in the hundreds."""
    folded, signed, per_spin = diagnostics(ORDERED, 1.0)
    assert signed > 10.0, (
        f"signed R-hat {signed:.4f} should be enormous for a ferromagnet split "
        f"across both ground states")
    assert folded <= RHAT_THRESHOLD, (
        f"folded R-hat {folded:.4f}: if folding no longer hides this, the "
        f"docstring's claim about what abs() costs needs rewriting")
    assert per_spin > RHAT_THRESHOLD


def test_both_scalar_order_parameters_go_blind_on_a_stuck_antiferromagnet():
    """WHAT THIS PINS: the stronger lesson, and the reason the verdict is
    per-spin rather than merely unfolded. On this bipartite lattice an
    antiferromagnet's two ground states are the two checkerboards -- and BOTH
    have net magnetisation zero. So the folded scalar is blind AND the signed
    scalar is blind, while every chain sits frozen in a different configuration.
    Unfolding the order parameter was necessary but NOT sufficient.

    HOW IT FAILS: reduce the verdict in `audit/codon_sampleability.py` to any
    scalar summary, folded or not, and it goes blind exactly here.

    PROVENANCE: the sign is confirmed against `audit/oracles/exact.py`, which
    does not import this package -- see `audit/diagnostic_control.py`. The mean
    neighbour correlation at IR weight -1.0 is -0.954 and <|m|> is 0.016."""
    folded, signed, per_spin = diagnostics(ORDERED, -1.0)
    assert folded <= RHAT_THRESHOLD
    assert signed <= RHAT_THRESHOLD, (
        f"signed R-hat {signed:.4f}: this test documents that the signed scalar "
        f"ALSO goes blind here; if that has changed, the claim must be rewritten")
    assert per_spin > RHAT_THRESHOLD, (
        f"max per-spin R-hat {per_spin:.4f} -- if this stops firing, nothing in "
        f"the project can detect a stuck antiferromagnet")
