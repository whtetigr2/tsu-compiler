"""Try to prove a model's own terms are doing nothing.

This is the control the whole project is built around, and until now it existed
only as scripts. Zero a term, sample again, and compare the change against a
NOISE FLOOR: the difference between two runs of the identical model under
different seeds.

Comparing against zero instead would call every term load-bearing, because
sampling is stochastic and no two runs agree exactly. That mistake is what R23
records: a lattice whose couplings could be deleted with no measurable effect
while the page credited them with the picture.

The threshold is 3x the noise floor, the same figure audit/game_on_thrml.py
uses, so a verdict here and a verdict there mean the same thing.
"""
from __future__ import annotations

import numpy as np

from .sampler_engine import SamplerConfig, SamplerEngine

RATIO_THRESHOLD = 3.0
VALID_TARGETS = ("couplings", "biases", "none")


def _marginals(receipt_id: str, *, seed: int, n_samples: int,
               zero_couplings: bool = False,
               zero_biases: bool = False) -> np.ndarray:
    """Per-spin mean state over a batch, as a vector."""
    engine = SamplerEngine(SamplerConfig(
        receipt_id=receipt_id, seed=seed, batch_size=n_samples))
    if zero_couplings or zero_biases:
        engine.zero_terms(couplings=zero_couplings, biases=zero_biases)
    batch = engine.sample_batch(n_samples=n_samples, warmup=64)
    states = np.asarray(batch["states"], dtype=float)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    return states.mean(axis=0)


def run_ablation(receipt_id: str, *, target: str = "couplings",
                 n_samples: int = 128, seed: int = 0) -> dict:
    """Zero `target` and measure the change against the noise floor.

    `target` of "none" is the control: it ablates nothing, so its ratio must come
    out below the threshold. A control that cannot fail is not a control, and
    this one can: if the noise floor were measured wrongly, "none" would report
    a large ratio and say so.
    """
    if target not in VALID_TARGETS:
        raise ValueError(
            f"unknown ablation target {target!r}; expected one of "
            f"{', '.join(VALID_TARGETS)}. Refusing rather than ablating nothing "
            f"and reporting 'not measurable', which would look like a result.")

    base = _marginals(receipt_id, seed=seed, n_samples=n_samples)
    # The floor: the SAME model, a different seed. Everything below is measured
    # against this rather than against zero.
    other = _marginals(receipt_id, seed=seed + 1_000, n_samples=n_samples)
    noise_floor = float(np.abs(base - other).mean())

    ablated = _marginals(
        receipt_id, seed=seed, n_samples=n_samples,
        zero_couplings=(target == "couplings"),
        zero_biases=(target == "biases"),
    )
    shift = float(np.abs(base - ablated).mean())
    ratio = shift / noise_floor if noise_floor > 0 else float("inf")
    load_bearing = ratio >= RATIO_THRESHOLD

    what = "nothing" if target == "none" else f"the {target}"
    return {
        "receipt_id": receipt_id,
        "target": target,
        "n_samples": n_samples,
        "seed": seed,
        "baseline_mean": float(base.mean()),
        "ablated_mean": float(ablated.mean()),
        "noise_floor": noise_floor,
        "shift": shift,
        "ratio": ratio,
        "threshold": RATIO_THRESHOLD,
        "verdict": "load-bearing" if load_bearing else "not measurable",
        "explanation": (
            f"Zeroing {what} moved the per-spin marginals by {shift:.4f}. "
            f"Two runs of the unchanged model under different seeds differ by "
            f"{noise_floor:.4f}, which is the noise floor. That is "
            f"{ratio:.1f}x the floor, "
            + (f"above the {RATIO_THRESHOLD:.1f}x threshold, so this term is "
               f"doing real work."
               if load_bearing else
               f"below the {RATIO_THRESHOLD:.1f}x threshold, so the change is "
               f"not distinguishable from noise and nothing should be credited "
               f"to this term.")),
    }
