"""Every statistic on screen must be one this project would publish.

CLAUDE.md carries a table headed "Before you write numpy, check this table",
because a numpy reimplementation of something the compiler already has is a
regression, not a shortcut. These check the live statistics against that rule and
against the compiler's own answers.

The finding that prompted them. `metrics.py` computed effective sample size from
lag-1 autocorrelation as n(1-rho)/(1+rho). The compiler ships
`tsu_compiler.ess.effective_sample_size`, a Sokal-window estimator that REFUSES
to answer when the chain is too short, against a threshold of N/tau >= 5000 its
own AR(1) validation established. The interface keeps 256 samples of history. At
that length the compiler declines every time:

    series                      N        shown    compiler
    white noise               256        237.1    REFUSES
    rho=0.5                   256        113.1    REFUSES
    rho=0.95 (near critical)  256          8.9    REFUSES
    white noise            20,000     19,985.5    20,561.3
    rho=0.5                20,000      6,791.9     6,760.4

The estimator is fine on long chains. The chains are too short, and nothing said
so. Every ESS this application displayed was one this project would not publish.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from tsu_compiler.ess import RELIABILITY_MIN_N_OVER_TAU  # noqa: E402

from backend.app.metrics import summarize_series  # noqa: E402


def _ar1(rho: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.normal(size=n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def test_short_chains_report_no_ess_rather_than_a_confident_number():
    """The live history is 256 samples. That is too short to publish an ESS."""
    out = summarize_series(_ar1(0.95, 256))
    assert out["ess"] is None, (
        f"a 256-sample chain reported ESS={out['ess']}. The compiler refuses "
        f"below N/tau >= {RELIABILITY_MIN_N_OVER_TAU}, and this is the number "
        f"the interface puts on screen.")
    assert out["ess_reliable"] is False
    assert "unavailable" in (out["ess_reason"] or "").lower()


def test_the_reason_names_why_rather_than_being_blank():
    out = summarize_series(_ar1(0.95, 256))
    reason = (out["ess_reason"] or "").lower()
    assert "tau" in reason or "short" in reason, reason


def test_a_long_enough_chain_does_report_an_ess():
    """A check that cannot pass proves as little as one that cannot fail."""
    out = summarize_series(_ar1(0.0, 40_000))
    assert out["ess"] is not None, out["ess_reason"]
    assert out["ess_reliable"] is True
    assert out["ess"] > 1_000


def test_ess_matches_the_compiler_not_a_reimplementation():
    from tsu_compiler.ess import effective_sample_size

    x = _ar1(0.5, 40_000)
    ours = summarize_series(x)
    theirs = effective_sample_size(x.reshape(1, -1))
    assert theirs.reliable
    assert ours["ess"] == pytest.approx(theirs.ess, rel=1e-9), (
        "the live ESS is not the compiler's ESS; it has been reimplemented")


def test_mean_std_and_last_still_work():
    """Cheap and always available. They must not have been lost."""
    out = summarize_series(np.array([1.0, 2.0, 3.0, 4.0]))
    assert out["mean"] == pytest.approx(2.5)
    assert out["last"] == pytest.approx(4.0)
    assert out["n"] == 4


def test_an_empty_series_is_unavailable_not_zero():
    out = summarize_series(np.array([]))
    assert out["ess"] is None
    assert out["mean"] is None, "an empty series has no mean; 0.0 is a lie"


def test_a_failed_receipt_load_records_why_it_substituted_a_model(monkeypatch):
    """Swapping the model is honest only if it says what went wrong.

    `_resolve_graph` catches bare Exception and samples a generic lattice2d
    instead. Measured: all packaged examples load today, so it does not fire.
    It is a trap waiting for the first receipt that fails to parse, and the
    person watching the energy trace would be watching a different model.
    """
    from backend.app import sampler_engine as se

    def boom(_rid):
        raise ValueError("receipt is malformed in some specific way")

    monkeypatch.setattr(se, "receipt_to_graph_arrays", boom, raising=False)
    import backend.app.receipt_loader as rl
    monkeypatch.setattr(rl, "receipt_to_graph_arrays", boom, raising=False)

    engine = se.SamplerEngine(se.SamplerConfig(receipt_id="small"))
    assert engine.config.sampling_fallback is True
    assert engine.config.fallback_reason, (
        "the engine swapped in a different model and recorded no reason")
    assert "malformed" in engine.config.fallback_reason
    assert "NOT this receipt" in engine.config.fallback_reason


def test_every_packaged_example_loads_without_substituting_a_model():
    """If this fails, somebody is watching the wrong model's statistics."""
    from backend.app.receipt_loader import examples_shelf, receipt_to_graph_arrays

    failed = []
    for item in examples_shelf():
        if not item["packaged"]:
            continue
        try:
            receipt_to_graph_arrays(item["id"])
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{item['id']}: {exc!r}")
    assert not failed, (
        "these examples would silently sample a generic lattice instead of "
        "themselves: " + "; ".join(failed))
