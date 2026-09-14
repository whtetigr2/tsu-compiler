"""Tests for receipt loader + receipt-backed sampler."""

from backend.app.receipt_loader import (
    list_receipts,
    load_receipt,
    program_drives_thrml,
    receipt_to_graph_arrays,
)
from backend.app.sampler_engine import SamplerConfig, SamplerEngine


def test_list_receipts_includes_small():
    items = list_receipts()
    ids = {r["id"] for r in items}
    assert "small" in ids
    small = next(r for r in items if r["id"] == "small")
    assert small["verdict"] == "COMPILED"
    assert small["encoding"] == "domain_wall"


def test_load_small_normalized():
    payload = load_receipt("small")
    assert payload["verdict"] == "COMPILED"
    assert payload["encoding"] == "domain_wall"
    assert payload["beta"] == 1.0
    assert payload["beta_fixed"] is True
    assert payload["kernel"] == "chromatic_block_gibbs"
    assert len(payload["gates"]) >= 4
    assert all("gate" in g for g in payload["gates"])
    assert payload["spins"]["n_nodes"] == 192
    assert payload["spins"]["world"] == 128
    assert payload["spins"]["mediators"] == 64
    assert payload["edges"]["count"] == 576
    assert payload["edges"]["weights"] is not None
    assert len(payload["edges"]["weights"]) == 576
    assert payload["sampling"]["thrml_ready"] is True
    assert payload["sampling"]["banner"] is None
    assert payload["label"] == "JAX/THRML simulation — not Extropic silicon"


def test_program_drives_thrml():
    payload = load_receipt("small")
    # Reconstruct minimal program dict check via arrays
    data = receipt_to_graph_arrays("small")
    assert data["n_nodes"] == 192
    assert len(data["edges"]) == 576
    assert len(data["weights"]) == 576
    assert len(data["biases"]) == 192
    assert len(data["color0"]) + len(data["color1"]) == 192


def test_missing_receipt_raises():
    try:
        load_receipt("__no_such_receipt__")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_receipt_backed_engine_samples():
    eng = SamplerEngine(
        SamplerConfig(
            receipt_id="small",
            warmup=2,
            batch_size=2,
            seed=7,
            steps_per_sample=1,
        )
    )
    g = eng.graph_payload()
    assert g["preset"] == "receipt"
    assert g["n_nodes"] == 192
    assert g["sampling_fallback"] is False
    assert g["beta_fixed"] is True
    assert len(g["mediator_idx"]) == 64
    batch = eng.sample_batch()
    assert batch["type"] == "batch"
    assert len(batch["last_state"]) == 192
    assert len(batch["energies"]) == 2


def test_program_drives_helper_rejects_empty():
    assert program_drives_thrml(None) is False
    assert program_drives_thrml({}) is False
    assert program_drives_thrml({"nodes": [1], "edges": []}) is False
