"""Phase 2: fabric tax, residuals, examples shelf, elev_band receipt."""

from backend.app.receipt_loader import (
    derive_fabric_tax,
    derive_residuals,
    examples_shelf,
    list_receipts,
    load_receipt,
    receipt_to_graph_arrays,
)
from backend.app.sampler_engine import SamplerConfig, SamplerEngine


def test_examples_shelf_has_small_elev_and_codon_stub():
    shelf = examples_shelf()
    by_id = {e["id"]: e for e in shelf}
    assert by_id["small"]["status"] == "ready"
    assert by_id["small"]["packaged"] is True
    assert by_id["elev_band"]["status"] == "ready"
    assert by_id["elev_band"]["packaged"] is True
    assert by_id["codon_opt"]["status"] == "stub"
    assert by_id["codon_opt"]["packaged"] is False
    assert "not packaged" in (by_id["codon_opt"]["message"] or "")


def test_list_receipts_excludes_codon_stub_dir():
    ids = {r["id"] for r in list_receipts()}
    assert "small" in ids
    assert "elev_band" in ids
    assert "codon_opt" not in ids  # README-only stub, no program/passes


def test_fabric_tax_small_64_pairs():
    payload = load_receipt("small")
    ft = payload["fabric_tax"]
    assert ft["derived"] is True
    assert ft["pair_count"] == 64
    assert len(ft["pairs"]) == 64
    sample = ft["pairs"][0]
    assert "mediator" in sample and "world_u" in sample and "world_v" in sample
    key = f"{min(sample['world_u'], sample['world_v'])}-{max(sample['world_u'], sample['world_v'])}"
    assert key in ft["by_edge_key"]


def test_fabric_tax_elev_band_unmediated():
    payload = load_receipt("elev_band")
    ft = payload["fabric_tax"]
    assert ft["pair_count"] == 0
    assert payload["spins"]["n_nodes"] == 64
    assert payload["sampling"]["thrml_ready"] is True
    assert payload["beta_fixed"] is False


def test_derive_fabric_tax_helper_empty():
    out = derive_fabric_tax([], [])
    assert out["pair_count"] == 0
    assert out["derived"] is True


def test_schedule_blocks_on_small():
    payload = load_receipt("small")
    sched = payload["schedule"]
    assert sched["n_colours"] == 2
    assert sched["blocks"][0]["size"] == 96
    assert sched["blocks"][1]["size"] == 96
    assert sched["kernel"] == "chromatic_block_gibbs"


def test_residuals_derived_not_invented():
    payload = load_receipt("small")
    res = payload["residuals"]
    assert res["has_receipt_residual_matrix"] is False
    ids = {b["id"] for b in res["bars"]}
    assert "edge_count_delta" in ids
    assert "headroom_degree" in ids
    assert all(b.get("derived") for b in res["bars"] if b["id"].startswith("headroom_"))
    assert any("unavailable" in u or "too large" in u for u in res["unavailable"])


def test_derive_residuals_empty_sources():
    out = derive_residuals(
        workload=None,
        physical=None,
        frontier=None,
        gates=None,
        verification=None,
        mediation=None,
    )
    assert out["bars"] == []
    assert out["has_receipt_residual_matrix"] is False


def test_elev_band_engine_samples():
    eng = SamplerEngine(
        SamplerConfig(
            receipt_id="elev_band",
            warmup=1,
            batch_size=2,
            seed=3,
            steps_per_sample=1,
        )
    )
    g = eng.graph_payload()
    assert g["preset"] == "receipt"
    assert g["n_nodes"] == 64
    assert g["sampling_fallback"] is False
    batch = eng.sample_batch()
    assert len(batch["last_state"]) == 64
    assert "active_block" in batch


def test_receipt_to_graph_elev_band():
    data = receipt_to_graph_arrays("elev_band")
    assert data["n_nodes"] == 64
    assert len(data["color0"]) + len(data["color1"]) == 64
