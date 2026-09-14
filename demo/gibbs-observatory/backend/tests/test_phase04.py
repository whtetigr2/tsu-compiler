"""Phase 4: snapshot export metadata + claim hygiene."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.snapshot import (
    APP_VERSION,
    STANDING_PROHIBITIONS,
    build_snapshot_slice,
    claim_hygiene_payload,
)
from backend.app.receipt_loader import load_receipt


def test_app_version_health():
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["version"] == APP_VERSION == "0.5.0"
    assert "not Extropic silicon" in r.json()["label"]


def test_claim_hygiene_standalone():
    out = claim_hygiene_payload(None)
    assert out["version"] == "0.5.0"
    assert len(out["standing_prohibitions"]) >= 7
    ids = {p["id"] for p in out["standing_prohibitions"]}
    assert "no_silicon" in ids and "no_energy" in ids and "thermalizers" in ids
    assert "no energy claims" in " ".join(out["claim_badges"]).lower() or any(
        "energy" in b.lower() for b in out["claim_badges"]
    )


def test_claim_hygiene_with_small_receipt():
    receipt = load_receipt("small")
    out = claim_hygiene_payload(receipt)
    assert out["live"]
    assert any(x["id"] == "beta" for x in out["live"])
    assert any(x["id"] == "silicon" for x in out["live"])
    client = TestClient(app)
    r = client.get("/api/claim-hygiene", params={"receipt_id": "small"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"].startswith("JAX/THRML")
    assert len(body["standing_prohibitions"]) == len(STANDING_PROHIBITIONS)


def test_claim_hygiene_missing_receipt():
    client = TestClient(app)
    r = client.get("/api/claim-hygiene", params={"receipt_id": "does-not-exist-xyz"})
    assert r.status_code == 404


def test_snapshot_slice_no_sim_dumps():
    receipt = load_receipt("small")
    slice_ = build_snapshot_slice(
        receipt,
        client={"step": 12, "view": "overview", "png_filename": "x.png"},
    )
    blob = str(slice_).lower()
    assert "program.json" not in blob or slice_.get("schema")
    assert "states" not in slice_
    assert "history" not in slice_
    assert "sample" not in slice_ or "no sim sample" in " ".join(slice_["notes"]).lower()
    assert slice_["verdict"] is not None
    assert slice_["spins"]["n_nodes"] == receipt["spins"]["n_nodes"]
    assert slice_["beta_fixed"] is True
    assert slice_["gates"]
    assert slice_["claim_badges"]
    assert slice_["png_filename"] == "x.png"
    assert slice_["version"] == "0.5.0"


def test_api_snapshot_with_receipt():
    client = TestClient(app)
    r = client.post(
        "/api/snapshot",
        json={
            "receipt_id": "small",
            "step": 3,
            "active_block": 1,
            "view": "overview",
            "png_filename": "demo.png",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    snap = data["snapshot"]
    assert snap["receipt_id"] == "small"
    assert snap["step"] == 3
    assert snap["schema"] == "gibbs-observatory.snapshot.v1"
    assert "no silicon" in " ".join(snap["claim_badges"]).lower() or any(
        "silicon" in b.lower() for b in snap["claim_badges"]
    )
    # Must not embed huge program / samples
    assert "nodes" not in snap
    assert "edges" not in snap or isinstance(snap.get("edges"), (type(None), dict))


def test_api_snapshot_empty_body():
    client = TestClient(app)
    r = client.post("/api/snapshot", json={})
    assert r.status_code == 200
    assert r.json()["snapshot"]["version"] == "0.5.0"


def test_receipt_badges_include_energy_and_ess():
    receipt = load_receipt("small")
    badges = " | ".join(receipt["claim_badges"]).lower()
    assert "no energy" in badges or "energy" in badges
    assert "ess" in badges
    assert "silicon" in badges
