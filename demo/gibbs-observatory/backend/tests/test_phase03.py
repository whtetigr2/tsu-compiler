"""Phase 3: Thermodynamic Program notepad API + state-space honesty helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.program_service import (
    NOTEPAD_RECEIPT_ID,
    ProgramServiceError,
    apply_program,
    compile_program,
    discover_tsu_root,
    ensure_tsu_importable,
    preflight_program,
    read_spec_yaml,
)
from backend.app.receipt_loader import DEFAULT_RECEIPTS_ROOT, load_receipt

TOY_YAML = """\
name: observatory_toy_test
description: pytest Thermodynamic Program
variables:
  a: {domain: binary}
  b: {domain: binary}
  c: {domain: categorical, k: 3}
terms:
  - {kind: product, a: {a: 1.0}, b: {b: 1.0}, weight: 2.0}
  - {kind: linear, form: {a: 1.0, b: 1.0}, weight: -0.5}
  - {kind: product, a: {"c=0": 1.0}, b: {a: 1.0}, weight: 1.5}
contract:
  validate:
    - {rule: forbid_both, vars: [a, b], message: "a and b must not both be occupied"}
    - {rule: forbid_value_with, var: c, value: 0, other: a,
       message: "c=0 forbidden while a is occupied"}
"""

BAD_YAML = """\
name: broken
variables:
  a: {domain: nonexistent}
terms: []
"""


@pytest.fixture(scope="module")
def tsu_ready():
    root = discover_tsu_root()
    if root is None:
        pytest.skip("tsu package not available")
    try:
        ensure_tsu_importable()
    except ProgramServiceError as exc:
        pytest.skip(str(exc))
    return root


@pytest.fixture
def client():
    return TestClient(app)


def test_tsu_discoverable(tsu_ready):
    assert (tsu_ready / "tsu" / "cli.py").is_file()


def test_read_spec_yaml_small():
    out = read_spec_yaml("small")
    assert out["ok"] is True
    assert "lattice_small" in (out["spec_yaml"] or "")
    assert "generate:" in out["spec_yaml"]


def test_load_receipt_includes_spec_yaml():
    payload = load_receipt("small")
    assert payload.get("spec_yaml")
    assert "lattice_small" in payload["spec_yaml"]


def test_preflight_toy(tsu_ready):
    rep = preflight_program(TOY_YAML)
    assert rep["ok"] is True
    assert rep["verdict"] in ("ok", "warn", "fail")
    assert isinstance(rep["gates"], list)
    assert len(rep["gates"]) >= 1
    assert "bipartite" in rep
    assert "mediators" in rep


def test_preflight_bad_yaml_refuses(tsu_ready):
    with pytest.raises(ProgramServiceError):
        preflight_program(BAD_YAML)


def test_apply_without_receipt_refuses():
    ghost = "definitely_missing_notepad_xyz"
    path = DEFAULT_RECEIPTS_ROOT / ghost
    if path.exists():
        shutil.rmtree(path)
    with pytest.raises(ProgramServiceError) as ei:
        apply_program(ghost)
    assert "Compile" in str(ei.value) or "refused" in str(ei.value).lower()


def test_compile_and_apply_toy(tsu_ready):
    # Clean previous notepad receipt
    out = DEFAULT_RECEIPTS_ROOT / NOTEPAD_RECEIPT_ID
    if out.exists():
        shutil.rmtree(out)
    result = compile_program(TOY_YAML, receipt_id=NOTEPAD_RECEIPT_ID)
    assert result["verdict"] == "COMPILED"
    assert result["ok"] is True
    assert result["receipt_id"] == NOTEPAD_RECEIPT_ID
    assert (DEFAULT_RECEIPTS_ROOT / NOTEPAD_RECEIPT_ID / "program.json").is_file()
    applied = apply_program(NOTEPAD_RECEIPT_ID)
    assert applied["ok"] is True
    assert applied["receipt_id"] == NOTEPAD_RECEIPT_ID
    payload = load_receipt(NOTEPAD_RECEIPT_ID)
    assert payload["verdict"] == "COMPILED"
    assert payload["sampling"]["thrml_ready"] is True


def test_api_program_status(client, tsu_ready):
    res = client.get("/api/program/status")
    assert res.status_code == 200
    body = res.json()
    assert body["importable"] is True
    assert body["tsu_root"]


def test_api_preflight(client, tsu_ready):
    res = client.post("/api/program/preflight", json={"yaml": TOY_YAML})
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "preflight"
    assert "gates" in body
    assert body["bipartite"] is not None


def test_api_preflight_empty_refuses(client):
    res = client.post("/api/program/preflight", json={"yaml": "   "})
    # pydantic min_length=1 may 422, or service 400
    assert res.status_code in (400, 422)


def test_api_apply_without_compile_refuses(client):
    res = client.post("/api/program/apply", json={"receipt_id": "no_such_receipt_ever"})
    assert res.status_code == 400
    detail = res.json()["detail"]
    msg = detail["message"] if isinstance(detail, dict) else str(detail)
    assert "refused" in msg.lower() or "Compile" in msg


def test_api_compile_apply_roundtrip(client, tsu_ready):
    out = DEFAULT_RECEIPTS_ROOT / NOTEPAD_RECEIPT_ID
    if out.exists():
        shutil.rmtree(out)
    res = client.post(
        "/api/program/compile",
        json={"yaml": TOY_YAML, "receipt_id": NOTEPAD_RECEIPT_ID, "target": "z1"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["receipt_id"] == NOTEPAD_RECEIPT_ID
    res2 = client.post("/api/program/apply", json={"receipt_id": NOTEPAD_RECEIPT_ID})
    assert res2.status_code == 200
    assert res2.json()["ok"] is True


def test_api_receipt_spec_endpoint(client):
    res = client.get("/api/receipts/small/spec")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert "generate:" in res.json()["spec_yaml"]


def test_health_includes_tsu(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["version"] == "0.5.0"
    assert "tsu" in res.json()
