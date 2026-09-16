"""The edge-list door: models whose biases are data, not a declaration.

`tsu_compiler.preflight.model.load_model` has always taken EITHER a workload
spec or an edge list. The Observatory only ever called the spec side, which
quietly excluded an entire class of model -- anything whose biases come from
input rather than from a rule.

The case that exposed it: the playable level's visibility lattice is 11,200
spins on a 200x56 grid, and every bias is set by whether that cell is occupied
in front of the player. It changes every frame. Expressed as notepad YAML that
is 11,200 `linear` terms regenerated per frame -- a data dump wearing a
program's clothes, and nothing anyone would call a Thermodynamic Program.

Through this door it is what it actually is: a grid, two distinct couplings,
and a bias vector. It preflights in well under a second.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.main import app  # noqa: E402

client = TestClient(app)


def _grid_edges(w: int, h: int, j_row: float, j_col: float,
                beta: float = 2.2) -> str:
    """A bipartite grid with two distinct couplings and alternating biases.

    Same shape as the level's visibility lattice, small enough to be a test.
    """
    idx = lambda x, y: y * w + x
    edges = []
    for y in range(h):
        for x in range(w):
            if x + 1 < w:
                edges.append([idx(x, y), idx(x + 1, y), j_row])
            if y + 1 < h:
                edges.append([idx(x, y), idx(x, y + 1), j_col])
    biases = [3.0 if (i % 7) else -4.0 for i in range(w * h)]
    return json.dumps({"nodes": w * h, "edges": edges,
                       "biases": biases, "beta": beta})


def test_edge_list_preflights():
    r = client.post("/api/program/preflight-edges",
                    json={"edges_json": _grid_edges(40, 24, 1.6, 2.4)})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["source"] == "edges", "the payload must say which door it came in"
    assert out["verdict"] == "ok", out
    assert out["n_spins"] == 40 * 24
    assert out["bipartite"] is True
    assert out["max_degree"] == 4
    assert out["placed"] is True


def test_edge_list_and_spec_return_the_same_shape():
    """Both doors share `_preflight_model`, so their payloads must not drift."""
    edges = client.post("/api/program/preflight-edges",
                        json={"edges_json": _grid_edges(8, 8, 1.2, 1.2)}).json()
    spec_yaml = (ROOT / "programs" / "ecology_lotka_lite.yaml").read_text(
        encoding="utf-8")
    spec = client.post("/api/program/preflight", json={"yaml": spec_yaml}).json()

    ignore = {"source"}
    assert set(edges) - ignore == set(spec) - ignore, (
        "the two preflight doors returned different fields; they are supposed "
        "to be the same function with different front ends")
    assert spec["source"] == "spec"


def test_malformed_edge_list_is_refused_not_guessed():
    """The compiler refuses to pad or truncate; that refusal must survive."""
    bad = json.dumps({"nodes": 4, "edges": [[0, 1, 1.0]],
                      "biases": [0.0, 0.0], "beta": 1.0})  # 2 biases, 4 nodes
    r = client.post("/api/program/preflight-edges", json={"edges_json": bad})
    assert r.status_code >= 400, (
        "a bias vector that does not match the node count must be refused, "
        "not silently padded into a different model")
    assert "refus" in r.text.lower() or "bias" in r.text.lower()


def test_not_json_is_refused_cleanly():
    r = client.post("/api/program/preflight-edges",
                    json={"edges_json": "this is not json"})
    assert r.status_code >= 400
    assert "500" not in str(r.status_code), "should be a refusal, not a crash"
