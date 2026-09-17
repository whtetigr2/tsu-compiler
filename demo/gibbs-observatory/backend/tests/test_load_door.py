"""The load door: point the Workbench at a file, or hand it pasted text.

Covers the two ways in. `/api/program/load` takes a name and text, which is
what a browser file picker and a paste both produce. `/api/program/load-path`
takes a path on disk, which is what "point it at the file I already wrote"
means on a desktop app.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GRID_YAML = ("name: demo\ngenerate:\n  kind: grid\n  width: 4\n  height: 4\n"
             "  variable_domain: {domain: binary}\nterms: []\n")


class TestLoadText:
    def test_yaml_loads_and_reports_itself(self):
        r = client.post("/api/program/load",
                        json={"filename": "m.yaml", "text": GRID_YAML})
        assert r.status_code == 200
        body = r.json()
        assert body["format"] == "yaml"
        assert body["yaml"] == GRID_YAML
        assert body["message"].startswith("loaded m.yaml")
        assert body["executed"] is False

    def test_json_is_converted_and_says_so(self):
        r = client.post("/api/program/load", json={
            "filename": "m.json",
            "text": '{"name": "demo", "generate": {"kind": "grid", "width": 4, "height": 4, "variable_domain": {"domain": "binary"}}, "terms": []}'})
        assert r.status_code == 200
        body = r.json()
        assert body["format"] == "json"
        assert "generate:" in body["yaml"]
        assert "json" in body["message"].lower()

    def test_python_is_read_as_data_and_says_it_was_not_run(self):
        r = client.post("/api/program/load", json={
            "filename": "m.py",
            "text": "SPEC = {'name': 'demo', 'generate': {'kind': 'grid', 'width': 2, 'height': 2, 'variable_domain': {'domain': 'binary'}}, 'terms': []}\n"})
        assert r.status_code == 200
        body = r.json()
        assert body["format"] == "python"
        assert body["executed"] is False
        assert "not run" in body["message"].lower()

    def test_a_file_that_is_not_a_program_is_refused_with_a_reason(self):
        r = client.post("/api/program/load",
                        json={"filename": "notes.yaml", "text": "title: hi\n"})
        assert r.status_code == 400
        detail = r.json()["detail"]
        text = detail if isinstance(detail, str) else detail.get("message", "")
        assert "generate" in text and "variables" in text

    def test_computed_python_spec_is_refused(self):
        r = client.post("/api/program/load", json={
            "filename": "m.py",
            "text": "def build():\n    return {}\nSPEC = build()\n"})
        assert r.status_code == 400

    def test_empty_text_is_refused(self):
        r = client.post("/api/program/load",
                        json={"filename": "m.yaml", "text": "  \n"})
        assert r.status_code == 400


class TestLoadPath:
    def test_reads_a_file_from_disk(self, tmp_path):
        p = tmp_path / "prog.yaml"
        p.write_text(GRID_YAML, encoding="utf-8")
        r = client.post("/api/program/load-path", json={"path": str(p)})
        assert r.status_code == 200
        body = r.json()
        assert body["yaml"] == GRID_YAML
        assert "prog.yaml" in body["message"]

    def test_reads_a_json_file_from_disk(self, tmp_path):
        p = tmp_path / "prog.json"
        p.write_text('{"name": "demo", "generate": {"kind": "grid", "width": 3, "height": 3, "variable_domain": {"domain": "binary"}}, "terms": []}', encoding="utf-8")
        r = client.post("/api/program/load-path", json={"path": str(p)})
        assert r.status_code == 200
        assert r.json()["format"] == "json"

    def test_missing_file_names_the_path(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        r = client.post("/api/program/load-path", json={"path": str(missing)})
        assert r.status_code == 400
        assert "nope.yaml" in str(r.json()["detail"])

    def test_a_directory_is_refused_as_a_directory(self, tmp_path):
        r = client.post("/api/program/load-path", json={"path": str(tmp_path)})
        assert r.status_code == 400
        assert "directory" in str(r.json()["detail"]).lower()

    def test_a_binary_file_is_refused_as_not_text(self, tmp_path):
        p = tmp_path / "blob.yaml"
        p.write_bytes(b"\x00\x01\x02\xff\xfe payload")
        r = client.post("/api/program/load-path", json={"path": str(p)})
        assert r.status_code == 400
        assert "text" in str(r.json()["detail"]).lower()

    def test_an_oversized_file_is_refused_with_its_size(self, tmp_path):
        p = tmp_path / "huge.yaml"
        p.write_text("# pad\n" * 400_000, encoding="utf-8")
        r = client.post("/api/program/load-path", json={"path": str(p)})
        assert r.status_code == 400
        assert "MB" in str(r.json()["detail"])
