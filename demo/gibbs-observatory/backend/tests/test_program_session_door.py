"""The HTTP surface for running a program: inspect, open, step, close.

The order matters and is the whole point of `inspect` existing. A reader points
at a file; the Workbench says what it is and whether running it is involved;
only then is there anything to consent to. Finding out by importing would mean
running the file to decide whether running it was acceptable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parents[1] / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "audit"))

from app.main import app  # noqa: E402

client = TestClient(app)

PROGRAM = str(ROOT / "programs" / "visibility_game.py")


class TestInspect:
    def test_it_names_a_program_as_a_program(self):
        r = client.post("/api/program/inspect", json={"path": PROGRAM})
        assert r.status_code == 200
        body = r.json()
        assert body["kind"] == "program"
        assert body["needs_consent"] is True

    def test_inspecting_does_not_execute(self):
        r = client.post("/api/program/inspect", json={"path": PROGRAM})
        assert r.json()["executed"] is False

    def test_it_says_why(self):
        r = client.post("/api/program/inspect", json={"path": PROGRAM})
        assert "build()" in r.json()["why"]

    def test_a_yaml_file_needs_no_consent(self, tmp_path):
        p = tmp_path / "m.yaml"
        p.write_text("name: demo\n", encoding="utf-8")
        r = client.post("/api/program/inspect", json={"path": str(p)})
        assert r.json()["needs_consent"] is False

    def test_a_missing_file_is_refused(self, tmp_path):
        r = client.post("/api/program/inspect",
                        json={"path": str(tmp_path / "nope.py")})
        assert r.status_code == 400


class TestOpenRequiresConsent:
    def test_opening_without_consent_is_forbidden(self):
        r = client.post("/api/program/open", json={"path": PROGRAM})
        assert r.status_code == 403, (
            "the request is well formed and was refused on purpose, which is "
            "403 rather than 400")

    def test_the_refusal_explains_itself(self):
        r = client.post("/api/program/open", json={"path": PROGRAM})
        detail = str(r.json()["detail"]).lower()
        assert "run" in detail or "execut" in detail

    def test_with_consent_it_opens(self):
        r = client.post("/api/program/open",
                        json={"path": PROGRAM, "consent": True})
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "visibility"
        assert body["n_spins"] == 48 * 32
        client.post("/api/program/close", json={"session": body["session"]})

    def test_it_reports_the_ports_and_their_modes(self):
        r = client.post("/api/program/open",
                        json={"path": PROGRAM, "consent": True})
        body = r.json()
        ports = {p["name"]: p for p in body["ports"]}
        assert ports["occupancy"]["mode"] == "bias"
        client.post("/api/program/close", json={"session": body["session"]})


class TestStepping:
    @pytest.fixture
    def session(self):
        r = client.post("/api/program/open",
                        json={"path": PROGRAM, "consent": True, "seed": 3})
        sid = r.json()["session"]
        yield sid
        client.post("/api/program/close", json={"session": sid})

    def test_a_step_returns_spins_and_a_decode(self, session):
        r = client.post("/api/program/step", json={"session": session})
        assert r.status_code == 200
        body = r.json()
        assert len(body["spins"]) == 48 * 32
        assert len(body["decoded"]) == 48

    def test_frames_advance(self, session):
        client.post("/api/program/step", json={"session": session})
        r = client.post("/api/program/step", json={"session": session})
        assert r.json()["frames"] == 2

    def test_the_trace_count_is_reported(self, session):
        """A recompile costs ~300ms a frame and is invisible in the output
        (R32), so it is a number on the wire rather than a hope."""
        for _ in range(12):
            r = client.post("/api/program/step", json={"session": session})
        assert r.json()["traces"] == 1

    def test_an_unknown_session_is_a_404(self):
        r = client.post("/api/program/step", json={"session": "nope"})
        assert r.status_code == 404

    def test_closing_twice_is_honest_about_it(self, session):
        assert client.post("/api/program/close",
                           json={"session": session}).json()["closed"] is True
        assert client.post("/api/program/close",
                           json={"session": session}).json()["closed"] is False
