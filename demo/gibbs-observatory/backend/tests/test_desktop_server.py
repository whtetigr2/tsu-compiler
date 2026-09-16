"""The API process serves the built frontend itself.

The existing launcher runs Vite as a second process, which means the end user
needs Node installed. Mounting the built output onto the same FastAPI app
removes that requirement entirely -- and it must do so without shadowing any
API route, which is what the ordering test below pins.
"""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from desktop.paths import FrontendMissingError  # noqa: E402
from desktop.server import build_desktop_app, mount_frontend  # noqa: E402


@pytest.fixture
def dist(tmp_path):
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>WB</title>",
                                  encoding="utf-8")
    (d / "app.js").write_text("console.log(1)", encoding="utf-8")
    return d


def test_index_is_served_at_root(dist):
    app = mount_frontend(FastAPI(), dist)
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "<title>WB</title>" in r.text


def test_assets_are_served(dist):
    app = mount_frontend(FastAPI(), dist)
    r = TestClient(app).get("/app.js")
    assert r.status_code == 200
    assert "console.log" in r.text


def test_api_routes_are_not_shadowed_by_the_static_mount(dist):
    """The mount is at '/', so ordering decides whether the API survives."""
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"service": "gibbs-observatory"}

    mount_frontend(app, dist)
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200, "the static mount swallowed an API route"
    assert r.json()["service"] == "gibbs-observatory"


def test_missing_build_is_refused_not_served_blank(tmp_path):
    with pytest.raises(FrontendMissingError):
        mount_frontend(FastAPI(), tmp_path / "does-not-exist")


def test_build_desktop_app_keeps_the_real_health_endpoint():
    """The packaged app is the real Observatory API, not a reduced copy."""
    try:
        app = build_desktop_app()
    except FrontendMissingError:
        pytest.skip("frontend not built in this checkout")
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json()["service"] == "gibbs-observatory"
