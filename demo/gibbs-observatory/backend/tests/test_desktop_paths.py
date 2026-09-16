"""Where the application finds its own files, frozen and unfrozen.

A PyInstaller bundle has no repository around it: data files land in a
temporary directory named by `sys._MEIPASS`. Getting this wrong produces a
window that opens onto a blank page with no error anywhere, which is the most
confusing failure this application can have. So `frontend_dist` refuses loudly
instead of returning a path that does not exist.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from desktop.paths import (  # noqa: E402
    FrontendMissingError,
    frontend_dist,
    is_frozen,
    resource_root,
)


def test_not_frozen_in_development():
    assert is_frozen() is False


def test_resource_root_is_the_observatory_directory_from_source():
    assert resource_root() == ROOT
    assert (resource_root() / "backend").is_dir(), (
        "resource_root must point at the directory holding backend/ and "
        "frontend/, or every other path resolution is wrong")


def test_resource_root_follows_meipass_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert is_frozen() is True
    assert resource_root() == tmp_path


def test_frontend_dist_refuses_when_index_html_is_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    (tmp_path / "frontend" / "dist").mkdir(parents=True)
    with pytest.raises(FrontendMissingError) as exc:
        frontend_dist()
    assert "index.html" in str(exc.value), (
        "the refusal must name what is missing, not just fail")


def test_frontend_dist_returns_the_directory_holding_index_html(
        monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert frontend_dist() == dist
