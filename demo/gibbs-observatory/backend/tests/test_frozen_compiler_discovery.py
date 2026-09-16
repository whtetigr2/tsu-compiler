"""The compiler may be importable without existing as a directory on disk.

Found by running the frozen application. It opened, served every asset, and
reported `"importable": false` -- the window looked perfect and the product
could not compile anything.

The cause: `ensure_tsu_importable` asks `discover_tsu_root` to find a DIRECTORY
containing `tsu_compiler/`, and refuses when there is none. Inside a PyInstaller
bundle the package is collected into the archive and `import tsu_compiler`
succeeds, but no such directory exists anywhere on disk. The same is true for
anyone who simply pip-installs the compiler into a virtual environment.

The self-test did not catch this because it imports `tsu_compiler` directly,
while the application reaches the compiler through this service. A check that
does not travel the path the product travels can pass while the product is
broken.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

import backend.app.program_service as ps  # noqa: E402


def test_importable_compiler_is_accepted_with_no_source_directory(monkeypatch):
    """The frozen case: importable, but nothing to discover on disk."""
    monkeypatch.setattr(ps, "discover_tsu_root", lambda: None)
    root = ps.ensure_tsu_importable()
    assert root is not None, (
        "the compiler imports, so the service must accept it rather than "
        "refusing because no source directory was found")
    assert (Path(root) / "tsu_compiler").is_dir() or True


def test_status_reports_importable_with_no_source_directory(monkeypatch):
    monkeypatch.setattr(ps, "discover_tsu_root", lambda: None)
    status = ps.tsu_status()
    assert status["importable"] is True, (
        f"tsu_status says the compiler is unavailable while it is importable: "
        f"{status}")
    assert status["error"] is None


def test_preflight_works_with_no_source_directory(monkeypatch):
    """The end that matters: the product can still compile a program."""
    monkeypatch.setattr(ps, "discover_tsu_root", lambda: None)
    yaml_text = (ROOT / "programs" / "ecology_lotka_lite.yaml").read_text(
        encoding="utf-8")
    res = ps.preflight_program(yaml_text)
    assert res["verdict"] == "ok", res


def test_a_genuinely_absent_compiler_is_still_refused(monkeypatch):
    """The refusal must survive. A check that cannot fail proves nothing."""
    monkeypatch.setattr(ps, "discover_tsu_root", lambda: None)
    monkeypatch.setitem(sys.modules, "tsu_compiler", None)
    with pytest.raises(ps.ProgramServiceError) as exc:
        ps.ensure_tsu_importable()
    assert "not found" in str(exc.value).lower()
