"""A frozen binary must be checkable without opening a window.

This is the acceptance test for the whole slice: if `--selftest` exits zero
inside the packaged application on a machine with no Python and no Node, the
bundle genuinely carries the compiler and THRML rather than merely claiming to.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from desktop.selftest import run_selftest  # noqa: E402


def test_selftest_passes_in_a_working_checkout():
    code, lines = run_selftest()
    report = "\n".join(lines)
    assert code == 0, f"self-test failed in a working checkout:\n{report}"


def test_selftest_reports_every_check_by_name():
    _, lines = run_selftest()
    report = "\n".join(lines).lower()
    for probe in ("compiler", "thrml", "frontend", "preflight"):
        assert probe in report, f"{probe} check is not reported by name"


def test_selftest_reports_a_real_preflight_verdict():
    """The compiler must actually run, not merely import."""
    _, lines = run_selftest()
    report = "\n".join(lines).lower()
    assert "verdict" in report
    assert "ok" in report


def test_failed_checks_produce_a_nonzero_exit(monkeypatch):
    """A self-test that cannot fail proves nothing."""
    import desktop.selftest as st

    monkeypatch.setattr(
        st, "_check_frontend",
        lambda: ("frontend", False, "unavailable: forced failure"))
    code, lines = run_selftest()
    assert code != 0, "a forced failure must not exit zero"
    assert "forced failure" in "\n".join(lines)
