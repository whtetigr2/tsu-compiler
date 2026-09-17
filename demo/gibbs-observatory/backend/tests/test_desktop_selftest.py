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


def test_the_load_door_check_fails_when_the_door_is_broken(monkeypatch):
    """A check that cannot fail proves nothing.

    The load door check was added after a build shipped without it. Force the
    door to raise and confirm the check reports a failure rather than passing
    on the strength of everything else working.
    """
    from desktop import selftest
    import backend.app.spec_formats as spec_formats

    def boom(*a, **k):
        raise RuntimeError("frozen bundle cannot read specs")

    monkeypatch.setattr(spec_formats, "load_text", boom)
    name, ok, detail = selftest._check_load_door()
    assert name == "load door"
    assert ok is False
    assert "frozen bundle cannot read specs" in detail


def test_the_load_door_check_fails_if_a_preview_claims_placement_ran():
    """The preview must never report that placement happened. If it ever does,
    a green gate strip becomes readable as COMPILED, which is the
    representation-versus-hardware confusion in a progress indicator."""
    from desktop import selftest
    import backend.app.program_service as program_service

    real = program_service.gate_preview
    try:
        program_service.gate_preview = lambda *a, **k: {
            "gates": [{"gate": "degree"}], "placement_checked": True}
        name, ok, detail = selftest._check_load_door()
        assert ok is False
        assert "placement" in detail.lower()
    finally:
        program_service.gate_preview = real
