"""Extropic's own workloads, as the compiler's known-good oracle.

The Observatory's stated mission is to inspect compiled thermodynamic programs
"with Extropic public workloads as known-good drop-ins". Three shelf entries are
flagged `extropic: True` and shipped as `ready` / `packaged`:

    prog_codon_opt_tiny      Extropic/THRML codon class
    prog_seq_design_longer   codon_opt scale-up, 8 AA positions
    prog_ecology_lotka_lite  discrete competitive exclusion (Torx LV)

Nothing verified that any of them compile. The only test touching the shelf
checked METADATA -- that the legacy `codon_opt` id is a stub -- which is a test
that the thing is absent. And every test able to reach the compiler was skipping
itself, because the package rename tsu -> tsu_compiler had never been applied to
the Observatory, so the whole integration was unexercised.

If somebody else's published workload stops compiling, that is the strongest
signal available that the compiler has regressed -- stronger than any toy this
project writes for itself, because nobody here chose the shape of it. That is
what an oracle is for, and it is why these run against the real shelf YAML
rather than a fixture.

PLACEMENT EFFORT IS PART OF THE RESULT. seq_design_longer does not place inside
the Observatory's default budget (6 restarts / 40k iters, ~2s) and reports
`fail` with every gate passing -- the refusal is placement, not a gate. It
places cleanly at 24 restarts / 250k iters with 8 mediators. The budget each one
needs is pinned below, so a future change that makes placement harder shows up
as a test failure rather than as a demo that quietly says "fail".
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROGRAMS = ROOT / "programs"
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from tsu_compiler.preflight.check import preflight  # noqa: E402
from tsu_compiler.preflight.model import load_model  # noqa: E402

# (shelf id, yaml stem, restarts, iters, expect_mediators)
# The effort figures are measured, not guessed: the first two place inside the
# Observatory's own default, the third does not and its real budget is recorded.
EXTROPIC = [
    ("prog_codon_opt_tiny", "codon_opt_tiny", 6, 40_000, True),
    ("prog_ecology_lotka_lite", "ecology_lotka_lite", 6, 40_000, False),
    ("prog_seq_design_longer", "seq_design_longer", 24, 250_000, True),
]


@pytest.mark.parametrize("shelf_id,stem,restarts,iters,expect_mediators", EXTROPIC,
                         ids=[e[0] for e in EXTROPIC])
def test_extropic_workload_compiles(shelf_id, stem, restarts, iters,
                                    expect_mediators):
    spec = PROGRAMS / f"{stem}.yaml"
    assert spec.is_file(), (
        f"{shelf_id} is advertised on the examples shelf as packaged and ready, "
        f"but {spec} does not exist")

    rep = preflight(load_model(spec=str(spec)), restarts=restarts, iters=iters)

    failed = [g.name for g in rep.gates if g.status == "fail"]
    assert not failed, (
        f"{shelf_id} fails hardware gates {failed}. A published Extropic "
        f"workload no longer fitting Z1's published limits means either the "
        f"target profile or a compiler pass has moved.")

    assert rep.placed, (
        f"{shelf_id} passed every gate but could not be placed within "
        f"{restarts} restarts / {iters} iters: {rep.place_error}. "
        f"Remediations offered: {rep.remediations}")

    assert rep.verdict == "ok", f"{shelf_id} verdict is {rep.verdict!r}"

    if expect_mediators:
        assert rep.mediators > 0, (
            f"{shelf_id} is non-bipartite and previously needed mediators; "
            f"getting none back means routing changed")


def test_seq_design_longer_still_needs_more_than_the_default_budget():
    """Pin the reason the Observatory shows `fail` for a workload that compiles.

    This is not a defect in the compiler -- the refusal is honest and its
    remediation says exactly what to do. It is recorded so that if placement
    ever gets cheap enough for the default budget, somebody notices and raises
    the Observatory's default rather than leaving a demo that says "fail" about
    a program that compiles.
    """
    rep = preflight(load_model(spec=str(PROGRAMS / "seq_design_longer.yaml")),
                    restarts=6, iters=40_000)
    if rep.placed:
        pytest.fail(
            "seq_design_longer now places inside the Observatory's default "
            "budget. Good news -- raise the default in preflight_program() and "
            "delete this test.")
    assert not [g.name for g in rep.gates if g.status == "fail"], (
        "the default-budget refusal should be placement effort, not a gate")
    assert "effort" in (rep.place_error or "").lower()


def test_notepad_escalates_instead_of_reporting_a_false_fail():
    """The service must reach `ok` on seq_design_longer without being asked.

    The raw compiler call above still refuses it at the cheapest budget, and
    that refusal is correct. What must not happen again is the Observatory
    surfacing that refusal as the final answer: a user pasting a published
    Extropic workload into the notepad saw "fail" for a program that compiles.
    """
    sys.path.insert(0, str(ROOT))
    from backend.app.program_service import PLACEMENT_TIERS, preflight_program

    yaml_text = (PROGRAMS / "seq_design_longer.yaml").read_text(encoding="utf-8")
    res = preflight_program(yaml_text)

    assert res["verdict"] == "ok", (
        f"notepad still reports {res['verdict']!r} for a workload that places "
        f"at a higher tier; attempts: {res['placement_attempts']}")
    assert res["placed"] is True
    assert res["placement_escalated"] is True, (
        "expected this one to need more than the cheapest tier")
    assert res["placement_attempts"][0]["placed"] is False, (
        "if the cheapest tier now places it, the escalation is untested here")
    assert PLACEMENT_TIERS[-1][1] == 800_000, (
        "the effort ceiling is a deliberate limit, not an accident")


def test_cheap_programs_are_not_made_slow_by_the_ceiling():
    """Escalation must not tax programs that place immediately."""
    sys.path.insert(0, str(ROOT))
    from backend.app.program_service import preflight_program

    yaml_text = (PROGRAMS / "ecology_lotka_lite.yaml").read_text(encoding="utf-8")
    res = preflight_program(yaml_text)
    assert res["verdict"] == "ok"
    assert res["placement_escalated"] is False, (
        "a program that places on the first tier must not climb the ladder")
    assert len(res["placement_attempts"]) == 1
