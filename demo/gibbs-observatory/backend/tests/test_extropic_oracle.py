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

# The workloads and their measured placement budgets come from
# backend/app/verified_workloads.py, which is ALSO what the examples shelf reads
# to decide which entries carry the "verified" badge. One list, so the claim the
# application makes on screen and the claim this suite enforces cannot drift
# apart. Add a workload there and it is tested here automatically; it cannot be
# advertised as verified without being tested.
sys.path.insert(0, str(ROOT))
from backend.app.verified_workloads import VERIFIED_WORKLOADS  # noqa: E402

EXTROPIC = [
    (w.shelf_id, w.spec_stem, w.restarts, w.iters, w.expect_mediators)
    for w in VERIFIED_WORKLOADS
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


@pytest.mark.parametrize("record", VERIFIED_WORKLOADS,
                         ids=[w.shelf_id for w in VERIFIED_WORKLOADS])
def test_the_real_compile_path_selects_the_encoding_the_receipt_shipped(record):
    """Compile the way the product compiles, not through a fixed encoding.

    audit/findings/R26.md. Every other test in this file calls
    `preflight(load_model(spec=...))`, and `load_model` hardcodes "domain_wall".
    So the project's strongest regression signal proved these specs compile
    under ONE encoding, never exercised the encoding search, and did not compile
    the model the shelf actually ships. A change that broke one-hot selection
    would have left it green.

    `compile_spec` searches SLICE_ENCODINGS and selects among the feasible
    candidates by physical p-bit count. It is what produced the receipts, so
    this is the only test that puts encoding selection under test at all.
    """
    from tsu_compiler.passes.search import compile_spec
    from tsu_compiler.spec import load_spec
    from tsu_compiler.target import PROFILES

    spec = PROGRAMS / f"{record.spec_stem}.yaml"
    comp = compile_spec(load_spec(str(spec)), PROFILES["z1"], allow_assumed=True)

    assert comp.verdict == "COMPILED", (
        f"{record.shelf_id} does not compile through the real path: "
        f"{comp.verdict}. Note that EFFORT means the placement search ran out "
        f"of budget, which is not a hardware refusal.")

    selected = [c for c in comp.repset.candidates if c.state.name == "SELECTED"]
    assert len(selected) == 1, f"expected one selected candidate, got {selected}"
    assert selected[0].encoding == record.receipt_encoding, (
        f"{record.shelf_id}: the shipped receipt is "
        f"{record.receipt_encoding!r} but the compiler now selects "
        f"{selected[0].encoding!r}. Either the receipt is stale or selection "
        f"changed. Both need a person to look, not a test to be adjusted.")


def test_encoding_selection_is_exercised_at_all():
    """At least one workload must select something other than the default.

    Without this, every assertion above could pass against a compiler that had
    quietly stopped searching and always returned domain_wall.
    """
    encodings = {w.receipt_encoding for w in VERIFIED_WORKLOADS}
    assert len(encodings) > 1, (
        f"every verified workload ships the same encoding ({encodings}), so "
        f"the tests above cannot tell a working encoding search from one that "
        f"returns a constant")
