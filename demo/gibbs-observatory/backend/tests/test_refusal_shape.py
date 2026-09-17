"""A refusal must carry enough to say WHICH kind of no it is.

The interface draws three different outcomes and they mean different things:

    a gate refused      the model does not fit. A real hardware limit.
    the search ran out  no embedding found in the budget. NOT a hardware limit,
                        and saying otherwise is a claim about Extropic's silicon
                        made on the basis of a timer (R27).
    the compiler faulted  our bug, which says nothing about the model.

Conflating the first two is the representation-versus-hardware confusion this
compiler exists to prevent, and it has been found in the codebase twice. These
tests pin the fields the panel needs to tell them apart, so a payload change
cannot quietly collapse the distinction back into "failed".
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.program_service import preflight_program  # noqa: E402

PROGRAMS = ROOT / "programs"


def _over_connected(n: int = 24) -> str:
    """A clique. Degree n-1 against a cap of 16, so a gate must refuse."""
    lines = ["name: too_dense", "description: >", "  Deliberately dense.",
             "variables:"]
    lines += [f"  v{i}: {{domain: binary}}" for i in range(n)]
    lines.append("terms:")
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(
                f"  - {{kind: product, a: {{v{i}: 1.0}}, b: {{v{j}: 1.0}}, "
                f"weight: 0.5}}")
    return "\n".join(lines)


def test_a_gate_refusal_names_the_gate_the_measurement_and_the_cap():
    res = preflight_program(_over_connected())
    failed = [g for g in res["gates"] if g["status"] == "fail"]
    assert failed, "a 24-clique must break the degree gate"
    g = failed[0]
    assert g["name"] == "max_degree"
    assert g["value"] == 23.0, "the measurement itself must be reported"
    assert g["limit"] == 16.0, "so must the cap it broke"
    assert g["note"], "a gate without a note cannot be read by a newcomer"
    assert res["remediations"], "a refusal with no remediation is a dead end"


def test_every_gate_declares_whether_its_cap_is_sourced_or_assumed():
    """A refusal resting on an assumption is a weaker claim than one resting on
    a published figure, and the reader is entitled to see which."""
    res = preflight_program(_over_connected())
    for g in res["gates"]:
        assert "assumed" in g, f"{g['name']} does not declare its provenance"
    by_name = {g["name"]: g for g in res["gates"]}
    assert by_name["max_abs_bias"]["assumed"] is True, (
        "the bias cap is this project's working value, not an Extropic figure, "
        "and target.py says so")
    assert by_name["max_degree"]["assumed"] is False, (
        "the degree cap is sourced; marking it assumed would understate it")


def test_an_effort_refusal_is_distinguishable_from_a_gate_refusal():
    """The distinction the whole panel exists to draw.

    seq_design_longer fails to place at the cheapest budget with EVERY gate
    passing. That is a search running out, not the hardware refusing.
    """
    res = preflight_program(
        (PROGRAMS / "seq_design_longer.yaml").read_text(encoding="utf-8"))
    attempts = res["placement_attempts"]
    assert len(attempts) > 1, "expected this one to need more than one tier"
    assert attempts[0]["placed"] is False
    assert attempts[-1]["placed"] is True
    assert not [g for g in res["gates"] if g["status"] == "fail"], (
        "no gate failed, so nothing here is a hardware limit")
    assert res["placement_escalated"] is True


def test_a_compiling_program_reports_what_it_costs():
    res = preflight_program(
        (PROGRAMS / "ecology_lotka_lite.yaml").read_text(encoding="utf-8"))
    assert res["verdict"] == "ok"
    assert res["placed"] is True
    for field in ("n_spins", "n_couplings", "max_degree", "mediators"):
        assert res.get(field) is not None, f"{field} missing from a success"


# ---------------------------------------------------------------------------
# The COMPILE path, which is the one the Program notepad actually calls.
#
# Everything above tests `preflight_program`, which has always carried `placed`
# and `place_error`. `compile_program` did not, and the panel keys its three
# branches off exactly those fields -- so on the real compile response the
# COMPILED branch and the effort branch were both unreachable and every
# successful compile rendered as "a verdict this panel does not recognise".
#
# The tests above could not have caught that: they exercised the path that
# already had the fields. These pin the other one.
# ---------------------------------------------------------------------------

from backend.app.program_service import compile_program  # noqa: E402

GRID_6 = (
    "name: two_metals_6x6\n"
    "generate:\n"
    "  kind: grid\n"
    "  width: 6\n"
    "  height: 6\n"
    "  variable_domain: {domain: binary}\n"
    "terms:\n"
    "- {kind: product_over_edges, a_value: 1, b_value: 1, weight: 0.4}\n"
)


def test_a_successful_compile_says_it_placed():
    """Without this the panel cannot tell a success from a verdict it does not
    recognise, because `placed` is the field it branches on."""
    res = compile_program(GRID_6, receipt_id="test_compiled")
    assert res["verdict"] == "COMPILED"
    assert res["ok"] is True
    assert res["placed"] is True, (
        "a COMPILED result placed by definition; the panel needs it said")


def test_a_successful_compile_reports_what_it_costs():
    res = compile_program(GRID_6, receipt_id="test_compiled")
    assert res["n_spins"] == 36
    assert res["n_couplings"] is not None
    assert res["max_degree"] is not None


def test_a_compile_carries_whether_the_hardware_question_was_reached():
    """`hardware_evaluated` is the compiler's own word for the R27 distinction.
    Dropping it on the way to the UI is how a timer becomes a claim about
    Extropic's silicon."""
    res = compile_program(GRID_6, receipt_id="test_compiled")
    assert res["hardware_evaluated"] is True


def test_a_gate_refusal_through_compile_is_not_placed_and_was_evaluated():
    res = compile_program(_over_connected(), receipt_id="test_gate_refused")
    assert res["ok"] is False
    assert res["placed"] is False
    assert res["hardware_evaluated"] is True, (
        "a gate refused, so the hardware question WAS reached")
    assert res["verdict"] != "EFFORT"


def test_the_compilers_own_reason_reaches_the_caller():
    """The compiler writes a sentence explaining each refusal. Throwing it away
    and rebuilding one in the UI is how the two refusals get conflated."""
    res = compile_program(_over_connected(), receipt_id="test_gate_refused")
    assert res.get("repset_reason"), "the compiler's own explanation is missing"


def test_a_gate_refusal_through_compile_still_shows_its_gate_table():
    """A refusal with no gate table is a dead end.

    The receipt is only loaded when the verdict is COMPILED, so on a refusal
    the gates came back empty and the reader was told "does not fit" with
    nothing saying which limit or by how much. The compiler carries the checks
    either way; they just use their own field names.
    """
    res = compile_program(_over_connected(), receipt_id="test_gate_refused")
    gates = res["gates"]
    assert gates, "a hardware refusal must still say which gate refused"
    failed = [g for g in gates if g.get("status") == "fail"]
    assert failed, "the failing gate must be marked failed, not left ambiguous"
    degree = [g for g in gates if (g.get("gate") or g.get("name")) == "degree"]
    assert degree, "the degree gate is the one a 24-clique breaks"
    assert degree[0]["measured"] == 23
    assert degree[0]["limit"] == 16


def test_gate_provenance_survives_the_compile_path_too():
    """SOURCED and ASSUMED render differently, and a refusal resting on an
    assumption is a weaker claim. Dropping the flag here would silently
    upgrade one."""
    res = compile_program(_over_connected(), receipt_id="test_gate_refused")
    by = {(g.get("gate") or g.get("name")): g for g in res["gates"]}
    assert by["degree"]["assumed"] is False
    assert any(g["assumed"] for g in res["gates"]), (
        "at least one Z1 cap is this project's assumption, and target.py says so")


def test_a_mediator_count_of_zero_is_reported_as_zero():
    """Zero mediators is a fact. Rendering it as "unavailable" tells the reader
    the number is unknown when it is known and is nothing."""
    res = compile_program(GRID_6, receipt_id="test_compiled")
    assert res["mediators"] == 0
    assert res["mediators"] is not None
