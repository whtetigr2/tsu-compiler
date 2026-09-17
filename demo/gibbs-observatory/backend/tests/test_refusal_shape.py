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
