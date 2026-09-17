"""Gates as you type, without paying for placement.

Compiling has two costs and they are nothing alike. Checking the gates is
milliseconds: encode, lower, look at the graph, compare against the caps.
Finding an embedding is seconds to minutes, and it is the reason pressing
Compile feels like a commitment.

So the gates run live and placement runs when asked. The whole value of that
split is that a reader writing a clique sees "degree 23 against a cap of 16"
while they are still typing, instead of after a placement search that was
never going to matter.

The honesty constraint is the interesting part. These gates measure the model
AS WRITTEN. Routing can insert mediators and split high-degree nodes, which
changes the degree the hardware actually sees. A preview that quietly presents
itself as the final answer would be the representation-versus-hardware
confusion again, in a new place, so every preview says which it is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.program_service import gate_preview  # noqa: E402

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


def _clique(n: int = 24) -> str:
    lines = ["name: too_dense", "variables:"]
    lines += [f"  v{i}: {{domain: binary}}" for i in range(n)]
    lines.append("terms:")
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(
                f"  - {{kind: product, a: {{v{i}: 1.0}}, b: {{v{j}: 1.0}}, "
                f"weight: 0.5}}")
    return "\n".join(lines)


class TestItAnswers:
    def test_a_fitting_model_passes_every_gate(self):
        res = gate_preview(GRID_6)
        assert res["ok"] is True
        assert res["gates"], "a preview with no gates has answered nothing"
        assert not [g for g in res["gates"] if g["status"] == "fail"]

    def test_it_reports_the_shape_it_measured(self):
        res = gate_preview(GRID_6)
        assert res["n_spins"] == 36
        assert res["max_degree"] == 4
        assert res["n_couplings"] == 60

    def test_a_clique_fails_the_degree_gate_with_its_numbers(self):
        res = gate_preview(_clique())
        assert res["ok"] is False
        failed = [g for g in res["gates"] if g["status"] == "fail"]
        assert failed
        degree = [g for g in failed if g["gate"] == "degree"][0]
        assert degree["measured"] == 23
        assert degree["limit"] == 16

    def test_provenance_survives_into_the_preview(self):
        """A cap this project assumed and one Extropic published are different
        strengths of claim, and the preview draws them differently."""
        res = gate_preview(GRID_6)
        by = {g["gate"]: g for g in res["gates"]}
        assert by["degree"]["assumed"] is False
        assert any(g["assumed"] for g in res["gates"])


class TestItIsHonestAboutWhatItSkipped:
    def test_it_says_placement_was_not_run(self):
        res = gate_preview(GRID_6)
        assert res["placement_checked"] is False

    def test_it_says_the_measurement_is_pre_routing(self):
        """Routing inserts mediators and splits high-degree nodes. A preview
        that let itself be read as the final degree would be claiming the
        hardware question was settled when it was not."""
        res = gate_preview(GRID_6)
        assert res["routed"] is False
        assert res["note"], "a preview without its caveat is a claim"
        note = res["note"].lower()
        assert "rout" in note or "mediator" in note

    def test_it_never_claims_a_verdict(self):
        """COMPILED means gates passed AND an embedding was found. A preview
        establishes only the first half, so it must not use the word."""
        res = gate_preview(GRID_6)
        assert res.get("verdict") != "COMPILED"


class TestItRefusesCleanly:
    def test_a_spec_the_compiler_cannot_read_is_an_error_not_a_pass(self):
        from backend.app.program_service import ProgramServiceError
        with pytest.raises(ProgramServiceError):
            gate_preview("name: x\nnot_a_program: true\n")

    def test_empty_text_is_refused(self):
        from backend.app.program_service import ProgramServiceError
        with pytest.raises(ProgramServiceError):
            gate_preview("   \n")


class TestItIsActuallyCheaper:
    """Where the split pays, measured rather than assumed.

    The first version of this test asserted the clique was the case that
    mattered, on the reasoning that placement is expensive and pointless once
    a gate has refused. Measurement said otherwise, and the reasoning was
    backwards. Warmed up, on this machine:

        6x6 grid              1 ms  vs  1401 ms    1069x
        24-clique            43 ms  vs    67 ms       2x
        seq_design_longer   152 ms  vs  4078 ms      27x
        ecology_lotka_lite    2 ms  vs  7198 ms    3085x

    The clique is where the split buys LEAST, because `compile_spec` already
    short-circuits placement when a gate refuses -- it was never paying that
    cost. The win is on models that PASS their gates, where the full compile
    goes on to spend seconds finding an embedding nobody asked for yet. That
    is the normal state of a program while someone is editing it, which is
    what makes a live tier worth having.
    """

    def test_the_preview_is_fast_enough_to_run_while_someone_types(self):
        import time
        gate_preview(GRID_6)  # warm the imports; the first call pays for them
        t0 = time.time()
        gate_preview(GRID_6)
        preview = time.time() - t0
        assert preview < 0.25, (
            f"the preview took {preview * 1000:.0f}ms; above a keystroke's "
            f"worth of budget it cannot run live and the split buys nothing")

    def test_it_is_cheaper_than_a_full_compile_on_a_model_that_passes(self):
        """The case the split exists for. A passing model is where a full
        compile goes on to spend seconds on placement, and where a reader
        editing one wants the gate answer now."""
        import time
        gate_preview(GRID_6)
        t0 = time.time()
        gate_preview(GRID_6)
        preview = time.time() - t0
        t0 = time.time()
        from backend.app.program_service import compile_program
        compile_program(GRID_6, receipt_id="test_preview_timing")
        full = time.time() - t0
        assert full > preview * 10, (
            f"preview {preview * 1000:.0f}ms vs compile {full * 1000:.0f}ms; "
            f"without a real gap the second tier is not worth its complexity")
