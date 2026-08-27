import json
from pathlib import Path

import pytest

from tsu.spec import load_spec, TaskContract, WorkloadSpec
from tsu.ir import Binary, Product, LinearForm, Var, VarRef
from tsu.passes.analyse import MAXCUT_EXACT_LIMIT
from tsu.target import IDEAL, Z1
from tsu.passes.search import compile_spec
from tsu.receipt import write_receipt, load_receipt, replay
from tsu.viz import render


def test_receipt_contains_every_required_artifact(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    for name in ("spec.yaml", "spec.sha256", "target.json", "passes.json",
                 "gates.json", "metrics.json", "verification.json",
                 "program.json", "environment.json", "regime.json"):
        assert (Path(d) / name).exists(), f"missing {name}"


def test_target_json_preserves_provenance_including_assumed(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    t = json.loads((Path(d) / "target.json").read_text())
    assert t["max_abs_coupling"]["source"] == "assumed"
    assert t["degree"]["source"] == "F-14"


def test_receipt_replays_and_hashes_match(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    r = replay(d)
    assert r.matches is True, f"replay diverged: {r.diffs}"


def test_verification_never_writes_a_number_it_does_not_have(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    v = json.loads((Path(d) / "verification.json").read_text())
    for value in v.values():
        assert not (isinstance(value, str) and value == ""), \
            "an empty string is not a verdict; write the number or 'unavailable: reason'"


def test_allow_assumed_receipt_records_the_override_and_replays_clean(tmp_path):
    """C4: --allow-assumed was invisible in the receipt (gates.json was always
    `[]` on a COMPILED verdict), and `replay` always recompiled WITHOUT the
    flag, so an override receipt hit the downgraded gate on replay and reported
    DIVERGED for a compile that was not actually wrong. too_strong.yaml's |J|=40
    term lowers (via the occupancy-to-spin substitution) to |J|max=10.0,
    still well past Z1's assumed cap of 6.0 -- it fails z1 outright without the
    override and compiles clean with it, exactly the case the spec's replay
    guarantee exists for."""
    c = compile_spec(load_spec("specs/too_strong.yaml"), Z1, allow_assumed=True)
    assert c.verdict == "COMPILED"
    d = write_receipt(c, tmp_path / "r")

    passes = json.loads((Path(d) / "passes.json").read_text())
    assert passes["allow_assumed"] is True

    gates = json.loads((Path(d) / "gates.json").read_text())
    downgraded = [g for g in gates if g["downgraded"]]
    assert downgraded, "the receipt must record which gate(s) allow_assumed downgraded"
    assert downgraded[0]["gate"] == "coupling_cap"
    assert downgraded[0]["measured"] == pytest.approx(10.0)
    assert downgraded[0]["limit"] == pytest.approx(6.0)

    # every gate the compile evaluated is present, not only failures/downgrades
    assert {g["gate"] for g in gates} >= {"degree", "coupling_cap", "field_cap",
                                          "node_budget", "colouring"}

    r = replay(d)
    assert r.matches is True, f"an override receipt must replay clean: {r.diffs}"


def test_gates_json_records_passed_gates_too_on_an_ordinary_compile(tmp_path):
    """gates.json used to be `[]` on EVERY COMPILED receipt -- the spec asks for
    every gate, passed or failed, with its measured value and threshold."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verdict == "COMPILED"
    d = write_receipt(c, tmp_path / "r")
    gates = json.loads((Path(d) / "gates.json").read_text())
    assert gates, "gates.json must not be empty on a COMPILED receipt"
    assert all(g["passed"] for g in gates)
    assert all(not g["downgraded"] for g in gates)


def _odd_ring_spec(n) -> WorkloadSpec:
    variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
    terms = tuple(
        Product(LinearForm({VarRef(f"x{i}"): 1.0}),
               LinearForm({VarRef(f"x{(i + 1) % n}"): 1.0}), 1.0)
        for i in range(n))
    return WorkloadSpec(name="odd_ring", variables=variables, terms=terms,
                        contract=TaskContract(()), source_text="name: odd_ring\n")


def test_metrics_json_never_publishes_the_uncomputed_mediators_sentinel(tmp_path):
    """I5: -1 means 'not computed', not zero. A 25-node odd ring exceeds
    MAXCUT_EXACT_LIMIT (20) and is never bipartite, so its mediator count is
    never computed -- compiled against IDEAL (which has no bipartite
    requirement, so this reaches COMPILED and metrics.json is actually
    written), the receipt must record `mediators: null` with a reason, never
    the raw sentinel."""
    n = MAXCUT_EXACT_LIMIT + 5
    spec = _odd_ring_spec(n)
    c = compile_spec(spec, IDEAL)
    assert c.verdict == "COMPILED"
    assert c.repset.selected.report.mediators == -1, \
        "the reproduction depends on this being uncomputed -- sanity-check it"

    d = write_receipt(c, tmp_path / "r")
    metrics = json.loads((Path(d) / "metrics.json").read_text())
    assert metrics["mediators"] is None, \
        "the uncomputed sentinel must never be published as a spin count"
    assert metrics["mediators_note"], "the receipt must say WHY it is missing"

    html_out = tmp_path / "r.html"
    render(d, html_out)
    text = html_out.read_text(encoding="utf-8")
    assert "<th>mediators</th><td>None</td>" not in text, \
        "mediators must render as its note, not the bare word None"
    assert metrics["mediators_note"] in text


def test_task_validity_and_noise_floor_render_their_reason_not_the_word_none(tmp_path):
    """I8: every verification field except task_validity carried its own note
    ('energy_note', 'execution_note', 'cross_check_note'); task_validity fell
    back to a bare 'unavailable' literal with no reason, and
    execution_noise_floor fell back to a raw null with no note at all -- the
    visualizer rendered both as the literal string 'None'. A model too large to
    enumerate exactly (compiled against IDEAL so it still reaches COMPILED)
    exercises the real path."""
    from tsu.backends.thrml_backend import EXACT_LIMIT

    n = EXACT_LIMIT + 2
    variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
    terms = tuple(
        Product(LinearForm({VarRef(f"x{i}"): 1.0}),
               LinearForm({VarRef(f"x{i + 1}"): 1.0}), 1.0)
        for i in range(n - 1))
    spec = WorkloadSpec(name="oversized", variables=variables, terms=terms,
                        contract=TaskContract(()), source_text="name: oversized\n")

    c = compile_spec(spec, IDEAL)
    assert c.verdict == "COMPILED"
    d = write_receipt(c, tmp_path / "r")

    verification = json.loads((Path(d) / "verification.json").read_text())
    assert verification["task_validity"].startswith("unavailable:"), \
        "task_validity must carry a REASON, not the bare word 'unavailable'"
    assert verification["execution_noise_floor"].startswith("unavailable:")

    html_out = tmp_path / "r.html"
    render(d, html_out)
    text = html_out.read_text(encoding="utf-8")
    assert "<th>task_validity</th><td>None</td>" not in text
    assert "<th>execution_noise_floor</th><td>None</td>" not in text
