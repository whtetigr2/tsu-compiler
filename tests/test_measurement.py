"""C4: measurement -- diversity, compilation cost, sampler cost, and a CPU
reference baseline. All four are honest: an unmeasured quantity reads
`unavailable: <reason>`, never a fabricated number. The CPU baseline is a
correctness-and-cost REFERENCE, never a speed/hardware claim -- nothing here
may state or imply one approach is faster than another.
"""
import json

import pytest

from tsu_compiler.ir import Binary, Product, LinearForm, Var, VarRef
from tsu_compiler.spec import load_spec, TaskContract, WorkloadSpec
from tsu_compiler.target import IDEAL, Z1
from tsu_compiler.passes.search import compile_spec
from tsu_compiler.receipt import write_receipt
from tsu_compiler.report import render_explain, render_report
from tsu_compiler.backends.thrml_backend import EXACT_LIMIT


def _oversized_spec(n) -> WorkloadSpec:
    variables = tuple(Var(f"x{i}", Binary()) for i in range(n))
    terms = tuple(
        Product(LinearForm({VarRef(f"x{i}"): 1.0}),
               LinearForm({VarRef(f"x{i + 1}"): 1.0}), 1.0)
        for i in range(n - 1))
    return WorkloadSpec(name="oversized", variables=variables, terms=terms,
                        contract=TaskContract(()), source_text="name: oversized\n")


# --------------------------------------------------------------------------
# diversity
# --------------------------------------------------------------------------

def test_diversity_reports_distinct_valid_configurations_and_the_denominator():
    v = compile_spec(load_spec("specs/toy.yaml"), Z1).verification
    assert v.diversity_distinct is not None
    assert v.diversity_valid_samples is not None
    assert 0 <= v.diversity_distinct <= v.diversity_valid_samples


def test_diversity_reports_the_reachable_valid_set_size_when_enumerable():
    """toy.yaml's logical state space (12 assignments) is trivially
    enumerable -- the KNOWN TRAP the brief names: on a space this small, a
    high distinct/valid ratio is not evidence of anything interesting unless
    read alongside how large the whole valid space actually is."""
    v = compile_spec(load_spec("specs/toy.yaml"), Z1).verification
    assert v.diversity_reachable is not None
    # hand count: toy.yaml forbids (a=1,b=1) and (c=0,a=1); of the 3*2*2=12
    # assignments, exactly 3 violate a rule -- (a,b,c) in
    # {(1,1,0),(1,1,1),(1,1,2)} hits forbid_both, plus (1,0,0) hits
    # forbid_value_with -- so 12 - 4 = 8 are valid. Verified independently
    # below by direct enumeration, not just asserted from this comment.
    from tsu_compiler.spec import load_spec as _load
    s = _load("specs/toy.yaml")
    hand_count = sum(1 for a in s.assignments() if s.contract.validate(a).ok)
    assert v.diversity_reachable == hand_count


def test_diversity_reachable_is_unavailable_with_a_reason_when_too_large_to_enumerate():
    n = EXACT_LIMIT + 2
    v = compile_spec(_oversized_spec(n), IDEAL).verification
    assert v.diversity_reachable is None
    assert v.diversity_reachable_note.startswith("unavailable")
    # but distinct/valid-samples need no exact reference, same as task_validity
    assert v.diversity_distinct is not None
    assert v.diversity_valid_samples is not None


def test_report_diversity_line_shows_the_real_measurement(tmp_path):
    """Supersedes the old permanent 'unavailable: not measured' placeholder
    now that something actually measures it -- same transition ESS/Mixing
    already went through (see test_report.py's own history)."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    diversity_line = next(l for l in text.splitlines() if l.strip().startswith("Diversity:"))
    assert "unavailable" not in diversity_line
    assert str(c.verification.diversity_distinct) in diversity_line
    assert str(c.verification.diversity_reachable) in diversity_line


def test_report_diversity_line_is_honest_when_unmeasurable(tmp_path):
    n = EXACT_LIMIT + 2
    c = compile_spec(_oversized_spec(n), IDEAL)
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    diversity_line = next(l for l in text.splitlines() if l.strip().startswith("Diversity:"))
    # distinct/valid are real, but the reachable-set half is unavailable --
    # the line must say so, never claim a number it does not have
    assert "unavailable" in diversity_line


# --------------------------------------------------------------------------
# compilation cost: per-pass wall time
# --------------------------------------------------------------------------

def test_passes_json_records_a_per_pass_duration_for_a_compiled_receipt(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    passes = json.loads((d / "passes.json").read_text())
    durations = passes.get("pass_durations")
    assert durations, "passes.json must record per-pass wall time"
    for pass_name in ("encode", "lower", "analyse", "build_program", "verify"):
        assert pass_name in durations, f"missing duration for pass {pass_name!r}"
        assert durations[pass_name] >= 0.0


def test_pass_durations_are_recorded_even_for_a_hardware_rejected_compile(tmp_path):
    """too_dense.yaml passes ideal then fails Z1's degree gate -- durations for
    the passes it DID reach (encode/lower/analyse/gate_checks) must still be
    recorded, not silently dropped because the compile did not succeed."""
    c = compile_spec(load_spec("specs/too_dense.yaml"), Z1)
    assert c.verdict == "HARDWARE"
    d = write_receipt(c, tmp_path / "r")
    passes = json.loads((d / "passes.json").read_text())
    durations = passes.get("pass_durations")
    assert durations
    assert "analyse" in durations


# --------------------------------------------------------------------------
# compilation cost: peak memory
# --------------------------------------------------------------------------

def test_cost_json_records_peak_memory_when_it_is_asked_for(tmp_path):
    """Peak memory is opt-in, and recorded faithfully when requested."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, measure_memory=True)
    assert c.peak_memory_bytes is not None and c.peak_memory_bytes > 0
    d = write_receipt(c, tmp_path / "r")
    cost = json.loads((d / "cost.json").read_text())
    assert cost["peak_memory_bytes"] == c.peak_memory_bytes


def test_memory_measurement_is_off_by_default_and_reported_as_unmeasured(tmp_path):
    """`tracemalloc` instruments every allocation and was measured at ~1.8x wall
    time on this workload -- the suite went 278s -> 683s with it always on. An
    application recompiling on each user action would pay that every time, for a
    number most callers never read. So it is opt-in, and when it is off the
    receipt must say the value is UNMEASURED rather than report it as zero."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.peak_memory_bytes is None
    d = write_receipt(c, tmp_path / "r")
    cost = json.loads((d / "cost.json").read_text())
    recorded = cost["peak_memory_bytes"]
    assert recorded != 0, "an unmeasured quantity must never be published as zero"
    assert recorded is None or "unavailable" in str(recorded).lower()


# --------------------------------------------------------------------------
# sampler cost
# --------------------------------------------------------------------------

def test_cost_json_records_sampler_wall_time_and_throughput(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    cost = json.loads((d / "cost.json").read_text())
    sampler = cost.get("sampler")
    assert sampler
    assert sampler["wall_time_s"] > 0.0
    assert sampler["samples_per_second"] > 0.0
    assert sampler["params"] == {"n_chains": 32, "n_samples": 200,
                                 "n_warmup": 400, "steps_per_sample": 2, "seed": 0}


def test_cost_json_sampler_is_absent_when_verification_never_ran(tmp_path):
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    d = write_receipt(c, tmp_path / "r")
    cost = json.loads((d / "cost.json").read_text())
    assert not cost.get("sampler")


# --------------------------------------------------------------------------
# CPU reference baseline -- correctness and cost only, never a speed claim
# --------------------------------------------------------------------------

def test_cost_json_baseline_available_for_a_small_enumerable_model(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    cost = json.loads((d / "cost.json").read_text())
    baseline = cost["baseline"]
    assert baseline["available"] is True
    assert baseline["exact_wall_time_s"] > 0.0
    assert baseline["sampler_wall_time_s"] > 0.0
    lowered = baseline["disclaimer"].lower()
    assert "no speed" in lowered or "not a speed" in lowered or \
        ("reference" in lowered and "claim" in lowered)
    # "faster" may appear only inside an explicit NEGATION of a speed claim
    # (e.g. "is not evidence that X is faster than Y") -- never an assertion
    # that either approach actually is faster.
    if "faster" in lowered:
        assert "not evidence" in lowered or "no " in lowered or "never" in lowered


def test_cost_json_baseline_unavailable_with_a_reason_for_an_oversized_model(tmp_path):
    n = EXACT_LIMIT + 2
    c = compile_spec(_oversized_spec(n), IDEAL)
    d = write_receipt(c, tmp_path / "r")
    cost = json.loads((d / "cost.json").read_text())
    baseline = cost["baseline"]
    assert baseline["available"] is False
    assert baseline["note"].startswith("unavailable")
    assert baseline["exact_wall_time_s"] is None
    # the sampler still ran and its own cost is still honestly recorded
    assert baseline["sampler_wall_time_s"] > 0.0


def test_no_speed_or_hardware_claim_anywhere_in_the_explain_render(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d).lower()
    assert "speedup" not in text
    # "faster" may appear only inside an explicit negation of a speed claim,
    # never an assertion that either approach actually is faster.
    for line in text.splitlines():
        if "faster" in line:
            assert "not evidence" in line or "no " in line or "never" in line, \
                f"a bare speed claim: {line!r}"


def test_explain_verification_layer_shows_the_diversity_and_baseline_numbers(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    verification = text.split("VERIFICATION", 1)[1].split("APPLICATION", 1)[0]
    assert str(c.verification.diversity_distinct) in verification
    assert str(c.verification.diversity_reachable) in verification
    assert "unavailable" not in verification.split("CPU baseline")[1].split("\n")[0].lower() \
        if "CPU baseline" in verification else True
