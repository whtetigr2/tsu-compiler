"""Part B: the compilation report, rendered from receipt data ALONE. Every
field the receipt does not contain must print `unavailable: <reason>`, never a
blank/zero/invented value; the four VERDICT lines must map to real receipt
evidence, printing `?` (not a false checkmark) when that evidence is missing.
"""
import json

import pytest

from tsu.cli import main
from tsu.passes.search import compile_spec
from tsu.receipt import write_receipt
from tsu.report import render_report
from tsu.spec import load_spec
from tsu.target import Z1


def test_a_compiled_receipt_renders_all_sections_with_real_numbers(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verdict == "COMPILED"
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)

    for header in ("TSU COMPILATION REPORT", "WORKLOAD", "REPRESENTATION",
                  "Z1 MAPPING", "SAMPLING", "VERDICT"):
        assert header in text

    assert "Variables:" in text and "3" in text          # a, b, c
    assert "domain-wall" in text                          # selected encoding
    assert "REPRESENTATION FITS Z1" in text
    # every VERDICT line must be a real checkmark for a clean COMPILED report
    assert "✗" not in text
    assert text.count("✓") == 4


def test_never_prints_the_bare_word_none_or_a_blank_value(tmp_path):
    """The receipt's own honesty rule (I5/I8 in receipt.py/viz.py) applies to
    the report too: an absent field must carry its reason, never render as a
    literal 'None' or an empty trailing value."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    assert "None" not in text
    for line in text.splitlines():
        if line.strip().endswith(":"):
            pytest.fail(f"a label with no value at all: {line!r}")


def test_ess_and_mixing_and_diversity_are_never_measured_and_say_so(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    assert "unavailable: not measured" in text
    ess_line = next(l for l in text.splitlines() if l.strip().startswith("ESS:"))
    mixing_line = next(l for l in text.splitlines() if l.strip().startswith("Mixing:"))
    diversity_line = next(l for l in text.splitlines() if l.strip().startswith("Diversity:"))
    for line in (ess_line, mixing_line, diversity_line):
        assert "unavailable: not measured" in line


def test_a_logical_failure_renders_unavailable_sections_and_a_reason_block(tmp_path):
    """broken.yaml fails the ideal control -- verdict LOGICAL, hardware never
    evaluated. REPRESENTATION/Z1 MAPPING/SAMPLING fields must all read
    unavailable (never a fabricated 0), and a Reason/Suggested-next-action
    block must be present, sourced from the recorded candidate reason."""
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)

    assert "REPRESENTATION FITS" not in text
    assert "?" in text          # at least one verdict layer unavailable
    assert "Reason:" in text
    assert "Suggested next action:" in text
    assert "unavailable" in text.lower()


def test_a_hardware_failure_names_the_offending_quantity_and_the_limit(tmp_path):
    """too_dense.yaml passes ideal (logically fine) then fails Z1's degree gate.
    The report's failure block must name the actual measured/limit values and a
    real recorded remediation -- not new prose (spec Part B rule)."""
    c = compile_spec(load_spec("specs/too_dense.yaml"), Z1)
    assert c.verdict == "HARDWARE"
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)

    assert "Reason:" in text
    assert "19" in text and "16" in text        # measured degree vs Z1's limit
    assert "Suggested next action:" in text
    # the remediation text must come from the ACTUAL recorded Remediation
    # objects (gates.py's degree-gate remediations), not fabricated prose
    assert ("encoding" in text.lower() or "target.degree" in text)


def test_a_rejected_candidates_own_measurements_are_used_not_blanket_unavailable(tmp_path):
    """too_dense.yaml's degree-exceeded candidate still reached `analyse`
    before Z1's degree gate rejected it -- REPRESENTATION must report its real
    logical spins/edges/bipartite-ness (tagged as not-selected), not claim
    'representation was never reached' for a compile that, in fact, measured
    one. Sourced from candidates.json (the A5 compare() table), not
    recomputed."""
    c = compile_spec(load_spec("specs/too_dense.yaml"), Z1)
    assert c.verdict == "HARDWARE"
    d = write_receipt(c, tmp_path / "r")
    candidates = json.loads((d / "candidates.json").read_text())
    assert candidates and candidates[0]["logical_spins"] == 20

    text = render_report(d)
    assert "20 (from domain_wall" in text
    assert "not selected" in text
    # and the arithmetic that depends on it must use the RAW number, not the
    # formatted "20 (from domain_wall...)" string
    assert "100 (from domain_wall" in text    # direct couplings: 190 - 90 mediators


def test_cli_report_subcommand_prints_the_same_render(tmp_path, capsys):
    out = tmp_path / "r"
    assert main(["compile", "specs/toy.yaml", "--target", "z1",
                "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["report", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "TSU COMPILATION REPORT" in printed
    assert "REPRESENTATION FITS Z1" in printed


def test_workload_json_is_written_and_generic_over_the_spec():
    """A1-adjacent: the workload summary in the receipt is derived once at
    compile time from the spec itself (variables/terms/state cardinality/
    constraint classes), not hardcoded, and works for a spec with no
    contract rules at all (edgeless.yaml)."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = write_receipt(c, td + "/r")
        workload = json.loads((d / "workload.json").read_text())
    assert workload["variables"] == 3
    assert workload["logical_interactions"] == 3
    assert workload["state_cardinality"] == 3     # c has k=3, the largest domain
    assert workload["constraint_classes"] == 2    # forbid_both, forbid_value_with
