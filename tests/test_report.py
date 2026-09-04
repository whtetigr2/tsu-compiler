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


def test_diversity_prints_the_real_measurement_now_that_something_computes_it(tmp_path):
    """C4: superseded from the old permanent 'unavailable: not measured'
    placeholder -- the SAME transition ESS/Mixing already went through below
    (that placeholder was honest only because nothing computed the field
    yet; now something does). toy.yaml's small state space is exactly the
    "known trap" C4's brief names: the reachable valid set's own size must
    print alongside the distinct/valid count so the ratio cannot be misread."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verification.diversity_distinct is not None
    assert c.verification.diversity_reachable is not None
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    diversity_line = next(l for l in text.splitlines() if l.strip().startswith("Diversity:"))
    assert "unavailable" not in diversity_line
    assert str(c.verification.diversity_distinct) in diversity_line
    assert str(c.verification.diversity_reachable) in diversity_line


def test_ess_and_mixing_print_the_real_measurement_when_the_chain_supports_it(tmp_path):
    """toy.yaml's compiled chain mixes fast enough (tau ~ 0.9 -- see
    tests/test_verify.py and tests/test_ess.py) that tsu.ess's reliability
    threshold is cleared: ESS/Mixing must now print REAL numbers sourced from
    the receipt (verification.json's `ess`, regime.json's `mixing_indicator`),
    not the old permanent placeholder -- that placeholder was honest only
    because nothing computed these fields yet; now something does."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verification.ess is not None, \
        "toy.yaml's chain should mix well enough for a reliable ESS estimate"
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    ess_line = next(l for l in text.splitlines() if l.strip().startswith("ESS:"))
    mixing_line = next(l for l in text.splitlines() if l.strip().startswith("Mixing:"))
    assert "unavailable" not in ess_line
    assert "unavailable" not in mixing_line
    # the numbers rendered must be the SAME numbers the receipt recorded --
    # report.py must never recompute, only format.
    assert f"{c.verification.ess:.6g}" in ess_line
    assert f"{c.regime.mixing_indicator:.6g}" in mixing_line


def test_ess_and_mixing_print_unavailable_with_a_reason_when_the_chain_is_too_short(
        tmp_path, monkeypatch):
    """The other honest half: when tsu.ess itself decides a chain is too
    short/too correlated to trust (see test_ess.py's validity-domain tests),
    the report must print `unavailable: <reason>`, never a fabricated number.
    Forcing this end-to-end (not just at the tsu.ess unit level) proves the
    wiring -- search.py, Verification, RegimeReport, report.py -- actually
    propagates a None-with-reason result all the way to the rendered text,
    rather than only the happy path ever being exercised."""
    import numpy as np
    from tsu.backends import thrml_backend

    real_sample_chains = thrml_backend.sample_chains

    def fake_sample_chains(prog, n_chains, n_samples, n_warmup, steps_per_sample, seed):
        # A short, strongly autocorrelated chain: N/tau is far below tsu.ess's
        # reliability threshold no matter what the real program mixes like.
        real = real_sample_chains(
            prog, n_chains, n_samples, n_warmup, steps_per_sample, seed)
        n = real.shape[-1]
        rng = np.random.default_rng(0)
        x = np.zeros((n_chains, n_samples))
        for t in range(1, n_samples):
            x[:, t] = 0.97 * x[:, t - 1] + rng.normal(size=n_chains)
        alternating = (x > np.median(x)).astype(int)
        forced = np.tile(alternating[:, :, None], (1, 1, n))
        return forced

    monkeypatch.setattr(thrml_backend, "sample_chains", fake_sample_chains)
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verification.ess is None
    assert c.verification.ess_note.startswith("unavailable")
    assert c.regime.mixing_indicator is None

    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    ess_line = next(l for l in text.splitlines() if l.strip().startswith("ESS:"))
    mixing_line = next(l for l in text.splitlines() if l.strip().startswith("Mixing:"))
    assert "unavailable" in ess_line
    assert "unavailable" in mixing_line


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


def test_a_downgraded_gate_does_not_read_as_a_hardware_failure_on_a_compiled_receipt(tmp_path):
    """too_strong_bias.yaml's bias exceeds Z1's ASSUMED |b| cap and fails
    outright, but compiles clean with --allow-assumed (test_search.py's own
    regression). A downgraded gate (passed=False, downgraded=True) did not
    block the compile -- the report's HARDWARE line must read as passing,
    and the Reason block must not blame a candidate for a compile that
    succeeded.

    P-3/F-A5 + I-9a/F-R7: was specs/too_strong.yaml, checking the "Coupling
    range:" line -- but max_abs_coupling (|J|) is now Extropic-documented,
    not assumed, so it is no longer downgradable via --allow-assumed at
    all (see tests/test_search.py::
    test_too_strong_coupling_violation_is_not_overridable_now_that_it_is_sourced).
    specs/too_strong_bias.yaml exercises the still-genuinely-assumed
    max_abs_bias (|b|) field instead, so this checks the "Field range:"
    line."""
    c = compile_spec(load_spec("specs/too_strong_bias.yaml"), Z1, allow_assumed=True)
    assert c.verdict == "COMPILED"
    d = write_receipt(c, tmp_path / "r")
    text = render_report(d)
    lines = text.splitlines()
    hardware_line = next(l for l in lines if l.strip().endswith("HARDWARE"))
    assert "✓" in hardware_line
    field_line = next(l for l in lines if l.strip().startswith("Field range:"))
    assert "downgraded" in field_line.lower()


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
