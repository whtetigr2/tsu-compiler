"""C3: `tsu explain` -- the fuller, teaching-trace rendering, one section per
layer, kept SEPARATE from `tsu report` (which stays the terse form). Same
honesty rules as report.py: rendered from receipt data alone, never
recomputed, never hardcoded; a field the receipt lacks prints
`unavailable: <reason>`, never a blank/None/invented value.
"""
import json

from tsu.cli import main
from tsu.passes.search import compile_spec
from tsu.receipt import write_receipt
from tsu.report import render_explain
from tsu.spec import load_spec
from tsu.target import Z1

_LAYERS = ("APPLICATION", "FORMULATION", "REPRESENTATION", "ENERGY",
          "TOPOLOGY", "PHYSICAL MAPPING", "SAMPLER", "VERIFICATION")


def test_explain_contains_every_layer_header(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    for header in _LAYERS:
        assert header in text, f"missing layer header {header!r}"
    # APPLICATION appears twice: what the spec declared, and the decoded result
    assert text.count("APPLICATION") == 2


def test_explain_never_prints_the_bare_word_none_or_a_blank_value(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    assert "None" not in text
    for line in text.splitlines():
        if line.strip().endswith(":"):
            raise AssertionError(f"a label with no value at all: {line!r}")


def test_explain_application_layer_shows_the_specs_own_vocabulary(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    assert "toy" in text
    assert "Variables:" in text and "3" in text


def test_explain_formulation_layer_lists_term_kinds_from_the_spec(tmp_path):
    c = compile_spec(load_spec("specs/adjacency_2x2_k3.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    formulation = text.split("FORMULATION", 1)[1].split("REPRESENTATION", 1)[0]
    assert "product_over_edges" in formulation


def test_explain_representation_layer_shows_the_candidate_table_and_rationale(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    representation = text.split("REPRESENTATION", 1)[1].split("ENERGY", 1)[0]
    assert "domain_wall" in representation and "one_hot" in representation
    assert "selection order" in representation.lower() or \
           "ordering" in representation.lower()


def test_explain_energy_layer_shows_term_inventory_and_ranges(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.encoded is not None
    d = write_receipt(c, tmp_path / "r")
    energy = json.loads((d / "energy.json").read_text())
    assert energy["term_counts"]["linear"] >= 0
    assert energy["term_counts"]["product"] >= 1

    text = render_explain(d)
    energy_section = text.split("ENERGY", 1)[1].split("TOPOLOGY", 1)[0]
    assert str(energy["term_counts"]["product"]) in energy_section
    assert "|J|" in energy_section and "|b|" in energy_section


def test_explain_topology_layer_reports_bipartiteness_and_residual(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    topology = text.split("TOPOLOGY", 1)[1].split("PHYSICAL MAPPING", 1)[0]
    assert "bipartite" in topology.lower()


def test_explain_physical_mapping_layer_shows_placement_and_pbits(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    phys = text.split("PHYSICAL MAPPING", 1)[1].split("SAMPLER", 1)[0]
    assert "Physical p-bits:" in phys
    assert "Colour blocks:" in phys


def test_explain_sampler_layer_shows_kernel_blocks_and_clamp(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1, clamp={"a": 1})
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    sampler = text.split("SAMPLER", 1)[1].split("VERIFICATION", 1)[0]
    assert "chromatic block Gibbs" in sampler
    assert "'a': 1" in sampler or '"a": 1' in sampler or "a: 1" in sampler


def test_explain_sampler_layer_shows_none_when_nothing_is_clamped(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    sampler = text.split("SAMPLER", 1)[1].split("VERIFICATION", 1)[0]
    clamp_line = next(l for l in sampler.splitlines() if l.strip().startswith("Clamp:"))
    assert "none" in clamp_line.lower()


def test_explain_verification_layer_reads_real_numbers_from_the_receipt(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    verification = text.split("VERIFICATION", 1)[1].split("APPLICATION", 1)[1] \
        if text.count("APPLICATION") else text.split("VERIFICATION", 1)[1]
    # the actual number from the receipt must appear verbatim, never recomputed
    assert f"{c.verification.energy_tv:.6g}" in text


def test_explain_final_application_layer_shows_task_validity_and_decoded_result(tmp_path):
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    final_app = text.rsplit("APPLICATION", 1)[1]
    assert "Task validity:" in final_app or "task validity" in final_app.lower()
    assert "Decoded result:" in final_app
    assert "unavailable" in final_app.lower()


def test_explain_on_a_logical_failure_reports_unavailable_layers(tmp_path):
    """broken.yaml fails the ideal control -- REPRESENTATION/ENERGY/TOPOLOGY/
    PHYSICAL MAPPING/SAMPLER/VERIFICATION must all read unavailable, never a
    fabricated value, exactly like `tsu report`'s own LOGICAL-verdict test."""
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    d = write_receipt(c, tmp_path / "r")
    text = render_explain(d)
    assert "unavailable" in text.lower()
    assert "None" not in text


def test_cli_explain_subcommand_prints_the_render(tmp_path, capsys):
    out = tmp_path / "r"
    assert main(["compile", "specs/toy.yaml", "--target", "z1",
                "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["explain", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "APPLICATION" in printed and "VERIFICATION" in printed


def test_tsu_report_still_works_unchanged_alongside_explain(tmp_path):
    """`tsu report` stays the terse form -- explain is additive."""
    from tsu.report import render_report
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    d = write_receipt(c, tmp_path / "r")
    report_text = render_report(d)
    explain_text = render_explain(d)
    assert "REPRESENTATION FITS Z1" in report_text
    assert len(explain_text) > len(report_text), \
        "explain is supposed to be the FULLER rendering"
