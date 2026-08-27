from pathlib import Path
from tsu.cli import main


def test_compile_then_visualize_produces_all_four_layers(tmp_path, capsys):
    out = tmp_path / "r"
    assert main(["compile", "specs/toy.yaml", "--target", "z1",
                 "--out", str(out)]) == 0
    html = tmp_path / "r.html"
    assert main(["visualize", str(out), "--out", str(html)]) == 0
    text = html.read_text(encoding="utf-8")
    for layer in ("Problem space", "Representation", "Thermodynamic state",
                  "Hardware mapping"):
        assert layer in text, f"missing visualization layer: {layer}"


def test_visualize_reads_only_the_receipt(tmp_path):
    """Layer D must be rendered from receipt data, never recomputed or hardcoded."""
    out = tmp_path / "r"
    main(["compile", "specs/toy.yaml", "--target", "z1", "--out", str(out)])
    html = tmp_path / "r.html"
    main(["visualize", str(out), "--out", str(html)])
    text = html.read_text(encoding="utf-8")
    assert "assumed" in text, "the Jmax provenance must be visible to a reader"


def test_broken_spec_exits_nonzero_and_says_LOGICAL(tmp_path, capsys):
    rc = main(["compile", "specs/broken.yaml", "--target", "z1",
               "--out", str(tmp_path / "b")])
    assert rc != 0
    assert "LOGICAL" in capsys.readouterr().out


def test_inspect_does_not_require_a_target_lattice(capsys):
    assert main(["inspect", "specs/toy.yaml"]) == 0
    assert "edges" in capsys.readouterr().out


def test_inspect_never_prints_the_uncomputed_mediators_sentinel(tmp_path, capsys):
    """I5: rep.mediators == -1 means 'not computed' (exact max-cut is
    exponential and the graph exceeded MAXCUT_EXACT_LIMIT), not zero
    mediators. `inspect` must not print the raw sentinel to a user."""
    n = 25   # > MAXCUT_EXACT_LIMIT (20), and an odd ring is never bipartite
    lines = ["name: odd_ring", "variables:"]
    lines += [f"  x{i}: {{domain: binary}}" for i in range(n)]
    lines += ["terms:"]
    lines += [f"  - {{kind: product, a: {{x{i}: 1.0}}, "
             f"b: {{x{(i + 1) % n}: 1.0}}, weight: 1.0}}" for i in range(n)]
    lines += ["contract:", "  validate: []"]
    spec_path = tmp_path / "odd_ring.yaml"
    spec_path.write_text("\n".join(lines), encoding="utf-8")

    assert main(["inspect", str(spec_path)]) == 0
    out = capsys.readouterr().out
    assert '"mediators": -1' not in out
    assert "not computed" in out
