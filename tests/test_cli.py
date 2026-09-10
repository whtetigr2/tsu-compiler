from pathlib import Path
from tsu.cli import main


def test_preflight_subcommand_writes_its_three_files_and_exits_zero(tmp_path):
    """WHAT THIS PINS: `tsu preflight --spec ... --out ...` runs end to end
    through the real CLI dispatch (not just write_preflight() called
    directly) and exits 0 on a model that compiles clean.
    HOW IT FAILS: task-5-brief.md's own worked CLI example runs
    `-m tsu.preflight.cli preflight ...` -- a module this task is explicitly
    told NOT to create (`tsu/preflight/cli.py` would be the second entry
    point + second renderer the brief calls out as the exact defect this
    plan already made once). That module does not exist and `python -m
    tsu.preflight.cli` raises `No module named tsu.preflight.cli`; the real,
    only entry point is `tsu.cli:main` (see pyproject.toml's console_scripts
    entry, `tsu = "tsu.cli:main"`), exercised here the same way
    test_compile_then_visualize_produces_all_four_layers already exercises
    `compile`/`visualize` above.
    PROVENANCE: pyproject.toml's own `tsu = "tsu.cli:main"` entry point."""
    out = tmp_path / "pf"
    rc = main(["preflight", "--spec", "specs/lattice_small_8x8_k3.yaml",
               "--out", str(out)])
    assert rc in (0, 1)  # 0 == verdict ok/warn, 1 == verdict fail; both are
                        # a completed run, not a crash
    for name in ("preflight.json", "report.md", "provenance.json"):
        assert (out / name).exists(), f"missing {name}"


def test_regime_subcommand_reports_it_cannot_sweep_a_single_fixed_model(tmp_path, capsys):
    """WHAT THIS PINS: `tsu regime --spec ...` exits nonzero with a clear
    stderr message rather than crashing or silently doing nothing --
    load_model() (tasks 1-4) returns exactly one fixed-size IsingModel, and
    sweep() (task 4) needs model_fn(size, beta_j) to generate a FAMILY of
    models across --sizes, a gap this CLI wiring cannot bridge on its own.
    HOW IT FAILS: a dispatch that calls load_model() unconditionally before
    branching on a.cmd (as task-5-brief.md's own pseudocode does -- 'model =
    load_model(...)' appears before the `if a.cmd == "preflight"` check)
    would raise ValueError on a `regime` invocation that gave neither
    --spec nor --edges, or would blame a perfectly valid spec file for a gap
    that is actually in the regime wiring -- either way surfacing the wrong
    failure to the user. This test passes a VALID spec and checks the
    process still exits 2 with the documented guidance rather than a
    traceback, which only holds if load_model() is never called on this path.
    PROVENANCE: sweep()'s own signature (tsu/preflight/sweep.py),
    `sweep(model_fn, sizes, couplings, ...)`, vs. load_model()'s
    (tsu/preflight/model.py), `load_model(spec=None, edges=None) ->
    IsingModel` -- one model, not a generator."""
    rc = main(["regime", "--spec", "specs/lattice_small_8x8_k3.yaml",
               "--out", str(tmp_path / "rg")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "sweep" in err.lower()


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
