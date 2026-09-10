import json
from pathlib import Path

import numpy as np
import pytest

from tsu.cli import main, sweep_single_model
from tsu.passes.analyse import analyse
from tsu.passes.lower import IsingModel
from tsu.passes.route import BetaMismatchError, insert_mediators


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


def _write_regime_edges(tmp_path, beta=0.2):
    """A tiny unmediated model: a 4-node path graph, small couplings. Small
    enough that sweep()'s own production defaults (16 chains x 2000 samples,
    measured ~1.2s for 2 beta points on a 2-node model) finish in a couple
    of seconds for a --beta-steps of 2, so this stays fast WITHOUT needing
    to expose n_chains/n_samples as new CLI flags the brief never asked
    for."""
    d = dict(nodes=4, edges=[[0, 1, 0.6], [1, 2, 0.6], [2, 3, 0.6]],
             biases=[0.0, 0.0, 0.0, 0.0], beta=beta)
    p = tmp_path / "regime_model.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_regime_subcommand_sweeps_a_single_model_and_exits_zero(tmp_path):
    """WHAT THIS PINS: `tsu regime --edges ... --out ...` actually runs the
    beta sweep (not the "cannot bridge the gap" placeholder this branch
    previously shipped) and writes all five of write_regime's output files,
    exiting 0. This test supersedes the old
    `test_regime_subcommand_reports_it_cannot_sweep_a_single_fixed_model`
    (see git history), which intentionally pinned that placeholder's
    refusal (rc == 2, "sweep" in stderr) -- exactly the behavior task 6
    exists to replace, so that assertion is now wrong on purpose.
    HOW IT FAILS: a `regime` dispatch that still prints the old "give one
    model; a sweep needs a family of sizes" message and returns 2 (i.e.
    plumbing added but the actual gap not closed) makes rc == 2 and none of
    the five files exist. Also fails if `sweep_single_model` sweeps at a
    hardcoded/default size instead of this model's own `len(model.nodes)`
    (the per-row `size` field would read something other than 4), or if a
    degenerate `--beta-steps` ever produced an empty `couplings` list,
    which would make `sweep` return no rows and `write_regime` operate on
    an empty sequence.
    PROVENANCE: `write_regime`'s own return value (`tsu/preflight/
    render.py`), `[regime.json, regime.png, diagnostics.png, report.md,
    provenance.json]` -- exactly the five files asserted below."""
    edges = _write_regime_edges(tmp_path)
    out = tmp_path / "rg"
    rc = main(["regime", "--edges", str(edges), "--out", str(out),
               "--beta-min", "0.1", "--beta-max", "0.3", "--beta-steps", "2"])
    assert rc == 0
    for name in ("regime.json", "regime.png", "diagnostics.png",
                "report.md", "provenance.json"):
        assert (out / name).exists(), f"missing {name}"
    data = json.loads((out / "regime.json").read_text(encoding="utf-8"))
    assert len(data["rows"]) == 2, "one row per --beta-steps point"
    assert {r["size"] for r in data["rows"]} == {4}, \
        "the sweep must run at this model's own size (4 nodes), not a default"


def test_regime_refuses_a_mediated_model_without_sampling(tmp_path, monkeypatch):
    """WHAT THIS PINS: a model carrying mediator spins (built through the
    real `route.insert_mediators`, not a hand-set `mediator_nodes` tuple)
    is refused by `sweep_single_model` via `assert_beta_consistent` (spec
    5.3.5) the moment a swept beta differs from the beta its mediator
    couplings were computed at, and NO sampling happens first -- `A =
    arccosh(exp(2*beta*|J|))/(2*beta)` is temperature-dependent, so
    resampling at a different beta would silently reproduce the WRONG
    physical couplings.
    HOW IT FAILS: a `model_fn` that builds the beta-replaced copy first
    (`dataclasses.replace(model, beta=beta_j)`) and calls
    `assert_beta_consistent` on THAT COPY instead of on the original
    `model` would never fail this check -- the copy's own `.beta` always
    equals `beta_j` by construction, so the guard degenerates into a
    tautology. Under that bug, `sample_chains` (stubbed below to raise) is
    reached instead, so `pytest.raises(BetaMismatchError)` fails with the
    stub's AssertionError propagating instead -- the wrong exception type,
    which is exactly how this test catches the bug. Also fails if the guard
    runs only inside `sweep`'s escalation loop (after a first
    `sample_chains` call) instead of in `model_fn` before it.
    PROVENANCE: `route.assert_beta_consistent`'s own docstring (spec 5.3.5)
    and `route.insert_mediators`'s gadget formula, exercised on the
    frustrated triangle `tests/test_mediator_insertion.py` already uses to
    verify mediation is exact -- reused here rather than hand-constructing
    a `mediator_nodes` tuple, per the task brief's own instruction."""
    import tsu.preflight.sweep as sweep_mod

    def boom(*a, **k):
        raise AssertionError(
            "sample_chains must not be called when the swept beta is "
            "refused before sampling")
    monkeypatch.setattr(sweep_mod, "sample_chains", boom)

    J = {(0, 1): -1.25, (0, 2): -1.25, (1, 2): -1.25}
    orig = IsingModel(nodes=("a", "b", "c"), edges=tuple(J),
                      weights=np.array([J[e] for e in J]),
                      biases=np.zeros(3), beta=4.0, offset=0.0)
    med, rep = insert_mediators(orig, analyse(orig))
    assert med.mediator_nodes, "fixture sanity: the triangle must have mediated"

    out = tmp_path / "rg"
    with pytest.raises(BetaMismatchError, match="mediator"):
        sweep_single_model(med, beta_min=0.05, beta_max=0.6, beta_steps=3,
                           out_dir=out)
    assert not out.exists(), \
        "a refused sweep must write nothing -- write_regime is never reached"


def test_regime_report_explains_why_no_binder_crossing_was_computed(tmp_path):
    """WHAT THIS PINS: report.md states WHY the Binder crossing is missing
    -- it requires a size family, which a single `--spec`/`--edges` model
    cannot provide -- not just that `binder_crossing` is null.
    `write_regime`'s own built-in crossing=None message ("the range may not
    bracket the transition") is true for a genuine multi-size sweep that
    simply never crossed, but MISLEADING here: this sweep never had a size
    family to begin with, so the crossing is not merely unobserved, it is
    structurally impossible from this input. A reader who saw only "may not
    bracket" would reasonably widen the beta range next, which cannot fix
    this.
    HOW IT FAILS: `sweep_single_model` calling `write_regime(...)` and
    returning without appending the size-family-specific explanation would
    leave report.md carrying only the generic message, which never mentions
    a size family at all -- the substring assertions below would then fail.
    PROVENANCE: the task brief's own required statement ("locating the
    transition by finite-size scaling requires a size family, which a
    single --spec/--edges model cannot provide") and
    `tsu.preflight.sweep.crossing`'s own docstring, which needs two sizes'
    Binder curves to find where they cross."""
    edges = _write_regime_edges(tmp_path)
    out = tmp_path / "rg"
    rc = main(["regime", "--edges", str(edges), "--out", str(out),
               "--beta-min", "0.1", "--beta-max", "0.3", "--beta-steps", "2"])
    assert rc == 0

    data = json.loads((out / "regime.json").read_text(encoding="utf-8"))
    assert data["binder_crossing"] is None

    text = (out / "report.md").read_text(encoding="utf-8")
    assert "finite-size scaling" in text
    assert "size family" in text.lower()
    assert "single" in text.lower()


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
