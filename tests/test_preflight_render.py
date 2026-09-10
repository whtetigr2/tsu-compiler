"""Task 5: the artifact a reader checks the work from.

Every test here carries WHAT THIS PINS / HOW IT FAILS / PROVENANCE in its
docstring, per this project's standing test-header convention.
"""
import json
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu.passes.lower import IsingModel
from tsu.preflight.check import preflight
from tsu.preflight.diagnostics import RHAT_THRESHOLD
from tsu.preflight.sweep import RegimeRow
from tsu.preflight.render import provenance, write_preflight, write_regime


def grid(n: int, j: float = 0.4, b: float = 0.0) -> IsingModel:
    """A 4-neighbour grid: bipartite, and a direct subgraph of Z1.

    Duplicated from tests/test_preflight_check.py rather than imported --
    there is no `tests/__init__.py` in this project and no existing
    precedent anywhere in the suite for `from tests.X import Y` (every other
    test file defines its own small fixtures, e.g. test_preflight_regime.py's
    own `_rows()`), so relying on implicit-namespace-package import behaviour
    here would be the first of its kind and the most fragile way to get ten
    lines of graph construction.
    """
    idx = {(x, y): y * n + x for y in range(n) for x in range(n)}
    e = []
    for (x, y), i in idx.items():
        for dx, dy in ((1, 0), (0, 1)):
            if (x + dx, y + dy) in idx:
                e.append((i, idx[(x + dx, y + dy)]))
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n * n)),
                      edges=tuple(e), weights=np.full(len(e), j),
                      biases=np.full(n * n, b), beta=1.0, offset=0.0)


def triangle() -> IsingModel:
    """An odd cycle: the smallest non-bipartite graph, and the fixture
    test_preflight_check.py already uses to force a placement failure via
    degree/mediation -- reused here (not re-derived) to get a PreflightReport
    with `place_error` and `remediations` populated."""
    return IsingModel(nodes=("a", "b", "c"), edges=((0, 1), (1, 2), (0, 2)),
                      weights=np.full(3, 0.4), biases=np.zeros(3),
                      beta=1.0, offset=0.0)


def _degree_violation() -> IsingModel:
    """A complete graph on 20 nodes: degree 19 exceeds Z1's cap, so place()
    raises CompileError and preflight() reports it (mirrors
    test_preflight_check.py's test_a_degree_violation_is_REPORTED_not_raised)."""
    n = 20
    e = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=tuple(e),
                      weights=np.full(len(e), 0.1), biases=np.zeros(n),
                      beta=1.0, offset=0.0)


def _row(beta_j, size, abs_m, abs_m_err, binder, r_hat, *, tau=2.0,
         n_eff=50.0, ess_reason="ok", ess_unavailable=False,
         n_samples_used=2000):
    """RegimeRow has 13 required fields (no defaults) -- three more than the
    brief's own `_rows()` fixture supplied (`ess_reason`, `ess_unavailable`,
    `n_samples_used` were missing there, which raises TypeError on
    construction; confirmed against the live dataclass via
    inspect.signature(RegimeRow.__init__) before writing this). This helper
    fills all 13 explicitly so every test below states only what it varies."""
    return RegimeRow(beta_j=beta_j, size=size, abs_m=abs_m,
                     abs_m_err=abs_m_err, chi=1.0, binder=binder, tau=tau,
                     n_eff=n_eff, r_hat=r_hat, ess_reason=ess_reason,
                     ess_unavailable=ess_unavailable,
                     provisional=r_hat > RHAT_THRESHOLD,
                     n_samples_used=n_samples_used)


def _rows():
    return [_row(0.2, 8, 0.05, 0.02, 0.02, 1.0),
            _row(0.4, 8, 0.40, 0.35, 0.35, 1.0),
            _row(0.6, 8, 0.95, 0.62, 0.58, 1.4)]


# --------------------------------------------------------------------------
# write_preflight
# --------------------------------------------------------------------------

def test_preflight_writes_its_three_files(tmp_path):
    """WHAT THIS PINS: write_preflight produces exactly preflight.json,
    report.md and provenance.json.
    HOW IT FAILS: dropping any one of the three write calls, or writing under
    a different filename, shrinks or renames the returned set.
    PROVENANCE: task-5-brief.md's own interface line for write_preflight."""
    names = {p.name for p in write_preflight(preflight(grid(6)), tmp_path)}
    assert names == {"preflight.json", "report.md", "provenance.json"}


def test_provenance_records_the_backend_and_versions():
    """WHAT THIS PINS: provenance() reports a real JAX backend and a real
    thrml version, not placeholders.
    HOW IT FAILS: a provenance() that hardcodes jax_backend="cpu" instead of
    calling jax.default_backend(), or that swallows an ImportError into
    "unavailable" for a module that is actually installed, passes a naive
    smoke test but silently misreports the environment on a GPU/TPU machine.
    PROVENANCE: this environment's own installed jax (0.11.1, cpu) and thrml
    (0.1.4), read directly from the modules -- not asserted as fixed strings,
    since the point is that the module must ask the library, not guess."""
    p = provenance()
    assert p["jax_backend"] in ("cpu", "gpu", "tpu")
    assert p["thrml"] != "unavailable"
    import jax
    assert p["jax_backend"] == jax.default_backend()
    assert p["thrml"] == __import__("thrml").__version__


def test_the_report_states_that_no_hardware_was_involved(tmp_path):
    """WHAT THIS PINS: report.md's front page says no Z1 hardware was used.
    HOW IT FAILS: this is the exact sentence the brief warns is 'most likely
    to be quietly dropped in a later edit' -- a header rewrite that keeps the
    verdict/gates table but drops the hardware disclaimer passes every other
    test in this file while silently letting a reader assume real hardware
    ran the sampling.
    PROVENANCE: task-5-brief.md, 'Constraints' section, sentence 1 of item 3."""
    write_preflight(preflight(grid(6)), tmp_path)
    txt = (tmp_path / "report.md").read_text(encoding="utf-8").lower()
    assert "no hardware" in txt or "simulat" in txt


def test_report_marks_an_assumed_gate_limit_visibly(tmp_path):
    """WHAT THIS PINS (F2, branch review): report.md's gates table renders
    a `note` column, and an ASSUMED (not Extropic-sourced) gate's note is
    marked visibly there -- before this, `write_preflight`'s table had NO
    note column at all (value/limit/%/status only), so `assumed`/
    `downgraded` were carried in preflight.json but invisible on the page
    a reader actually reads, which was itself part of the F2 defect: an
    Extropic-documented fact (max_abs_coupling) and a genuine project
    guess (max_abs_bias) rendered IDENTICALLY on report.md's table.
    HOW IT FAILS: a report.md whose gates table has no note column (or
    one that never says ASSUMED for max_abs_bias) makes the substring
    assertions below fail.
    PROVENANCE: target.py's Sourced(6.0, "assumed", ...) for max_abs_bias;
    branch review F2."""
    n = 3
    im = IsingModel(nodes=tuple(f"n{i}" for i in range(n)),
                    edges=((0, 1), (1, 2)), weights=np.full(2, 0.5),
                    biases=np.array([9.0, 0.0, 0.0]), beta=1.0, offset=0.0)
    rep = preflight(im)
    write_preflight(rep, tmp_path)
    txt = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| note |" in txt, "the gates table must render a note column"
    assert "max_abs_bias" in txt and "ASSUMED" in txt
    lines = [ln for ln in txt.splitlines() if ln.startswith("| max_abs_bias")]
    assert lines, "max_abs_bias must appear as a gate row"
    assert "ASSUMED" in lines[0]
    assert "the hardware's" not in lines[0].lower()


def test_a_placement_failure_shows_its_remediations_in_the_report(tmp_path):
    """WHAT THIS PINS: when placement fails, the compiler's own remediation
    text (not just a bare 'fail' verdict) reaches report.md.
    HOW IT FAILS: check.py's own docstring says a pre-flight report that
    prints only 'degree exceeded' throws away the actionable half of an
    answer place() already computed; a renderer that prints the gates table
    and verdict but never touches `report.remediations` passes every test
    that only checks the verdict string, while hiding the one thing a user
    needs to actually fix the model.
    PROVENANCE: PreflightReport.remediations, populated by check.py's
    _remediations() from place()'s own CompileError.failures; independently
    confirmed non-empty and degree-related in
    test_preflight_check.py::test_a_placement_failure_carries_the_compiler_s_own_remediations."""
    rep = preflight(_degree_violation())
    assert rep.verdict == "fail" and rep.remediations
    write_preflight(rep, tmp_path)
    txt = (tmp_path / "report.md").read_text(encoding="utf-8").lower()
    assert "degree" in txt
    assert any(r.lower() in txt for r in
              [s.split(":", 1)[-1].strip().lower() for s in rep.remediations]) \
        or any(s.lower() in txt for s in rep.remediations)


# --------------------------------------------------------------------------
# write_regime
# --------------------------------------------------------------------------

def test_regime_writes_json_and_both_plots(tmp_path):
    """WHAT THIS PINS: write_regime produces regime.json, regime.png and
    diagnostics.png.
    HOW IT FAILS: dropping either plot call, or writing it under a different
    name, shrinks the returned set below the required subset.
    PROVENANCE: task-5-brief.md's interface line for write_regime."""
    names = {p.name for p in write_regime(
        _rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)}
    assert {"regime.json", "regime.png", "diagnostics.png"} <= names


def test_regime_also_writes_its_provenance_file(tmp_path):
    """WHAT THIS PINS: write_regime actually WRITES provenance.json to
    out_dir, not just returns a path to one.
    HOW IT FAILS: the brief's own sample write_regime() body never calls
    `.write_text(...)` for provenance.json anywhere -- it only appends
    `out / "provenance.json"` to the returned list. Confirmed by inspection:
    write_preflight's sample body has a matching `p.write_text(...)` line for
    provenance.json; write_regime's does not. Returning a path to a file
    nobody wrote means a reader who trusts the return value gets a
    FileNotFoundError, or silently reads a stale provenance.json left over
    from a previous run in the same out_dir.
    PROVENANCE: direct comparison of task-5-brief.md's write_preflight vs
    write_regime sample bodies."""
    paths = write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)
    prov_path = tmp_path / "provenance.json"
    assert prov_path in paths
    assert prov_path.exists() and prov_path.stat().st_size > 0
    d = json.loads(prov_path.read_text(encoding="utf-8"))
    assert "jax_backend" in d


def test_every_row_reaches_the_json_with_its_diagnostics(tmp_path):
    """WHAT THIS PINS: all three rows appear in regime.json, each carrying
    tau/n_eff/r_hat/provisional.
    HOW IT FAILS: a renderer that filters out provisional rows before writing
    JSON (rather than only excluding them from usable_band, which happens
    upstream in sweep.py) would make the band look better-supported than it
    is -- this is the brief's own stated concern for this test.
    PROVENANCE: task-5-brief.md, test docstring for this exact test."""
    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)
    d = json.loads((tmp_path / "regime.json").read_text(encoding="utf-8"))
    assert len(d["rows"]) == 3
    for r in d["rows"]:
        assert {"tau", "n_eff", "r_hat", "provisional"} <= set(r)


def test_a_provisional_row_is_marked_in_the_markdown(tmp_path):
    """WHAT THIS PINS: the word 'provisional' appears in report.md when a row
    has R-hat above threshold.
    HOW IT FAILS: a table that renders r_hat as a bare number without a flag
    column would require the reader to know RHAT_THRESHOLD=1.01 by heart and
    scan every row for it -- this test would still pass under an
    `if band is not None` style guard only if 'provisional' is unconditionally
    printed somewhere in the boilerplate, so it is paired with the next test
    which checks the flag tracks the FIELD, not just the word's presence."""
    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)
    txt = (tmp_path / "report.md").read_text(encoding="utf-8").lower()
    assert "provisional" in txt


def test_a_non_provisional_sweep_does_not_claim_any_row_is_provisional(tmp_path):
    """WHAT THIS PINS: the 'provisional' marker in the markdown table tracks
    the FIELD, not a static word baked into the template regardless of data.
    HOW IT FAILS: a template that always prints the literal string
    'provisional' in a legend/caption (satisfying the previous test
    unconditionally) would still pass if it also incorrectly tagged a
    non-provisional row as provisional, or vice versa -- this test uses an
    all-R-hat-1.0 fixture and checks the per-row flag cell specifically.
    PROVENANCE: RegimeRow.provisional's own definition, r_hat > RHAT_THRESHOLD
    (1.01); every row here has r_hat=1.0, strictly below it."""
    rows = [_row(0.2, 8, 0.05, 0.02, 0.02, 1.0),
            _row(0.4, 8, 0.40, 0.35, 0.35, 1.0)]
    assert all(not r.provisional for r in rows)
    write_regime(rows, None, None, tmp_path, onsager=None)
    lines = (tmp_path / "report.md").read_text(encoding="utf-8").splitlines()
    table_rows = [ln for ln in lines if ln.startswith("| 0.")]
    assert len(table_rows) == 2
    assert not any("**provisional**" in ln for ln in table_rows)


def test_a_row_with_no_reliable_ess_prints_its_reason_not_a_number(tmp_path):
    """WHAT THIS PINS: a row whose abs_m_err/tau/n_eff are None (tsu.ess
    refused an estimate) must render its ess_reason text in report.md, and
    must keep None (JSON null) in regime.json -- never 0.0, never a blank
    table cell.
    HOW IT FAILS: the brief's own sample write_regime body formats the table
    row unconditionally as f"{r.abs_m_err:.3f}" / f"{r.tau:.2f}" /
    f"{r.n_eff:.0f}" -- calling that on this fixture raises
    `TypeError: unsupported format string passed to NoneType.__format__`
    (confirmed by hand: `f'{None:.3f}'` raises exactly this in Python 3.14).
    A defensive fix that substitutes `r.abs_m_err or 0.0` would instead print
    '0.000', which reads as an exact zero measurement -- the precise
    violation of this task's rule 1 ('a zero error bar reads as an exact
    measurement, which is the opposite of what happened').
    PROVENANCE: RegimeRow's own docstring in sweep.py -- 'abs_m_err, tau and
    n_eff are None together... They are optional rather than zero-filled on
    purpose.'"""
    reason = ("unavailable: N/tau=812 (N=6400, tau~=7.88) is below the "
              "reliability threshold 5000 the AR(1) validation established "
              "(largest attempt: n_samples=32000, budget max_samples=32000)")
    unavailable_row = _row(0.44, 16, 0.50, None, 0.30, 1.0, tau=None,
                          n_eff=None, ess_reason=reason, ess_unavailable=True,
                          n_samples_used=32_000)
    rows = [_row(0.2, 16, 0.05, 0.02, 0.02, 1.0), unavailable_row]

    write_regime(rows, None, None, tmp_path, onsager=None)

    d = json.loads((tmp_path / "regime.json").read_text(encoding="utf-8"))
    row_044 = next(r for r in d["rows"] if r["beta_j"] == 0.44)
    assert row_044["abs_m_err"] is None
    assert row_044["tau"] is None
    assert row_044["n_eff"] is None

    txt = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "N/tau=812" in txt or "below the reliability threshold" in txt
    assert "0.000" not in txt, \
        "a None error bar must never render as the literal zero 0.000"


def test_n_samples_used_is_rendered_so_escalated_rows_are_visible(tmp_path):
    """WHAT THIS PINS: two rows with different n_samples_used (one at the
    base draw count, one escalated 16x by sweep()'s doubling loop) must show
    THOSE DIFFERENT NUMBERS in report.md, not a single constant.
    HOW IT FAILS: the brief's own sample markdown table has NO n_samples_used
    column at all -- confirmed by re-reading the table's header line in
    task-5-brief.md, which lists beta*J/L/<|m|>/chi/U/tau/N_eff/R-hat/flag
    and nothing else. Rendering only n_eff (a downstream, tau-DIVIDED number)
    would not let a reader distinguish 'this row needed 32x the draws to
    certify' from 'this row settled fast', which rule 4 explicitly names as
    a direct physics readout that must be visible.
    PROVENANCE: RegimeRow's own docstring -- 'n_samples_used is itself a
    physics readout... the rows that needed the most draws to certify are
    the ones nearest the transition.'"""
    rows = [_row(0.2, 16, 0.05, 0.02, 0.02, 1.0, n_samples_used=2000),
            _row(0.44, 16, 0.50, 0.30, 0.30, 1.0, n_samples_used=32_000)]
    write_regime(rows, None, None, tmp_path, onsager=None)
    txt = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "2000" in txt or "2,000" in txt
    assert "32000" in txt or "32,000" in txt


def test_onsager_is_printed_only_when_it_applies(tmp_path):
    """WHAT THIS PINS: Onsager's Kc appears in report.md only when the caller
    passes a numeric `onsager`, and not at all when it passes None.
    HOW IT FAILS: printing the constant unconditionally (or hedging it with
    a caveat instead of omitting it entirely) is 'how this project got its
    band wrong for months' per the brief -- this project measured the same
    ferromagnet ordering at beta*J ~ 0.075 on degree-16 connectivity, six
    times below Onsager's value for a square lattice, because the constant
    was treated as a property of the chip rather than the specific graph.
    PROVENANCE: Kc = ln(1+sqrt(2))/2, computed here from the closed form
    (Onsager 1944), not copied as a literal -- so this test would also catch
    a transcription error in whatever literal the implementation prints."""
    kc = math.log(1 + math.sqrt(2)) / 2
    assert kc == pytest.approx(0.4406867935, abs=1e-9)  # sanity on the oracle itself

    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)
    without = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "0.4407" not in without and "0.440687" not in without

    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=kc)
    with_it = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "0.4407" in with_it or "0.440687" in with_it


def test_a_closed_usable_band_renders_as_a_plain_pair(tmp_path):
    """WHAT THIS PINS: a normal, fully-closed band (both edges are real
    numbers) still renders as a simple `(lo, hi)` pair -- the guard against
    regression is
    `test_an_open_above_band_renders_as_an_explicit_interval_not_a_python_tuple`
    below (the open case), but that fix must not change this, far more
    common, closed case.
    HOW IT FAILS: a `_format_band` that always appends the "open above"
    wording, or that mishandles a plain `(float, float)` tuple, makes the
    "0.4" / "0.6" substring checks below fail or pollutes the line with
    "open above" text that does not apply here.
    PROVENANCE: `usable_band`'s own closed-band return shape,
    `(float(lo), float(hi))` (src/tsu/preflight/sweep.py)."""
    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    band_line = next(ln for ln in text.splitlines()
                     if ln.startswith("Usable band:"))
    assert "(0.4, 0.6)" in band_line
    assert "open" not in band_line.lower()
    assert "None" not in band_line


def test_an_open_above_band_renders_as_an_explicit_interval_not_a_python_tuple(tmp_path):
    """WHAT THIS PINS: `usable_band`'s open-above return, `(lo, None)`
    (no row at or above the crossing reached SATURATION within the
    sweep -- see `tsu.preflight.sweep.usable_band`), renders in report.md
    as an explicit half-open interval a reader can parse on sight -- not
    Python's own tuple repr, which would print the literal substring
    "None" next to a beta value and read as a formatting bug rather than
    "unbounded". A coordinator review flagged this as a formatting bug
    waiting for the first multi-size caller (no CLI path reaches this
    today, but `usable_band`/`write_regime` are public and library-usable
    on their own).
    HOW IT FAILS: `write_regime`'s markdown line reverting to the plain
    f-string it used before this fix, `f"Usable band: **{band}**"`, makes
    this fail: with `band=(0.2, None)`, that renders "Usable band:
    **(0.2, None)**", and the `"None" not in band_line` assertion below
    catches exactly that regression.
    PROVENANCE: `usable_band`'s own `(lo, None)` open-above convention
    (src/tsu/preflight/sweep.py, the "no row saturates" branch)."""
    write_regime(_rows(), (0.2, None), 0.2, tmp_path, onsager=None)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    band_line = next(ln for ln in text.splitlines()
                     if ln.startswith("Usable band:"))
    assert "None" not in band_line
    assert "0.2" in band_line
    assert "open" in band_line.lower(), \
        "an open-above band must read as an explicit open interval"


def test_the_regime_report_also_states_no_hardware_was_involved(tmp_path):
    """WHAT THIS PINS: write_regime's report.md carries the same no-hardware
    disclaimer as write_preflight's, since it is the file most likely to be
    shown to a reader after a sweep.
    HOW IT FAILS: if the two renderers share one _HEADER template today but a
    later edit gives write_regime its own header (e.g. to add regime-specific
    framing) without carrying the disclaimer over, this test catches it
    independently of test_the_report_states_that_no_hardware_was_involved,
    which only exercises write_preflight.
    PROVENANCE: task-5-brief.md, 'Constraints' item 3, applied to both
    report.md outputs this task produces."""
    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)
    txt = (tmp_path / "report.md").read_text(encoding="utf-8").lower()
    assert "no hardware" in txt or "simulat" in txt


def test_sweep_config_is_recorded_when_supplied_and_unavailable_when_not(tmp_path):
    """WHAT THIS PINS: provenance.json for a regime run records the sweep
    configuration (seed, n_chains, n_samples, ...) verbatim when the caller
    supplies it, and states 'unavailable: ...' -- never a fabricated or
    silently omitted config -- when it does not.
    HOW IT FAILS: RegimeRow carries no seed or sweep-parameter fields at all
    (confirmed against the live dataclass above), and task-5-brief.md's
    stated write_regime signature, `write_regime(rows, band, cross, out_dir,
    *, onsager)`, has no parameter through which a seed or sweep config could
    reach provenance.json either -- so a literal reading of that signature
    makes half of this task's own rule 3 ('provenance.json records ... seeds,
    and the full sweep configuration') structurally impossible to satisfy.
    This test pins the resolution taken here: write_regime gains an
    additional keyword-only `sweep_config` parameter, defaulted to None so
    every call in this file and in task-5-brief.md's own examples still
    works unchanged, and the value is recorded honestly either way rather
    than invented.
    PROVENANCE: task-5-brief.md 'Constraints' item 3 vs. RegimeRow's actual
    field list (sweep.py) and write_regime's stated signature -- a genuine
    interface gap, reported rather than silently worked around."""
    cfg = {"seed": 0, "n_chains": 16, "n_samples": 2000, "n_warmup": 4000,
           "steps": 8, "max_samples": 32_000, "sizes": [8, 16],
           "couplings": [0.2, 0.4, 0.6]}
    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None,
                sweep_config=cfg)
    d = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert d["sweep_config"] == cfg

    write_regime(_rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)
    d2 = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert d2["sweep_config"] != cfg
    assert isinstance(d2["sweep_config"], str) \
        and d2["sweep_config"].startswith("unavailable")


def test_regime_png_has_three_panels_and_diagnostics_png_has_two(tmp_path):
    """WHAT THIS PINS: regime.png is the 3-panel (<|m|>, chi, Binder) figure
    and diagnostics.png is the 2-panel (tau, R-hat) figure -- not two copies
    of the same plot, and not swapped.
    HOW IT FAILS: swapping which figure gets `plt.subplots(1, 3, ...)` vs
    `plt.subplots(1, 2, ...)` (or collapsing both to the same panel count)
    changes each PNG's saved pixel width in a way this test can detect
    without re-running any sampling: at dpi=120 with no bbox_inches='tight',
    a matplotlib Figure saves at exactly figsize*dpi pixels, so a 3-panel
    figsize=(15, 4.2) and a 2-panel figsize=(10.5, 4.2) are unambiguously
    different widths (1800px vs 1260px here) despite sharing the same height.
    PROVENANCE: matplotlib's documented savefig behaviour (pixel size =
    figsize * dpi when bbox_inches is not 'tight'); this repo's own
    guidance to keep plot tests fast by not sampling inside them."""
    import matplotlib.image as mpimg
    paths = {p.name: p for p in write_regime(
        _rows(), (0.4, 0.6), 0.45, tmp_path, onsager=None)}
    regime_img = mpimg.imread(paths["regime.png"])
    diag_img = mpimg.imread(paths["diagnostics.png"])
    assert paths["regime.png"].stat().st_size > 0
    assert paths["diagnostics.png"].stat().st_size > 0
    # height, width, channels
    assert regime_img.shape[1] > diag_img.shape[1], \
        "the 3-panel regime figure must be wider than the 2-panel diagnostics figure"
