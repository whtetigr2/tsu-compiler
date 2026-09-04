"""Task B1 (capacity frontier) + B2 (regime/traces, pure math only).

B1's arithmetic is TDD'd against the spec's own recorded measurement table
(`2026-08-28-lattice-design.md` sections 4.2 and 4.8) BEFORE it is trusted
for anything else -- every number in that table is a MEASUREMENT the spec
already made; these tests only check that `predicted_degree_one_hot`/
`predicted_field_one_hot_floor` reproduce it as arithmetic.

B2's pure math (Onsager beta_c siting and the trace ring buffer) lives in
demo/lattice_app.py per the brief ("extend demo/lattice_app.py; test any
pure math in tests/test_frontier.py") -- imported here, not reimplemented.
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import frontier as fr  # noqa: E402
import lattice_app as la  # noqa: E402
import theme  # noqa: E402 -- Task 11: gauge pixel-colour tests


# ===========================================================================
# B1 Step 1: TDD the frontier arithmetic against the spec's measured table.
# ===========================================================================

# ---------------------------------------------------------------------------
# Degree law (spec section 4.2): degree(v) = (k-1) + g*|partners(v)|, g=4.
# "Confirmed on five independent configurations, three of them as
# predictions made before measurement" -- the table itself, reproduced here.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,partners,expected", [
    (5, 3, 16),   # 8 rules, k=5
    (5, 3, 16),   # 10 rules arranged for p<=3 (same k,p; law is a function
                  # of k and p alone, so it must predict the same number)
    (5, 4, 20),   # 12 rules, p=4
    (4, 2, 11),   # L0 elevation k=4, gradual only
    (4, 3, 15),   # L0 elevation k=4, +smoothing
])
def test_predicted_degree_one_hot_matches_the_spec_table(k, partners, expected):
    assert fr.predicted_degree_one_hot(k, partners) == expected


def test_predicted_degree_increment_is_flat_g_regardless_of_starting_point():
    """p -> p+1 on an existing value: the law predicts +g no matter what k
    or the starting p is (spec section 4.2 -- the partner term is linear
    in p with slope g)."""
    assert fr.predicted_degree_increment_one_hot() == fr.GRID_DEGREE == 4


def test_degree_law_is_size_independent_in_partners_alone():
    """Spec: 'Degree is size-independent: 16 at 3x3, 4x4 and 5x5 alike' --
    the law has no width/height term at all, only k and partners."""
    assert fr.predicted_degree_one_hot(5, 3) == fr.predicted_degree_one_hot(5, 3)


# ---------------------------------------------------------------------------
# Field law (spec section 4.8): P*(k-2)/2, a LOWER bound on |b|max.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("penalty,k,expected", [
    (10.0, 5, 15.00),
    (4.0, 5, 6.00),
    (3.0, 5, 4.50),
    (2.0, 5, 3.00),
])
def test_predicted_field_floor_matches_the_spec_table(penalty, k, expected):
    assert fr.predicted_field_one_hot_floor(penalty, k) == pytest.approx(expected)


def test_predicted_field_floor_at_k3_is_5_as_stated_in_the_brief():
    """'at P=10, k=5 that is 15 against an assumed cap of 6.0; at k=3 it is
    5.0' -- the brief's own worked example."""
    assert fr.predicted_field_one_hot_floor(10.0, 3) == pytest.approx(5.0)


def test_predicted_field_floor_is_a_lower_bound_not_an_exact_value():
    """Spec's own table: observed ran ABOVE the floor at every P tried
    (15.00->15.31, 6.00->6.31, 4.50->4.81, 3.00->3.19) -- this function
    must document (and this test locks in) that it is the floor, not the
    exact figure; callers must never treat it as an equality."""
    floor = fr.predicted_field_one_hot_floor(10.0, 5)
    observed_examples = (15.31, 16.25)
    assert all(o >= floor for o in observed_examples)


# ===========================================================================
# B1 Step 2: headroom display, model shape, verification against real
# encodes of demo/receipts/small.
# ===========================================================================

SMALL = REPO_ROOT / "demo" / "receipts" / "small"


def test_headroom_from_receipt_matches_small_receipts_own_gates_and_metrics():
    import json
    gates = json.loads((SMALL / "gates.json").read_text())
    metrics = json.loads((SMALL / "metrics.json").read_text())
    headroom = fr.headroom_from_receipt(gates, metrics)
    by_gate = {h.gate: h for h in headroom}

    assert by_gate["degree"].measured == 9
    assert by_gate["degree"].limit == 16
    assert by_gate["degree"].pct_used == pytest.approx(100.0 * 9 / 16)

    assert by_gate["coupling_cap"].measured == pytest.approx(2.5)
    assert by_gate["coupling_cap"].limit == pytest.approx(6.0)

    assert by_gate["field_cap"].measured == pytest.approx(1.6)
    assert by_gate["field_cap"].limit == pytest.approx(6.0)

    # node_budget's MEASURED side is the receipt's final placed node count
    # (metrics.json n_nodes = 192, post-mediation), not gates.json's own
    # pre-mediation node_budget row (128) -- see headroom_from_receipt's
    # own docstring for why.
    assert by_gate["node_budget"].measured == 192
    assert by_gate["node_budget"].limit == 250000


def test_gate_headroom_describe_reads_as_x_of_y_pct_used():
    h = fr.GateHeadroom("degree", 9, 16, 100.0 * 9 / 16)
    text = h.describe()
    assert "9" in text and "16" in text and "%" in text


def test_pct_used_is_none_for_a_non_finite_limit():
    h = fr.GateHeadroom("ideal_degree", 3, float("inf"), None)
    assert "%" not in h.describe()


def test_read_model_shape_matches_lattice_smalls_own_rules():
    """lattice_small_8x8_k3.yaml: water(0)-rock(1), water-water, rock-rock,
    grass(2)-grass. water and rock each partner {0,1} (p=2); grass partners
    only {2} (p=1) -- hand-verified against the spec.yaml term list."""
    from tsu.spec import load_spec
    spec = load_spec(str(SMALL / "spec.yaml"))
    shape = fr.read_model_shape(spec)
    assert shape.k == 3
    assert shape.partners[0] == frozenset({0, 1})
    assert shape.partners[1] == frozenset({0, 1})
    assert shape.partners[2] == frozenset({2})
    assert shape.worst_partners == 2
    assert shape.worst_value in (0, 1)


def test_observe_domain_wall_k3_matches_the_receipts_own_measured_numbers():
    """The receipt's own compile measured degree=9, |J|=2.5, |b|=1.6 for
    this exact spec under domain_wall -- `observe` must reproduce it
    exactly (same encode/lower/analyse pipeline the compiler itself runs)."""
    from tsu.spec import load_spec
    spec = load_spec(str(SMALL / "spec.yaml"))
    rep = fr.observe(spec, "domain_wall", 1.0)
    assert rep.max_degree == 9
    assert rep.max_abs_J == pytest.approx(2.5)
    assert rep.max_abs_b == pytest.approx(1.6)
    assert rep.n_nodes == 128


def test_verify_k_increment_on_one_hot_matches_the_law_exactly():
    """one_hot is the encoding the law was confirmed on -- verifying it on
    THIS model (not just re-citing the spec's own past table) must match
    the law's prediction exactly, not merely approximately."""
    from tsu.spec import load_spec
    spec = load_spec(str(SMALL / "spec.yaml"))
    shape = fr.read_model_shape(spec)
    v = fr.verify_k_increment(spec, shape, "one_hot", fr.ONE_HOT_PENALTY, 1.0)
    assert v.degree_matches, (
        f"one-hot law predicted degree {v.predicted_degree}, observed "
        f"{v.observed_degree} -- the law should hold exactly for one_hot")
    assert v.field_at_or_above_floor


def test_verify_p_increment_on_one_hot_matches_the_law_exactly():
    from tsu.spec import load_spec
    spec = load_spec(str(SMALL / "spec.yaml"))
    shape = fr.read_model_shape(spec)
    v = fr.verify_p_increment(spec, shape, "one_hot", 1.0)
    assert v.degree_matches


def test_verify_k_increment_on_domain_wall_diverges_from_the_one_hot_law():
    """This is the honest, reported divergence: domain_wall (this receipt's
    OWN encoding) does not follow the one-hot degree law -- confirmed
    directly (not just cited from a prior report) by actually re-encoding.
    A future maintainer changing encode.py's domain_wall structure and
    making this test fail should treat that as NEWS, not tighten it back to
    'passes' without re-measuring."""
    from tsu.spec import load_spec
    spec = load_spec(str(SMALL / "spec.yaml"))
    shape = fr.read_model_shape(spec)
    v = fr.verify_k_increment(spec, shape, "domain_wall", fr.ONE_HOT_PENALTY, 1.0)
    assert not v.degree_matches, (
        "domain_wall was expected to diverge from the one-hot law here "
        "(task-3-report.md); if this now matches, the divergence this "
        "module documents no longer holds and the report text must change")


def test_binding_forecast_predicts_field_cap_first_for_small_receipt():
    """At k=3, P=10: field floor is already 5.0 (k=3) and crosses the 6.0
    cap at k=4 (10.0), only one increment away -- long before the degree
    law would predict a failure (k=10, or p=4). This directly reproduces
    the project's own established finding (spec 4.8/4.9: the field gate,
    not degree, is what actually blocks larger k)."""
    from tsu.spec import load_spec
    from tsu.target import Z1
    spec = load_spec(str(SMALL / "spec.yaml"))
    shape = fr.read_model_shape(spec)
    forecast = fr.predict_first_binding_gate(
        shape, fr.ONE_HOT_PENALTY, Z1.degree.value, Z1.max_abs_coupling.value)
    assert forecast.field_binds_at_k == 4
    assert forecast.degree_binds_at_p == 4
    assert forecast.degree_binds_at_k == 10
    assert "field_cap" in forecast.headline


def test_build_frontier_report_runs_end_to_end_on_the_small_receipt():
    report = fr.build_frontier_report(SMALL)
    assert report.encoding == "domain_wall"
    assert report.shape.k == 3
    assert len(report.headroom) == 4
    assert len(report.verified) == 4  # domain_wall k/p + one_hot k/p
    text = fr.render_text(report)
    assert "FRONTIER" in text
    assert "domain_wall" in text


# ===========================================================================
# B2: Onsager beta_c siting and the trace ring buffer (pure math only --
# the Tk canvas plots themselves are verified by inspection per the brief).
# ===========================================================================

def test_onsager_betac_matches_the_textbook_constant():
    """betac = ln(1+sqrt(2)) / (2 * |J|max) -- the standard Onsager critical
    coupling for the UNIFORM 2-D square-lattice Ising model (sinh(2*Kc)=1,
    Kc = arcsinh(1) = ln(1+sqrt(2))). Computed from the formula every time,
    never a hardcoded decimal, so a transcription slip can't silently ship."""
    j_max = 2.5
    betac = la.onsager_betac(j_max)
    assert betac == pytest.approx(math.log(1 + math.sqrt(2)) / 2 / j_max)


def test_onsager_betac_scales_inversely_with_j_max():
    assert la.onsager_betac(1.0) == pytest.approx(2 * la.onsager_betac(2.0))


def test_onsager_betac_rejects_nonpositive_j_max():
    with pytest.raises(ValueError):
        la.onsager_betac(0.0)
    with pytest.raises(ValueError):
        la.onsager_betac(-1.0)


def test_beta_regime_reports_beta_betac_and_ratio():
    regime = la.beta_regime(beta=1.0, j_max=2.5)
    assert regime.beta == pytest.approx(1.0)
    assert regime.betac == pytest.approx(math.log(1 + math.sqrt(2)) / 2 / 2.5)
    assert regime.ratio == pytest.approx(regime.beta / regime.betac)
    assert "not the uniform" in regime.assumption_note.lower() \
        or "not a derived" in regime.assumption_note.lower()


def test_trace_ring_buffer_bounds_length():
    t = la.Trace(maxlen=5)
    for i in range(10):
        t.append(i, float(i))
    assert len(t) == 5
    assert list(t.xs) == [5, 6, 7, 8, 9]


def test_trace_bounds_reports_axis_range():
    t = la.Trace(maxlen=10)
    assert t.bounds() is None
    t.append(0, 3.0)
    t.append(1, -2.0)
    t.append(2, 7.0)
    assert t.bounds() == (0, 2, -2.0, 7.0)


# ---------------------------------------------------------------------------
# I-2/F-R6 + F1/F-R10: the continuity-implying polyline. `_draw_trace`
# (energy) and `render_line_plot` (magnetization) each used to draw ONE
# connected polyline through the whole trace regardless of where the data
# actually came from -- but SampleWorker's own draws are chain-major
# flattened (a clamped batch) or restart-interleaved (every unclamped
# tick), so consecutive trace points are routinely NOT successive draws of
# one physical chain. A connected line asserts continuous single-trajectory
# dynamics that do not exist. Trace.chain_break is the boundary metadata
# both renderers now read instead of each re-deriving (or forgetting to
# derive -- see F-R10's own history) what counts as "the same chain".
# ---------------------------------------------------------------------------

def test_trace_append_defaults_chain_break_true():
    """The safe default: a point appended with no explicit provenance must
    NOT be assumed continuous with the point before it -- a caller that
    forgets to state chain_break must never silently get a connected line
    (the exact failure mode this whole fix exists to remove)."""
    t = la.Trace(maxlen=10)
    t.append(0, 1.0)
    assert list(t.chain_breaks) == [True]


def test_trace_append_records_explicit_chain_break_value():
    t = la.Trace(maxlen=10)
    t.append(0, 1.0, chain_break=True)
    t.append(1, 2.0, chain_break=False)
    assert list(t.chain_breaks) == [True, False]


def test_trace_chain_breaks_ring_buffer_truncates_alongside_xs_ys():
    t = la.Trace(maxlen=3)
    for i in range(5):
        t.append(i, float(i), chain_break=(i % 2 == 0))
    assert list(t.xs) == [2, 3, 4]
    assert list(t.chain_breaks) == [True, False, True]


def test_trace_runs_groups_points_into_contiguous_connectable_segments():
    """The ONE shared boundary-computation helper both _draw_trace and
    render_line_plot call, rather than each re-deriving the run-grouping
    logic separately -- that duplication-without-a-shared-helper is
    literally how F-R10 came to exist (Wave 2 fixed _draw_trace's version
    of this bug; nobody checked whether render_line_plot had its own copy
    until Wave 4). chain_break=True at index i means index i must not be
    connected to index i-1; index 0 always starts a run regardless of its
    own flag (there is nothing before it to connect to)."""
    assert la.trace_runs([True, False, False, True, False]) == [(0, 3), (3, 5)]
    assert la.trace_runs([True, True, True]) == [(0, 1), (1, 2), (2, 3)]
    assert la.trace_runs([False, False, False]) == [(0, 3)]
    assert la.trace_runs([]) == []


def test_render_line_plot_never_connects_across_a_chain_break():
    """F1/F-R10: render_line_plot (the magnetization trace's own renderer)
    must not draw a connecting line between the last point of one physical
    chain and the first point of the next. Two chains of two points each,
    at opposite ends of the fixed [-1, 1] y range, so the OLD single
    polyline (connecting all 4 points in arrival order) would pass directly
    through the plot's centre; that pixel must stay background once the
    break between chains is honoured."""
    xs = [0, 1, 10, 11]
    ys = [-1.0, -1.0, 1.0, 1.0]
    chain_breaks = [True, False, True, False]
    img = la.render_line_plot(300, 150, xs, ys, "draw", "M",
                              y_range=(-1.0, 1.0), chain_breaks=chain_breaks)
    # Geometry replicated from render_line_plot's own pad_l=40/pad_r=8/
    # pad_t=8/pad_b=16 constants: px(5.5)=166, py(0.0)=71 is the midpoint
    # of the (never-drawn) segment connecting the two chains.
    cx, cy = 166, 71
    found_accent = any(
        _close(img.getpixel((cx + dx, cy + dy)), la.PLOT_ACCENT, tol=10)
        for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2))
    assert not found_accent, (
        "a line was drawn connecting two different chains across the "
        "plot's centre -- F1/F-R10 regression")


def test_render_line_plot_still_connects_within_one_chain():
    """The fix must not degrade into scatter-only: two points in the SAME
    run (chain_break=False on the second) still get a connecting line --
    otherwise a genuine, honestly-continuous clamped-batch chain would be
    drawn with no more information than restart-interleaved noise."""
    xs = [0, 1]
    ys = [-1.0, 1.0]
    img = la.render_line_plot(300, 150, xs, ys, "draw", "M",
                              y_range=(-1.0, 1.0), chain_breaks=[True, False])
    # Only 2 points -> xmin=0, xmax=1, xspan=1: px(0.5) = 40 + 0.5/1*252 =
    # 166; py(0.0) = 71 (same y math as the test above). The midpoint of
    # the ONE genuine segment.
    cx, cy = 166, 71
    found_accent = any(
        _close(img.getpixel((cx + dx, cy + dy)), la.PLOT_ACCENT, tol=10)
        for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2))
    assert found_accent, "a genuine within-chain segment was not drawn"


def test_energy_of_draw_matches_hand_computed_ising_energy():
    """E = offset - sum(b*s) - sum(J*s_u*s_v), s = 2*bit-1 (the same sign
    convention IsingModel/lower.py document: 'sum b s + sum J s s == -E')."""
    from tsu.passes.lower import IsingModel
    import numpy as np
    im = IsingModel(nodes=("a", "b"), edges=((0, 1),),
                    weights=np.array([2.0]), biases=np.array([1.0, -1.0]),
                    beta=1.0, offset=0.5)
    # bits (1, 0) -> spins (+1, -1)
    row = np.array([1, 0])
    # -b.s = -(1*1 + (-1)*-1) = -(1+1) = -2; -J*s0*s1 = -2*(1*-1) = 2
    # E = offset - b.s - J*s0*s1 ... using the module's own convention:
    expected = im.offset - (im.biases[0]*1 + im.biases[1]*-1) - (im.weights[0]*1*-1)
    assert la.energy_of_draw(im, row) == pytest.approx(expected)


# ===========================================================================
# Task 11: FRONTIER redesign -- load gauges. frontier_gauge_specs is
# PRESENTATION ONLY (no new arithmetic): these tests lock in that every
# number on demo/receipts/small's own report survives unchanged into the
# gauges, and that the two SELECTIONS (highest current utilisation,
# predicted-to-bind-first) land on the gates the underlying report already
# implies -- degree is highest currently used (56.25%), field_cap is what
# BindingForecast's own headline names as binding first.
# ===========================================================================

def _small_report():
    return fr.build_frontier_report(SMALL)


def test_frontier_gauge_specs_preserves_every_headroom_number_unchanged():
    """No new maths -- Step 1 of the brief. Every measured/limit/pct_used
    on each GaugeSpec must be the EXACT same value headroom_from_receipt
    already produced, not a re-derived one."""
    report = _small_report()
    specs = la.frontier_gauge_specs(report)
    by_gate = {s.gate: s for s in specs}
    for h in report.headroom:
        s = by_gate[h.gate]
        assert s.measured == h.measured
        assert s.limit == h.limit
        assert s.pct_used == h.pct_used


def test_frontier_gauge_specs_covers_all_four_gates_in_headroom_order():
    report = _small_report()
    specs = la.frontier_gauge_specs(report)
    assert [s.gate for s in specs] == [h.gate for h in report.headroom]
    assert [s.gate for s in specs] == ["degree", "coupling_cap", "field_cap", "node_budget"]


def test_frontier_gauge_specs_tags_degree_as_highest_current_utilisation():
    """56.25% (degree) > 41.67% (coupling_cap) > 26.67% (field_cap) >
    0.08% (node_budget) on the small receipt -- degree is the highest
    CURRENT utilisation, even though it is not what's predicted to bind
    first (see the next test)."""
    report = _small_report()
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    assert by_gate["degree"].tag_kind == "highest_util"
    assert by_gate["degree"].tag_text == "HIGHEST CURRENT UTILISATION"


def test_frontier_gauge_specs_tags_field_cap_as_predicted_to_bind_first():
    """Matches report.binding.headline (fr.predict_first_binding_gate):
    field_cap crosses its cap at k=4, one increment away -- long before
    degree would (k=10 or p=4)."""
    report = _small_report()
    assert "field_cap" in report.binding.headline
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    assert by_gate["field_cap"].tag_kind == "bind_first"
    assert by_gate["field_cap"].tag_text == "PREDICTED TO BIND FIRST"
    assert by_gate["field_cap"].predicted_exceeds_cap


def test_frontier_gauge_specs_keeps_both_tags_when_one_gate_earns_both():
    """Minor #5 (final review): frontier_gauge_specs used to be if/elif
    over (bind-first, highest-utilisation, unbound), which silently
    DROPPED the highest-utilisation tag whenever a single gate was both
    the one predicted to bind first AND the one with the highest current
    utilisation -- neither real receipt happens to exercise that overlap,
    which is exactly why it went unnoticed. Force the overlap directly by
    monkeypatching a `report` whose headroom/binding agree on the same
    gate, and check BOTH labels survive in tag_text, not just one."""
    report = _small_report()
    # field_cap is the real report's bind-first gate; give it the highest
    # pct_used too so highest-utilisation ALSO selects it.
    headroom = list(report.headroom)
    by_gate = {h.gate: i for i, h in enumerate(headroom)}
    fc = headroom[by_gate["field_cap"]]
    headroom[by_gate["field_cap"]] = fr.GateHeadroom(
        fc.gate, fc.measured, fc.limit, 99.0)  # highest pct_used by far
    forced = replace(report, headroom=headroom)
    by_gate2 = {s.gate: s for s in la.frontier_gauge_specs(forced)}
    spec = by_gate2["field_cap"]
    assert "PREDICTED TO BIND FIRST" in spec.tag_text
    assert "HIGHEST CURRENT UTILISATION" in spec.tag_text
    assert spec.tag_kind == "bind_first"


def test_frontier_gauge_specs_node_budget_tagged_unbound_and_untagged_by_utilisation():
    """node_budget's 0.08% is nowhere near the highest (degree's 56.25%
    is), so it must get its OWN "unbound" tag, not the utilisation one."""
    report = _small_report()
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    assert by_gate["node_budget"].tag_kind == "ok"
    assert by_gate["node_budget"].tag_text == "UNBOUND ON Z1-CLASS"


def test_frontier_gauge_specs_coupling_cap_has_no_predicted_value():
    """frontier.py has no named law predicting coupling_cap's next value --
    a gauge must never fabricate a hairline/overrun frontier.py itself
    never computed."""
    report = _small_report()
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    assert by_gate["coupling_cap"].predicted_value is None
    assert by_gate["coupling_cap"].predicted_label is None
    assert not by_gate["coupling_cap"].predicted_exceeds_cap


def test_frontier_gauge_specs_degree_predicted_value_matches_report_field():
    report = _small_report()
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    assert by_gate["degree"].predicted_value == report.law_predicted_degree_next_k
    assert not by_gate["degree"].predicted_exceeds_cap  # 11 <= 16


def test_frontier_gauge_specs_field_cap_predicted_value_matches_report_field():
    report = _small_report()
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    assert by_gate["field_cap"].predicted_value == report.law_predicted_field_floor_next_k
    assert by_gate["field_cap"].predicted_exceeds_cap  # 10.0 > 6.0


def test_frontier_gauge_specs_predicted_labels_name_their_law_never_bare():
    """Every prediction must stay labelled as a prediction from a named
    law, never presented as a measurement (the panel's entire credibility,
    per the brief) -- both predicted gauges must carry a non-empty
    predicted_law naming the spec section."""
    report = _small_report()
    for s in la.frontier_gauge_specs(report):
        if s.predicted_value is not None:
            assert s.predicted_law and "section" in s.predicted_law
            assert s.predicted_label and (
                s.gate.replace("_cap", "") in s.predicted_label
                or "degree" in s.predicted_label
                or "floor" in s.predicted_label)


def test_frontier_gauge_specs_assumed_cap_note_appears_for_coupling_and_field():
    """|J| <= 6.0 and |b| <= 6.0 are ASSUMED project values, not sourced
    Extropic figures -- every gauge that shows either cap must say so."""
    report = _small_report()
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    assert "assumed project value" in by_gate["coupling_cap"].foot_text.lower()
    assert "assumed project value" in by_gate["field_cap"].foot_text.lower()


def test_binding_gate_name_matches_the_reports_own_headline_exactly():
    report = _small_report()
    assert la._binding_gate_name(report.binding.headline, report.headroom) == "field_cap"


def test_binding_gate_name_is_none_when_headline_predicts_no_binding_gate():
    headroom = [fr.GateHeadroom("degree", 1, 16, 6.25),
               fr.GateHeadroom("field_cap", 1, 6.0, 16.7)]
    headline = ("under the one-hot law, no gate is predicted to bind "
               "within 128 increments of k or p")
    assert la._binding_gate_name(headline, headroom) is None


def test_safe_frac_clamps_and_never_raises_on_degenerate_limits():
    assert la._safe_frac(9, 16) == pytest.approx(9 / 16)
    assert la._safe_frac(20, 16) == 1.0    # clamped, never > 1
    assert la._safe_frac(5, 0) == 0.0      # zero limit -> 0, never ZeroDivisionError
    assert la._safe_frac(5, float("inf")) == 0.0
    assert la._safe_frac(5, float("nan")) == 0.0


def test_gauge_spec_frac_current_and_frac_predicted_properties():
    report = _small_report()
    by_gate = {s.gate: s for s in la.frontier_gauge_specs(report)}
    degree = by_gate["degree"]
    assert degree.frac_current == pytest.approx(9 / 16)
    assert degree.frac_predicted == pytest.approx(11 / 16)
    coupling = by_gate["coupling_cap"]
    assert coupling.frac_predicted is None  # no predicted value at all


# ---------------------------------------------------------------------------
# render_frontier_gauge_track -- the IMAGE track itself. Pixel-level checks,
# same convention as render_temperature_track's own tests
# (tests/test_lattice_app_logic.py): sample real pixels, don't just check
# the image doesn't crash.
# ---------------------------------------------------------------------------

def _hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _close(a, b, tol=24):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_gauge_track_is_the_requested_size():
    img = la.render_frontier_gauge_track(300, 14, 0.5)
    assert img.size == (300, 14)


def test_gauge_track_fill_reaches_frac_current_and_stops():
    img = la.render_frontier_gauge_track(300, 14, 0.5)
    mid_y = 7
    filled_px = img.getpixel((100, mid_y))     # inside the 50% fill
    empty_px = img.getpixel((250, mid_y))       # past the fill, inside the track
    assert _close(filled_px, _hexrgb(theme.GOLD_DIM))
    assert not _close(empty_px, _hexrgb(theme.GOLD_DIM))


def test_gauge_track_predicted_hairline_is_gold_and_distinct_from_fill():
    """The predicted hairline must actually BE a dashed GOLD line, not
    merely "some pixels changed at that column" (which would also pass for
    a hairline drawn in the fill colour, the background colour, or solid
    rather than dashed). Sample the column at the predicted fraction (0.6,
    well past the 0.2 current fill so no pixel there can be GOLD_DIM by
    construction) and check: at least one pixel is GOLD (the dash strokes),
    at least one pixel is NOT GOLD (the dash gaps -- i.e. actually dashed,
    not a solid fill), and no pixel is GOLD_DIM (distinct from the measured
    fill colour)."""
    predicted = la.render_frontier_gauge_track(300, 14, 0.2, frac_predicted=0.6)
    x = int(round(0.6 * 299))
    col = [predicted.getpixel((x, y)) for y in range(14)]
    gold = _hexrgb(theme.GOLD)
    gold_dim = _hexrgb(theme.GOLD_DIM)
    assert any(_close(px, gold, tol=8) for px in col), \
        "the predicted hairline must draw GOLD pixels in its own column"
    assert any(not _close(px, gold, tol=8) for px in col), \
        "the predicted hairline must be DASHED (gaps), not a solid line"
    assert all(not _close(px, gold_dim, tol=8) for px in col), \
        "the predicted hairline must never be the same colour as the fill"


def test_gauge_track_overrun_draws_a_hatched_red_region_past_the_compressed_track():
    """When overrun_ratio is given, the track compresses to
    GAUGE_OVERRUN_TRACK_FRAC and a red-ish hatched region fills the rest --
    must be visually distinct from the gold fill (never a second gold bar)."""
    img = la.render_frontier_gauge_track(300, 18, 0.3, overrun_ratio=1.67)
    mid_y = 9
    track_w = int(round(300 * la.GAUGE_OVERRUN_TRACK_FRAC))
    hatch_px = img.getpixel((track_w + 10, mid_y))
    fill_px = img.getpixel((30, mid_y))
    assert not _close(hatch_px, _hexrgb(theme.GOLD_DIM))
    assert not _close(fill_px, hatch_px, tol=10)


def test_gauge_track_no_overrun_never_draws_past_full_width():
    """Sanity: the plain (non-overrun) branch uses the FULL image width as
    the track, not the compressed GAUGE_OVERRUN_TRACK_FRAC -- so the fill at
    frac_current=0.9 should extend almost to the right edge, not stop at
    62% of it."""
    img = la.render_frontier_gauge_track(300, 14, 0.9)
    mid_y = 7
    # fill extends to ~0.9*300=270px; sample just inside that, well past
    # where GAUGE_OVERRUN_TRACK_FRAC (62%, ~186px) would have cut it off.
    near_edge_px = img.getpixel((260, mid_y))
    assert _close(near_edge_px, _hexrgb(theme.GOLD_DIM))
