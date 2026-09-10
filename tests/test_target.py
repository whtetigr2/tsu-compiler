from dataclasses import fields

from tsu.target import Z1, IDEAL, Sourced, TargetProfile


def test_z1_degree_is_sixteen_and_sourced_to_a_fact_id():
    assert Z1.degree.value == 16
    assert Z1.degree.source == "F-14"


def test_z1_offsets_are_degree_sixteen_and_all_bipartite():
    offs = Z1.offsets.value
    assert len(offs) == 16, f"Z1 degree must be 16, got {len(offs)}"
    assert all((abs(dx) + abs(dy)) % 2 == 1 for dx, dy in offs), \
        "every Z1 offset must have odd L1 length, or the lattice is not bipartite"


# P-3/F-A5 + I-9a/F-R7: retired test_coupling_cap_is_marked_assumed_not_sourced,
# which asserted Z1.max_abs_coupling.source == "assumed" -- that is now known to
# be false. |J| <= 6.0 is an Extropic-documented Z1 hardware cap (the
# Thermalizers paper, 2608.01615v1.pdf, Fig. 12's cap-sweep axis annotates the
# value 6 as "(Z1)"), independently re-extracted and confirmed by A3/R9. It was
# disclaimed identically to the genuinely-assumed |b| cap only because
# TargetProfile had exactly one Sourced field (max_abs_coupling) feeding both
# gates.py's coupling_cap and field_cap checks -- see the two tests below,
# which replace it now that max_abs_bias exists as its own field.

def test_max_abs_coupling_is_sourced_to_the_thermalizers_cap_sweep_not_assumed():
    """|J| <= 6.0 is Extropic-documented, not a project assumption. The
    numeric VALUE is unchanged (still 6.0) -- only its provenance and how
    that provenance renders change."""
    assert Z1.max_abs_coupling.value == 6.0
    assert Z1.max_abs_coupling.source != "assumed"
    assert Z1.is_assumed("max_abs_coupling") is False
    assert Z1.is_assumed("degree") is False
    assert "2608.01615v1" in Z1.max_abs_coupling.source
    assert "6 (Z1)" in Z1.max_abs_coupling.source


def test_max_abs_bias_is_a_separate_genuinely_assumed_field():
    """|b| <= 6.0 (max_abs_bias) is the genuinely unsourced cap -- h_max is
    named symbolically in the Thermalizers paper (|h_i| <= h_max) but its
    numeric value appears nowhere in either primary source. Same numeric
    value as max_abs_coupling (6.0, unchanged), but now a SEPARATE Sourced
    field so the two caps' provenance can never again be conflated -- one
    shared field feeding both gates is exactly how P-3/I-9a happened."""
    assert Z1.max_abs_bias.value == 6.0
    assert Z1.max_abs_bias.source == "assumed"
    assert Z1.is_assumed("max_abs_bias") is True
    assert Z1.max_abs_bias is not Z1.max_abs_coupling


def test_target_profile_declares_max_abs_bias_as_its_own_dataclass_field():
    names = {f.name for f in fields(TargetProfile)}
    assert "max_abs_bias" in names
    assert "max_abs_coupling" in names


def test_ideal_target_has_no_constraints():
    assert IDEAL.degree.value == float("inf")
    assert IDEAL.max_abs_coupling.value == float("inf")
    assert IDEAL.max_abs_bias.value == float("inf")
    assert IDEAL.offsets.value == ()


def test_profiles_are_frozen():
    import pytest
    with pytest.raises(Exception):
        Z1.name = "not-z1"


def test_node_budget_is_the_exact_taped_out_269568_pbit_figure_not_the_approximate_250k():
    """WHAT THIS PINS (A-2, external review 2026-09-09): `node_budget` is
    269,568 -- the taped-out Z1 die's own exact pbit count, cited to
    `billion` ("From One to One Billion: Torx, Thermalizers, and Z1" -
    Extropic.pdf) p.7, Fig. 05, where BOTH the callout panel ("PBITS
    269,568 IN 8 CORES") and the figure caption ("The Z1 die: eight cores,
    269,568 pbits, ...") independently agree on the same number -- an
    unambiguous, citable figure (audit/provenance.md Part 1 row 9). This
    SUPERSEDES the older `thermalizers` p.5 figure ("~250,000 nodes"), a
    rounded estimate from a different document, used there only to size a
    per-iteration energy-cost calculation, never presented as a hardware
    spec (Finding P-1). The quote must still name the superseded figure
    explicitly, so a reader who encounters "~250,000"/"F-15" elsewhere in
    this project's history (e.g. an old committed receipt) is told which
    number is current and why it changed, rather than seeing the old
    number quietly vanish with no trace.

    HOW IT FAILS: a silent revert of the numeric literal back to 250_000
    (restoring the earlier conservative-but-superseded figure) fails the
    `== 269_568` assertion directly. Losing the citation to the newer
    document, or dropping the explicit mention of the superseded ~250,000
    figure from the quote (e.g. reverting to `Sourced(269_568, "F-15",
    "the entire chip has ~250,000 nodes")` -- an internally inconsistent
    value/quote pairing that would still pass a bare `value == 269_568`
    check), fails the citation/quote assertions below. This distinction
    matters operationally, not just editorially: 269,568 is LARGER than
    250,000, so reverting it lowers the node_budget gate's threshold and
    would newly start REJECTING (false FAIL) models between 250,000 and
    269,568 nodes that actually fit the real, taped-out die -- the failure
    mode a silent revert must be caught before it ships.
    PROVENANCE: audit/provenance.md Part 1 row 9 and "Finding P-1"; the
    primary source itself is `billion` p.7, Fig. 05 (not independently
    re-verified from this test, which trusts the ledger's own citation
    check -- see that document's own "Method" section for how it was
    verified against the extracted PDF text)."""
    nb = Z1.node_budget
    assert nb.value == 269_568
    assert nb.source != "F-15"
    assert "269,568" in nb.source or "Fig. 05" in nb.source
    assert "billion" in nb.source.lower()
    assert "269,568" in nb.quote
    assert "250,000" in nb.quote, \
        "the quote must still name the superseded approximate figure"
    assert "supersede" in nb.quote.lower()
    assert Z1.is_assumed("node_budget") is False


def test_node_budget_coupling_counts_are_untouched_by_the_a2_fix():
    """WHAT THIS PINS: A-2 changes `node_budget` (a pbit COUNT) only.
    TargetProfile carries no field for the die's coupling/coupler COUNT at
    all (`max_abs_coupling` is a |J| MAGNITUDE cap, a different quantity
    entirely) -- the same Fig. 05 that gives 269,568 pbits also gives two
    unreconciled coupling-count figures (2,135,904 couplers vs 215,904
    coupling parameters, audit/findings/R2.md, UNRESOLVED) that this task
    is explicitly out of scope for.
    HOW IT FAILS: this would fail if a future change added a coupling-COUNT
    field to TargetProfile seeded from either of those two unreconciled
    Fig. 05 numbers without first resolving R2.md -- there is no such field
    today, which this test pins by enumerating exactly what IS present."""
    from dataclasses import fields
    names = {f.name for f in fields(Z1)}
    assert "max_abs_coupling" in names          # |J| magnitude cap -- untouched
    assert not any("coupler" in n or "n_coupling" in n for n in names)
