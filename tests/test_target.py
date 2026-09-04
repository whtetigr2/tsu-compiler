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
