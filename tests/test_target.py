from tsu.target import Z1, IDEAL, Sourced, TargetProfile


def test_z1_degree_is_sixteen_and_sourced_to_a_fact_id():
    assert Z1.degree.value == 16
    assert Z1.degree.source == "F-14"


def test_z1_offsets_are_degree_sixteen_and_all_bipartite():
    offs = Z1.offsets.value
    assert len(offs) == 16, f"Z1 degree must be 16, got {len(offs)}"
    assert all((abs(dx) + abs(dy)) % 2 == 1 for dx, dy in offs), \
        "every Z1 offset must have odd L1 length, or the lattice is not bipartite"


def test_coupling_cap_is_marked_assumed_not_sourced():
    """Jmax is this project's working value, NOT an Extropic figure."""
    assert Z1.max_abs_coupling.source == "assumed"
    assert Z1.is_assumed("max_abs_coupling") is True
    assert Z1.is_assumed("degree") is False


def test_ideal_target_has_no_constraints():
    assert IDEAL.degree.value == float("inf")
    assert IDEAL.max_abs_coupling.value == float("inf")
    assert IDEAL.offsets.value == ()


def test_profiles_are_frozen():
    import pytest
    with pytest.raises(Exception):
        Z1.name = "not-z1"
