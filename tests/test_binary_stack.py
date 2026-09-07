"""Task 4 of the all-bipartite-world plan
(SPR/docs/superpowers/plans/2026-09-07-all-bipartite-world.md): composing
the three all-binary-stack layers (Task 2's specs/binary_stack_l0/l1/l2.yaml)
into one of 8 distinguishable cell states.

`compose_state` is a pure function of three already-decoded 0/1 layer
values -- no sampling, no compilation, no thrml/torx involved -- so this
file tests only that composition, per Task 4's own file-responsibility
table ("composition and decode logic"). The violation-rate MEASUREMENT
(Task 4 Steps 4-6) is a separate, sampling-based script
(demo/binary_world.py's own __main__), not a pytest test -- it calls
place()/thrml_sample() against real compiled programs, which is
measurement work, not composition-logic verification.
"""
import sys

sys.path.insert(0, "demo")


def test_three_binary_layers_compose_to_eight_states():
    """Three binary layers give 2^3 = 8 distinguishable states per cell --
    more than the k=3 base they replace. The composition is a pure function
    of the three decoded layers, testable without any sampling."""
    from binary_world import compose_state
    assert compose_state([0, 0, 0]) == 0
    assert compose_state([1, 0, 0]) == 1
    assert compose_state([1, 1, 1]) == 7


def test_compose_state_covers_every_one_of_the_eight_combinations():
    """Every one of the 2**3 = 8 bit combinations decodes to a DISTINCT
    state 0..7 -- not just the three combinations the plan's own snippet
    happens to name. bits[0] is least-significant (matches
    specs/binary_stack_l0.yaml being "bit 0" in that file's own docstring)."""
    from binary_world import compose_state
    seen = set()
    for b2 in (0, 1):
        for b1 in (0, 1):
            for b0 in (0, 1):
                state = compose_state([b0, b1, b2])
                assert state == b0 + 2 * b1 + 4 * b2
                seen.add(state)
    assert seen == set(range(8))


def test_compose_state_rejects_a_non_binary_or_wrong_length_input():
    """compose_state is a decode step for exactly three binary layers --
    silently accepting a 2-bit or 4-bit list, or a non-0/1 value, would
    launder a caller's bug (e.g. forgetting one layer) into a wrong-but-
    plausible-looking state instead of an error."""
    import pytest
    from binary_world import compose_state
    with pytest.raises(ValueError):
        compose_state([0, 1])
    with pytest.raises(ValueError):
        compose_state([0, 1, 2])
