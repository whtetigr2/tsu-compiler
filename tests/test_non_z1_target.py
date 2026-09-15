"""Are the gate and mediation layers really profile-driven?

The README claims the gate and mediation layers "have been tested working
against a non-Z1 target". That claim was written after testing it interactively
in a session, with no committed test behind it -- a claims-vs-evidence audit
found the gap the same day. This file is the evidence the sentence asserts.

The target below is DELIBERATELY NOT NAMED after any real device. Inventing
plausible-looking specifications for hardware nobody has sourced is the exact
failure this project refuses everywhere else; the point here is only that the
passes read the profile rather than Z1 constants baked into them.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu_compiler.gates import gate_checks
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.route import insert_mediators
from tsu_compiler.target import PROFILES, Sourced, TargetProfile

# Every limit differs from Z1's, so a pass that ignored the profile and used a
# Z1 constant would produce visibly wrong numbers rather than passing by luck.
HYPOTHETICAL = TargetProfile(
    name="hypothetical",
    degree=Sourced(8, "test"),                    # Z1: 16
    offsets=Sourced((), "test"),                  # Z1: 16 lattice offsets
    bipartite=Sourced(False, "test"),
    schedule=Sourced("chromatic_block_gibbs", "test"),
    max_abs_coupling=Sourced(2.0, "test"),        # Z1: 6.0
    max_abs_bias=Sourced(1.0, "test"),            # Z1: 6.0
    coupling_bits=Sourced(4, "test"),
    node_budget=Sourced(1000, "test"),            # Z1: 269,568
    connected_fabric=Sourced(True, "test"),
    coupling_parameters=Sourced(500, "test"),     # Z1: 215,904
    per_edge_independent_J=Sourced(True, "test"),
)


def odd_cycle_grid(n: int = 12, weight: float = 1.5) -> IsingModel:
    """A 4-neighbour grid plus one chord, which makes it non-bipartite and so
    forces mediation to do real work."""
    idx = {(x, y): y * n + x for y in range(n) for x in range(n)}
    e = [(i, idx[(x + dx, y + dy)])
         for (x, y), i in idx.items()
         for dx, dy in ((1, 0), (0, 1)) if (x + dx, y + dy) in idx]
    e.append((0, idx[(2, 2)]))
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n * n)), edges=tuple(e),
                      weights=np.full(len(e), weight), biases=np.zeros(n * n),
                      beta=1.0, offset=0.0)


def test_gates_evaluate_against_the_profile_they_are_given():
    """WHAT THIS PINS: `gate_checks` reads its limits from the TargetProfile
    argument, not from Z1 constants. This is what makes "a hardware target is
    declarative" true rather than aspirational, and it is the whole basis for
    the claim that retargeting is writing a profile.

    HOW IT FAILS: hardcode any limit in gates.py -- say `16` for degree instead
    of `target.degree.value` -- and the degree gate here reports a limit of 16
    against a profile that says 8. Nothing else in the suite would catch it,
    because every other test uses z1 or ideal, where a Z1 constant and the
    profile value coincide.

    PROVENANCE: the limits asserted below are the HYPOTHETICAL profile's own
    fields, read back; the test compares the gate's reported limit to the
    profile, never to a number typed twice."""
    model = odd_cycle_grid()
    checks = {g.gate: g for g in gate_checks(model, analyse(model),
                                             HYPOTHETICAL, False)}
    assert checks["degree"].limit == HYPOTHETICAL.degree.value
    assert checks["coupling_cap"].limit == HYPOTHETICAL.max_abs_coupling.value
    assert checks["field_cap"].limit == HYPOTHETICAL.max_abs_bias.value
    assert checks["node_budget"].limit == HYPOTHETICAL.node_budget.value
    # and none of them accidentally reports a Z1 number
    z1 = PROFILES["z1"]
    assert checks["degree"].limit != z1.degree.value
    assert checks["coupling_cap"].limit != z1.max_abs_coupling.value


def test_a_cap_that_only_the_hypothetical_target_violates_actually_fails():
    """WHAT THIS PINS: the profile's limits are enforced, not merely reported.
    The fixture uses |J| = 2.5, which is legal on Z1 (cap 6.0) and illegal on
    this target (cap 2.0), so the two profiles give OPPOSITE verdicts.

    HOW IT FAILS: evaluate the cap against Z1's 6.0 and this passes, because
    2.5 is legal there. The fixture is chosen so the two profiles DISAGREE --
    a gate reading the wrong profile gives the wrong verdict, not a wrong
    number that still happens to pass.

    PROVENANCE: 2.5 sits between this target's 2.0 cap and Z1's 6.0, so the
    verdict is determined entirely by which profile was consulted."""
    model = odd_cycle_grid(weight=2.5)
    rep = analyse(model)
    hypo = {g.gate: g for g in gate_checks(model, rep, HYPOTHETICAL, False)}
    z1 = {g.gate: g for g in gate_checks(model, rep, PROFILES["z1"], False)}
    assert hypo["coupling_cap"].passed is False, "2.5 exceeds this target's 2.0"
    assert z1["coupling_cap"].passed is True, "2.5 is legal on Z1's 6.0"


def test_mediation_is_graph_theoretic_and_needs_no_z1_at_all():
    """WHAT THIS PINS: `insert_mediators` fixes bipartite parity using only the
    graph, so it works identically whatever hardware is targeted -- and takes no
    target argument at all, which is the strongest form of the claim.

    HOW IT FAILS: introduce any Z1-specific behaviour into route.py -- an offset
    table, a degree assumption -- and mediation stops being portable while every
    existing test, all of which mediate for z1, keeps passing.

    PROVENANCE: the input is non-bipartite by construction (a 4-neighbour grid
    plus one chord); the assertion is that the output is bipartite and larger,
    which is mediation's own stated contract."""
    model = odd_cycle_grid()
    rep = analyse(model)
    assert rep.bipartite is False, "fixture must be non-bipartite to test this"
    mediated, _ = insert_mediators(model, rep)
    mrep = analyse(mediated)
    assert mrep.bipartite is True
    assert mrep.n_nodes > rep.n_nodes, "mediation spends spins to buy parity"


def test_the_hypothetical_target_is_not_named_after_real_hardware():
    """WHAT THIS PINS: this file must not invent specifications for a real
    device. A profile called "dwave" or "pegasus" carrying numbers nobody
    sourced would be exactly the fabrication this project refuses, and it would
    be quoted back as if it were a supported target.

    HOW IT FAILS: rename HYPOTHETICAL after any vendor and this catches it.

    PROVENANCE: the project's own rule that an unsourced figure is labelled
    assumed or not stated at all -- see target.py's Sourced fields."""
    banned = ("dwave", "d-wave", "pegasus", "zephyr", "chimera", "fujitsu",
              "toshiba", "hitachi", "normal", "z1")
    assert not any(b in HYPOTHETICAL.name.lower() for b in banned)
    assert all(f.source == "test" for f in (
        HYPOTHETICAL.degree, HYPOTHETICAL.max_abs_coupling,
        HYPOTHETICAL.node_budget)), "every field must be marked as a test value"


def test_a_degree_between_the_two_targets_gives_opposite_verdicts():
    """WHAT THIS PINS: the degree gate's COMPARISON reads the profile, not just
    its reported limit. Those are separate lines in gates.py, and a mutation
    hardcoding only the comparison keeps the reported limit correct -- so the
    limit assertions above pass while the verdict is silently wrong.

    HOW IT FAILS: hardcode the comparison (`report.max_degree <= 16` instead of
    `<= target.degree.value`) and a degree-10 model wrongly PASSES against a
    target that allows 8. Found by mutation testing this very file: the first
    mutation attempted hit the comparison, and nothing caught it.

    PROVENANCE: degree 10 sits between this target's 8 and Z1's 16, so the
    verdict is decided entirely by which profile the comparison consulted."""
    n = 40
    # a node wired to 10 distinct partners -> max_degree 10
    edges = tuple((0, k) for k in range(1, 11))
    model = IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=edges,
                       weights=np.full(len(edges), 0.5), biases=np.zeros(n),
                       beta=1.0, offset=0.0)
    rep = analyse(model)
    assert rep.max_degree == 10, f"fixture must sit between 8 and 16, got {rep.max_degree}"

    hypo = {g.gate: g for g in gate_checks(model, rep, HYPOTHETICAL, False)}
    z1 = {g.gate: g for g in gate_checks(model, rep, PROFILES["z1"], False)}
    assert hypo["degree"].passed is False, "10 ports exceed this target's 8"
    assert z1["degree"].passed is True, "10 ports are legal on Z1's 16"
