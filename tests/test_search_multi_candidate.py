"""A5: real multi-candidate search over SLICE_ENCODINGS, and `compare()`.

Every candidate is retained with its terminal state and reason -- rejected ones
included, as evidence. SEMANTICALLY_INVALID and HARDWARE_INFEASIBLE must stay
distinct (a defect in the representation vs. a statement about the substrate).
Selection order is physical p-bit count, then colour blocks, then |J|max, ties
broken by declaration order and stated in `ordering_rationale`.
"""
from tsu.spec import load_spec
from tsu.target import IDEAL, Z1
from tsu.passes.search import SLICE_ENCODINGS, compare, compile_spec
from tsu.states import CandidateState


def test_both_encodings_are_generated_as_candidates():
    assert set(SLICE_ENCODINGS) == {"domain_wall", "one_hot"}


def test_toy_on_z1_domain_wall_selected_one_hot_mediated_but_not_selected():
    """one_hot's exactly-one penalty makes a clique per categorical variable,
    which for toy.yaml's c (k=3) is a triangle -- not bipartite. Task 6:
    `place` no longer rejects that outright with `parity_conflict`; it
    mediates the triangle's one within-side edge through a hidden spin
    (making it bipartite, 6 physical spins total) and places it cleanly.
    one_hot still loses selection to domain_wall's smaller physical p-bit
    count (4 < 6), retained as VIABLE_NOT_SELECTED -- the honest, expected
    cost of one_hot, reported rather than silently dropped."""
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    assert c.verdict == "COMPILED"
    by_encoding = {cand.encoding: cand for cand in c.repset.candidates}
    assert set(by_encoding) == {"domain_wall", "one_hot"}

    assert by_encoding["domain_wall"].state == CandidateState.SELECTED
    oh = by_encoding["one_hot"]
    assert oh.state == CandidateState.VIABLE_NOT_SELECTED
    assert oh.reason, "a non-selected candidate without its reason is not evidence"
    assert oh.placement.mediation is not None
    assert oh.placement.mediation.mediator_count == 1
    assert oh.placement.mediation.bipartite_after is True


def test_both_feasible_lower_physical_pbit_count_wins_and_loser_is_recorded():
    """Under IDEAL (no bipartite requirement, no degree/coupling cap) both
    encodings reach HARDWARE_FEASIBLE for toy.yaml. domain_wall costs 4 spins
    (a, b, c__dw0, c__dw1), one_hot costs 5 (a, b, c__oh0..2) -- domain_wall
    must be selected on physical p-bit count, and one_hot must be retained as
    VIABLE_NOT_SELECTED, not simply discarded."""
    c = compile_spec(load_spec("specs/toy.yaml"), IDEAL)
    assert c.verdict == "COMPILED"
    by_encoding = {cand.encoding: cand for cand in c.repset.candidates}

    assert by_encoding["domain_wall"].state == CandidateState.SELECTED
    assert c.repset.selected.encoding == "domain_wall"
    assert by_encoding["one_hot"].state == CandidateState.VIABLE_NOT_SELECTED
    assert by_encoding["one_hot"].reason


def test_ordering_rationale_states_the_full_tie_break_order():
    c = compile_spec(load_spec("specs/toy.yaml"), IDEAL)
    r = c.repset.ordering_rationale.lower()
    assert "p-bit" in r or "pbit" in r
    assert "colour" in r or "color" in r
    assert "|j|max" in r or "j|max" in r or "jmax" in r
    assert "declaration order" in r


def test_ideal_control_still_runs_first_and_still_cannot_be_disabled():
    """Unchanged from the single-encoding slice: a broken spec must fail the
    ideal control and report LOGICAL before any real (per-target) candidate
    search happens at all."""
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    assert c.verdict == "LOGICAL"
    assert c.hardware_evaluated is False


def test_compare_returns_a_row_per_candidate_with_every_required_column():
    c = compile_spec(load_spec("specs/toy.yaml"), Z1)
    rows = compare(c.repset)
    assert len(rows) == 2
    required = {"encoding", "state", "reason", "logical_spins", "logical_edges",
               "bipartite", "mediators", "physical_pbits", "colour_blocks",
               "max_abs_J", "max_abs_b"}
    for row in rows:
        assert required <= set(row)

    by_encoding = {row["encoding"]: row for row in rows}
    dw = by_encoding["domain_wall"]
    assert dw["state"] == "SELECTED"
    assert dw["logical_spins"] == 4          # a, b, c__dw0, c__dw1
    assert dw["bipartite"] is True

    # Task 6: one_hot's clique (a, b, c__oh0, c__oh1, c__oh2 -- 5 spins,
    # not bipartite) is mediated before placement, so `logical_spins`/
    # `bipartite` here describe the PLACED, post-mediation graph (5 + 1
    # mediator, bipartite) -- `mediators_inserted` is the actual mediation
    # pass's own count, distinct from `mediators` (analyse()'s theoretical
    # floor for the ALREADY-bipartite mediated graph, which is 0).
    oh = by_encoding["one_hot"]
    assert oh["state"] == "VIABLE_NOT_SELECTED"
    assert oh["logical_spins"] == 6
    assert oh["bipartite"] is True
    assert oh["mediators_inserted"] == 1


def test_compare_handles_a_candidate_with_no_report_gracefully():
    """A candidate that never reached `analyse` (e.g. encode itself failed)
    has report=None; compare() must degrade to `None`/unavailable fields, never
    crash and never fabricate a number."""
    c = compile_spec(load_spec("specs/broken.yaml"), Z1)
    rows = compare(c.repset)
    assert len(rows) >= 1
    for row in rows:
        if row["state"] != "SELECTED" and row["logical_spins"] is None:
            assert row["reason"]   # still explains itself
