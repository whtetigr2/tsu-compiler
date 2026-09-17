"""Parity pruning must delete only folds that cannot exist.

audit/findings/R28.md. A square lattice is bipartite, so a walk alternates
sublattices at every step: after i steps the monomer sits on a site whose
(x + y) parity matches i. Half of all (monomer, site) pairs therefore describe
configurations no chain can reach.

Deleting them takes the model's degree from 35 to 15 and turns a Z1 refusal into
a placement. That is only legitimate if it is EXACT, so the first test enumerates
every self-avoiding walk of the sequence and requires that none violates the
rule. A pruning that quietly discarded a real fold would make the whole result
worthless while still looking like a win.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from audit.protein_folding import (  # noqa: E402
    PARITY_PRUNING, SEQUENCE, all_self_avoiding_walks, build_qubo,
    contact_count, known_optimum, reachable, sites,
)


def test_parity_pruning_discards_no_reachable_fold():
    """The claim the whole encoding rests on."""
    violations = []
    for walk in all_self_avoiding_walks(len(SEQUENCE)):
        ox, oy = walk[0]
        for i, (x, y) in enumerate(walk):
            if (abs(x - ox) + abs(y - oy)) % 2 != i % 2:
                violations.append((i, (x - ox, y - oy)))
    assert not violations, (
        f"{len(violations)} monomer placements in real self-avoiding walks "
        f"break the parity rule, so pruning by it deletes reachable folds")


def test_the_known_optimum_is_a_property_of_the_sequence():
    """Ground truth must not move when the encoding changes."""
    best, n_opt, n_walks = known_optimum(SEQUENCE)
    assert (best, n_opt, n_walks) == (3, 8, 2172), (
        "the enumeration changed; it depends on the sequence and the lattice "
        "only, so either one of those moved or the enumerator is wrong")


def test_pruning_halves_the_variables():
    lin, quad, idx = build_qubo(SEQUENCE)
    full = len(SEQUENCE) * len(sites())
    assert PARITY_PRUNING, "these numbers assume pruning is on"
    assert len(idx) == full // 2, f"expected {full // 2} variables, got {len(idx)}"


def test_the_model_fits_the_degree_cap():
    """The point of the pruning. Without it the degree is 35 against a cap of 16."""
    _, quad, _ = build_qubo(SEQUENCE)
    degree: dict[int, int] = {}
    for a, b in quad:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    assert max(degree.values()) <= 16, (
        f"max degree {max(degree.values())} exceeds Z1's cap of 16")


def test_reachable_rejects_the_impossible_and_keeps_the_possible():
    """A control: the predicate must say no to something."""
    assert reachable(0, (0, 0)), "monomer 0 can start at the origin"
    assert not reachable(1, (0, 0)), "monomer 1 cannot be back at the origin"
    assert reachable(1, (1, 0)), "monomer 1 can be one step away"
    assert not reachable(2, (1, 0)), "monomer 2 cannot sit on odd parity"


def test_contact_count_ignores_chain_neighbours():
    """Consecutive monomers touch by construction and must never count.

    A straight chain has every pair of neighbours touching and no contacts at
    all. This test originally used a closed square and asserted zero, which was
    wrong: in a square, monomers 0 and 3 sit next to each other and are three
    apart along the chain, so that is a real contact worth exactly one.
    """
    straight = ((0, 0), (1, 0), (2, 0), (3, 0))
    assert contact_count("HHHH", straight) == 0, (
        "a straight chain has no non-consecutive contacts")

    square = ((0, 0), (1, 0), (1, 1), (0, 1))
    assert contact_count("HHHH", square) == 1, (
        "monomers 0 and 3 close the square: adjacent on the lattice, three "
        "apart along the chain, so exactly one contact")

    assert contact_count("HPPH", square) == 1, "both ends are H, still one"
    assert contact_count("PHHP", square) == 0, (
        "monomers 1 and 2 are chain-adjacent, so their touching is not a "
        "contact, and the H pair here is only those two")
