"""Workloads this project did not write, that compile here, with a test saying so.

This module is the single place that decides what "verified" means in the
application. The examples shelf reads it to decide which entries carry the
badge, and `backend/tests/test_extropic_oracle.py` parametrizes from it to prove
each one still compiles. Neither can drift from the other, because there is only
one list.

That matters because the badge used to be a flag somebody typed into a catalog.
A label is not a fact. Presented this way, "verified" means something a person
can check: somebody else published this problem, it compiles here, and a
standing test fails if that stops being true.

Every figure below is MEASURED, by running preflight on the shipped YAML, not
estimated. The placement effort each one needs is part of the record: a workload
that places only at a higher budget is not a failure, and reporting it as one
was a real defect in this application.

Nothing here is an energy or performance claim. These compile under Z1's
published limits in simulation. No silicon has run them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerifiedWorkload:
    """One externally published problem, with what it is and why it fits."""

    shelf_id: str
    spec_stem: str
    plain_name: str
    """What it is, in words a newcomer can read. Not the directory name."""

    one_line: str
    source: str
    """Who published the problem. The credential is that it is not ours."""

    citation: str
    problem: str
    """The real-world question, before any physics."""

    math: str
    """How the question becomes an energy. Written out, not gestured at."""

    hardware: str
    """What it costs on Z1, measured."""

    # Measured placement effort. A higher budget is a fact about the model,
    # not a failure of the compiler.
    restarts: int
    iters: int
    expect_mediators: bool

    # What the SHIPPED RECEIPT contains, which is what the user opens. These
    # are checked against the receipt on disk by a test, because the numbers in
    # `hardware` above and the spin count the picker shows beside the name come
    # from different places and were caught disagreeing: the receipt for the
    # eight-position workload is one-hot encoded, while a fresh preflight is
    # domain-wall, and the two produce different models of the same spec.
    receipt_encoding: str
    receipt_spins: int
    receipt_mediators: int


_ISING = (
    "Everything lowers to the same energy: E(s) = sum_i b_i s_i + "
    "sum_(i<j) J_ij s_i s_j, over spins s_i in {-1,+1}. In this project's IR a "
    "weight of +1.0 is FERROMAGNETIC and the IR weight is J itself, not its "
    "negation. Sampling draws states with probability proportional to "
    "exp(-beta E)."
)

VERIFIED_WORKLOADS: tuple[VerifiedWorkload, ...] = (
    VerifiedWorkload(
        shelf_id="prog_codon_opt_tiny",
        spec_stem="codon_opt_tiny",
        plain_name="DNA codon choice, six positions",
        one_line=(
            "Pick synonymous DNA codons for a short peptide so the host "
            "expresses it well."),
        source="Extropic",
        citation=(
            "Extropic's THRML codon_opt class. Library: "
            "github.com/extropic-ai/thrml. The published problem also carries a "
            "GC-content term; this instance keeps host usage and adjacent-pair "
            "penalties only, which is a scale-down stated in the program's own "
            "description rather than a silent simplification."),
        problem=(
            "Several DNA triplets code for the same amino acid. Which you pick "
            "changes how well a host organism manufactures the protein. Some "
            "synonyms are common in that host and translate smoothly; rare ones "
            "stall. Certain adjacent pairs also create repeat-like runs that "
            "are awkward to synthesise. So: choose one synonym per position, "
            "preferring common ones, avoiding bad neighbours."),
        math=(
            "Six positions, each a categorical variable over three synonyms. "
            "Domain-wall encoding uses two spins per position, so 12 spins.\n"
            "  Host usage, a LINEAR term: weight -0.4 on choosing the host's "
            "favourite synonym, +0.3 on the rare one. Negative is a reward.\n"
            "  Adjacent pairs, PRODUCT terms: +0.8 where two neighbouring "
            "positions both take the rare synonym, which is the repeat-like "
            "case.\n" + _ISING),
        hardware=(
            "The shipped receipt is domain-wall encoded: 12 spins for the six "
            "positions, plus 6 mediator spins inserted to make the update "
            "schedule legal, so 18 in total. Max degree 4. Places at 6 restarts "
            "and 40,000 iterations in well under a second. Every Z1 gate "
            "passes: degree, coupling cap, bias cap, colouring, node budget."),
        restarts=6,
        iters=40_000,
        expect_mediators=True,
        receipt_encoding="domain_wall",
        receipt_spins=18,
        receipt_mediators=6,
    ),
    VerifiedWorkload(
        shelf_id="prog_ecology_lotka_lite",
        spec_stem="ecology_lotka_lite",
        plain_name="Two competing species on a 4x4 grid",
        one_line=(
            "Two species cannot share a border, and each prefers its own kind "
            "nearby. Where do the patches form?"),
        source="Extropic",
        citation=(
            "Extropic's Torx Lotka-Volterra class, as a discrete Ising "
            "instance. The continuous Lotka-Volterra differential equations "
            "are a different tool and are out of scope for this shelf, which "
            "the program's own description says rather than blurring."),
        problem=(
            "Competitive exclusion: two species occupying the same niche "
            "cannot coexist at a boundary. Give each cell of a landscape one "
            "of two species, forbid them from abutting, and give same-species "
            "neighbours a mild reward so territory clumps. The question is "
            "what spatial arrangement that produces."),
        math=(
            "A 4x4 grid, each cell categorical over two species, so 16 spins.\n"
            "  Unlike neighbours, PRODUCT_OVER_EDGES: +1.2. Positive is a "
            "cost, so opposite species sharing an edge is expensive.\n"
            "  Like neighbours: -0.25 for either species. A mild reward, so "
            "patches clump without freezing solid.\n"
            "  A CONTRACT additionally forbids the exclusion violation "
            "outright, so a draw that breaks it is rejected rather than "
            "rendered.\n" + _ISING),
        hardware=(
            "16 spins, max degree 4, and BIPARTITE, so zero mediators are "
            "needed and the two-colour update schedule is exact. Places "
            "immediately. Every Z1 gate passes. This is the cheapest of the "
            "three and the best one to read first."),
        restarts=6,
        iters=40_000,
        expect_mediators=False,
        receipt_encoding="domain_wall",
        receipt_spins=16,
        receipt_mediators=0,
    ),
    VerifiedWorkload(
        shelf_id="prog_seq_design_longer",
        spec_stem="seq_design_longer",
        plain_name="DNA codon choice, eight positions",
        one_line=(
            "The same codon problem, made longer, to find where the hardware "
            "starts to strain."),
        source="Extropic",
        citation=(
            "Extropic's SEQ class, same grammar as the six-position instance "
            "above, stretched to eight amino-acid positions."),
        problem=(
            "Identical to the six-position problem, with two more positions "
            "and a fuller set of adjacent-pair penalties. It exists to answer "
            "a scaling question honestly: does this still compile when the "
            "sequence grows, and what does it cost?"),
        math=(
            "Eight positions over three synonyms. Same terms as the shorter "
            "instance, with adjacent-pair penalties on every neighbouring pair "
            "rather than a subset, plus a milder +0.35 penalty on a second "
            "synonym pairing.\n"
            "  ENCODING MATTERS HERE, and this is the one place two numbers on "
            "screen were caught disagreeing. The compiler searches encodings "
            "and chose ONE-HOT for the shipped receipt: three spins per "
            "position, 24 in total. Domain-wall would use two per position, "
            "16, and also compiles. They are different models of the same "
            "problem, so the count depends on which one you are looking at.\n"
            + _ISING),
        hardware=(
            "The shipped receipt is one-hot encoded: 24 spins for the eight "
            "positions plus 8 mediators, so 32 in total. Max degree 4. This one "
            "does NOT place at the cheapest budget. It needs 24 restarts and "
            "250,000 iterations, roughly 25 seconds. That refusal at low effort "
            "is honest, and the application escalates automatically rather than "
            "reporting a failure. Every Z1 gate passes at both budgets: the "
            "constraint is search effort, not hardware."),
        restarts=24,
        iters=250_000,
        expect_mediators=True,
        receipt_encoding="one_hot",
        receipt_spins=32,
        receipt_mediators=8,
    ),
)


def verified_ids() -> tuple[str, ...]:
    """Shelf ids an oracle test covers. The badge derives from this."""
    return tuple(w.shelf_id for w in VERIFIED_WORKLOADS)


def by_shelf_id(shelf_id: str) -> VerifiedWorkload | None:
    for w in VERIFIED_WORKLOADS:
        if w.shelf_id == shelf_id:
            return w
    return None
