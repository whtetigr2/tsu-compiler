"""Placement -> SamplingProgram: the colour blocks and the schedule.

A block containing two adjacent nodes is SILENTLY WRONG in thrml, not an error, so
the partition is asserted here and again in the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass

from .analyse import GraphReport
from .lower import IsingModel


@dataclass(frozen=True)
class SamplingProgram:
    """ONE program form: a single Boltzmann energy sampled by chromatic block Gibbs.

    NOT the only form a stochastic program may take. A stochastic program can also be
    a COMPOSITION OF CONDITIONAL KERNELS -- which is what torx's directed factor graph
    expresses, and it is a genuinely different computational object from a stationary
    distribution (C-033, U-008). Forcing every workload into one monolithic energy is
    an assumption, not a requirement.

    The vertical slice implements this form only. `ProgramForm` exists so the
    composition form can be added beside it without rewriting every consumer; the
    field is carried through the receipt so a reader always knows which form produced
    a result. See the plan's "Deferred by design" section.
    """
    ising: IsingModel
    blocks: tuple[tuple[int, ...], ...]
    schedule: str = "chromatic_block_gibbs"
    form: str = "monolithic_energy"


def build_program(ising: IsingModel, report: GraphReport) -> SamplingProgram:
    blocks = []
    for c in range(report.colour_blocks):
        members = tuple(sorted(n for n, col in report.colouring.items() if col == c))
        if members:
            blocks.append(members)
    blocks = tuple(blocks)

    members = set()
    for b in blocks:
        for u, v in ising.edges:
            assert not (u in b and v in b), \
                f"invalid colouring: {u} and {v} are adjacent and share a block"
        members |= set(b)
    assert members == set(range(len(ising.nodes))), "blocks must cover every node once"

    return SamplingProgram(ising, blocks)
