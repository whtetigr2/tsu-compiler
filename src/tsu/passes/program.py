"""Placement -> SamplingProgram: the colour blocks and the schedule.

A block containing two adjacent nodes is SILENTLY WRONG in thrml, not an error, so
the partition is asserted here and again in the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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
    # C1: clamping. `clamped` is the tuple of PHYSICAL node indices held fixed
    # -- never resampled, and therefore never a member of any block above.
    # `clamp_values` is that same set of nodes' fixed 0/1 values. Both default
    # empty so an unclamped program is bit-for-bit the program this pass
    # always built.
    clamped: tuple[int, ...] = ()
    clamp_values: Mapping[int, int] = field(default_factory=dict)


def build_program(ising: IsingModel, report: GraphReport,
                  clamp: Mapping[int, int] | None = None) -> SamplingProgram:
    """`clamp`: physical node index -> its fixed 0/1 value (C1). A clamped
    node is not resampled, so it must never appear in a free block -- the
    colouring is partitioned into blocks over the FREE nodes only, and that
    non-membership is asserted below, not assumed from how the blocks happen to be
    built."""
    clamp = dict(clamp or {})
    clamped_idx = frozenset(clamp)

    blocks = []
    for c in range(report.colour_blocks):
        members = tuple(sorted(n for n, col in report.colouring.items()
                               if col == c and n not in clamped_idx))
        if members:
            blocks.append(members)
    blocks = tuple(blocks)

    free_members = set()
    for b in blocks:
        for n in b:
            assert n not in clamped_idx, \
                f"clamped node {n} appeared in a free block: {b}"
        for u, v in ising.edges:
            assert not (u in b and v in b), \
                f"invalid colouring: {u} and {v} are adjacent and share a block"
        free_members |= set(b)

    assert free_members.isdisjoint(clamped_idx), \
        "a clamped node must never appear in a free block"
    assert free_members | clamped_idx == set(range(len(ising.nodes))), \
        "free blocks plus the clamp must cover every node exactly once"

    return SamplingProgram(ising, blocks, clamped=tuple(sorted(clamped_idx)),
                           clamp_values=clamp)
