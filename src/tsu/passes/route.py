"""Mediate couplings the substrate cannot place directly.

In the vertical slice this is the IDENTITY: the slice's toy spec places without
frustration, and actual mediator insertion is exercised by the acceptance suite
(spec section 13), not by the slice. It raises rather than silently passing through a
graph that DOES need routing, so the gap cannot be mistaken for support.
"""
from __future__ import annotations

from ..target import TargetProfile
from .analyse import GraphReport
from .lower import IsingModel


def route(ising: IsingModel, report: GraphReport, target: TargetProfile) -> IsingModel:
    if target.bipartite.value and not report.bipartite:
        raise NotImplementedError(
            "this graph needs mediator routing, which is not implemented in the "
            "vertical slice; `place` should have raised parity_conflict first")
    return ising
