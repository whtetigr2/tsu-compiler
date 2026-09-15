"""What does the coupling gate actually bound -- and what does it not?

This pins a property that is easy to assume the other way, and that was in fact
assumed the other way in review: the coupling gate reads the RAW weight and
ignores beta entirely.

The distribution depends on beta and J only through their product -- `p ~
exp(-beta * E)` with `E = -sum J s s - sum b s` -- so scaling beta with J fixed
specifies exactly the same family as scaling J with beta fixed. The gate does
not see that. It compares `|J|max` against the target cap with no beta factor.

Whether that is correct is NOT settled by any source available to this project.
Extropic's Thermalizers states `|J| <= Jmax` without saying whether Jmax bounds
the dimensionless product `beta*J` or a programmable register value at a fixed
hardware operating temperature. Those two readings agree exactly at beta = 1 and
disagree everywhere else. Every model this project compiles or publishes is
beta = 1.0, so nothing shipped depends on the answer.

These tests therefore do NOT assert the gate is right. They assert the gate is
what it is, loudly, so that:

  - nobody concludes from a passing coupling verdict that a beta-scaled model
    is representable on the hardware (that inference was made in review and was
    wrong -- audit/findings/R22.md), and
  - if the semantics are ever resolved by a real source, the change is a
    deliberate edit to a documented property rather than a silent drift.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from tsu_compiler.gates import gate_checks
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.target import PROFILES

Z1 = PROFILES["z1"]


def model(beta: float, weight: float = 2.5) -> IsingModel:
    n = 8
    edges = tuple((i, i + 1) for i in range(n - 1))
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=edges,
                      weights=np.full(len(edges), weight), biases=np.zeros(n),
                      beta=beta, offset=0.0)


def coupling_gate(beta: float, weight: float = 2.5):
    m = model(beta, weight)
    return {c.gate: c for c in gate_checks(m, analyse(m), Z1, False)}["coupling_cap"]


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5, 2.0, 6.0, 100.0])
def test_the_coupling_gate_does_not_move_when_beta_does(beta):
    """WHAT THIS PINS: the gate reports the raw |J| and the raw cap at every
    beta. At beta=100 a weight of 2.5 specifies a dimensionless coupling of 250
    and the gate still passes against a cap of 6.0.

    HOW IT FAILS: fold beta into `peak_J` in gates.py and every row here moves.
    That may one day be the right change -- but it is a change to what the tool
    CLAIMS about hardware, and it must not happen silently.

    PROVENANCE: gates.py reads `np.abs(ising.weights).max()`; analyse.py does
    not reference beta at all."""
    c = coupling_gate(beta)
    assert c.measured == pytest.approx(2.5), (
        f"at beta={beta} the gate reported |J|max={c.measured}; if beta is now "
        f"folded in, audit/findings/R22.md and the gates.py note need rewriting")
    assert c.limit == Z1.max_abs_coupling.value
    assert c.passed is True


def test_a_beta_scaled_model_and_a_J_scaled_model_are_gated_differently():
    """WHAT THIS PINS: the asymmetry itself, which is the whole point. Two
    models specifying the SAME distribution get OPPOSITE coupling verdicts
    depending on which knob expressed it.

    beta=4, |J|=2.5 and beta=1, |J|=10.0 both give a dimensionless coupling of
    10.0. The first passes a cap of 6.0; the second fails it.

    HOW IT FAILS: if a future change makes these agree, this test says so --
    and that would be a genuine improvement, not a regression. It should be
    made deliberately, with the source that settles the Jmax semantics.

    PROVENANCE: p ~ exp(-beta*E) depends on beta and J only through beta*J."""
    scaled_beta = coupling_gate(beta=4.0, weight=2.5)
    scaled_j = coupling_gate(beta=1.0, weight=10.0)

    assert scaled_beta.passed is True, "beta=4, |J|=2.5 passes the cap"
    assert scaled_j.passed is False, "beta=1, |J|=10.0 fails the same cap"
    assert scaled_beta.measured != scaled_j.measured, (
        "the two models specify the same distribution but the gate sees "
        "different numbers -- that asymmetry IS the documented limit")


def test_every_workload_this_project_ships_is_beta_one():
    """WHAT THIS PINS: why the limit above is currently harmless. The two
    readings of Jmax coincide exactly at beta = 1, so no published verdict
    depends on resolving them. The moment a beta != 1 model is shipped, that
    stops being true and R22 becomes load-bearing.

    HOW IT FAILS: add a workload at any other beta and this fires, which is the
    signal to resolve the Jmax semantics before publishing a verdict about it.

    PROVENANCE: the edge files in out/extropic-verify/ are the shipped pack."""
    from pathlib import Path

    from tsu_compiler.preflight.model import load_model
    pack = Path(__file__).resolve().parents[1] / "out" / "extropic-verify"
    files = sorted(pack.glob("*.edges.json"))
    assert files, "the shipped pack must not be empty"
    offenders = {f.name: load_model(edges=f).beta
                 for f in files if load_model(edges=f).beta != 1.0}
    assert not offenders, (
        f"these shipped workloads are not beta=1.0: {offenders}. The coupling "
        f"gate is beta-independent (R22), so their coupling verdicts do not "
        f"mean what a reader will assume. Resolve the Jmax semantics first.")
