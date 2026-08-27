import numpy as np
import pytest

from tsu.target import Z1, IDEAL
from tsu.passes.lower import IsingModel
from tsu.passes.analyse import analyse
from tsu.gates import check_gates, gate_checks


def ising(n_nodes, edges, w, b=None):
    return IsingModel(tuple(f"x{i}" for i in range(n_nodes)), tuple(edges),
                      np.array(w, dtype=float),
                      np.array(b if b is not None else [0.0] * n_nodes, dtype=float),
                      1.0, 0.0)


def test_coupling_cap_gate_fires_and_is_tagged_assumed():
    im = ising(2, [(0, 1)], [40.0])
    fails = check_gates(im, analyse(im), Z1)
    caps = [f for f in fails if f.gate == "coupling_cap"]
    assert len(caps) == 1
    assert caps[0].assumed is True, "Jmax is a project assumption, not a sourced fact"
    assert caps[0].measured == pytest.approx(40.0)
    assert "--allow-assumed" in caps[0].remediations[-1].detail


def test_allow_assumed_downgrades_the_gate():
    im = ising(2, [(0, 1)], [40.0])
    fails = check_gates(im, analyse(im), Z1, allow_assumed=True)
    assert not any(f.gate == "coupling_cap" for f in fails)


def test_degree_gate_fires_on_a_clique_of_twenty():
    edges = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    im = ising(20, edges, [1.0] * len(edges))
    fails = check_gates(im, analyse(im), Z1)
    deg = [f for f in fails if f.gate == "degree"]
    assert len(deg) == 1
    assert deg[0].measured == 19
    assert deg[0].assumed is False, "degree 16 is sourced to F-14 and not overridable"


def test_sourced_gate_is_not_overridable_by_allow_assumed():
    edges = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    im = ising(20, edges, [1.0] * len(edges))
    fails = check_gates(im, analyse(im), Z1, allow_assumed=True)
    assert any(f.gate == "degree" for f in fails), \
        "--allow-assumed must never suppress a gate backed by a fact id"


def test_ideal_target_passes_everything_logical():
    edges = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    im = ising(20, edges, [1000.0] * len(edges))
    assert check_gates(im, analyse(im), IDEAL) == ()


def test_bias_cap_gate_fires_on_a_large_bias_with_a_small_coupling():
    """C2: the spec's hardware gate is 'any |J| OR |b| exceeds
    target.max_abs_coupling' (section 7.1). A model with a small coupling but a
    bias far past the cap must fail -- the existing `ising()` helper defaults
    biases to zero, which is exactly why this was missed for fifteen review
    passes; this is the first test in the suite to pass a nonzero bias."""
    im = ising(2, [(0, 1)], [1.0], b=[50.25, 0.0])
    fails = check_gates(im, analyse(im), Z1)
    caps = [f for f in fails if f.gate == "field_cap"]
    assert len(caps) == 1, "a bias of 50.25 against a cap of 6.0 must gate"
    assert caps[0].measured == pytest.approx(50.25)
    assert caps[0].assumed is True


def test_bias_cap_gate_is_checked_even_with_zero_edges():
    """`if len(ising.weights):` used to wrap the ENTIRE magnitude-gate block, so a
    model with no couplings at all skipped the bias check too. A single node with
    a large bias and no edges must still gate."""
    im = ising(1, [], [], b=[50.25])
    fails = check_gates(im, analyse(im), Z1)
    assert any(f.gate == "field_cap" for f in fails)


def test_bias_cap_gate_is_downgradable_like_the_coupling_cap():
    im = ising(2, [(0, 1)], [1.0], b=[50.25, 0.0])
    fails = check_gates(im, analyse(im), Z1, allow_assumed=True)
    assert not any(f.gate == "field_cap" for f in fails)


def test_gate_checks_reports_every_gate_not_only_failures():
    """The receipt needs 'every gate, passed/failed, with measured value and
    threshold' (spec section 10) -- gate_checks is the full evaluation
    check_gates filters down to failures only."""
    im = ising(2, [(0, 1)], [1.0], b=[0.5, 0.0])
    checks = gate_checks(im, analyse(im), Z1)
    names = {c.gate for c in checks}
    assert {"degree", "coupling_cap", "field_cap", "node_budget",
           "colouring"} <= names
    assert all(c.passed for c in checks), "this model should pass every gate"
