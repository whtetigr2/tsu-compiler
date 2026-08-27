import numpy as np
import pytest

from tsu.target import Z1
from tsu.passes.lower import IsingModel
from tsu.passes.analyse import analyse
from tsu.regime import analyse_regime


def ising(n_nodes, edges, w, b=None):
    return IsingModel(tuple(f"x{i}" for i in range(n_nodes)), tuple(edges),
                      np.array(w, dtype=float),
                      np.array(b if b is not None else [0.0] * n_nodes, dtype=float),
                      1.0, 0.0)


def test_coupling_utilisation_reflects_the_bias_not_only_the_coupling():
    """C2: `RegimeReport.coupling_utilisation` read only max_abs_J, so a model
    whose bias (not its coupling) was the thing actually near the cap reported a
    misleadingly low utilisation -- the exact reproduction case (|b|=50.25
    against a cap of 6.0) reported '4% of range used' beside a receipt that had
    every reason to say otherwise."""
    im = ising(2, [(0, 1)], [0.24], b=[50.25, 0.0])   # |J| tiny, |b| huge
    r = analyse(im)
    regime = analyse_regime(r, Z1)
    assert regime.coupling_utilisation == pytest.approx(50.25 / 6.0)
