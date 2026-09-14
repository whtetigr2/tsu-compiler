import numpy as np
import pytest

from tsu_compiler.target import Z1
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.regime import analyse_regime


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


def test_precision_headroom_is_not_a_constant_reported_as_a_measurement():
    """I1: `step = 2*|J|max / 2**bits` then `headroom = |J|max / step` reduces
    algebraically to `2**(bits-1)` -- ALWAYS, independent of the model (every z1
    receipt read 32.0). It was also not the spec's quantity, which is
    "quantisation step vs smallest distinct |J| gap": two models with different
    couplings, both on Z1, must be able to report DIFFERENT headroom -- if the
    field is a real measurement it varies with the model that produced it."""
    im_tight = ising(3, [(0, 1), (1, 2)], [1.0, 1.05])     # gap 0.05: tight
    im_loose = ising(3, [(0, 1), (1, 2)], [1.0, 5.0])      # gap 4.0: loose

    h_tight = analyse_regime(analyse(im_tight), Z1, weights=im_tight.weights) \
        .precision_headroom
    h_loose = analyse_regime(analyse(im_loose), Z1, weights=im_loose.weights) \
        .precision_headroom

    assert h_tight is not None and h_loose is not None
    assert h_tight != pytest.approx(h_loose), \
        "two different models must not report the same headroom"
    assert h_tight < h_loose, "the tighter gap must report the smaller headroom"


def test_precision_headroom_is_honestly_none_with_fewer_than_two_distinct_couplings():
    """A single |J| value (or none) has nothing for quantisation to collide with
    -- the field must read None with a reason, never a guessed number."""
    im = ising(2, [(0, 1)], [2.0])
    regime = analyse_regime(analyse(im), Z1, weights=im.weights)
    assert regime.precision_headroom is None
    assert regime.precision_headroom_note != ""
    d = regime.to_dict()
    assert d["precision_headroom"] == regime.precision_headroom_note
