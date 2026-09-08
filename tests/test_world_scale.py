"""Task 4: array upsampling. Pure array logic, no sampling."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "demo")

from world.scale import nearest, bilinear, upsample


def test_nearest_replicates_each_cell_into_a_block():
    a = np.array([[0, 3], [1, 2]])
    out = nearest(a, 2)
    assert out.shape == (4, 4)
    assert np.array_equal(out, np.array([[0, 0, 3, 3],
                                         [0, 0, 3, 3],
                                         [1, 1, 2, 2],
                                         [1, 1, 2, 2]]))


def test_nearest_is_not_transposed():
    """A distinct value in every corner, so an x/y swap cannot pass. The
    cascade plan shipped a fixture where both off-diagonal values were 0 and
    a transpose bug would have gone unnoticed -- this is that lesson."""
    a = np.array([[10, 20], [30, 40]])
    out = nearest(a, 2)
    assert out[0, 0] == 10 and out[0, -1] == 20
    assert out[-1, 0] == 30 and out[-1, -1] == 40


def test_bilinear_matches_at_the_corners_and_smooths_between():
    a = np.array([[0.0, 4.0], [0.0, 4.0]])
    out = bilinear(a, 4)
    assert out.shape == (8, 8)
    assert out[0, 0] < out[0, 3] < out[0, -1]
    assert np.all(out >= 0.0) and np.all(out <= 4.0)


def test_bilinear_never_exceeds_the_input_range():
    """Interpolation must not overshoot -- an out-of-range parameter would push
    the spline past its knots and silently clamp."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=(8, 8)).astype(float)
    out = bilinear(a, 8)
    assert out.min() >= a.min() - 1e-9
    assert out.max() <= a.max() + 1e-9


def test_bilinear_on_a_constant_field_is_constant():
    a = np.full((4, 4), 2.0)
    assert np.allclose(bilinear(a, 4), 2.0)


def test_factor_one_is_the_identity_for_both_modes():
    a = np.array([[1, 2], [3, 4]])
    assert np.array_equal(nearest(a, 1), a)
    assert np.allclose(bilinear(a.astype(float), 1), a)


def test_upsample_dispatches_and_rejects_an_unknown_mode():
    a = np.array([[1, 2], [3, 4]])
    assert np.array_equal(upsample(a, 2, "nearest"), nearest(a, 2))
    assert np.allclose(upsample(a, 2, "bilinear"), bilinear(a, 2))
    with pytest.raises(ValueError):
        upsample(a, 2, "bicubic")
