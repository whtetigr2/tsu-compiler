"""Upsampling a coarse parameter field to the output resolution.

WHY THIS IS NOT THE CASCADE THAT FAILED. The coarse-to-fine cascade was refuted
(tsu-compiler 5a21393) because it used upsampling to CONDITION a sampled fine
layer through soft bias patches: lossy, with a degenerate zone, and it
transmitted per-cell information while transmitting no structure. Here
upsampling only SETS A PARAMETER FIELD'S SCALE. It is deterministic, adds no
energy terms, creates nothing that can be violated, and is applied to a
parameter rather than to terrain. Same operation, different job.

WHY SCALE COMES FROM HERE AND NOT FROM COUPLING. Correlation length is tunable
only over roughly 0.75 to 2.66 cells -- a 3.5x range -- against the 10-100x
feature-scale ratios a world needs. Criticality cannot deliver scale
separation; sampling each parameter on its own grid and upsampling can.

TWO ARMS, DECIDED BY MEASUREMENT. `bilinear` matches the reference pipeline,
which evaluates noise coarsely and interpolates. `nearest` carries values
without inventing them. Smoothing is exactly the operation that could turn
sampled structure into mush, so audit/routing.py measures both and the simpler
one wins on a tie.

This is the ARRAY path. demo/cascade.py's own `upsample` operates on decode
dicts and stays untouched -- different data shape, not a different algorithm.
"""
from __future__ import annotations

import numpy as np


def nearest(a: np.ndarray, factor: int) -> np.ndarray:
    """Replicate each cell across its factor x factor block. Values are carried,
    never blended -- dtype is preserved."""
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)


def bilinear(a: np.ndarray, factor: int) -> np.ndarray:
    """Bilinear interpolation onto a factor-times-finer grid, always float.

    Sample points sit at cell centres and are CLAMPED to the input's extent, so
    the output never overshoots the input range -- an out-of-range parameter
    would push the spline past its outermost knot and be silently clamped there,
    losing information without any error.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    a = np.asarray(a, dtype=float)
    if factor == 1:
        return a.copy()
    n, m = a.shape
    # Cell centres of the fine grid, expressed in coarse-cell coordinates.
    ys = (np.arange(n * factor) + 0.5) / factor - 0.5
    xs = (np.arange(m * factor) + 0.5) / factor - 0.5
    ys = np.clip(ys, 0.0, n - 1.0)
    xs = np.clip(xs, 0.0, m - 1.0)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, n - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, m - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    top = a[np.ix_(y0, x0)] * (1 - fx) + a[np.ix_(y0, x1)] * fx
    bot = a[np.ix_(y1, x0)] * (1 - fx) + a[np.ix_(y1, x1)] * fx
    return top * (1 - fy) + bot * fy


def upsample(a: np.ndarray, factor: int, mode: str) -> np.ndarray:
    """Dispatch to `nearest` or `bilinear`. An unknown mode raises rather than
    falling back, so a typo cannot silently change which arm was measured."""
    if mode == "nearest":
        return nearest(a, factor)
    if mode == "bilinear":
        return bilinear(a, factor)
    raise ValueError(f"unknown upsample mode {mode!r}; expected 'nearest' or "
                     f"'bilinear'")
