"""The structure factor, S(k): what a lattice looks like in reciprocal space.

S(k) is the squared magnitude of the spatial Fourier transform of the spin
field, averaged over draws. It is the standard way physics asks "what kind of
order is this", and it answers with a picture: diffuse haze when the system is
hot and disordered, a ring blooming out of the noise near a critical point, and
sharp Bragg peaks once order sets in. Where the peaks sit says WHICH order.

    k = (0, 0)     ferromagnetic, everything aligned
    k = (pi, pi)   antiferromagnetic, checkerboard

Checked against a known answer rather than asserted. The shipped alloy program is
an ordering alloy: unlike neighbours are preferred, which is a textbook
antiferromagnet, and its peak must land on the zone corner. `tests/` requires it.

TWO HONESTY CONSTRAINTS, both load-bearing.

The transform needs REAL-SPACE positions, meaning where a spin sits in the
model's own lattice. It must NOT use placement coordinates, which say where the
spin was laid out on the die. Those are different things and a transform over
die positions would be a picture of the router's output, not of the physics. The
coordinates here come from the node's own name, which the grid generator writes
as `g{x}_{y}`.

When the lattice cannot be recovered there is no structure factor, and this says
so instead of transforming an arbitrary ordering of spins into a plausible
looking square.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

_GRID_NAME = re.compile(r"^g(\d+)_(\d+)$")


@dataclass(frozen=True)
class Lattice:
    """Where each spin sits in the model's own space."""

    width: int
    height: int
    index_to_xy: dict[int, tuple[int, int]]
    source: str


def recover_lattice(node_names: Sequence[Any] | None) -> Lattice | None:
    """Real-space coordinates from the node's own name, or None.

    The grid generator names a site `g{x}_{y}`, so the name carries the
    coordinate and nothing has to be inferred from index order or from where the
    router happened to place it.

    Returns None rather than guessing. A model whose spins are not on a lattice
    has no structure factor, and inventing one would produce a convincing square
    picture of nothing.
    """
    if not node_names:
        return None
    coords: dict[int, tuple[int, int]] = {}
    for i, name in enumerate(node_names):
        m = _GRID_NAME.match(str(name))
        if m:
            coords[i] = (int(m.group(1)), int(m.group(2)))
    if not coords:
        return None

    width = max(x for x, _ in coords.values()) + 1
    height = max(y for _, y in coords.values()) + 1
    if width < 2 or height < 2:
        return None
    if len(coords) != width * height:
        # A partially covered grid would transform into edge artefacts that
        # look like structure. Refuse rather than explain them away.
        return None
    return Lattice(width, height, coords,
                   source="site names written by the grid generator")


def structure_factor(states: Sequence[Sequence[float]], lattice: Lattice,
                     *, drop_k0: bool = True) -> dict[str, Any]:
    """Average |FFT(spin field)|^2 over draws, with k = 0 at the centre.

    `drop_k0` zeroes the origin. S(0) is the square of the net magnetisation and
    is usually enormous compared with everything else, so leaving it in makes
    every other feature invisible. It is reported separately rather than hidden.
    """
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return {"available": False,
                "reason": "unavailable: no draws to transform yet"}

    W, H = lattice.width, lattice.height
    total = np.zeros((H, W), dtype=float)
    for row in arr:
        field = np.zeros((H, W), dtype=float)
        for i, (x, y) in lattice.index_to_xy.items():
            if i < len(row):
                field[y, x] = 1.0 if row[i] > 0 else -1.0
        total += np.abs(np.fft.fft2(field)) ** 2
    spectrum = np.fft.fftshift(total / len(arr))

    cx, cy = W // 2, H // 2
    k0 = float(spectrum[cy, cx])
    if drop_k0:
        spectrum[cy, cx] = 0.0

    ky, kx = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    peak = ((kx - cx) / (W / 2.0), (ky - cy) / (H / 2.0))

    return {
        "available": True,
        "width": W,
        "height": H,
        "values": spectrum.tolist(),
        "max": float(spectrum.max()),
        # float() explicitly: numpy scalars do not survive JSON encoding.
        "peak_k_over_pi": [float(round(peak[0], 4)), float(round(peak[1], 4))],
        "peak_label": _label(peak),
        "k0_intensity": k0,
        "n_draws": int(len(arr)),
        "source": lattice.source,
        "note": ("k = 0 sits at the centre and is excluded from the peak "
                 "search, because S(0) is the squared net magnetisation and "
                 "would swamp everything else."),
    }


def _label(peak: tuple[float, float]) -> str:
    """Name the order, when the peak sits somewhere with a name."""
    x, y = abs(peak[0]), abs(peak[1])
    near = lambda v, t: abs(v - t) < 0.2  # noqa: E731
    if near(x, 1.0) and near(y, 1.0):
        return "peak at the zone corner, k = (pi, pi): checkerboard order"
    if near(x, 0.0) and near(y, 0.0):
        return "peak at k = 0: uniform order, everything aligned"
    if near(x, 1.0) and near(y, 0.0) or near(x, 0.0) and near(y, 1.0):
        return "peak on a zone edge: stripe order"
    return "peak away from the high-symmetry points"
