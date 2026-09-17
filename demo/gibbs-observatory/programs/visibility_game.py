"""First-hit visibility, as a loadable Workbench program.

This is the file the spec's section 8 is about. The game cannot be a static
model: its biases are rewritten every frame from where the player is standing,
so no file IS its model and something has to build one on demand. That is what
`build(inputs)` is for.

The physics is not invented here. Every constant and both helper functions are
imported from `audit/visibility_as_inference.py`, which is checked against an
exact raycast, and the same energy is sampled through THRML in
`audit/visibility_on_thrml.py`. Retyping the numbers is how two files stop
describing the same model, so they are imported.

WHY THE INPUT IS A BIAS AND NOT A CLAMP. The design document says input is
clamping. Here it is not, deliberately, and this is the one program verified
against ground truth. A wall is a strong bias toward "stopped", not a pinned
spin, so the couplings are able to outvote the input. Clamping made the
couplings inert, that version's claims were wrong, and R23 is the retraction.
The port below says `mode: "bias"` for exactly that reason.

What that buys, measured in `audit/visibility_on_thrml.py`: on a corrupted
input -- spurious walls in free space, the failure a scan cannot recover from
-- the couplings let neighbouring columns outvote a wrong cell. A clamped
model cannot do that, because a clamped wrong cell stays wrong.

No hardware claim. THRML simulates on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# The audit directory holds the verified reference. Import the constants rather
# than retyping them: if they drift, this program and the thing that checks it
# stop describing the same model.
_AUDIT = Path(__file__).resolve().parents[3] / "audit"
if _AUDIT.is_dir() and str(_AUDIT) not in sys.path:
    sys.path.insert(0, str(_AUDIT))

from visibility_as_inference import (  # noqa: E402
    W_COH,
    W_MONO,
    biases,
    decode as decode_depths,
)

from tsu_compiler.preflight.model import IsingModel  # noqa: E402

NAME = "visibility"
DECODER = "visibility"

#: How the spins are laid out: columns x depth. Not a port shape.
LATTICE_SHAPE = [48, 32]

#: Occupancy in front of the player, one cell per (column, depth). It enters as
#: a BIAS, not a clamp: see the module docstring and R23.
INPUT_PORTS = [
    {
        "name": "occupancy",
        "shape": [48, 32],
        "mode": "bias",
        "doc": "1 where a cell is solid, as the player currently sees it. "
               "A wall biases its cell toward 'stopped' rather than pinning "
               "it, so neighbouring columns can outvote a wrong reading.",
    },
]


def build(inputs) -> IsingModel:
    """The energy as a real IsingModel, from this frame's occupancy.

    Two couplings, both orthogonal so the lattice stays bipartite and the
    compiler can colour it:

      along depth    W_MONO/2 per neighbour, which makes a column's "stopped"
                     state monotone: once it stops it stays stopped.
      across columns W_COH, which makes neighbouring columns prefer similar
                     depth. This is the term that outvotes a wrong cell.

    SIGN: this project's IR weight IS the oracle's J, not its negation,
    measured in `audit/diagnostic_control.py`. Writing a negative here produces
    a perfectly alternating chain, which is what antiferromagnetic looks like,
    and is a trap this project has walked into three times.
    """
    observed = np.asarray(inputs["occupancy"])
    if observed.ndim != 2:
        raise ValueError(
            f"occupancy must be a 2D (columns x depth) grid, got shape "
            f"{observed.shape}")

    n_cols, n_depth = observed.shape
    coupled = bool(inputs.get("couplings", True))

    def idx(c: int, k: int) -> int:
        return c * n_depth + k

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for c in range(n_cols):
        for k in range(n_depth):
            if k + 1 < n_depth:
                edges.append((idx(c, k), idx(c, k + 1)))
                weights.append(W_MONO / 2.0 if coupled else 0.0)
            if c + 1 < n_cols:
                edges.append((idx(c, k), idx(c + 1, k)))
                weights.append(W_COH if coupled else 0.0)

    order = sorted(range(len(edges)), key=lambda i: edges[i])
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(n_cols * n_depth)),
        edges=tuple(edges[i] for i in order),
        weights=np.array([weights[i] for i in order]),
        biases=biases(observed).reshape(-1).astype(float),
        beta=float(inputs.get("beta", 1.0)),
        offset=0.0,
    )


def decode(spins, shape):
    """Spins to the thing the player sees: one visible depth per column.

    `shape` is (columns, depth). The depth of the first hit is the length of
    the leading run of 1s, which is `visibility_as_inference.decode` and is
    imported rather than rewritten.
    """
    v = np.asarray(spins).reshape(shape)
    return decode_depths(v)
