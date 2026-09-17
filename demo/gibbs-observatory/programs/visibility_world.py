"""A walkable level whose visibility is solved by sampling, not by raycasting.

`visibility_game.py` takes occupancy along each ray as its input, which is the
form the audit scripts verify. That is the physics, and it is not yet a game:
something has to turn "where the player is standing" into those rays.

This does that. The port is the player's POSE -- three numbers -- and the
program owns the world and the camera. So a frame costs three floats on the
wire rather than 1536, and the game logic lives in the program, which is what
the program contract is for.

The energy is not rewritten here. `build` below casts the rays and then hands
them to `visibility_game.build`, so there is exactly one definition of this
model in the repo and it is the one checked against an exact raycast. Copying
those forty lines to avoid an import is how two files stop describing the same
model.

WHAT IS AND IS NOT BEING CLAIMED. The ray march is ordinary raycasting: it
decides which cells of the world lie along each column, which is geometry and
is not the interesting part. What the sampler does is decide, from noisy and
possibly WRONG occupancy, where each column actually stops -- and it can
overrule a bad reading because neighbouring columns couple. A plain raycaster
cannot: it stops at the first thing it is told about, right or wrong. That
difference is the whole reason this is a thermodynamic program rather than a
loop.

No hardware claim. THRML simulates on CPU.
"""
from __future__ import annotations

import numpy as np

from visibility_game import build as build_lattice, decode as decode_depths

NAME = "visibility_world"
DECODER = "firstperson"

#: How the spins are laid out. The port is a three-number pose, so the
#: decoder cannot get this from the port and the program has to say it.
LATTICE_SHAPE = [48, 32]

N_COLS, N_DEPTH = 48, 32

#: Horizontal field of view, radians. 60 degrees, the usual choice, and wide
#: enough that the coupling between neighbouring columns has something to do.
FOV = np.pi / 3.0

#: How far one depth step carries, in world cells. N_DEPTH * STEP is the far
#: plane, so this and N_DEPTH together set how far the player can see.
STEP = 0.35

#: A room with things in it. 1 is solid. The outer ring is wall so the player
#: cannot walk out of the world, and the interior has pillars at different
#: distances so that near and far geometry are both on screen at once.
_W, _H = 24, 24


def _make_world() -> np.ndarray:
    world = np.zeros((_H, _W), dtype=bool)
    world[0, :] = world[-1, :] = True
    world[:, 0] = world[:, -1] = True
    # Pillars, deliberately of different sizes and distances.
    world[6:9, 6:9] = True
    world[5:7, 16:19] = True
    world[15:20, 5:7] = True
    world[17:19, 15:17] = True
    world[11:13, 11:13] = True
    # A short wall that a column of rays will graze, which is where the
    # coupling between neighbouring columns shows up most clearly.
    world[13, 3:10] = True
    return world


WORLD = _make_world()

#: Where the player starts, and which way they face. Chosen to be inside the
#: room and not inside a pillar.
START_POSE = (3.5, 3.5, 0.6)

#: Spurious walls per cell, in the occupancy the sampler is handed. NOT zero
#: on purpose. With clean input this program reproduces a raycast, which is not
#: the claim being made: a raycaster stops at the first thing it is told about,
#: right or wrong, and the reason to sample is that neighbouring columns couple
#: and can overrule a cell that was read wrong. With no noise there is nothing
#: to overrule and the demonstration demonstrates nothing.
#:
#: 0.08 is the rate audit/visibility_on_thrml.py measures at, where the
#: couplings take agreement with the exact raycast from 10.4% to 60.4%.
DEFAULT_NOISE = 0.08

INPUT_PORTS = [
    {
        "name": "pose",
        "shape": [3],
        "mode": "bias",
        "doc": "Player pose as (x, y, heading in radians). It enters as a "
               "bias: the rays it produces push each cell toward 'stopped' or "
               "'keep going' rather than pinning it, so the couplings can "
               "overrule a wrong reading.",
    },
    {
        "name": "noise",
        "shape": [1],
        "mode": "bias",
        "doc": "Rate of spurious walls in the occupancy handed to the "
               "sampler. At 0 this reproduces a raycast; above 0 the coupling "
               "between neighbouring columns has something to do.",
    },
]

DEFAULT_INPUTS = {
    "pose": np.array(START_POSE, dtype=float),
    # Declared explicitly: the runtime fills an unsupplied port with
    # ZEROS of its shape, and zero noise is the one setting under
    # which this program demonstrates nothing.
    "noise": DEFAULT_NOISE,
}


def occupancy_from_pose(pose, world=None, *, n_cols: int = N_COLS,
                        n_depth: int = N_DEPTH) -> np.ndarray:
    """March one ray per column and record what it passes through.

    This is the geometry half and it is deliberately dumb: it says which world
    cells lie along each column, nothing about where the column stops. Deciding
    where it stops is the sampler's job, and that is the half that can be wrong
    about a cell and recover.
    """
    world = WORLD if world is None else world
    x, y, heading = (float(v) for v in np.asarray(pose).reshape(-1)[:3])
    h, w = world.shape

    cols = np.arange(n_cols)
    angles = heading + FOV * (cols / max(n_cols - 1, 1) - 0.5)
    steps = (np.arange(n_depth) + 1) * STEP

    # (n_cols, n_depth) sample points, then one lookup for the whole grid.
    px = x + np.cos(angles)[:, None] * steps[None, :]
    py = y + np.sin(angles)[:, None] * steps[None, :]
    ix = np.clip(np.floor(px).astype(int), 0, w - 1)
    iy = np.clip(np.floor(py).astype(int), 0, h - 1)
    occ = world[iy, ix]

    # Anything past the edge of the world reads as solid, so a ray that leaves
    # the room stops at the boundary rather than running to the far plane.
    outside = (px < 0) | (px >= w) | (py < 0) | (py >= h)
    occ = occ | outside

    # The far plane is always solid: a column that hits nothing stops at the
    # end of its own range rather than reporting "infinitely far", which the
    # decoder would have no way to draw.
    occ[:, -1] = True
    return occ


def _scalar(value, default: float) -> float:
    """One number from a port.

    Ports carry arrays -- the runtime fills an unsupplied one with zeros of
    its declared shape -- and a one-element array is not a Python float.
    """
    if value is None:
        return float(default)
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else float(default)


def _corrupt(occ: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Spurious walls in free space: the failure a scan cannot recover from.

    Only ever ADDS walls, never removes them, which is the asymmetry that
    matters. A missing wall makes a ray run long and the next cell catches it;
    a phantom wall stops the ray dead and nothing downstream can undo it. That
    is precisely what a raycaster cannot recover from and what coupled columns
    can.
    """
    if rate <= 0:
        return occ
    rng = np.random.default_rng(seed)
    return occ | ((rng.random(occ.shape) < rate) & ~occ)


def build(inputs):
    """Rays from the pose, corrupted, then the verified energy over them."""
    occ = occupancy_from_pose(inputs.get("pose", START_POSE))
    occ = _corrupt(occ, _scalar(inputs.get("noise"), DEFAULT_NOISE),
                   int(_scalar(inputs.get("noise_seed"), 0)))
    # The far plane stays solid whatever the noise did, so a column still has
    # somewhere to stop.
    occ[:, -1] = True
    return build_lattice({
        "occupancy": occ,
        "beta": inputs.get("beta", 1.0),
        "couplings": inputs.get("couplings", True),
    })


def decode(spins, shape):
    """Spins to one visible depth per column, which is what the view draws."""
    return decode_depths(spins, shape)
