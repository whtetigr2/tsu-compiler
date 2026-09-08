"""Walk an exported world.

    python demo/world_walk.py worlds/1234/world.json

Reads an EXPORTED world, never `world.generate`'s internals -- the generator is
a component with an interface, not a place to reach into. That boundary is what
lets the generator change without touching this file.

Movement is pure and unit-tested; only the key loop is interactive. Every cell
is enterable, so no move is ever refused -- terrain changes what a move COSTS,
which is the whole design.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LoadedWorld:
    size: int
    terrain: np.ndarray
    cost: np.ndarray
    glyphs: list[str]
    names: list[str]


@dataclass(frozen=True)
class PlayerState:
    x: int
    y: int
    spent: int
    steps: int


def load(path) -> LoadedWorld:
    """Glyphs travel IN THE ARTIFACT rather than being hardcoded here or
    imported from `world.spline`. Hardcoding duplicates a literal that nothing
    would catch drifting; importing spline would break this module's rule of
    consuming only the export. The artifact carries them, so neither is needed."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return LoadedWorld(size=d["size"],
                       terrain=np.array(d["terrain"], dtype=int),
                       cost=np.array(d["cost"], dtype=int),
                       glyphs=d["terrain_glyphs"],
                       names=d["terrain_names"])


def step(state: PlayerState, dx: int, dy: int, world: LoadedWorld) -> PlayerState:
    """Move one cell, charging the cost of the cell ENTERED. A move into the
    boundary is refused entirely -- no position change, no cost, no step."""
    nx, ny = state.x + dx, state.y + dy
    if not (0 <= nx < world.size and 0 <= ny < world.size):
        return state
    return replace(state, x=nx, y=ny,
                   spent=state.spent + int(world.cost[ny, nx]),
                   steps=state.steps + 1)


def render(state: PlayerState, world: LoadedWorld, radius: int = 12) -> str:
    """A (2*radius+1) square window centred on the player, edge-clamped so the
    window is always the same size no matter where the player stands."""
    n = world.size
    side = 2 * radius + 1
    y0 = min(max(0, state.y - radius), max(0, n - side))
    x0 = min(max(0, state.x - radius), max(0, n - side))
    rows = []
    for y in range(y0, y0 + side):
        row = []
        for x in range(x0, x0 + side):
            row.append("@" if (x, y) == (state.x, state.y)
                       else world.glyphs[world.terrain[y, x]])
        rows.append("".join(row))
    return "\n".join(rows)


_KEYS = {"w": (0, -1), "s": (0, 1), "a": (-1, 0), "d": (1, 0)}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python demo/world_walk.py <path to world.json>")
        raise SystemExit(2)
    world = load(sys.argv[1])
    state = PlayerState(x=world.size // 2, y=world.size // 2, spent=0, steps=0)
    print("WASD then Enter to move, q to quit.\n")
    while True:
        print(render(state, world))
        here = world.names[world.terrain[state.y, state.x]]
        print(f"  ({state.x},{state.y}) {here}  cost here "
              f"{world.cost[state.y, state.x]}  steps {state.steps}  "
              f"spent {state.spent}")
        try:
            keys = input("> ").strip().lower()
        except EOFError:
            return
        if keys.startswith("q"):
            return
        for k in keys:
            if k in _KEYS:
                state = step(state, *_KEYS[k], world)


if __name__ == "__main__":
    main()
