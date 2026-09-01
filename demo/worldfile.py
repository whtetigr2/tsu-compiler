"""A2: save and load a world as evidence, not just pixels.

A saved world is a claim about a sampled, decoded, contract-checked grid --
so the file carries the provenance that claim rests on (which spec, which
receipt, which seed, which clamp, which sampler params, when) alongside the
grid values themselves. One JSON file per world.

Nothing here calls `simulate`, `thrml`, or `torx` -- this module only reads
and writes the already-decoded result of a sampling run; it never samples.

Honesty rule enforced here (not just documented): a world whose recorded
`task_valid` is False must never silently reload as if it were a good one.
`load_world` raises unless the caller explicitly passes `allow_invalid=True`
-- the same "no fabrication, no laundering a failure into a success" spirit
CLAUDE.md's epistemic-discipline rule and this project's own
verification-never-fabricates rule apply everywhere else in this codebase.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FORMAT_FIELDS = (
    "spec_name", "receipt_dir", "seed", "clamp", "width", "height",
    "values", "value_names", "task_valid", "violations", "sampler_params",
    "saved_at",
)


@dataclass(frozen=True)
class World:
    """Everything `load_world` read back out of a saved world file, verbatim
    -- no field here is recomputed or reinterpreted from the others."""
    spec_name: str
    receipt_dir: str
    seed: int
    clamp: dict[str, int]
    width: int
    height: int
    values: list[int]              # row-major, len == width * height
    value_names: list[str]
    task_valid: bool
    violations: list[str] = field(default_factory=list)
    sampler_params: dict[str, Any] = field(default_factory=dict)
    saved_at: str = ""

    def grid(self) -> np.ndarray:
        """Reshape `values` (row-major: row = y, column = x) into a
        (height, width) array -- the same orientation
        `demo/render_world.py`/`demo/lattice_app.py` build their grids in
        (`grid[y, x]`)."""
        return np.array(self.values, dtype=int).reshape(self.height, self.width)


def save_world(path, *, spec_name: str, receipt_dir: str, seed: int,
                clamp: Mapping[str, int], width: int, height: int,
                values: Sequence[int], value_names: Sequence[str],
                task_valid: bool, violations: Sequence[str],
                sampler_params: Mapping[str, Any], saved_at: str) -> Path:
    """Write one world as JSON to `path`. Raises ValueError if `values`'
    length doesn't match `width * height` -- a shape mismatch is a bug in
    the caller, caught here rather than silently misread back at load time.
    """
    values = list(values)
    expected = width * height
    if len(values) != expected:
        raise ValueError(
            f"values length {len(values)} does not match width*height "
            f"({width}*{height}={expected})")

    doc = {
        "spec_name": spec_name,
        "receipt_dir": str(receipt_dir),
        "seed": int(seed),
        "clamp": dict(clamp),
        "width": int(width),
        "height": int(height),
        "values": [int(v) for v in values],
        "value_names": list(value_names),
        "task_valid": bool(task_valid),
        "violations": list(violations),
        "sampler_params": dict(sampler_params),
        "saved_at": str(saved_at),
    }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2))
    return p


def load_world(path, *, allow_invalid: bool = False) -> World:
    """Read a world back from `path`.

    Raises ValueError if the recorded `task_valid` is False and
    `allow_invalid` is not True -- a saved invalid world must never be
    silently reloaded as a good one; the caller has to explicitly ask for
    it, in which case World.task_valid still reports False so the caller
    can't lose track of it either.
    """
    p = Path(path)
    doc = json.loads(p.read_text())

    missing = [f for f in FORMAT_FIELDS if f not in doc]
    if missing:
        raise ValueError(f"world file {p} is missing field(s): {missing}")

    if not doc["task_valid"] and not allow_invalid:
        raise ValueError(
            f"world file {p} has task_valid=False (violations="
            f"{doc['violations']!r}) -- pass allow_invalid=True to load it "
            f"anyway; a saved invalid world must never silently reload as "
            f"a good one")

    return World(
        spec_name=doc["spec_name"],
        receipt_dir=doc["receipt_dir"],
        seed=doc["seed"],
        clamp=dict(doc["clamp"]),
        width=doc["width"],
        height=doc["height"],
        values=list(doc["values"]),
        value_names=list(doc["value_names"]),
        task_valid=bool(doc["task_valid"]),
        violations=list(doc["violations"]),
        sampler_params=dict(doc["sampler_params"]),
        saved_at=doc["saved_at"],
    )
