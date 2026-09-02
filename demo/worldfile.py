"""A2: save and load a world as evidence, not just pixels.

A saved world is a claim about a sampled, decoded, contract-checked grid --
so the file carries the provenance that claim rests on (which spec, which
receipt, which seed, which clamp, which sampler params, when) alongside the
grid values themselves. One JSON file per world.

Task 4 EXTENDS this with a STACK format (`save_stack`/`load_stack`,
`StackLayer`/`Stack`) for saving several layers of the same world (a base
plus its overlay bands) in one file, each layer carrying the same kind of
provenance plus which layer it is and what conditioning patch it was
sampled under. This is an ADDITION alongside the single-grid format below,
never a replacement -- every single-grid save/load test in
tests/test_worldfile.py keeps passing against the unmodified format.

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


STACK_LAYER_FIELDS = (
    "name", "spec_name", "receipt_dir", "seed", "clamp", "conditioning_patch",
    "width", "height", "values", "value_names", "task_valid", "violations",
    "sampler_params",
)
STACK_FIELDS = ("layers", "saved_at")


@dataclass(frozen=True)
class StackLayer:
    """One layer's worth of evidence within a saved stack -- the SAME
    per-layer provenance `World` carries above, plus the two things a
    LAYER specifically needs and a single world does not: `name` (which
    layer this is -- "base", "band0", "band1", ...) and
    `conditioning_patch` (the {(cell_name, value): weight} bias patch --
    see demo/layers.bias_patch -- this layer was sampled under; an empty
    dict for a layer with no conditioning, i.e. the base)."""
    name: str
    spec_name: str
    receipt_dir: str
    seed: int
    clamp: dict[str, int]
    conditioning_patch: dict[tuple[str, int], float]
    width: int
    height: int
    values: list[int]
    value_names: list[str]
    task_valid: bool
    violations: list[str] = field(default_factory=list)
    sampler_params: dict[str, Any] = field(default_factory=dict)

    def grid(self) -> np.ndarray:
        """Same row-major reshape World.grid() does -- see that method's
        own docstring for the orientation convention."""
        return np.array(self.values, dtype=int).reshape(self.height, self.width)


@dataclass(frozen=True)
class Stack:
    """Everything `load_stack` read back out of a saved stack file,
    verbatim -- no field here is recomputed or reinterpreted from the
    others, same honesty rule `World`/`load_world` follow."""
    layers: tuple[StackLayer, ...]
    saved_at: str


def save_stack(path, *, layers: Sequence[Mapping[str, Any]],
                saved_at: str) -> Path:
    """Write a STACK of layers (base + N overlay bands, or any ordered
    sequence of layers over the same grid shape) as ONE JSON file at
    `path` -- an ADDITIONAL format alongside `save_world`/`load_world`
    above, never a replacement: every single-grid test in this module
    keeps passing against the unmodified single-grid format.

    Each element of `layers` is a mapping carrying the fields listed in
    STACK_LAYER_FIELDS -- the same per-layer provenance `save_world`
    already requires (spec_name, receipt_dir, seed, clamp, width, height,
    values, value_names, task_valid, violations, sampler_params) plus
    `name` and `conditioning_patch` (see `StackLayer`'s own docstring).
    `clamp` and `conditioning_patch` may be omitted/empty for a layer that
    carries neither (e.g. the base layer's `conditioning_patch` is always
    empty -- nothing conditions the base on anything).

    Raises ValueError, naming the offending layer's index and name, if
    that layer's `values` length doesn't match `width * height` -- the
    same shape guard `save_world` applies to a single grid, per layer here.
    """
    docs = []
    for i, layer in enumerate(layers):
        name = str(layer["name"])
        values = list(layer["values"])
        width, height = int(layer["width"]), int(layer["height"])
        expected = width * height
        if len(values) != expected:
            raise ValueError(
                f"layer {i} ({name!r}): values length {len(values)} does "
                f"not match width*height ({width}*{height}={expected})")

        patch = layer.get("conditioning_patch") or {}
        patch_list = [[str(cell), int(value), float(weight)]
                      for (cell, value), weight in patch.items()]

        docs.append({
            "name": name,
            "spec_name": str(layer["spec_name"]),
            "receipt_dir": str(layer["receipt_dir"]),
            "seed": int(layer["seed"]),
            "clamp": dict(layer.get("clamp") or {}),
            "conditioning_patch": patch_list,
            "width": width,
            "height": height,
            "values": [int(v) for v in values],
            "value_names": list(layer["value_names"]),
            "task_valid": bool(layer["task_valid"]),
            "violations": list(layer.get("violations") or []),
            "sampler_params": dict(layer.get("sampler_params") or {}),
        })

    doc = {"layers": docs, "saved_at": str(saved_at)}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2))
    return p


def load_stack(path, *, allow_invalid: bool = False) -> Stack:
    """Read a stack back from `path`.

    Raises ValueError, naming every layer whose recorded `task_valid` is
    False, unless `allow_invalid` is True -- the SAME "a saved invalid
    [layer] must never silently reload as a good one" rule `load_world`
    applies to a single world, applied per layer here: a stack with one
    bad layer among several good ones is still not silently loadable,
    because a caller reading `stack.layers[i]` has no other cue that layer
    is the one that broke.
    """
    p = Path(path)
    doc = json.loads(p.read_text())

    missing = [f for f in STACK_FIELDS if f not in doc]
    if missing:
        raise ValueError(f"stack file {p} is missing field(s): {missing}")

    layers: list[StackLayer] = []
    invalid = []
    for i, ld in enumerate(doc["layers"]):
        layer_missing = [f for f in STACK_LAYER_FIELDS if f not in ld]
        if layer_missing:
            raise ValueError(
                f"stack file {p} layer {i} is missing field(s): {layer_missing}")
        if not ld["task_valid"]:
            invalid.append((i, ld["name"], ld["violations"]))
        patch = {(str(cell), int(value)): float(weight)
                 for cell, value, weight in ld["conditioning_patch"]}
        layers.append(StackLayer(
            name=ld["name"],
            spec_name=ld["spec_name"],
            receipt_dir=ld["receipt_dir"],
            seed=ld["seed"],
            clamp=dict(ld["clamp"]),
            conditioning_patch=patch,
            width=ld["width"],
            height=ld["height"],
            values=list(ld["values"]),
            value_names=list(ld["value_names"]),
            task_valid=bool(ld["task_valid"]),
            violations=list(ld["violations"]),
            sampler_params=dict(ld["sampler_params"]),
        ))

    if invalid and not allow_invalid:
        raise ValueError(
            f"stack file {p} has {len(invalid)} invalid layer(s) "
            f"(task_valid=False): {invalid!r} -- pass allow_invalid=True "
            f"to load it anyway; a saved invalid layer must never "
            f"silently reload as a good one")

    return Stack(layers=tuple(layers), saved_at=doc["saved_at"])


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
