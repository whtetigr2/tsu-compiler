"""Loading an Ising model from either input this tool accepts.

Two forms, because a tool usable only on this project's own YAML is a tool for
one user. The edge-list path takes a model built anywhere -- by hand, by another
compiler, by a solver someone already trusts -- and puts it through exactly the
same checks.

Every validation here REFUSES rather than repairs. Padding a short bias vector or
collapsing a duplicate edge would change the model the user asked about, and they
would get a confident report on something they did not submit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tsu_compiler.passes.lower import IsingModel, lower
from tsu_compiler.passes.encode import encode
from tsu_compiler.spec import load_spec

EDGE_SCHEMA = ('{"nodes": int, "edges": [[i, j, w], ...], '
               '"biases": [b0, ...], "beta": float}')


def _from_edges(path: Path) -> IsingModel:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        n = int(d["nodes"])
        raw = d["edges"]
        biases = [float(b) for b in d["biases"]]
        beta = float(d["beta"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: expected {EDGE_SCHEMA} ({exc})") from exc

    if len(biases) != n:
        raise ValueError(
            f"{path}: {len(biases)} biases for {n} nodes -- refusing to pad or "
            f"truncate, which would change the model you asked about")

    edges, weights, seen = [], [], set()
    for pos, item in enumerate(raw):
        try:
            i, j, w = int(item[0]), int(item[1]), float(item[2])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: edge {pos} is malformed ({item!r}) -- expected "
                f"[i, j, w] and the file must match {EDGE_SCHEMA} ({exc})"
            ) from exc
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"{path}: edge ({i},{j}) has a node index outside "
                             f"0..{n - 1}")
        if i == j:
            raise ValueError(f"{path}: edge ({i},{j}) is a self-coupling, which "
                             f"is a bias in disguise and would double-count")
        key = (min(i, j), max(i, j))
        if key in seen:
            raise ValueError(f"{path}: duplicate edge {key} -- (i,j) and (j,i) "
                             f"are the same coupling")
        seen.add(key)
        edges.append(key)
        weights.append(w)

    return IsingModel(
        nodes=tuple(f"n{i}" for i in range(n)),
        edges=tuple(edges),
        weights=np.asarray(weights, dtype=float),
        biases=np.asarray(biases, dtype=float),
        beta=beta, offset=0.0)


def load_model(spec: str | Path | None = None,
               edges: str | Path | None = None) -> IsingModel:
    """Load from a workload spec OR an edge-list JSON, never both.

    The spec path runs the same `encode` -> `lower` the rest of the toolchain
    uses, so preflight reports on the model that would actually be compiled
    rather than on a parallel reconstruction of it.
    """
    if (spec is None) == (edges is None):
        raise ValueError(
            "provide exactly one of --spec or --edges "
            f"(edge-list schema: {EDGE_SCHEMA})")
    if spec is not None:
        return lower(encode(load_spec(str(spec)), "domain_wall").model)
    return _from_edges(Path(edges))
