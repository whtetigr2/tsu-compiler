"""Task 1 of the all-bipartite-world plan
(SPR/docs/superpowers/plans/2026-09-07-all-bipartite-world.md).

`audit/placement_curve.py` already measured the bipartite binary overlay
placing in 0.0s through 20x20 (400 spins), against a non-bipartite k=3 base
layer that costs 6-32 minutes to FAIL above 8x8. This module pushes the
overlay curve further -- 16x16, 24x24, 32x32, 40x40, 64x64 -- to find where
(if anywhere) it stops placing instantly, and to confirm two things that
would change the conclusion if they didn't hold: that every size actually
takes the deterministic `_try_grid_embed` code path (not a lucky fast
annealer run), and that |J|max/|b|max don't drift with size toward the 6.0
cap.

SAFETY RULE (non-negotiable, see the plan's Global Constraints and Task 1):
`place()` on a non-bipartite graph above 8x8 costs 6-32 minutes to FAIL.
`bipartite_at` checks `analyse(...).bipartite is True` and returns a
SKIPPED row WITHOUT calling `place()` whenever it is False. This is a
finding, not a launch signal, for any future kind added to `_KINDS` that
turns out non-bipartite at some size.

Zero workload-specific logic: `kind` selects a generic term SHAPE (a
self-rule `product_over_edges` on a grid), never a domain concept. Only
"overlay" (binary, self-rule) is defined here -- Tasks 2/3 add more kinds
to this same table under their own commits, not this one.

Writes incrementally (json.dump after every row) so a killed run keeps
every completed measurement -- run with `python -u` for unbuffered stdout
on top of that.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "src")

import networkx as nx

from tsu.failures import CompileError
from tsu.passes.analyse import analyse
from tsu.passes.encode import encode
from tsu.passes.lower import lower
from tsu.passes.place import _try_grid_embed, place
from tsu.spec import load_spec
from tsu.target import PROFILES

# Same spec shape audit/placement_curve.py's OVL already uses: one binary
# variable per grid cell, a single self-rule (a_value == b_value == 1) at
# negative weight -- a clumping term, not a forbidden-pair cross-rule, which
# is exactly what keeps it bipartite (see the plan's "measured situation").
_OVERLAY_BODY = """name: o
generate: {{kind: grid, width: {n}, height: {n}, variable_domain: {{domain: binary}}}}
terms:
  - {{kind: product_over_edges, a_value: 1, b_value: 1, weight: -0.4}}
"""

_KINDS = {"overlay": _OVERLAY_BODY}


def bipartite_at(n: int, kind: str = "overlay") -> dict:
    """Measure one grid size for one layer kind. Returns the fields
    audit/bipartite_matrix.md reports: n_nodes, n_edges, max_degree,
    bipartite, max_abs_J, max_abs_b, place_s, mediators, result, and
    code_path (which branch of place() fired -- see module docstring).

    THE SAFETY RULE lives here: place() is called only when
    `analyse(...).bipartite is True`. A False comes back as its own row
    (result="SKIPPED_non_bipartite", place_s=None) with no placement
    attempted -- verified BEFORE every place() call, never inferred
    afterward.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; available: {sorted(_KINDS)}")

    f = os.path.join(tempfile.gettempdir(), f"bipartite_routes_{kind}_{n}.yaml")
    with open(f, "w") as fh:
        fh.write(_KINDS[kind].format(n=n))

    m = lower(encode(load_spec(f), "domain_wall").model)
    rep = analyse(m)

    row = dict(grid=n, kind=kind, n_nodes=rep.n_nodes, n_edges=rep.n_edges,
               max_degree=rep.max_degree, bipartite=rep.bipartite,
               max_abs_J=rep.max_abs_J, max_abs_b=rep.max_abs_b)

    if not rep.bipartite:
        # THE SAFETY RULE. place() on a non-bipartite graph above 8x8 is
        # measured at 6-32 minutes to FAIL (audit/placement_curve.json,
        # base_k3 rows). Never launch it speculatively -- a False here IS
        # the finding for this (kind, n); report it and stop.
        row.update(place_s=None, result="SKIPPED_non_bipartite",
                    mediators=None, code_path="not_attempted")
        return row

    target = PROFILES["z1"]

    # Confirm -- don't infer from a fast place_s -- which branch of
    # _embed_on_lattice fires. A 0.0s result is consistent with the
    # deterministic grid embed, but it would ALSO be consistent with the
    # seeded annealer getting lucky on a small/sparse graph; those are
    # different (one has no seed dependence or budget limit, the other
    # does) and the plan calls out exactly this ambiguity to resolve.
    # `_try_grid_embed` is the same function, called on the same graph
    # `_embed_on_lattice` would build, so this reproduces its verdict
    # exactly rather than approximating it.
    G = nx.Graph()
    G.add_nodes_from(range(rep.n_nodes))
    G.add_edges_from(m.edges)
    grid_coords = _try_grid_embed(G)
    code_path = "grid_embed" if grid_coords is not None else "annealer_fallback"

    t0 = time.perf_counter()
    try:
        p = place(m, rep, target, restarts=12, iters=200_000)
        el = time.perf_counter() - t0
        row.update(place_s=round(el, 4), result="PLACED",
                    mediators=(p.mediation.mediator_count if p.mediation else 0),
                    code_path=code_path)
    except CompileError as e:
        el = time.perf_counter() - t0
        row.update(place_s=round(el, 4), result=e.failures[0].failure_class,
                    mediators=None, code_path=code_path)
    return row


if __name__ == "__main__":
    out = "audit/bipartite_matrix.json"
    rows = []
    for n in (16, 24, 32, 40, 64):
        r = bipartite_at(n, "overlay")
        rows.append(r)
        print(f"{n}x{n} overlay nodes={r['n_nodes']:>5} bip={r['bipartite']} "
              f"path={r['code_path']:<17} {r['place_s']}s {r['result']} "
              f"med={r['mediators']} |J|max={r['max_abs_J']} |b|max={r['max_abs_b']}",
              flush=True)
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=1)
    print("done ->", out)
