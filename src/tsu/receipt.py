"""Receipts are first-class and replayable. A receipt that does not replay is a
defect in the compiler (spec section 10)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .spec import load_spec
from .target import PROFILES, Sourced


def _sourced(t):
    out = {}
    for f in fields(t):
        v = getattr(t, f.name)
        if isinstance(v, Sourced):
            val = list(v.value) if isinstance(v.value, tuple) else v.value
            out[f.name] = {"value": val, "source": v.source, "quote": v.quote}
        else:
            out[f.name] = v
    return out


def _placement(p):
    """JSON-safe serialisation of a Placement: coords keyed by string node index,
    realized/unrealized as lists of [u, v] pairs. None when the compile never
    reached placement (LOGICAL/HARDWARE verdicts)."""
    if p is None:
        return None
    return {
        "coords": {str(k): list(v) for k, v in p.coords.items()},
        "realized": [list(e) for e in p.realized],
        "unrealized": [list(e) for e in p.unrealized],
    }


def _env():
    # Nothing outside src/tsu/backends/ may import thrml directly; obtain its
    # version via importlib.metadata instead of `import thrml`.
    import networkx, sympy
    try:
        tv = importlib.metadata.version("thrml")
    except importlib.metadata.PackageNotFoundError:
        tv = "unavailable: thrml package metadata not found"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "networkx": networkx.__version__,
            "sympy": sympy.__version__, "thrml": tv}


@dataclass(frozen=True)
class ReplayResult:
    matches: bool
    diffs: tuple


def write_receipt(c, out_dir) -> Path:
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)

    (d / "spec.yaml").write_text(c.spec.source_text, encoding="utf-8")
    digest = hashlib.sha256(c.spec.source_text.encode("utf-8")).hexdigest()
    (d / "spec.sha256").write_text(digest, encoding="utf-8")

    (d / "target.json").write_text(json.dumps(_sourced(c.target), indent=2))

    # allow_assumed must be recorded so a result obtained under it can never be
    # mistaken for one obtained under sourced constraints (spec section 10), and
    # so `replay` can pass the SAME flag back to compile_spec -- without this a
    # receipt written with --allow-assumed replays without it, hits the gate it
    # was downgraded past, and reports DIVERGED (C4).
    passes = {"verdict": c.verdict, "ideal_passed": c.ideal_passed,
              "hardware_evaluated": c.hardware_evaluated,
              "allow_assumed": c.allow_assumed,
              "candidates": [{"encoding": x.encoding, "state": x.state.value,
                              "reason": x.reason} for x in c.repset.candidates],
              "ordering_rationale": c.repset.ordering_rationale}
    (d / "passes.json").write_text(json.dumps(passes, indent=2))

    # Every gate this compile evaluated against `target`, passed or failed --
    # not only the ones that aborted it (spec section 10: "every gate,
    # passed/failed, with the measured value and threshold"). `downgraded` marks
    # a gate that would have failed but was overridden by --allow-assumed, so a
    # receipt written under the override is visibly distinguishable from one
    # obtained under sourced constraints (C4).
    gates = [{"gate": g.gate, "passed": g.passed, "measured": g.measured,
             "limit": g.limit, "assumed": g.assumed, "downgraded": g.downgraded}
            for g in c.gate_checks]
    (d / "gates.json").write_text(json.dumps(gates, indent=2))

    rep = c.repset.selected.report if c.repset.selected else None
    # rep.mediators == -1 means "not computed" (the graph exceeded
    # MAXCUT_EXACT_LIMIT), not zero mediators. Publishing -1 verbatim is the same
    # sentinel-as-a-published-fact error C1 fixes for placement's own
    # remediation, one field over (I5): a reader of metrics.json cannot tell "-1
    # mediators" from a real (impossible) measurement without already knowing
    # this file's internals. `mediators_note` makes the "not computed" case
    # explicit instead.
    metrics = {} if rep is None else {
        "n_nodes": rep.n_nodes, "n_edges": rep.n_edges,
        "max_degree": rep.max_degree, "bipartite": rep.bipartite,
        "mediators": rep.mediators if rep.mediators >= 0 else None,
        "mediators_note": ("" if rep.mediators >= 0 else
                           "not computed: graph exceeds the exact max-cut "
                           "limit (MAXCUT_EXACT_LIMIT)"),
        "colour_blocks": rep.colour_blocks,
        "max_abs_J": rep.max_abs_J, "max_abs_b": rep.max_abs_b}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))

    (d / "regime.json").write_text(json.dumps(
        c.regime.to_dict() if c.regime else {}, indent=2))
    (d / "verification.json").write_text(json.dumps(
        c.verification.to_dict() if c.verification else {}, indent=2))

    # Deviation from the brief: the placement must reach the receipt too, not
    # just the sampling program -- coords/realized/unrealized carry the
    # geometric evidence behind a COMPILED verdict (or None otherwise).
    prog = {}
    if c.program is not None:
        im = c.program.ising
        prog = {"form": c.program.form,
                "nodes": list(im.nodes), "edges": [list(e) for e in im.edges],
                "weights": im.weights.tolist(), "biases": im.biases.tolist(),
                "beta": im.beta, "offset": im.offset,
                "blocks": [list(b) for b in c.program.blocks],
                "schedule": c.program.schedule,
                "placement": _placement(c.placement)}
    (d / "program.json").write_text(json.dumps(prog, indent=2))

    (d / "environment.json").write_text(json.dumps(_env(), indent=2))
    return d


def load_receipt(d) -> dict:
    d = Path(d)
    return {p.stem: (json.loads(p.read_text()) if p.suffix == ".json"
                     else p.read_text(encoding="utf-8"))
            for p in d.iterdir() if p.is_file()}


def replay(d) -> ReplayResult:
    from .passes.search import compile_spec
    d = Path(d)
    recorded = json.loads((d / "program.json").read_text())
    target = PROFILES[json.loads((d / "target.json").read_text())["name"]]
    # Read the SAME allow_assumed flag the receipt was compiled with -- recompiling
    # without it would hit a gate that was legitimately downgraded and report
    # DIVERGED for a receipt that is not actually wrong (C4).
    allow_assumed = json.loads((d / "passes.json").read_text()).get(
        "allow_assumed", False)

    spec = load_spec(str(d / "spec.yaml"))
    if hashlib.sha256(spec.source_text.encode()).hexdigest() != \
            (d / "spec.sha256").read_text().strip():
        return ReplayResult(False, ("spec hash changed",))

    fresh = compile_spec(spec, target, allow_assumed=allow_assumed)
    if fresh.program is None:
        return ReplayResult(not recorded, ("no program on replay",))

    im = fresh.program.ising
    diffs = []
    if list(im.nodes) != recorded["nodes"]:
        diffs.append("nodes")
    if [list(e) for e in im.edges] != recorded["edges"]:
        diffs.append("edges")
    if not np.allclose(im.weights, recorded["weights"], atol=1e-12):
        diffs.append("weights")
    if not np.allclose(im.biases, recorded["biases"], atol=1e-12):
        diffs.append("biases")
    return ReplayResult(not diffs, tuple(diffs))
