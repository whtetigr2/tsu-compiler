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

from .ir import Categorical, Linear, Product
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


def _failure_dict(f):
    """JSON-safe serialisation of a GateFailure or PlacementFailure -- the SAME
    object `search.py` used to reject a candidate, not a re-derived summary.
    None when the candidate carries no failure (e.g. SELECTED, or rejected by a
    bare exception message with no structured failure object behind it)."""
    if f is None:
        return None
    out = {"measured": f.measured, "limit": f.limit, "assumed": f.assumed,
           "remediations": [{"action": r.action, "detail": r.detail}
                            for r in f.remediations]}
    if hasattr(f, "gate"):
        out.update(gate=f.gate, cause=f.cause)
    else:
        out.update(failure_class=f.failure_class,
                   offending=[{"kind": o.kind, "detail": o.detail}
                             for o in f.offending])
    return out


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

    # WORKLOAD-layer counts, generic over any spec and computed ONCE here from
    # `c.spec` -- so a report renderer (report.py) reads them from the receipt
    # rather than re-deriving them by re-parsing spec.yaml at render time.
    # "state cardinality" is the largest number of distinct values any single
    # declared variable can take (a Binary variable has 2, a Categorical(k) has
    # k); "constraint classes" is the number of DISTINCT contract rule kinds in
    # use (not a raw rule count -- two `forbid_value_pair_over_edges` rules with
    # different values are the same class of check).
    workload = {
        "variables": len(c.spec.variables),
        "logical_interactions": len(c.spec.terms),
        "state_cardinality": max(
            (v.domain.k if isinstance(v.domain, Categorical) else 2
             for v in c.spec.variables), default=0),
        "constraint_classes": len({r["rule"] for r in c.spec.contract.rules}),
    }
    (d / "workload.json").write_text(json.dumps(workload, indent=2))

    (d / "target.json").write_text(json.dumps(_sourced(c.target), indent=2))

    # G1 (FORMULATION layer of `tsu explain`): WHY each spec construct
    # expanded into the IR shape it did -- recorded by spec.py's own term
    # expansion (`c.spec.formulation`), not reconstructed here. Present
    # regardless of verdict (it is a property of the SPEC, computed at parse
    # time, before any compilation stage that could fail); empty for a spec
    # whose constructs produced no terms to explain.
    (d / "formulation.json").write_text(json.dumps(
        [{"construct": r.construct, "ir_shape": r.ir_shape, "count": r.count,
          "scope": r.scope, "reason": r.reason} for r in c.spec.formulation],
        indent=2))

    # allow_assumed must be recorded so a result obtained under it can never be
    # mistaken for one obtained under sourced constraints (spec section 10), and
    # so `replay` can pass the SAME flag back to compile_spec -- without this a
    # receipt written with --allow-assumed replays without it, hits the gate it
    # was downgraded past, and reports DIVERGED (C4).
    #
    # `failure` carries the GateFailure/PlacementFailure a rejected candidate
    # hit, if any -- previously only `reason` (its free-text cause) reached the
    # receipt, so a report renderer had no structured remediation to point a
    # reader at. `_failure_dict` serialises the SAME object the compiler used
    # to reject the candidate, not a re-derived summary of it.
    passes = {"verdict": c.verdict, "ideal_passed": c.ideal_passed,
              "hardware_evaluated": c.hardware_evaluated,
              "allow_assumed": c.allow_assumed,
              "candidates": [{"encoding": x.encoding, "state": x.state.value,
                              "reason": x.reason,
                              "failure": _failure_dict(x.failure)}
                             for x in c.repset.candidates],
              "ordering_rationale": c.repset.ordering_rationale,
              # C4: wall time (seconds) per pass this compile's own verdict
              # is sourced from -- the selected candidate's pipeline on
              # COMPILED, else the best-reached candidate's on LOGICAL/
              # HARDWARE, so a rejected compile still shows the cost of the
              # passes it actually ran, not an empty dict.
              "pass_durations": dict(getattr(c, "pass_durations", None) or {}),
              # Task 2: the uniform coefficient_scale this compile ran under
              # (1.0 == unscaled) and the beta it was compensated to
              # (spec_beta(spec) / coefficient_scale) so p(x) ~
              # exp(-beta*E(x)) is unchanged by the scale -- both
              # deterministic given the spec and the scale alone, so present
              # on every verdict, never "unavailable". A scaled receipt must
              # never be mistakable for an unscaled one (search.py's own
              # `compile_spec` docstring); recording both here, verbatim
              # from `Compilation`, is what makes that true for a receipt
              # read back later, e.g. by `tsu report`/`tsu explain` or by a
              # human diffing two receipts.
              "coefficient_scale": getattr(c, "coefficient_scale", 1.0),
              "scaled_beta": getattr(c, "scaled_beta", None)}
    (d / "passes.json").write_text(json.dumps(passes, indent=2))

    # The A5 comparison table over EVERY candidate (encoding, state, reason,
    # logical spins/edges, bipartite, mediators, physical p-bits, colour
    # blocks, |J|max, |b|max) -- persisted so a report renderer can use a
    # REJECTED candidate's own measured numbers (e.g. too_dense.yaml's degree-
    # exceeded candidate still reached `analyse`) instead of reporting
    # "representation never reached" for a compile that, in fact, measured one.
    from .passes.search import compare
    (d / "candidates.json").write_text(json.dumps(compare(c.repset), indent=2))

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

    # C3 (ENERGY layer of `tsu explain`): the term inventory of the ENCODED
    # model that was actually lowered and placed -- how many of the IR's own
    # two term kinds (Linear/Product; ir.py's own docstring: "nothing else")
    # -- so `tsu explain` can state the energy model's shape without
    # re-parsing spec.yaml or recomputing anything the compiler has not
    # already computed. Empty when the compile never reached `encode` (a
    # LOGICAL verdict).
    energy = {}
    if c.encoded is not None:
        counts = {"linear": 0, "product": 0}
        for t in c.encoded.model.terms:
            counts["linear" if isinstance(t, Linear) else "product"] += 1
        energy = {"term_counts": counts, "n_terms": len(c.encoded.model.terms)}
    (d / "energy.json").write_text(json.dumps(energy, indent=2))

    (d / "regime.json").write_text(json.dumps(
        c.regime.to_dict() if c.regime else {}, indent=2))
    (d / "verification.json").write_text(json.dumps(
        c.verification.to_dict() if c.verification else {}, indent=2))

    # G2 (final APPLICATION layer of `tsu explain`): ONE concrete decoded
    # sample from this compile's own verification run -- the workload's own
    # variable names/values, never raw physical spins -- so an application
    # rendering a world has a real state to show, not just a validity
    # fraction over the whole run. {} exactly when verification never ran
    # (LOGICAL/HARDWARE verdict), same convention as regime.json/
    # verification.json above.
    (d / "sample.json").write_text(json.dumps(
        c.sample.to_dict() if c.sample else {}, indent=2))

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
                "placement": _placement(c.placement),
                # C1: the clamp this compile was run under, at BOTH layers --
                # `clamp` is the WORKLOAD-level request (what the caller
                # asked to pin, in the spec's own variable names -- present
                # even on a LOGICAL/HARDWARE receipt with no program.ising),
                # `clamped_nodes`/`clamp_values` are the physical spins it
                # translated to on the SELECTED program. A sample obtained
                # under a clamp must never be mistakable for an unconditioned
                # one, and the converse: an unclamped receipt must visibly
                # record that nothing was clamped, not omit the field.
                "clamped_nodes": list(c.program.clamped),
                "clamp_values": {str(k): v for k, v in c.program.clamp_values.items()}}
    prog["clamp"] = dict(getattr(c, "clamp", None) or {})
    (d / "program.json").write_text(json.dumps(prog, indent=2))

    # C4: compilation/sampler cost, and the CPU reference baseline. Peak
    # memory is measured (via `tracemalloc`) around the WHOLE compile_spec
    # call, so it is present regardless of verdict. `sampler`/`baseline` come
    # straight from `_verify`'s own measurement and are empty ({}) whenever
    # verification never ran (a LOGICAL or HARDWARE verdict) -- never a
    # fabricated number standing in for one.
    cost = {"peak_memory_bytes": getattr(c, "peak_memory_bytes", None),
            "sampler": dict((getattr(c, "sample_cost", None) or {}).get("sampler") or {}),
            "baseline": dict((getattr(c, "sample_cost", None) or {}).get("baseline") or {})}
    (d / "cost.json").write_text(json.dumps(cost, indent=2))

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
