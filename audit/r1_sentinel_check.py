"""R1 -- INDEPENDENT re-verification of dependency_map.md's "5 arrows never
fire" finding, by a METHOD DIFFERENT FROM sys.settrace.

dependency_map.md (Task A1) used sys.settrace: a global frame-event hook that
watches every call event and classifies it in-scope/out-of-scope after the
fact. That is a frame-tracing method. This script instead uses DIRECT
MONKEYPATCHED SENTINELS: it replaces the specific function objects under
question (insert_mediators, place, route, the search-module entry points,
torx_cross_check, and torx's own public constructors) with counting wrappers
BEFORE the app ever runs, then calls the exact same SampleWorker actions A1
used (one unclamped tick, one clamped batch, same receipt, same clamp, same
speed_idx) and reads the wrapper's own invocation count. This does not walk
frames at all -- it does not need to classify anything as "in scope" or
"out of scope" after the fact, because only the functions under test are
wrapped. If settrace's frame classification heuristic had a blind spot, a
direct wrapper around the function object itself would not share it: a
wrapped function's counter increments if and only if that exact object is
ever called, full stop.

A second, independent purpose folded in here per R1's brief ("name every
parameter actually passed to the sampler"): thrml_backend.sample and
.sample_chains are ALSO wrapped, capturing their real call kwargs directly
at the call site -- not read back off a pushed message dict the way A2's
truth table did (that reads `classify_draw`'s own sampler_params dict, which
is itself constructed by lattice_app.py, not captured at the sampler's own
call boundary). This is a second, independently-sourced confirmation of the
same numbers.

IMPORT-ORDER CAUTION (this is what makes monkeypatching correct here, and
is recorded because getting it wrong would silently produce a false
"never called" result): `insert_mediators`/`place`/`route` are each
re-imported BY NAME into other modules (`from .route import insert_mediators,
route` in search.py; `from .route import ... insert_mediators` in place.py).
A name-import binds a NEW reference in the importing module's namespace at
IMPORT TIME. Patching `route.insert_mediators` AFTER `search`/`place` have
already imported their own copy would silently miss any call routed through
those copies. This script therefore imports route/place/search/torx_backend/
thrml_backend explicitly FIRST, patches EVERY namespace holding a copy of
each target function (the defining module AND every re-importing module),
and only THEN imports `lattice_app` -- so lattice_app's own
`from tsu_compiler.backends.thrml_backend import sample as thrml_sample` (its only
top-level import of a target symbol) picks up the WRAPPED function.
`tsu_compiler.simulate.simulate` and `tsu_compiler.passes.search._verify` both import their
sampler / torx_cross_check via a LOCAL import statement INSIDE the function
body (`from .backends.thrml_backend import sample_chains`,
`from ..backends.torx_backend import torx_cross_check`) -- those re-resolve
the module attribute on every call, so patching the defining module's
attribute is sufficient for them regardless of import order.

Run with:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/r1_sentinel_check.py
from the repo root. Never runs `tsuc compile`; never starts a Tk mainloop.
"""
from __future__ import annotations

import json
import queue as queue_mod
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
SRC = REPO_ROOT / "src"
DEMO = REPO_ROOT / "demo"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DEMO))

CALLS: dict[str, list[dict]] = {}


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def wrap_fn(namespace, attr_name: str, label: str, capture_kwargs: bool = False,
            param_names: tuple[str, ...] = ()):
    """Replace namespace.attr_name with a counting wrapper; CALLS[label]
    accumulates one entry per invocation. `param_names` lets a positional
    call (e.g. sample()'s own internal `sample_chains(prog, n_chains,
    n_samples, ...)` call, which passes every scalar positionally, not by
    keyword) be captured by real parameter name too -- without this, a
    purely-kwargs-based capture would silently show an empty dict for any
    positional call, which is not "no parameters were passed", it is "this
    capture method cannot see them" -- a false negative this script must
    not produce. Returns the original for reference (unused, kept for
    symmetry / possible restoration)."""
    orig = getattr(namespace, attr_name)

    def wrapper(*args, **kwargs):
        entry = {"n_positional_args": len(args)}
        if capture_kwargs:
            named = dict(zip(param_names, args))  # positional -> name
            named.update(kwargs)  # keyword args win / fill the rest
            entry["params"] = {k: _jsonable(v) for k, v in named.items()
                               if k != "prog"}  # prog is a SamplingProgram, not JSON-able/interesting here
        CALLS.setdefault(label, []).append(entry)
        return orig(*args, **kwargs)

    wrapper.__name__ = getattr(orig, "__name__", attr_name)
    setattr(namespace, attr_name, wrapper)
    return orig


def wrap_init(cls, label: str):
    if cls is None or not hasattr(cls, "__init__"):
        return
    orig_init = cls.__init__

    def wrapper(self, *args, **kwargs):
        CALLS.setdefault(label, []).append({"n_positional_args": len(args)})
        return orig_init(self, *args, **kwargs)

    cls.__init__ = wrapper


def main() -> None:
    print(f"[R1-sentinel] repo root      : {REPO_ROOT}")
    print(f"[R1-sentinel] interpreter    : {sys.executable}")
    print(f"[R1-sentinel] python version : {sys.version.split()[0]}")

    # ------------------------------------------------------------------
    # Step 1: import the defining modules FIRST, in dependency order, and
    # patch every namespace that holds a copy of a target symbol -- see
    # module docstring's IMPORT-ORDER CAUTION.
    # ------------------------------------------------------------------
    import tsu_compiler.passes.route as route_mod
    import tsu_compiler.passes.place as place_mod
    import tsu_compiler.passes.search as search_mod
    import tsu_compiler.backends.torx_backend as torx_backend_mod
    import tsu_compiler.backends.thrml_backend as thrml_backend_mod
    import torx  # the real, installed torx package
    import torx.psc as torx_psc

    wrap_fn(route_mod, "insert_mediators", "route.insert_mediators (defining module)")
    wrap_fn(route_mod, "route", "route.route (defining module)")
    wrap_fn(place_mod, "insert_mediators", "place.insert_mediators (re-imported alias)")
    wrap_fn(place_mod, "place", "place.place (defining module)")
    wrap_fn(search_mod, "insert_mediators", "search.insert_mediators (re-imported alias)")
    wrap_fn(search_mod, "route", "search.route (re-imported alias)")
    wrap_fn(search_mod, "place", "search.place (re-imported alias)")
    wrap_fn(search_mod, "compile_spec", "search.compile_spec (public entry point)")
    wrap_fn(search_mod, "_compile_spec_impl", "search._compile_spec_impl")
    wrap_fn(search_mod, "_verify", "search._verify (where torx_cross_check is called from)")
    wrap_fn(torx_backend_mod, "torx_cross_check", "torx_backend.torx_cross_check")

    # The real torx package's own public constructors -- catches ANY use of
    # torx's documented API surface, independent of torx_backend.py's own
    # wrapper, in case some future/undiscovered call site uses torx
    # directly rather than through torx_backend.
    for cls_name in ["DFG", "ChainFactor", "TiledFactor", "DeterministicFactor",
                     "AbstractFactor", "AbstractChainFactor", "AbstractTiledFactor"]:
        wrap_init(getattr(torx, cls_name, None), f"torx.{cls_name}.__init__")
    for name in ["PISING", "DiscretePCircuit", "StateVectorSimulator"]:
        wrap_init(getattr(torx_psc, name, None), f"torx.psc.{name}.__init__")

    # The sampler itself -- independent capture of real call-site kwargs.
    # param_names matches thrml_backend.py's own signature order exactly
    # (read from source, src/tsu_compiler/backends/thrml_backend.py:181,232) so a
    # POSITIONAL call is captured by name just as accurately as a keyword
    # one.
    _sample_params = ("prog", "n_chains", "n_samples", "n_warmup",
                      "steps_per_sample", "seed")
    wrap_fn(thrml_backend_mod, "sample", "thrml_backend.sample",
           capture_kwargs=True, param_names=_sample_params)
    wrap_fn(thrml_backend_mod, "sample_chains", "thrml_backend.sample_chains",
           capture_kwargs=True, param_names=_sample_params)

    # ------------------------------------------------------------------
    # Step 2: NOW import lattice_app -- its own
    # `from tsu_compiler.backends.thrml_backend import sample as thrml_sample`
    # executes here, AFTER the patch, so it binds the WRAPPED function.
    # ------------------------------------------------------------------
    from lattice_app import Receipt, SampleWorker, RECEIPT_DIR  # noqa: E402

    print(f"[R1-sentinel] receipt dir    : {RECEIPT_DIR}")

    # ------------------------------------------------------------------
    # Step 3: run the IDENTICAL two actions A1's trace used -- same
    # receipt, same clamp, same speed_idx -- so the two methods are
    # directly comparable on the same real workload, not two different
    # workloads that happen to agree by accident.
    # ------------------------------------------------------------------
    out_q: "queue_mod.Queue[dict]" = queue_mod.Queue(maxsize=64)
    receipt = Receipt(RECEIPT_DIR)
    worker = SampleWorker(receipt, out_q, seed_base=12345, clamp=None, speed_idx=0)
    print("[R1-sentinel] running SampleWorker._run_unclamped_tick(is_step=False) ...")
    worker._run_unclamped_tick(is_step=False)

    drained = []
    while True:
        try:
            drained.append(out_q.get_nowait())
        except queue_mod.Empty:
            break
    print(f"[R1-sentinel] unclamped: {len(drained)} message(s), "
          f"kinds={[d.get('kind') for d in drained]}")

    out_q2: "queue_mod.Queue[dict]" = queue_mod.Queue(maxsize=256)
    CLAMP = {"g0_0": 0}
    worker2 = SampleWorker(receipt, out_q2, seed_base=67890, clamp=CLAMP, speed_idx=0)
    print(f"[R1-sentinel] running SampleWorker._run_clamped_batch(is_step=False), "
          f"clamp={CLAMP} ...")
    worker2._run_clamped_batch(is_step=False)

    drained2 = []
    while True:
        try:
            drained2.append(out_q2.get_nowait())
        except queue_mod.Empty:
            break
    print(f"[R1-sentinel] clamped: {len(drained2)} message(s), "
          f"kinds={[d.get('kind') for d in drained2]}")

    # ------------------------------------------------------------------
    # Step 4: report. Every label defined above is listed explicitly, even
    # at zero, so "not in the report" can never be confused with "zero
    # calls" -- an absent key would be ambiguous; a present key with an
    # empty list is not.
    # ------------------------------------------------------------------
    all_labels = [
        "route.insert_mediators (defining module)",
        "route.route (defining module)",
        "place.insert_mediators (re-imported alias)",
        "place.place (defining module)",
        "search.insert_mediators (re-imported alias)",
        "search.route (re-imported alias)",
        "search.place (re-imported alias)",
        "search.compile_spec (public entry point)",
        "search._compile_spec_impl",
        "search._verify (where torx_cross_check is called from)",
        "torx_backend.torx_cross_check",
        "torx.DFG.__init__", "torx.ChainFactor.__init__",
        "torx.TiledFactor.__init__", "torx.DeterministicFactor.__init__",
        "torx.AbstractFactor.__init__", "torx.AbstractChainFactor.__init__",
        "torx.AbstractTiledFactor.__init__",
        "torx.psc.PISING.__init__", "torx.psc.DiscretePCircuit.__init__",
        "torx.psc.StateVectorSimulator.__init__",
        "thrml_backend.sample", "thrml_backend.sample_chains",
    ]
    report = {
        "method": "monkeypatched sentinel wrappers (NOT sys.settrace)",
        "receipt_dir": str(RECEIPT_DIR),
        "unclamped_pushed_kinds": [d.get("kind") for d in drained],
        "clamped_pushed_kinds": [d.get("kind") for d in drained2],
        "sentinel_call_counts": {label: len(CALLS.get(label, [])) for label in all_labels},
        "sentinel_call_detail": {label: CALLS.get(label, []) for label in all_labels},
    }
    out_path = AUDIT_DIR / "traces" / "r1_sentinel_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[R1-sentinel] wrote {out_path.relative_to(REPO_ROOT)}")

    print("\n[R1-sentinel] SUMMARY (call counts)")
    for label in all_labels:
        n = len(CALLS.get(label, []))
        flag = "  <-- ZERO" if n == 0 else ""
        print(f"  {n:3d}  {label}{flag}")

    print("\n[R1-sentinel] sampler params actually observed at the call boundary:")
    for label in ("thrml_backend.sample", "thrml_backend.sample_chains"):
        for i, entry in enumerate(CALLS.get(label, [])):
            print(f"  {label} call #{i}: {entry.get('params', entry)}")


if __name__ == "__main__":
    main()
