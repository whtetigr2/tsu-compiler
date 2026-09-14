"""Task A1 -- the RUNTIME dependency map, evidence-gathering script.

Import graphs lie: a module can be imported and never called, or reached only
through a branch nobody takes. This script does NOT read the code and infer
the call path -- it INSTRUMENTS it, using `sys.settrace`, while the app does
exactly what a user does: one UNCLAMPED draw, then one CLAMPED draw, both
starting from `demo.lattice_app.SampleWorker` (never `tsuc compile` -- the
receipt at demo/receipts/small is loaded exactly the way `demo/lattice_app.
main()` loads it, read-only, and the SampleWorker's own `_run_unclamped_tick`
/ `_run_clamped_batch` methods are called directly, off the Tk event loop,
which `tests/test_lattice_app_logic.py` already establishes is safe: those
methods touch no Tk state, only `self.receipt` / `self.q` / thrml).

WHY THE TRACE IS BOUNDED, NOT FREE-RUNNING (read this before trusting the
"opaque" claim in dependency_map.md): thrml's sampler goes through
`jax.jit`. The FIRST time a given (n_chains, n_samples, ...) shape is jitted
in a process, JAX Python-traces the function to build an XLA computation
graph -- this can mean hundreds of thousands of Python-level calls into
jax's own tracing machinery, none of which correspond to a step of the
Markov chain a physicist would recognise. After that shape is cached,
JAX dispatches through a fast C++ path with very few Python frames, so a
naive `sys.settrace` may see almost nothing on a warm call. Both regimes
are real; which one you hit depends on cache state, not on the sampler.
`BoundedTracer` below therefore:

  1. Records the REAL call sequence (caller -> callee, file:line, in
     delivery order, capped at a few thousand raw entries but with an
     UNCAPPED de-duplicated edge-count summary covering the whole run) for
     every call whose code object lives under `src/` or `demo/` -- OUR
     code, the thing this task is actually mapping.

  2. The instant execution crosses INTO external code (jax, thrml, torx,
     numpy, PIL, tkinter, the stdlib -- anything under site-packages or
     the standard library), it stops recording individual frames and
     instead tallies, per top-level package, how many calls landed there,
     plus the exact (our function -> external package) boundary edge that
     was crossed, deduplicated with a hit count. This is the "bound the
     trace at the backend boundary" the audit brief calls for: the
     interior of a jit-compiled kernel is not where this compiler's own
     logic lives, and is not lied about by being silently omitted -- its
     volume is reported, never its (nonexistent, from Python's view)
     detail.

  3. Two hard caps (MAX_CALL_EVENTS, MAX_WALL_SECONDS) guarantee this
     script terminates and reports an honest "trace aborted: <reason>"
     finding rather than hanging or exhausting memory if a call turns out
     to be an unbounded flood. If a cap fires, `aborted`/`abort_reason` in
     the written JSON says so plainly -- the audit brief is explicit that
     an opaque sampler is itself a finding, not something to paper over by
     silently truncating output and pretending the trace was complete.

Everything this script measures is written verbatim to
`audit/traces/*.json` -- `dependency_map.md` is written FROM those files
(cited by file:line), not from a re-reading of the source with the trace
used as decoration.

Run with:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/trace_runtime.py
from the repo root.
"""
from __future__ import annotations

import json
import os
import queue as queue_mod
import sys
import time
from collections import Counter
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
SRC = REPO_ROOT / "src"
DEMO = REPO_ROOT / "demo"
TRACES_DIR = AUDIT_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DEMO))

# --------------------------------------------------------------------------
# Safety caps -- see module docstring point 3.
# --------------------------------------------------------------------------
MAX_CALL_EVENTS = 4_000_000
MAX_WALL_SECONDS = 150.0
_CHECK_EVERY = 20_000
_RAW_EVENT_CAP = 4_000

_SRC_R = os.path.realpath(str(SRC)) + os.sep
_DEMO_R = os.path.realpath(str(DEMO)) + os.sep


def _in_scope(filename: str) -> bool:
    """True iff `filename` is a file under src/ or demo/ of THIS repo --
    the code this audit is mapping, as opposed to jax/thrml/numpy/PIL/
    tkinter/stdlib, which are out of scope by construction (see module
    docstring point 2)."""
    try:
        rp = os.path.realpath(filename)
    except Exception:
        return False
    return rp.startswith(_SRC_R) or rp.startswith(_DEMO_R)


def _short(filename: str) -> str:
    try:
        rp = os.path.realpath(filename)
        return os.path.relpath(rp, REPO_ROOT).replace("\\", "/")
    except Exception:
        return filename


def _pkg_of(filename: str) -> str:
    """Best-effort top-level package name for an out-of-scope frame (jax,
    thrml, torx, numpy, scipy, PIL, tkinter, ...), for the per-package
    call tally. Anything not under a `site-packages` directory is
    attributed to `<stdlib/builtin>`."""
    rp = os.path.realpath(filename).replace("\\", "/")
    marker = "/site-packages/"
    if marker in rp:
        tail = rp.split(marker, 1)[1]
        return tail.split("/", 1)[0]
    return "<stdlib/builtin>"


class BoundedTracer:
    """A sys.settrace-compatible callable. See module docstring for the
    full rationale. Only 'call' events are examined; the local trace
    function is always None, meaning 'line'/'return'/'exception' events
    are never enabled for any frame -- but note this does NOT reduce the
    number of 'call' events observed: Python's global tracer, once
    installed via sys.settrace, is invoked for the 'call' event of EVERY
    new frame pushed on this thread for as long as tracing is active,
    regardless of what the local trace function returns. That is exactly
    why the hard caps exist."""

    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()
        self.n_events = 0
        self.n_in_scope_calls = 0
        self.n_out_of_scope_calls = 0
        self.ordered_events: list[dict] = []
        self.edge_counts: Counter = Counter()
        self.boundary_out: dict[tuple, dict] = {}
        self.pkg_tally: Counter = Counter()
        self.aborted = False
        self.abort_reason = ""

    def __call__(self, frame, event, arg):
        if event != "call":
            return None
        self.n_events += 1

        if self.n_events % _CHECK_EVERY == 0:
            if time.perf_counter() - self.t0 > MAX_WALL_SECONDS:
                self.aborted = True
                self.abort_reason = (
                    f"wall time exceeded {MAX_WALL_SECONDS}s after "
                    f"{self.n_events} call events -- treating the interior "
                    f"as an unbounded flood per this script's own docstring")
                sys.settrace(None)
                return None
        if self.n_events > MAX_CALL_EVENTS:
            self.aborted = True
            self.abort_reason = (
                f"exceeded MAX_CALL_EVENTS={MAX_CALL_EVENTS} -- treating "
                f"the interior as an unbounded flood per this script's own "
                f"docstring")
            sys.settrace(None)
            return None

        code = frame.f_code
        fname = code.co_filename
        in_scope = _in_scope(fname)
        back = frame.f_back

        if in_scope:
            self.n_in_scope_calls += 1
            callee = f"{_short(fname)}:{code.co_firstlineno}:{code.co_name}"
            if back is not None:
                bcode = back.f_code
                caller = f"{_short(bcode.co_filename)}:{bcode.co_firstlineno}:{bcode.co_name}"
            else:
                caller = "<root>"
            self.edge_counts[(caller, callee)] += 1
            if len(self.ordered_events) < _RAW_EVENT_CAP:
                self.ordered_events.append({
                    "idx": self.n_events, "caller": caller, "callee": callee,
                    "call_line": frame.f_lineno,
                })
        else:
            self.n_out_of_scope_calls += 1
            pkg = _pkg_of(fname)
            self.pkg_tally[pkg] += 1
            if back is not None and _in_scope(back.f_code.co_filename):
                bcode = back.f_code
                caller = f"{_short(bcode.co_filename)}:{bcode.co_firstlineno}:{bcode.co_name}"
                callee = f"{pkg}:{os.path.basename(fname)}:{code.co_name}"
                key = (caller, pkg)
                if key not in self.boundary_out:
                    self.boundary_out[key] = {
                        "callee_example": callee, "count": 0,
                        "first_seen_at_event": self.n_events,
                    }
                self.boundary_out[key]["count"] += 1
        return None

    def summary(self) -> dict:
        wall = time.perf_counter() - self.t0
        return {
            "label": self.label,
            "tracer_wall_seconds": wall,
            "total_call_events": self.n_events,
            "in_scope_calls": self.n_in_scope_calls,
            "out_of_scope_calls": self.n_out_of_scope_calls,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "out_of_scope_package_tally": dict(self.pkg_tally.most_common()),
            "boundary_crossings_ours_to_external": [
                {"caller": c, "external_package": p, **v}
                for (c, p), v in sorted(
                    self.boundary_out.items(), key=lambda kv: -kv[1]["count"])
            ],
            "distinct_in_scope_edges": len(self.edge_counts),
            "in_scope_call_edges": [
                {"caller": c, "callee": e, "count": n}
                for (c, e), n in sorted(
                    self.edge_counts.items(), key=lambda kv: -kv[1])
            ],
            "raw_event_sequence_prefix": self.ordered_events,
            "raw_event_sequence_prefix_capped_at": _RAW_EVENT_CAP,
        }


def _trace_call(label: str, fn, *args, **kwargs):
    tracer = BoundedTracer(label)
    sys.settrace(tracer)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.settrace(None)
    wall = time.perf_counter() - t0
    data = tracer.summary()
    data["fn_wall_seconds"] = wall
    return result, data


def _scrub_message(d: dict) -> dict:
    """A message pushed by SampleWorker onto its queue may carry a PIL
    Image and a numpy grid (only on kind=='valid') -- drop those (not
    JSON-serialisable / not evidence this map needs) but keep everything
    else, including a `grid_shape` marker so a reader can see a world WAS
    decoded without the image bytes cluttering the trace file."""
    out = {}
    for k, v in d.items():
        if k == "image":
            continue
        if k == "grid":
            out["grid_shape"] = list(v.shape)
            continue
        out[k] = v
    return out


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"[A1] wrote {path.relative_to(REPO_ROOT)} "
          f"({path.stat().st_size} bytes)")


def main() -> None:
    print(f"[A1] repo root       : {REPO_ROOT}")
    print(f"[A1] interpreter     : {sys.executable}")
    print(f"[A1] python version  : {sys.version.split()[0]}")

    from lattice_app import Receipt, SampleWorker, RECEIPT_DIR  # noqa: E402

    print(f"[A1] receipt dir     : {RECEIPT_DIR}")

    # ------------------------------------------------------------------
    # UNCLAMPED trace: Receipt() load (the "receipt -> representation"
    # arrows) followed by exactly one SampleWorker._run_unclamped_tick
    # call (the "energy model -> ... -> THRML sampling -> raw state ->
    # codeword test -> decode -> contract validation" arrows). speed_idx=0
    # (SPEED_LEVELS' own "Slow" entry, n_chains=1) is used deliberately --
    # a real, existing code path (n_warmup/steps_per_sample are UNCHANGED
    # by speed, see SampleWorker's own docstring), chosen only to bound
    # trace volume; not a modification of anything.
    # ------------------------------------------------------------------
    out_q: "queue_mod.Queue[dict]" = queue_mod.Queue(maxsize=64)

    def _unclamped_run():
        receipt = Receipt(RECEIPT_DIR)
        worker = SampleWorker(receipt, out_q, seed_base=12345, clamp=None,
                              speed_idx=0)
        worker._run_unclamped_tick(is_step=False)
        return receipt, worker

    print("[A1] tracing UNCLAMPED path: Receipt() + "
          "SampleWorker._run_unclamped_tick ...")
    (receipt, _worker), unclamped_data = _trace_call(
        "unclamped_draw (Receipt load + one tick, speed_idx=0)", _unclamped_run)

    drained = []
    while True:
        try:
            drained.append(out_q.get_nowait())
        except queue_mod.Empty:
            break
    unclamped_data["pushed_messages"] = [_scrub_message(d) for d in drained]
    kinds = [d.get("kind") for d in drained]
    print(f"[A1] unclamped: {len(drained)} message(s) pushed, kinds={kinds}, "
          f"{unclamped_data['total_call_events']} call events, "
          f"aborted={unclamped_data['aborted']}")
    _write(TRACES_DIR / "unclamped_trace.json", unclamped_data)

    # ------------------------------------------------------------------
    # CLAMPED trace: one SampleWorker._run_clamped_batch call on the SAME
    # already-loaded receipt (a pin change never reloads the receipt --
    # see demo/lattice_app.py's _restart_sampling_for_clamp_change), pin
    # g0_0=WATER(0) -- the same {"g0_0": 0} clamp convention
    # tests/test_clamp.py's own
    # test_execution_tv_is_measured_against_the_conditional_exact_reference
    # uses, valid for this receipt's 8x8 grid variable names. speed_idx=0
    # again, for the same reason as above (n_chains=1, clamp_samples=3;
    # n_warmup=CLAMP_N_WARMUP=600 and steps_per_sample are UNCHANGED).
    # ------------------------------------------------------------------
    out_q2: "queue_mod.Queue[dict]" = queue_mod.Queue(maxsize=256)
    CLAMP = {"g0_0": 0}

    def _clamped_run():
        worker2 = SampleWorker(receipt, out_q2, seed_base=67890, clamp=CLAMP,
                               speed_idx=0)
        worker2._run_clamped_batch(is_step=False)
        return worker2

    print(f"[A1] tracing CLAMPED path: SampleWorker._run_clamped_batch "
          f"(clamp={CLAMP}) ...")
    _worker2, clamped_data = _trace_call(
        f"clamped_draw (clamp={CLAMP}, speed_idx=0)", _clamped_run)

    drained2 = []
    while True:
        try:
            drained2.append(out_q2.get_nowait())
        except queue_mod.Empty:
            break
    clamped_data["pushed_messages"] = [_scrub_message(d) for d in drained2]
    kinds2 = [d.get("kind") for d in drained2]
    print(f"[A1] clamped: {len(drained2)} message(s) pushed, kinds={kinds2}, "
          f"{clamped_data['total_call_events']} call events, "
          f"aborted={clamped_data['aborted']}")
    _write(TRACES_DIR / "clamped_trace.json", clamped_data)

    # ------------------------------------------------------------------
    # Bonus (not one of the two required traces, written separately so it
    # cannot be mistaken for either): re-trace a SECOND unclamped tick on
    # the SAME process, after the shapes above have already been jitted
    # once. This directly tests the module docstring's "warm dispatch may
    # show almost nothing" claim against the "cold compile floods" result
    # already captured above, rather than asserting it untested.
    # ------------------------------------------------------------------
    def _warm_run():
        q3 = queue_mod.Queue(maxsize=64)
        worker3 = SampleWorker(receipt, q3, seed_base=999, clamp=None,
                               speed_idx=0)
        worker3._run_unclamped_tick(is_step=False)

    print("[A1] bonus: re-tracing a SECOND unclamped tick (warm JIT cache, "
          "same shape as the first) ...")
    _, warm_data = _trace_call(
        "unclamped_draw_SECOND_CALL_warm_jit_cache (bonus, not one of the "
        "two required traces)", _warm_run)
    _write(TRACES_DIR / "unclamped_trace_warm_cache_bonus.json", warm_data)

    print("\n[A1] SUMMARY")
    print(f"  unclamped : {unclamped_data['total_call_events']} call events "
          f"({unclamped_data['in_scope_calls']} in-scope, "
          f"{unclamped_data['out_of_scope_calls']} out-of-scope), "
          f"aborted={unclamped_data['aborted']}, "
          f"wall={unclamped_data['fn_wall_seconds']:.3f}s")
    print(f"  clamped   : {clamped_data['total_call_events']} call events "
          f"({clamped_data['in_scope_calls']} in-scope, "
          f"{clamped_data['out_of_scope_calls']} out-of-scope), "
          f"aborted={clamped_data['aborted']}, "
          f"wall={clamped_data['fn_wall_seconds']:.3f}s")
    print(f"  warm-cache: {warm_data['total_call_events']} call events "
          f"({warm_data['in_scope_calls']} in-scope, "
          f"{warm_data['out_of_scope_calls']} out-of-scope), "
          f"aborted={warm_data['aborted']}, "
          f"wall={warm_data['fn_wall_seconds']:.3f}s")

    torx_hits = [
        e for e in unclamped_data["boundary_crossings_ours_to_external"]
        + clamped_data["boundary_crossings_ours_to_external"]
        if e["external_package"] == "torx"
    ] + [
        e for e in unclamped_data["in_scope_call_edges"]
        + clamped_data["in_scope_call_edges"]
        if "torx_backend" in e["callee"]
    ]
    print(f"  torx observed in either live path: {bool(torx_hits)} "
          f"({len(torx_hits)} matching edge(s))")
    print("[A1] done.")


if __name__ == "__main__":
    main()
