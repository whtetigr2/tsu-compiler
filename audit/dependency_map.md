# A1 -- The Runtime Dependency Map

**Method:** instrumented, not inferred. `audit/trace_runtime.py` uses `sys.settrace`
to record the real call sequence of one UNCLAMPED draw and one CLAMPED draw,
both starting from `demo.lattice_app.SampleWorker` (the same worker
`demo/lattice_app.py`'s Tk UI drives), invoked directly off the Tk event
loop -- `tests/test_lattice_app_logic.py`'s own docstring already
establishes that `SampleWorker`'s methods touch no Tk state, only
`self.receipt`/`self.q`/thrml, so calling `_run_unclamped_tick` /
`_run_clamped_batch` synchronously (instead of via `.start()` on a
background thread) exercises the identical function calls with the
identical arguments the real background thread would use. The one thing
this does NOT exercise is the real thread/Tk-poll interleaving; the call
graph itself is faithful.

Traces were captured on `tsu-compiler` branch `lattice-audit`,
`HEAD=2ee3a54`, with
`PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/trace_runtime.py`,
run once, in the foreground, from the repo root. Raw output:

| file | label | total call events | in-scope (ours) | out-of-scope (external) | aborted |
|---|---|---:|---:|---:|---|
| `audit/traces/unclamped_trace.json` | Receipt() load + 1 unclamped tick (cold JIT) | 314,197 | 16,785 | 297,412 | No |
| `audit/traces/clamped_trace.json` | 1 clamped batch, clamp=`{"g0_0": 0}` | 520,761 | 156,230 | 364,531 | No |
| `audit/traces/unclamped_trace_warm_cache_bonus.json` | 2nd unclamped tick, same shape, warm JIT cache (bonus, not one of the two required traces) | 85,137 | 4,495 | 80,642 | No |

Both required traces completed within the script's own safety caps
(`MAX_CALL_EVENTS=4,000,000`, `MAX_WALL_SECONDS=150` per call) without
tripping either -- `aborted=False` in both files. `speed_idx=0`
(`SPEED_LEVELS[0]`, "Slow", `demo/lattice_app.py:219-228`) was passed to
`SampleWorker` for both draws: a real, existing, unmodified code path,
chosen only to bound trace volume by minimising `n_chains`/`n_samples`
-- `n_warmup` and `steps_per_sample`, the two parameters that actually
determine the sampled distribution, are identical at every speed
(`SampleWorker`'s own docstring, `demo/lattice_app.py:1300-1309`) and were
NOT altered for this trace. One asymmetry worth recording precisely: at
`speed_idx=0` the UNCLAMPED tick used `n_chains=1` (speed-controlled, see
`_run_unclamped_tick`, `demo/lattice_app.py:1401-1402`), while the CLAMPED
batch still used `n_chains=CLAMP_N_CHAINS=6` (a constant, NOT
speed-controlled -- only `n_samples` scales with speed for a clamped
batch, `demo/lattice_app.py:1426-1431`) with `n_samples=3` (Slow's
`clamp_samples`). The clamped trace's 18 raw rows (6 chains x 3 samples,
chain-major) is exactly consistent with this.

Both traces' `pushed_messages` are drained straight from the worker's own
output queue, unmodified (only the PIL `image` and the numpy `grid` are
dropped for JSON-safety, see `_scrub_message` in `trace_runtime.py`) --
these are the literal dicts `classify_draw` (`demo/lattice_app.py:1238`)
produced.

- Unclamped: 1 message, `kind=contract-fail`, `seed=12345`, violation
  `"water directly adjacent to rock with no grass between: g2_4-g2_5"`.
- Clamped: 19 messages (18 draws + 1 `batch_summary`): 5 `valid`, 12
  `contract-fail`, 2 `non-codeword`; every `valid` row's `g0_0` decodes to
  `0` (water, the pinned value) -- the clamp is honoured in every valid
  decode, consistent with `thrml`'s `state_clamp` mechanism
  (`src/tsu/backends/thrml_backend.py:210-217`). Final
  `batch_summary`: `{"infeasible": false, "reason": "5/18 draws valid",
  "valid": 5, "total": 18}`.

---

## The arrow-by-arrow chain, as observed

```
LATTICE APP -> receipt -> representation -> energy model (J, b, beta)
  -> graph/topology/mediators -> physical mapping/placement/routing
  -> program construction -> THRML/Torx sampling path -> raw state
  -> codeword test -> decode -> contract validation -> UI panels
```

| # | Arrow | Function(s) observed in the trace | file:line | Evidence |
|---|---|---|---|---|
| 1 | LATTICE APP -> receipt | `main()` constructs `Receipt(RECEIPT_DIR)` once at startup; `Receipt.__init__` reads 9 JSON files verbatim via a local `load()` closure | `demo/lattice_app.py:4501` (call site, not traced -- see caveat below), `demo/lattice_app.py:1111-1125` | `demo/lattice_app.py:1111:__init__ -> demo/lattice_app.py:1114:load`, count=9 (unclamped trace) |
| 2 | receipt -> representation | `load_spec` parses `spec.yaml`; `encode(spec, encoding_name)` rebuilds the categorical/binary NAME MAPPING (`Encoded.categorical`/`.binary_names`) via `_encode_domain_wall` -> `_build_categorical`, `_guard_penalty_dominates`, `_scale_terms`, `_rewrite_terms` -> `_rewrite_form_domain_wall` | `src/tsu/spec.py:354` (`load_spec`), `src/tsu/passes/encode.py:398` (`encode`), `:323` (`_encode_domain_wall`), `:233`, `:258`, `:288`, `:309`, `:190` | `demo/lattice_app.py:1111:__init__ -> src/tsu/spec.py:354:load_spec` (1x); `-> src/tsu/passes/encode.py:398:encode` (1x); `encode.py:309:_rewrite_terms -> encode.py:190:_rewrite_form_domain_wall` (1,792x) |
| 3 | representation -> energy model (J, b, beta) | `reconstruct_program()` reads `program.json`'s `nodes/edges/weights/biases/beta/offset/mediator_nodes` **verbatim** into an `IsingModel` -- no `lower()` pass runs live | `src/tsu/simulate.py:49-81` | `demo/lattice_app.py:1111:__init__ -> src/tsu/simulate.py:49:reconstruct_program` (1x); internal genexprs at `simulate.py:66`/`:77` build the `edges`/`blocks` tuples (577/3 calls, matching 576 edges + 1) |
| 4 | graph/topology/mediators | **Not recomputed.** Mediator identity (which of the 192 physical nodes are the 64 hidden mediator spins) is read from `program.json`'s `mediator_nodes` field, not re-derived by `insert_mediators` | `demo/lattice_app.py:1133` (`mediator_set = set(self.program.get("mediator_nodes", ()))`) | `insert_mediators` (`src/tsu/passes/route.py:150`): **0 edges in either trace** |
| 5 | physical mapping/placement/routing | **Not observed in either draw.** `place()` and `route()` never appear. The only compile-adjacent passes that DO run live are `analyse()` and `build_program()`, and only on the clamped path, only because `simulate(..., clamp={...})` differs from the receipt's own clamp -- a graph-theoretic REPARTITION of the already-fixed edge set into chromatic blocks, never a re-placement | `src/tsu/passes/place.py:240`, `src/tsu/passes/route.py:235` | `place`: 0 edges either trace. `route`: 0 edges either trace. `analyse` (`src/tsu/passes/analyse.py:43`): 0 edges unclamped, 1 edge clamped (`src/tsu/simulate.py:89:simulate -> src/tsu/passes/analyse.py:43:analyse`) |
| 6 | program construction | `build_program()` -- **clamped path only**, called from `simulate()`; on the unclamped path the program (blocks/schedule/clamp) is instead read verbatim inside `reconstruct_program` (arrow 3) | `src/tsu/passes/program.py:43` | `build_program`: 0 edges unclamped, 1 edge clamped (`src/tsu/simulate.py:89:simulate -> src/tsu/passes/program.py:43:build_program`) |
| 7 | THRML/Torx sampling path | `sample()` (unclamped) / `sample_chains()` (both) -> `_model()` builds the `thrml.models.IsingEBM`, then `jax.jit(jax.vmap(lambda i, k: sample_states(...)))(...)` crosses into jax/thrml. **`torx` never appears, either direction, in either trace.** | `src/tsu/backends/thrml_backend.py:232` (`sample`), `:181` (`sample_chains`), `:38` (`_model`), `:226` (the jitted lambda) | `demo/lattice_app.py:1401:_run_unclamped_tick -> thrml_backend.py:232:sample` (1x); `thrml_backend.py:181:sample_chains -> thrml_backend.py:38:_model` (1x, both traces); boundary crossings into pkg `thrml`/`jax`/`jaxlib`/`equinox` (see JIT-opacity section below); `torx_cross_check`/pkg `torx`: **0 matches in 1,229,987 combined call events across all three trace files** |
| 8 | raw state | The flattened `(n_chains*n_samples, n_spins)` 0/1 numpy array `sample()`/`sample_chains()` returns | `src/tsu/backends/thrml_backend.py:234-236` | Return value shape confirmed via `pushed_messages`: clamped trace's 18 rows = 6 chains x 3 samples (chain-major, matches `chains.reshape(-1, chains.shape[-1])`) |
| 9 | codeword test | `enc.is_codeword(bits)` -- called from `classify_draw`, **and independently again from `simulate()` itself** on the clamped path (see Finding I-1) | `src/tsu/passes/encode.py:163` | `demo/lattice_app.py:1238:classify_draw -> encode.py:163:is_codeword` (18x clamped, 1x unclamped); `src/tsu/simulate.py:89:simulate -> encode.py:163:is_codeword` (18x clamped) |
| 10 | decode | `enc.decode(bits)` -- same duplicate-call pattern as arrow 9 | `src/tsu/passes/encode.py:131` | `classify_draw -> decode` (16x clamped, 1x unclamped); `simulate -> decode` (16x clamped) |
| 11 | contract validation | `spec.contract.validate(decoded)` -- same duplicate-call pattern | `src/tsu/spec.py:38` | `classify_draw -> validate` (16x clamped, 1x unclamped); `simulate -> validate` (16x clamped) |
| 12 | UI panels | `_classify_and_push` pushes the classified dict onto `SampleWorker.q`; `_put` (blocking-with-timeout put) is the last in-scope frame this trace observes | `demo/lattice_app.py:1357` (`_classify_and_push`), `:1349` (`_put`) | `_run_unclamped_tick -> _classify_and_push -> classify_draw`, `-> _put` (both traces). **Not traced past this point**: the Tk-side consumer (`_poll_queue`/`_handle_msg`, `demo/lattice_app.py:3402-3527`) runs on Tk's `after`-loop, which this script never starts -- see Methodology caveat below. The `valid` branch of `classify_draw` also calls `render_world_image` (`demo/lattice_app.py:1166`), observed 5x in the clamped trace (once per `valid` draw), 0x in the unclamped trace (its single draw was `contract-fail`, never reaching that branch). |

### Methodology caveat on arrow 12

This script calls `SampleWorker._run_unclamped_tick`/`_run_clamped_batch`
directly and drains the resulting queue with `queue.Queue.get_nowait()` in
the same process -- it never starts `LatticeApp`'s Tk `after`-loop, so
`_handle_msg`'s own downstream fan-out (energy/magnetization/valid-fraction
traces, the DECODED WORLD canvas, the SCOPE panel) is not part of either
`trace_runtime.py` trace. That consumption path is read (not traced, since
it requires a live Tk mainloop this audit is not standing up) and reported
with file:line citations in `audit/truth_table.md` instead, per Task A2.

---

## Arrows that do NOT appear in the trace

Recorded as findings, not omissions:

1. **`insert_mediators`** (`src/tsu/passes/route.py:150`) -- 0 occurrences.
   Mediator topology is fixed at compile time and read back as data
   (`program.json`'s `mediator_nodes`), never recomputed by either draw.
2. **`place`** (`src/tsu/passes/place.py:240`) -- 0 occurrences in either
   draw. Confirms the docstring claim at the top of `demo/lattice_app.py`
   ("a receipt whose `place` pass alone took ~201s to produce -- read once
   at startup here, NEVER regenerated") against live instrumentation, not
   just against the comment.
3. **`route`** (`src/tsu/passes/route.py:235`) -- 0 occurrences.
4. **`search`** (`src/tsu/passes/search.py`, `compile_spec`'s
   representation search) -- 0 occurrences. `_verify`/`_measure_mixing`
   (search.py:494-520), which is where the receipt's own frozen `ess`
   figure and the `torx_cross_check` call live, ran exactly once, at
   compile time, and is never re-entered by sampling.
5. **`torx_backend.torx_cross_check`** / package `torx` -- 0 occurrences,
   either direction, across all 1,229,987 call events in the three trace
   files combined. See the dedicated answer below.
6. **`analyse`** (`src/tsu/passes/analyse.py:43`) and **`build_program`**
   (`src/tsu/passes/program.py:43`) -- 0 occurrences on the UNCLAMPED
   path; each appears exactly once on the CLAMPED path (see arrows 5-6
   above). Not an absence in the "never runs" sense, but the trace shows
   precisely which of the two draw types it runs under, rather than
   assuming both.

---

## The two specific questions the brief asks

**Does `torx` appear anywhere in the live path at all? No.** Confirmed by
direct instrumentation, not by absence-of-import-search: `torx_backend.py`
(`src/tsu/backends/torx_backend.py`) is in-scope for this tracer (it lives
under `src/`), so if `torx_cross_check` had been called, its `call` event
would have been recorded as an ordinary in-scope edge, and its own
subsequent call into the real `torx` package would have been recorded as a
boundary crossing with `external_package="torx"`. Neither occurred, in
either the unclamped, clamped, or bonus warm-cache trace. Static reading
corroborates this independently: `torx_cross_check` is called from exactly
one place in `src/`, `search.py:520,724`, inside `_verify`
(`src/tsu/passes/search.py:517`) -- a compile-time-only pass that this
audit is barred from invoking (`tsu compile` is never run). `demo/` never
imports `tsu.backends.torx_backend` at all (`grep -r torx demo/` returns
nothing). Torx is, in this receipt's live sampling path, entirely inert:
it exists only as a compile-time independent oracle for a single-bond
model (its own docstring, `torx_backend.py:1-29`, states it is exact only
for a 2-node/1-edge model -- this 192-node/576-edge receipt could not use
it even at compile time beyond that narrow role).

**Does placement/routing run at sample time, or only at compile time?
Only at compile time.** `place`/`route` are 0/0 across both draws (see
above). What DOES run at sample time, and only for a clamped draw whose
clamp differs from the receipt's own, is `analyse`/`build_program` --
repartitioning the ALREADY-placed, ALREADY-routed edge set into new
chromatic colour blocks so the clamped spins can be held fixed during
Gibbs updates. This is graph bookkeeping over a fixed embedding, not a
re-placement onto different hardware sites; `src/tsu/simulate.py`'s own
module docstring says exactly this ("clamping changes which nodes may
share a chromatic block, and that repartition is graph-theoretic
bookkeeping over the ALREADY-fixed edge set, not a re-derivation of the
energy model itself") and the trace confirms the docstring's claim rather
than merely repeating it.

---

## JIT opacity: what settrace actually saw inside the sampler

The brief warned this could go either way -- "may see nothing inside it,
or may see an unusable flood." Both were observed, in the SAME process, on
the SAME shape, one call apart:

| trace | total call events | in-scope (ours) | out-of-scope | jax | thrml | jaxlib | equinox |
|---|---:|---:|---:|---:|---:|---:|---:|
| unclamped, cold (1st call, this shape) | 314,197 | 16,785 | 297,412 | 220,839 | 4,193 | 5,745 | 4,108 |
| unclamped, warm (2nd call, same shape) | 85,137 | 4,495 | 80,642 | 63,153 | 4,187 | 2,866 | 4,108 |

The cold call is a flood: 220,839 Python-level calls attributed to the
`jax` package alone, none of them named anything a physicist would
recognise (block-Gibbs update, spin flip, local field) -- every boundary
crossing this script recorded resolves to jax's OWN tracing/dispatch
machinery (`jax/_src/interpreters/batching.py`, `jax/_src/core.py`, ...),
because `jax.jit(jax.vmap(...))` Python-traces the function once per
distinct input shape to build an XLA computation graph before it ever
executes a single Gibbs step. The warm call drops jax's share by ~3.5x
(220,839 -> 63,153) but does not go anywhere near zero: even a cached
dispatch still round-trips through a real (if much smaller) amount of
Python. Neither regime exposes a single frame corresponding to a
chromatic-block update, a spin flip, or a warmup sweep -- once
`sample_states` (thrml) hands off to XLA, the actual Gibbs kernel executes
as compiled machine code with no Python frames at all, cold or warm. This
is the honest boundary this map draws: `thrml_backend.py:181`
(`sample_chains`) is the last point from which the compiler's OWN logic is
observable; everything past it is reported as a volume and a boundary
edge, never fabricated as a step-by-step trace it does not have. One
real, in-scope crossing back INTO our code during jax's tracing was
captured and is worth naming directly, as positive evidence the boundary
is real and not just a name filter with nothing on the other side:
`.../site-packages/jax/_src/interpreters/batching.py:118:flatten_fun_for_vmap
-> src/tsu/backends/thrml_backend.py:226:<lambda>` -- jax's own `vmap`
machinery calling back into the lambda `sample_chains` passed it
(`thrml_backend.py:226`), during graph construction.

A caveat on the out-of-scope tally's own accuracy: entries attributed to
package `<stdlib/builtin>` include a large contingent (~18,800 calls in
the unclamped trace, concentrated at `encode.py:190`/`spec.py:213`/
`spec.py:328`) whose frames report `co_filename="<string>"` -- almost
certainly `sympy`'s dynamically-`exec`'d class/cache machinery (`lower.py`
imports `sympy as sp` for the same symbolic energy-model construction
`encode()` performs, see Finding I-2 below), not genuine stdlib code. This
script's `_pkg_of` heuristic (looks for `/site-packages/` in the realpath)
cannot distinguish a `<string>`-sourced frame's true origin, so this
number is reported as `<stdlib/builtin>` rather than mis-attributed to a
specific package it cannot verify -- an honest "don't know exactly which
package" rather than a guess.

---

## Findings

### Important

**I-1: Every clamped-batch draw is codeword-tested, decoded, and
contract-validated TWICE, by two independent call sites, on the same
data.** `src/tsu/simulate.py:157-169` (inside `simulate()`) and
`demo/lattice_app.py:1238-1255` (`classify_draw`, called once per row from
`_run_clamped_batch`) each independently call `enc.is_codeword`,
`enc.decode`, and `spec.contract.validate` on every row of the SAME
returned array. Confirmed by the trace's exact call counts: `simulate`
calls `is_codeword` 18x/`decode` 16x/`validate` 16x; `classify_draw` calls
the identical trio the identical number of times, on the identical 18
rows (`src/tsu/simulate.py:89:simulate -> encode.py:163:is_codeword`
count=18; `demo/lattice_app.py:1238:classify_draw -> encode.py:163:is_codeword`
count=18, both in `audit/traces/clamped_trace.json`). **Concrete failure
scenario:** this is not a correctness bug -- both call sites use the
compiler's own canonical `enc`/`spec.contract` (no reimplementation, so
R16's "duplication vs canonical source" concern does not apply in its
sharper form) -- but it doubles the CPU cost of decode+validate on every
pin change for no visible benefit (`simulate()`'s own return value, `got`,
is discarded by `_run_clamped_batch` in favour of re-deriving the same
classification from scratch), and it is a place a future edit could let
the two independently-computed verdicts drift (e.g. if one call site's
`enc`/`spec` object is later swapped for a differently-configured one and
the other is not). **What was checked:** live trace edge counts in both
`audit/traces/clamped_trace.json` (in_scope_call_edges) and direct
reading of both call sites; confirmed the row order is identical (both
iterate the same `got`/`rows` array in the same order) so this is exact
duplication, not two different subsets being checked.

**I-2: `Receipt.__init__` re-derives the full symbolic domain-wall energy
representation on every app launch, not just a name mapping, contradicting
the narrower claim in `tsu.simulate`'s own module docstring.**
`demo/lattice_app.py:1129` (`self.enc = encode(self.spec,
self.encoding_name)`) calls `src/tsu/passes/encode.py:398`'s `encode()`,
which the trace shows reaching `_rewrite_form_domain_wall` 1,792 times,
`_max_energy_contribution`/`_max_abs_form` (the `MONOTONE_PENALTY`
dominance guard) 1,792/3,584 times, and `_guard_penalty_dominates` once
per categorical variable -- i.e. the full linear-form rewriting and
penalty-dominance verification `lower.py` performs at compile time,
re-run at every `Receipt()` construction (`src/tsu/simulate.py:11-19`'s
own docstring calls `encode`'s re-run "a pure, deterministic re-derivation
of a NAME MAPPING ... not of the compiled program" -- true of the NUMERIC
Ising model, which genuinely is read verbatim via `reconstruct_program`,
but the trace shows `encode()` itself does materially more work than a
name-mapping lookup to get there). **Concrete failure scenario:** none
today -- this runs once at startup (measured: this app launches in well
under a second in practice), and its result is deterministic and
side-effect-free, so it cannot desynchronise from the receipt. It is
recorded because the audit brief requires flagging every label/behaviour
gap "however slightly," and a future reader trusting "pure name-mapping
re-derivation" literally would underestimate what changing `encode()`'s
cost profile could do to app startup time. **What was checked:** the
`in_scope_call_edges` list in `audit/traces/unclamped_trace.json`,
specifically the `encode.py` internal call graph rooted at
`demo/lattice_app.py:1111:__init__ -> encode.py:398:encode`; cross-read
against `encode.py`'s own docstring (lines 1-21) confirming this is the
SAME linear-form/penalty machinery `lower.py` documents using for the
original compile.

### Minor

**M-1: `boundary_crossings_ours_to_external`'s `<stdlib/builtin>` bucket
silently absorbs `sympy`'s dynamically-executed (`<string>`-filenamed)
frames because this script's own package-attribution heuristic cannot see
past them.** See the JIT-opacity section's caveat above for the full
explanation and the ~18,800-call figure. Not a finding about the
application under audit -- a limitation of this script's own
instrumentation, recorded per this task's own "do not fake a trace"
instruction rather than presented as a clean stdlib tally it is not.

**M-2: The two required traces exercise different `n_chains` scaling
rules without it being obvious from `SPEED_LEVELS` alone.** `speed_idx=0`
gave the unclamped tick `n_chains=1` (`SPEED_LEVELS[0]["chains"]`,
`demo/lattice_app.py:220-221`) but left the clamped batch at
`n_chains=CLAMP_N_CHAINS=6` (a module constant, `demo/lattice_app.py:136`,
never read from `SPEED_LEVELS`) -- only `clamp_samples` is
speed-controlled for a clamped batch (`demo/lattice_app.py:1429`). This
matches the code exactly (`_run_clamped_batch` never reads
`speed_level(...)["chains"]`, only `["clamp_samples"]`) and is not a
defect, but a reader of `SPEED_LEVELS`' own inline comment ("This varies
n_chains and n_samples per worker call") could reasonably expect BOTH
knobs to move together on both paths; only the unclamped tick's `n_chains`
actually does. **What was checked:** `audit/traces/clamped_trace.json`'s
`pushed_messages[*].sampler_params` (`"n_chains": 6` on every row, despite
`speed_idx=0`), cross-read against `_run_unclamped_tick`/
`_run_clamped_batch`'s source (`demo/lattice_app.py:1401-1454`).

### What was tried and did not turn up a finding

- Attempted to find ANY compile-time pass (`encode`'s SYMBOLIC lowering
  aside, per I-2) re-entering at sample time by grepping every in-scope
  edge in both traces for `place`, `route`, `search`, `gate_checks`,
  `regime` (the other five `PASS_ORDER` stages) -- none appear in either
  trace, and `regime`/`gate_checks` are read from `regime.json`/
  `gates.json` as static receipt data (`Receipt.__init__`,
  `demo/lattice_app.py:1119,1123`), never recomputed live. No finding: the
  "compiled once, at load" claim (`demo/lattice_app.py:2389`,
  `:2835-2841`) holds under live instrumentation for every pass except the
  `encode()` re-derivation already flagged as I-2.
- Attempted to trigger a THIRD sampling regime (an overlay/band draw,
  which uses a SEPARATE receipt, `demo/receipts/elev_band`) to check
  whether it shares this map's arrows or diverges. Deliberately NOT done:
  out of scope for A1, which the plan binds to "starting from
  `demo/lattice_app.py`'s worker" against the ONE unclamped + ONE clamped
  draw it specifies; the overlay path is `demo/elevation_world.py`/
  `demo/layers.py` machinery layered on the SAME `SampleWorker`-adjacent
  pattern and belongs to a later wave (R13, "layers/overlays/composite")
  if the controller wants it traced too.
