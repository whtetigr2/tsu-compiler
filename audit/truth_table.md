# A2 -- The Execution Truth Table

**Method:** for each row, the UI text is quoted verbatim from `demo/lattice_app.py`
(or `demo/explainer.py`/`demo/scope.py` where the UI text lives there), and
the actual behaviour is stated with concrete parameter values read from the
implementation and, where available, from `audit/traces/*.json` (Task A1's
live instrumentation) or directly from `demo/receipts/small/*.json`
(`PYTHONIOENCODING=utf-8
"C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe"` used
for every computed figure below, never a hand-derived decimal). Verdict is
one of **MATCH** (label and behaviour agree), **MATCH-WITH-CAVEAT** (they
agree but a reader could still be misled without the caveat stated here),
or **MISMATCH** (they differ) -- every MATCH-WITH-CAVEAT and MISMATCH is
also written up as a finding at the end of this file, per the brief's
"however slightly" instruction.

Receipt under audit: `demo/receipts/small` (192 nodes = 128 world spins +
64 mediator spins, 576 edges, `beta=1.0`, `verdict=COMPILED`).

---

## `compile`

**UI says** (PIPELINE panel, `demo/lattice_app.py:2389,2835-2841`):
`"PIPELINE (compiled once, at load)"` / `"Compilation (encode..verify) ran
ONCE, when this receipt was written -- the durations above are that one
run's own numbers, read from passes.json. Only SAMPLING re-runs per world;
the pipeline itself is not re-executing."`

**Actual behaviour:** `main()` (`demo/lattice_app.py:4501`) constructs
`Receipt(RECEIPT_DIR)` once; `Receipt.__init__` (`:1111-1130`) only reads
9 JSON files and calls `load_spec`/`encode`/`reconstruct_program` -- it
never calls `compile_spec`, `place`, `route`, or `search`. Confirmed live
by A1's trace, not merely by static reading: `place`/`route`/`search`
appear **0 times** across 834,958 combined call events in the unclamped
and clamped traces (`audit/traces/unclamped_trace.json`,
`audit/traces/clamped_trace.json`). Real compile-time durations, read
verbatim from `demo/receipts/small/passes.json`: `place=201.351s`,
`route=3.10us`, `verify=2.931s`, `encode=2.92ms`, `lower=1.57ms`,
`analyse=5.94ms`, `gate_checks=57.0us`, `build_program=0.53ms`,
`regime=0.17ms` -- `fmt_duration`'s own docstring (`:988-990`) cites this
exact spread ("place: 201s, route: 3.1us") as the reason it never
normalises units, and the live figures match that docstring precisely.

**Verdict: MATCH.** One caveat, not severe enough to break the match:
`encode()` DOES run again live, once per `Receipt()` construction (see the
`sample`/`J,b` rows below and A1 Finding I-2) -- but this is the workload
NAME-MAPPING re-derivation `tsu.simulate`'s docstring describes, not a
re-run of `encode` as a PIPELINE STAGE with its own gate/verdict; the
PIPELINE panel's claim is specifically about the compiled VERDICT and its
nine `PASS_ORDER` stages never re-executing, and that claim holds.

---

## `sample`

**UI says** (SAMPLER panel, `demo/lattice_app.py:3039-3055`): `"live
sampler settings (this app, not the receipt) -- UNPINNED at Full speed:
n_chains/call=6, n_warmup=300, n_samples/call=1"` / `"PINNED at Full
speed ...: n_chains/call=6, n_warmup=600, n_samples/call=30"` / `"both:
steps_per_sample(thinning)=4 -- n_warmup and steps_per_sample are FIXED at
every speed: only n_chains/call and n_samples/call scale down below Full,
which changes batch size/display rate, never the sampled distribution."`

**Actual behaviour:** confirmed EXACTLY by A1's traced `sampler_params`,
which are read directly off the classified draw, not off a constant the
UI could drift from: unclamped tick pushed `{"n_chains": 1,
"n_samples_per_call": 1, "n_warmup": 300, "steps_per_sample": 4}` at
`speed_idx=0` (`audit/traces/unclamped_trace.json`); clamped batch pushed
`{"n_chains": 6, "n_samples": 3, "n_warmup": 600, "steps_per_sample": 4}`
at the same `speed_idx=0` (`audit/traces/clamped_trace.json`) -- `n_warmup`
(300/600) and `steps_per_sample` (4) match the SAMPLER panel's claimed
constants exactly, at a speed level OTHER than Full, which is precisely
the claim under test ("FIXED at every speed"). One asymmetry the panel
text does not surface (see A2 Finding M-1 below): only the unclamped
tick's `n_chains` is itself speed-controlled (`speed_level(self.speed_idx)
["chains"]`, `demo/lattice_app.py:1402`); the clamped batch's `n_chains`
is the constant `CLAMP_N_CHAINS=6` regardless of speed
(`demo/lattice_app.py:1430`, confirmed live: `speed_idx=0` still produced
`n_chains=6` in every pushed message) -- only `clamp_samples` scales for a
clamped batch.

**Verdict: MATCH**, with the `n_chains` asymmetry noted as a finding
(M-1) because a reader of the SAMPLER panel's "n_chains/call=6" line for
the pinned case could reasonably read that as ALSO being Full-speed-only,
when it is in fact the pinned batch's `n_chains` at every speed.

---

## `decode`

**UI says** (`classify_draw`'s own docstring, `demo/lattice_app.py:1239-1241`):
`"One raw physical sample -> a classified draw. Uses ONLY the compiler's
own enc.is_codeword / enc.decode / spec.contract.validate -- no decode or
contract logic is reimplemented here."`

**Actual behaviour:** confirmed live. `classify_draw`
(`demo/lattice_app.py:1238`) calls `receipt.enc.is_codeword(bits)`
(`src/tsu_compiler/passes/encode.py:163`), and iff that passes,
`receipt.enc.decode(bits)` (`encode.py:131`) -- both traced exactly once
per row processed (`audit/traces/clamped_trace.json`'s
`in_scope_call_edges`: `classify_draw -> is_codeword` count=18,
`classify_draw -> decode` count=16, i.e. every row is codeword-tested,
and every codeword among them is decoded -- the 2 non-codeword rows are
correctly never decoded, matching `encode.py:131-142`'s own warning that
decoding a non-codeword "still returns a legal-looking value with no
error" for domain-wall, so `is_codeword` must gate it, which it does
here). `Encoded.decode`'s domain-wall branch (`encode.py:145-146`) is a
pure summation projection (`sum(bits[s] for s in chain)`), confirmed
traced 4,288+192 times across the two chain-position generator
expressions at `encode.py:159`/`:146` in the unclamped trace alone.

**Verdict: MATCH.** No reimplementation found anywhere in the traced
decode path; `demo/render_world.py`'s own decode call (not traced by this
script, but read) uses the identical `enc.decode` entry point per the
module docstring's own claim (`demo/lattice_app.py:8-10`).

---

## `verify` / `contract`

**UI says:** the word "verify"/"valid" covers TWO distinct things in this
app, and the UI does not use different words for them:

1. VERIFICATION panel (`demo/lattice_app.py:2430,2972-3013`): static
   receipt metrics -- `task_validity`, `codeword_violation_rate`, `ess`,
   `energy_tv`, `execution_tv`, `cross_check_tv`, `diversity_*` -- plus
   four hardware gates (`DEGREE`/`COUPLING CAP`/`FIELD CAP`/`NODE
   BUDGET`), each `PASS`/`FAIL`.
2. The live "valid fraction" label (`demo/lattice_app.py:3506-3511`):
   `f"valid fraction: {frac:.1f}% (valid=... contract-fail=...
   non-codeword=... total=...)"`.

**Actual behaviour:** these are computed by DIFFERENT code, at DIFFERENT
times, and never reconciled on screen. (1) is written ONCE, by
`_populate_static_panels` (`demo/lattice_app.py:2817`, called exactly
once, from `__init__` at `:2168` -- confirmed by `grep`: no second call
site anywhere in the file), from `verification.json`, itself produced
ONCE at compile time by `_verify`/`_measure_mixing`
(`src/tsu_compiler/passes/search.py:517,494`) using FIXED
`_VERIFY_SAMPLE_PARAMS=dict(n_chains=32, n_samples=200, n_warmup=400,
steps_per_sample=2, seed=0)` (`search.py:513-514`) -- a completely
different sampler configuration than either live draw type uses (compare:
live unpinned `n_warmup=300`, live pinned `n_warmup=600`, both
`steps_per_sample=4`). This panel NEVER updates again for the life of the
session, however many draws occur. (2) is recomputed on every single
`_handle_msg` call (`demo/lattice_app.py:3449-3511`), from `spec.contract.
validate` (`src/tsu_compiler/spec.py:38`) run fresh on that draw's own decode --
confirmed live: A1's clamped trace shows `classify_draw -> validate`
called 16 times, once per codeword row (`audit/traces/clamped_trace.json`).
Sanity cross-check, not a discrepancy: the receipt's frozen
`task_validity=0.24703125` (`demo/receipts/small/verification.json`) and
the live clamped batch's own observed 5/18=27.8% valid
(`audit/traces/clamped_trace.json`'s `batch_summary`) are close, as
expected for two independent measurements of a real ~25% rate at small N.

**Verdict: MATCH-WITH-CAVEAT.** Both numbers are individually honest and
neither is fabricated -- but nothing on screen tells a viewer that the
VERIFICATION panel's `task_validity`/`ess`/etc. are a FROZEN, ONE-TIME
measurement taken under different sampler parameters than anything
currently running, while the nearby "valid fraction" label is a LIVE,
continuously-updating measurement of the CURRENT session. See Finding
M-2.

---

## `beta`

**UI says** (`demo/lattice_app.py:20-28`, module docstring; SAMPLER panel
`:3032-3038`): `"NO beta slider. This model carries 64 mediator spins
whose couplings were baked in at beta=1 ... Sampling a mediated model at
any other beta is refused by tsu.passes.route.assert_beta_consistent
(raises BetaMismatchError, spec 5.3.5) ... This app never calls
simulate(beta=...), so it never exercises that refusal path directly --
but it displays beta as FIXED and states why."`

**Actual behaviour:** verified true on two independent counts. (a) Grep
of every `simulate(...)` call site in `demo/lattice_app.py`
(`_run_clamped_batch:1435-1440`, `_regenerate_band_async` and the
elevation-band paths) confirms none pass `beta=`; `simulate`'s own
signature (`src/tsu_compiler/simulate.py:89-91`) defaults `beta: float | None =
None`, and only calls `assert_beta_consistent` `if beta is not None`
(`simulate.py:123-124`) -- so the docstring's claim that this app "never
exercises that refusal path directly" is exactly right, and confirmed
live: `assert_beta_consistent` (`src/tsu_compiler/passes/route.py:82`) appears 0
times in either trace. (b) `temperature_control_state`
(`demo/scope.py:118-144`) returns `("fixed", reason)` iff
`ising.mediator_nodes` is non-empty, keyed on the SAME fact
`assert_beta_consistent` gates on (`scope.py:121-127`, cross-checked by
`tests/test_scope.py`'s own `test_temperature_control_state_keys_on_the_
same_fact_...`); `demo/receipts/small`'s `program.json` records
`mediator_nodes` with 64 entries, so this receipt's temperature control is
"fixed" -- displayed `beta=1.0` (`program.json`'s own `"beta": 1.0`),
matching the receipt.

**Verdict: MATCH.**

---

## `J` / `b`

**UI says** (FOOTER_TEXT, `demo/lattice_app.py:125-127`;
`ASSUMED_CAP_NOTE`, `:592-596`): `"|J| and |b| caps are assumed project
values, not sourced Extropic figures."` REGIME panel
(`:3178`): `f"|J|max (this program) = {j_max:.4g}"`.

**Actual behaviour:** the CAP labelling is honest and consistently
reused (one literal string, `ASSUMED_CAP_NOTE`, not independently
re-typed per panel -- confirmed by grep, both FOOTER_TEXT and the
FRONTIER gauges cite the identical phrase). The displayed `j_max` VALUE,
however, is computed as `float(np.abs(r.im.weights).max())`
(`demo/lattice_app.py:3170`) over `r.im`, the RECONSTRUCTED (post-
mediation) program -- i.e. it includes the 64 mediator spins' OWN
couplings, not just the workload's own declared weights. Read directly
from `demo/receipts/small/program.json`: workload terms are `{1.0, -0.4,
-0.4, -0.4}` (`spec.yaml:39-42`, so the workload's own `|J|max=1.0`), but
the FULL mediated model's `|J|max=2.8467` (`np.abs(program.json
["weights"]).max()`, verified independently) -- **2.85x larger** than the
workload's own declared coupling, because the domain-wall mediator gadget
coupling `A = arccosh(exp(2*beta*|J|))/(2*beta)`
(`src/tsu_compiler/passes/lower.py`'s own comment, `:34-36`) is not bounded by the
workload's own term weights. `|b|` similarly ranges `0.0..1.6`
(`np.abs(program.json["biases"])`, mean `0.617`) over 192 world+mediator
biases.

**Verdict: MATCH-WITH-CAVEAT.** Nothing displayed is WRONG -- `j_max` is
honestly the largest coupling in the model actually being sampled, which
is arguably the physically correct quantity to site a critical-coupling
estimate against -- but the label `"|J|max (this program)"` does not say
"including mediator gadget couplings, ~2.85x the workload's own largest
declared weight," and a viewer has no way to tell, from that one line
alone, that the number is dominated by an artifact of the domain-wall
encoding rather than by the workload's own physics. See Finding I-1
(shared with the `beta/beta_c` row below, since this is where the
inflated `j_max` propagates to).

---

## `tau` / `ACF`

**UI says** (SCOPE panel, `demo/lattice_app.py:2451`; `render_acf_plot`,
`:1519-1569`): a semi-log autocorrelation plot with a marked `"tau~{iat.
tau:.1f}"` vertical line, built from `energy_ys = list(self.energy_trace.
ys)` (`:3327,3354`) -- the running history of `energy_of_draw` over
"EVERY draw, valid or not" (`:3454-3458`'s own comment).

**Actual behaviour -- this is the row the brief specifically asked to be
verified hardest, and it does NOT hold up.** `render_acf_plot` calls
`autocorrelation(list(series), max_lag=lag_cap)`
(`demo/lattice_app.py:1538`) and `integrated_autocorrelation_time(np.
asarray(series, dtype=float))` (`:1539`), both of which pass straight to
`tsu.ess`'s `_as_chains` (`src/tsu_compiler/ess.py:65-71`), which treats a 1-D
input as a SINGLE chain of shape `(1, N)`. But `energy_trace.ys` is NOT
one chain's trajectory -- it is the concatenation, in arrival order, of
draws from a sequence of INDEPENDENT restarts:

- **Unclamped tick** (`_run_unclamped_tick`, `:1401-1424`): every single
  tick calls `thrml_sample(..., n_warmup=N_WARMUP=300, seed=self.
  seed_base + self.draw_counter)` -- a FRESH 300-step warmup from a FRESH
  `jax.random.bernoulli(0.5, ...)` uniform-random initial state
  (`src/tsu_compiler/backends/thrml_backend.py:222-225`, "uniform-random init, NOT
  hinton_init"), with a NEW seed every tick. Nothing in `thrml_backend.
  sample`/`sample_chains` persists chain state between calls -- each tick
  is a from-scratch restart, not a continuation of the previous tick's
  chain. Within one tick, `n_chains` (1 at `speed_idx=0`, up to 6 at Full)
  parallel, mutually-independent chains each contribute exactly 1 sample
  (`N_SAMPLES_PER_CALL=1`, `:118`) -- there is no "later time in the same
  chain" relationship between rows even WITHIN a tick.
- **Clamped batch** (`_run_clamped_batch`, `:1426-1454`): within ONE
  batch call, `sample_chains` DOES produce a genuine per-chain trajectory
  -- `n_samples` recorded samples per chain, `steps_per_sample=4` real
  Gibbs sweeps apart (confirmed live: `audit/traces/clamped_trace.json`'s
  18 pushed rows are exactly `CLAMP_N_CHAINS=6` chains x `n_samples=3`
  samples, chain-major, matching `chains.reshape(-1, ...)`,
  `thrml_backend.py:234-236`). But `simulate()`'s return value is
  flattened chain-major (`simulate.py:151`, `got = chains.reshape(-1,
  chains.shape[-1])`) BEFORE `_run_clamped_batch` iterates it row-by-row
  into `_classify_and_push` -- so the 3 real trajectory samples of chain 0
  are immediately followed, in `energy_trace.ys`, by chain 1's 3
  INDEPENDENT samples, chain 2's, etc.: a hard discontinuity every
  `n_samples` entries that the flat series carries no marker for. And the
  NEXT pin change or Slow/Full speed batch restarts the whole thing with a
  new seed and a fresh `n_warmup=600` (`_restart_sampling_for_clamp_
  change`, `:4456-4492`, which explicitly resets `energy_trace`/
  `raw_draws` -- but nothing resets them BETWEEN batches under the SAME
  clamp, so multiple clamped batches' worth of chain-major-flattened,
  mutually-independent-chain draws also concatenate into the one series
  the ACF plot reads).

`tsu.ess`'s OWN module docstring states the rule this violates in as many
words: `"callers hand this module a plain (n_chains, n_samples) array ...
autocorrelation is only meaningful within a chain"`
(`src/tsu_compiler/ess.py:4-6`), and `thrml_backend.sample_chains`'s docstring
says the same thing from the producer side: `"autocorrelation/
effective-sample-size (tsu.ess) is meaningless across that [chain]
boundary"` (`thrml_backend.py:187-189`). `render_acf_plot` calls the
FLAT, single-chain-shaped `autocorrelation`/`integrated_autocorrelation_
time` entry points on exactly the kind of series both of those docstrings
warn against. This is NOT hypothetical: `demo/ess_run.py` exists
specifically to do this correctly -- it calls `sample_chains` UNFLATTENED
once, with a genuine `(n_chains=32, n_samples>=200)` shape, and feeds
THAT to `effective_sample_size` (`demo/ess_run.py:1-30`) -- proving the
correct pattern is known and implemented elsewhere in this same codebase,
just not in the code path the live SCOPE panel actually uses.

**Verdict: MISMATCH (Critical).** The plot's own `tau~X.X` label implies
a real integrated autocorrelation time of the sampler's Markov chain.
What is actually measured is the autocorrelation structure of a series
built from independent restarts interleaved with (in the clamped case)
short, immediately-truncated-and-mixed-together real trajectory segments
-- a quantity `tsu.ess`'s own author does not consider meaningful, in
that module's own words. See Finding C-1.

---

## `ESS`

**UI says** (VERIFICATION panel, `demo/lattice_app.py:3005-3013`):
`f"ess: {text}"` where `text` is `verification.json["ess"]` verbatim or
`"unavailable: field absent from receipt"`.

**Actual behaviour:** `demo/receipts/small/verification.json` records
`"ess": "unavailable: N/tau=1147 (N=6400, tau~=5.58) is below the
reliability threshold 5000 the AR(1) validation established (see
ess-report.md); chain too short/too correlated for a trustworthy
estimate"` -- computed ONCE, at compile time, by `_measure_mixing`
(`src/tsu_compiler/passes/search.py:494-507`) on a GENUINE unflattened `(n_chains=
32, n_samples=200, n_spins)` array from `sample_chains`
(`_VERIFY_SAMPLE_PARAMS`, `search.py:513-514`), through `tsu.ess.
effective_sample_size` (`src/tsu_compiler/ess.py:189-242`) exactly as that
module's own docstring requires. This is the CORRECT, contract-respecting
use of `tsu.ess` in this codebase, and it is honestly reported as
`unavailable` rather than a manufactured number, per this project's own
"never fabricate" rule. It never updates live and carries no relationship
to the live SCOPE panel's `tau` mark (previous row) -- two readers could
easily assume the SCOPE panel's `tau~5.6`-ish-looking mark (should one
ever land near that value) IS this `ess` figure's `tau~=5.58`, when they
are computed by entirely different code, on entirely different data, and
one is validated while the other is not.

**Verdict: MATCH** for this row taken alone (the receipt's own frozen
`ess` field is genuinely `unavailable`, honestly labelled, and correctly
computed via the reliability-gated `effective_sample_size` entry point).
The juxtaposition with the previous row is exactly Finding C-1's
substance, not a separate mismatch.

---

## `beta` / `beta_c`

**UI says** (REGIME panel, `demo/lattice_app.py:3173-3174,3182-3184`):
`f"beta={regime.beta:.4g} beta_c(Onsager)={regime.betac:.4g}
beta/beta_c={regime.ratio:.3g}"`, immediately followed on screen by
`ONSAGER_ASSUMPTION_NOTE` (`:1020-1025`, WARN-coloured):
`"Onsager's beta_c is EXACT for the UNIFORM 2-D square-lattice Ising
model with no field. This model is NOT that: couplings vary by rule,
mediator spins are hidden nodes, and the graph is not a plain square
lattice. beta_c below is an ORIENTING estimate only -- not a derived
critical point for this graph."`

**Actual behaviour:** `onsager_betac(j_max)` (`demo/lattice_app.py:998-
1017`) computes `math.log(1.0 + math.sqrt(2.0)) / 2.0 / j_max`.
Independently verified (SymPy, this task): `sinh(2 * (ln(1+sqrt(2))/2)) =
1.0000000000` to machine precision, satisfying Onsager's own condition
`sinh(2*K_c)=1` -- the CODE is numerically correct. Fed this receipt's own
mediated `j_max=2.846567915192133` (see the `J`/`b` row above), `beta_c=
0.15481337759686886` and `beta/beta_c=6.459390109063968` -- i.e. this
receipt sits **6.46x** past the orienting Onsager threshold. Had the
DISPLAYED `j_max` instead been the workload's own bare `|J|max=1.0`
(ignoring mediator couplings), the SAME formula gives `beta_c=
0.44068679350977147` and `beta/beta_c=2.269185314213022` -- **a 2.85x
difference in the headline ratio depending on which "j_max" is used**,
entirely because of which couplings are counted, not because of any
choice about beta itself. The caveat text ("mediator spins are hidden
nodes") gestures at the reason without stating the magnitude.

One further item, flagged during Phase A by the controller ahead of this
task and independently re-verified here rather than taken on faith
(`audit/findings/R10-controller-prefinding.md`): `onsager_betac`'s own
DOCSTRING (`demo/lattice_app.py:999-1001`) misderives the constant its
own code correctly computes -- it states `"Kc = arcsinh(1) =
ln(1+sqrt(2))"`, but `sinh(2*Kc)=1` implies `Kc = arcsinh(1)/2 =
ln(1+sqrt(2))/2`, not `arcsinh(1)` bare (a factor-of-2 error, present only
in the comment, not in the `return` statement, which correctly divides by
2). This is R10's finding to adjudicate in full (Wave 3); it is recorded
here only because it sits directly in this row's own file:line range and
this task's brief requires verifying from implementation rather than
inheriting a claim -- re-derivation confirms the pre-finding's read of
both the bug and the fact that it is docstring-only.

**Verdict: MATCH-WITH-CAVEAT.** The screen text is not wrong -- the ratio
shown IS `beta/onsager_betac(j_max)` for the `j_max` actually used, and
the caveat correctly says the estimate is orienting, not derived. But
"orienting" undersells how sensitive the headline number is to a modelling
choice (which couplings count toward `j_max`) that is not itself surfaced
anywhere near the ratio. See Finding I-1.

---

## `valid fraction`

**UI says** (`demo/lattice_app.py:3506-3511`): `f"valid fraction:
{frac:.1f}% (valid={self.valid_count} contract-fail={self.
contract_fail_count} non-codeword={self.noncodeword_count}
total={self.total_draws})"`.

**Actual behaviour:** `frac = 100.0 * self.valid_count /
self.total_draws` (`:3506`); `self.total_draws` increments on EVERY
message except `batch_summary`/`band_regenerated`/`band_regenerate_error`
(`:3449`, the increment happens before the kind-specific branch at
`:3468-3504`) -- i.e. the denominator is every raw physical draw
classified this session (valid + contract-fail + non-codeword), exactly
as the parenthetical breakdown states, and the breakdown's own three
counters sum to the total by construction (no fourth silent bucket).
Confirmed live by direct re-count (`sum(1 for m in pushed_messages if
m["kind"]==k)`, not eyeballed) over
`audit/traces/clamped_trace.json`'s 19 pushed messages: 18 classified
rows (`contract-fail`x11, `valid`x5, `non-codeword`x2 -- 11+5+2=18) plus 1
`batch_summary` message, which is handled in a SEPARATE branch
(`:3421-3435`) that explicitly `return`s before reaching the
`total_draws` increment at `:3449` -- so it correctly never enters the
valid-fraction denominator. `batch_summary`'s own self-reported
`{"valid": 5, "total": 18}` (`audit/traces/clamped_trace.json`) matches
this direct re-count exactly: `batch_feasibility`
(`demo/lattice_app.py:973-984`) and the live `valid fraction` label are
counting the identical thing, correctly.

**Verdict: MATCH.** The displayed formula, its denominator, and the
three-counter breakdown are exactly what the code computes, and the
live trace's own numbers reconcile with `batch_summary`'s self-report
with no discrepancy.

---

## `energy_of_draw()`

**UI says** (`demo/lattice_app.py:1072-1086`, its own docstring): `"The
physical energy E(x) of one raw {0,1} draw under IsingModel im, from ITS
OWN documented sign convention (lower.py module docstring: 'sum b s + sum
J s s == -E(x)', spins s = 2*occupancy - 1). The same formula tsu.passes.
search._from_ising uses ... reimplemented here ... rather than importing a
leading-underscore name from another module."`

**Actual behaviour:** `energy_of_draw(im, row)` computes `s = 2*row - 1;
total = im.offset; total -= sum(biases[i]*s[i]); total -=
sum(weights[k]*s[u]*s[v] for edges)` (`:1080-1086`) -- algebraically
`E = offset - b.s - s^T J s`, matching `src/tsu_compiler/passes/lower.py:8-9`'s
stated convention `"sum b s + sum J s s == -E(x)"` (i.e.
`E = -(b.s + s^T J s) + offset`, the SAME expression). `tsu.passes.
search._from_ising` (`src/tsu_compiler/passes/search.py`, used by `_measure_mixing`
for the receipt's OWN frozen `ess`/mixing diagnostic) and
`demo/ess_run.py`'s own `_energy_of` (`:83-91`) are both independently
confirmed, by direct reading, to compute the identical arithmetic. This
row's claim ("the SAME formula") is therefore a claim about three
SEPARATE, independently-typed implementations of one formula (this
function, `search._from_ising`, `ess_run._energy_of`) rather than one
shared function -- a deliberate choice per this function's own docstring
("rather than importing a leading-underscore name"), but three
opportunities for the formulas to silently diverge that a shared function
would not have. A byte-for-byte numeric cross-check of `energy_of_draw`
against `tsu.passes.search._from_ising` on a live draw (not merely a
reading of the two function bodies) is R3's mandatory falsification test
("energy_of_draw() must equal the compiler's energy AND the independent
oracle's energy on an enumerable instance," plan Wave 2) -- out of A2's
scope, flagged here as the natural next step rather than re-done
partially in this table.

**Verdict: MATCH** (docstring's convention claim reads correctly against
`lower.py`'s own stated sign convention); the "SAME formula" phrasing is
accurate about the MATH, not about a SHARED implementation -- worth
distinguishing, not severe enough alone to file as a finding (R16 already
tracks this general "canonical source vs re-typed duplicate" pattern
project-wide and R3 owns the numeric cross-check).

---

## `world generation`

**UI says** (`demo/explainer.py:255-272`, "02 SAMPLING, NOT COMPUTING"):
`"LATTICE does not calculate a single correct map ... It compiles your
workload into an ENERGY LANDSCAPE E(x) over every possible world x, then
draws worlds from the probability distribution p(x) proportional to
exp(-beta*E(x)) ... every world you see in DECODED WORLD is one draw from
that distribution ... roughly THREE QUARTERS of raw draws are discarded
... What's shown is drawn from p(x | valid), never the raw, unfiltered
chain."` `Run LATTICE.bat:2`'s window title calls the whole app a
"stochastic world generator."

**Actual behaviour:** the 8x8 grid TOPOLOGY (which variable is `g{x}_{y}`,
which cells are adjacent) is fixed, structural, and was fixed once at
compile time from `spec.yaml`'s `generate: {kind: grid, width: 8,
height: 8, ...}` (`demo/receipts/small/spec.yaml:33-37`) -- "world
generation" in the sense of laying out NEW terrain geometry never happens
live; only the per-cell VALUE assignment (water/rock/grass) varies draw
to draw, exactly as explainer.py describes. `classify_draw`
(`demo/lattice_app.py:1238-1255`) is the literal mechanism: raw physical
state -> `is_codeword` gate -> `decode` -> `spec.contract.validate` gate
-> (iff both gates pass) `render_world_image`. Live-traced task_validity
this session (clamped batch): 5/18=27.8%; receipt's own frozen
`task_validity=0.24703125`, `codeword_violation_rate=0.01328125` --
both consistent with explainer.py's stated "roughly three quarters...
codeword_violation_rate ~1%" claim, independently re-measured here rather
than taken on faith. A rejected (non-codeword or contract-failing) draw
never overwrites `last_valid_grid` (`demo/lattice_app.py:3468-3504`: only
the `kind=="valid"` branch touches it) -- confirmed structurally: the
unclamped trace's single draw was `contract-fail`, and no `_update_world`
call appears in that trace's edge list, consistent with "a non-codeword or
contract-failing draw is NEVER rendered as a world" (module docstring
point 3, `:33-34`).

**Verdict: MATCH.**

---

## `clamped path`

**UI says** (`SampleWorker`'s own docstring, `demo/lattice_app.py:1286-
1298`): `"CLAMPED ... repeated calls to tsu.simulate.simulate(...,
clamp=clamp) -- clamping is a WORKLOAD-level concept ... that simulate
already knows how to translate through enc.encode_clamp and repartition
via analyse/build_program; this worker never touches that machinery
itself, only calls simulate and classifies what comes back with the SAME
classify_draw the unclamped path uses."`

**Actual behaviour:** confirmed exactly by the trace.
`_run_clamped_batch` (`demo/lattice_app.py:1426-1454`) calls `simulate()`
(`src/tsu_compiler/simulate.py:89`) exactly once per batch
(`audit/traces/clamped_trace.json`: `_run_clamped_batch -> simulate`
count=1) with `clamp=self.clamp` (the workload-level `{"g0_0": 0}` dict
used for this trace); `simulate()` internally calls `enc.encode_clamp`
(via `_physical_clamp`, `simulate.py:84-86,139`), `analyse`
(`src/tsu_compiler/passes/analyse.py:43`, count=1) and `build_program`
(`src/tsu_compiler/passes/program.py:43`, count=1) ONLY because `clamp is not None`
(`simulate.py:131-141`'s branch) -- confirmed these two passes are ABSENT
from the unclamped trace entirely (0 occurrences), matching "this worker
never touches that machinery itself." Every clamped row IS honoured:
every `valid`-kind pushed message's decoded `g0_0` is `0`
(water, the pinned value) across all 5 valid draws in the clamped trace,
via `thrml`'s own `state_clamp` mechanism (`thrml_backend.py:210-217`),
not a post-hoc filter. `classify_draw` is confirmed to be the SAME
function object for both paths (both `_run_unclamped_tick` and
`_run_clamped_batch` call `self._classify_and_push`, which calls the one
module-level `classify_draw`, `:1357-1367`) -- there is no separate
"clamped classify" path that could silently diverge from the unclamped
one.

**Verdict: MATCH.**

---

## Findings

### Critical

**C-1: The SCOPE panel's live `tau`/ACF plot computes Sokal's estimator
over a series that is not a Markov-chain trajectory, directly
contradicting `tsu.ess`'s own documented calling contract, and displays a
confident-looking `tau~X.X` number regardless.** `demo/lattice_app.py:
1538-1539` (`render_acf_plot`, called from `_refresh_scope_panel`,
`:3354`) feeds `energy_trace.ys` -- a flat history built by appending
`energy_of_draw` once per `_handle_msg` call, `:3458` -- to `tsu.ess.
autocorrelation`/`integrated_autocorrelation_time`
(`src/tsu_compiler/ess.py:96-145`), both of which treat a 1-D input as ONE chain
(`_as_chains`, `ess.py:65-71`). This series interleaves: (a) on the
unclamped path, draws from a NEW, independently-seeded, freshly-`n_warmup`
-ed chain every single tick (`_run_unclamped_tick`, `:1401-1412`, fresh
`jax.random.bernoulli` init every call, `thrml_backend.py:222-225`,
confirmed no chain-state object persists between `thrml_sample` calls);
(b) on the clamped path, `CLAMP_N_CHAINS=6` genuinely-independent parallel
chains' worth of samples, flattened chain-major (`simulate.py:151`) and
concatenated end-to-end with no discontinuity marker. `tsu.ess`'s own
module docstring: `"callers hand this module a plain (n_chains, n_samples)
array ... this module never sees the chain's internal representation"`
and `"Callers must NOT flatten multiple chains into one series before
calling this"` (`ess.py:1-6`, and `thrml_backend.py:187-189` from the
producer side) -- `render_acf_plot` does precisely the thing both
docstrings warn against. **Concrete failure scenario:** a user watches the
SCOPE panel, sees `tau~4.2` (or any value) marked on the ACF plot, and
reasonably concludes this measures how many samples the sampler needs
between independent draws -- a number that could inform, e.g., how long to
wait before trusting a fresh DECODED WORLD as independent of the last one.
The number instead reflects the autocorrelation structure of a sequence
built from restarts and interleaved independent chains, which is not the
same physical quantity and has no established relationship to true
single-chain mixing time. The compiler's OWN correct pattern for this
exact computation exists in the same codebase (`demo/ess_run.py`, unused
by the live SCOPE panel) and the receipt's own frozen, correctly-computed
`ess` field is honestly `unavailable` for precisely the reason (short/
correlated chain) that this live plot papers over with a number every
time it has >=12 accumulated draws (`render_acf_plot`'s only gate,
`:1533-1535`). **What was checked:** live trace evidence for both draw
types (`audit/traces/unclamped_trace.json`, `audit/traces/clamped_trace.
json`) confirming fresh-seed/fresh-warmup per unclamped tick and
chain-major flattening per clamped batch; direct reading of `_handle_msg`
(`:3449-3527`) confirming `energy_trace` is a single flat ring buffer with
no chain-boundary metadata; direct reading of `tsu/ess.py`'s and
`thrml_backend.py`'s own docstrings stating the contract being violated;
cross-read of `demo/ess_run.py` as the SAME codebase's own correct
counter-example, proving the distinction is already understood elsewhere
in this project. **Recommended scope for Phase F (not performed here, per
this task's no-fix rule):** either gate `render_acf_plot`/the `tau` mark
behind a genuine `(n_chains, n_samples)` buffer (e.g. only compute it from
`raw_draws`-equivalent per-chain-tagged storage collected within ONE
`sample_chains` call, the way `demo/ess_run.py` already does), or relabel
the plot to state plainly what it actually measures and drop the `tau`
claim.

### Important

**I-1: The displayed `beta/beta_c` ratio is 2.85x more extreme (6.46 vs
2.27) depending on whether mediator-gadget couplings count toward
`j_max`, and the app always uses the mediator-inclusive value without
stating the magnitude of that choice.** `demo/lattice_app.py:3170`
(`j_max = float(np.abs(r.im.weights).max())`, over the RECONSTRUCTED,
post-mediation program) feeds `beta_regime`/`onsager_betac`
(`:1041-1043,998-1017`). Verified independently (SymPy) that
`onsager_betac`'s formula is numerically correct; verified from
`demo/receipts/small/program.json` that the workload's own declared
`|J|max=1.0` (`spec.yaml:39-42`) vs the mediated model's `|J|max=
2.846567915192133` differ by 2.85x, propagating to `beta/beta_c=6.459`
vs `2.269`. **Concrete failure scenario:** `ONSAGER_ASSUMPTION_NOTE`
(`:1020-1025`) tells a viewer the estimate is "orienting" and lists
qualitative reasons (non-uniform couplings, hidden mediator spins,
non-square graph) without saying that swapping which couplings count
toward `j_max` alone moves the headline ratio by nearly 3x -- a viewer
could reasonably treat "beta/beta_c=6.46" as a stable, if approximate,
description of "how far into the ordered phase" this model sits, when the
number is this sensitive to an unstated modelling choice about mediator
inclusion. **What was checked:** independent SymPy re-derivation of
`onsager_betac`'s formula; direct computation of both `j_max` variants and
both resulting ratios from `program.json`/`spec.yaml`; re-read of
`ONSAGER_ASSUMPTION_NOTE`'s exact text for whether it already covers this
(it does not -- it names mediator spins as A source of non-uniformity, not
as the DOMINANT term in the displayed `j_max`). Independently cross-
checked against `audit/findings/R10-controller-prefinding.md`'s own
finding that `onsager_betac`'s DOCSTRING (not its code, not its on-screen
output) misderives the constant by a factor of 2 -- a separate defect,
correctly scoped to R10, noted here only for completeness since it shares
this row's file:line range.

### Minor

**M-1: The SAMPLER panel's `n_chains/call` line for the PINNED case reads
as speed-dependent but is a constant.** See the `sample` row above for the
full citation. Recorded separately here because it is a UI-wording
precision issue, not a computation error: `SPEED_LEVELS`' own inline
comment (`demo/lattice_app.py:199-201`) says the speed control "varies
n_chains and n_samples per worker call," which is true of the unclamped
tick but not of the clamped batch's `n_chains` specifically.

**M-2: Nothing on screen distinguishes the VERIFICATION panel's frozen,
compile-time-only metrics from the live, continuously-updating "valid
fraction" label, though both use overlapping vocabulary ("valid").** See
the `verify`/`contract` row above. Not a numeric error -- both figures are
independently honest -- but the two panels sit near each other on screen
with no inline note that one was measured once, at compile time, under a
different sampler configuration (`n_chains=32, n_samples=200, n_warmup=
400, steps_per_sample=2`, `search.py:513-514`) than either live draw type
currently running (`n_warmup=300` unclamped / `600` clamped, `steps_per_
sample=4` both).

### What was tried and did not turn up a finding

- Checked whether `magnetization_trace`/`raw_draws` (SCOPE panel's other
  two live series, `demo/lattice_app.py:3465-3466`) are used anywhere to
  compute a chain-dependent statistic the way `energy_trace` is for `tau`
  -- `magnetization` (`demo/scope.py:161-171`) and `local_field_response`
  (`scope.py:240-315`) are both computed as simple pooled/per-draw
  statistics with no autocorrelation-style chain assumption baked in, so
  the C-1 finding does not extend to them; `per_cell_occupancy`
  (`scope.py:182-226`) is likewise a plain pooled mean. No finding: only
  the ACF/tau computation specifically assumes chain structure it does not
  have.
- Checked whether `render_energy_histogram_plot`
  (`demo/lattice_app.py:3372-3376`) makes any claim that would be
  undermined by the same independent-restarts fact as C-1 -- it does not;
  a histogram of energies pooled across independent restarts is exactly
  what a histogram is for, and the caption makes no chain-trajectory
  claim (confirmed by reading `render_energy_histogram_plot`'s caption
  text, which speaks only of "the distribution the energy TRACE only
  samples one point of at a time," `demo/scope.py:34-36`).
- Checked whether the DECODED MIX panel's percentages
  (`demo/lattice_app.py:1253`, `mix = {TERRAIN_NAMES[k]: 100.0*(grid==k).
  sum()/total ...}`) could mislabel a per-draw quantity as a
  session-aggregate one -- confirmed it is explicitly per-draw (recomputed
  and overwritten by every new `valid` message, `:3490`, never averaged
  across draws), and the panel title "DECODED MIX" carries no
  aggregate-over-session implication. No finding.
