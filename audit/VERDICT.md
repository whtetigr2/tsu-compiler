# R20 -- FINAL VERDICT

**Subject:** LATTICE, branch `lattice-audit`, HEAD `1b13405`, 560 tests.
**Scope:** the twenty review sections R1-R19 (+ the R10 controller pre-finding), the
three Phase-A ground-truth documents (`dependency_map.md`, `truth_table.md`,
`provenance.md`), and the controller's adjudication ledger
(`.superpowers/sdd/2026-09-03-lattice-pre-theme-audit/progress.md`).
**Mandate of this section:** assemble the verdict. No re-running of the audit, no fixes,
no behaviour change. Nothing below was measured by this section; every number is
attributed to the section that measured it.

---

## 0. The shape of the result, stated before the table

This audit did not find "a lot of problems." It found a specific and unusual shape, and
flattening it would misreport the outcome in both directions:

**The core physics passed, and passed hard.** Three-way energy agreement to `4.4e-16` on
an instance chosen to be asymmetric precisely so a double-sign-flip could not hide (R3).
A sign-flip falsification test with real teeth -- `TV = 0.276210`, one state inverting
from second-least to most likely -- proving the agreement test was capable of failing
(R3). A **pre-registered** Pearson chi-square goodness-of-fit at `alpha = 0.01`, criterion
written into the script before execution and not retuned, run against the app's *real*
sampler at the app's own constants: `p = 0.176471`, `TV = 0.0122`, `N = 10,000` -- PASS
(R15). Mediator insertion preserving the exact marginal to `8.3e-17` (one mediator) and
`1.110e-16` (K4, two mediators, with `offset_correction` accumulating correctly) (R7).
Domain-wall `is_codeword`/`decode` exact over the *complete* bit-pattern space for
k = 2..6, not a sample (R7). And an independent oracle -- written from the physics, never
importing `src/tsu_compiler` -- cross-checked against Wolfram|Alpha to **11 significant figures**
(`7.3890560989306495` vs `7.38905609893`) (A4).

**Almost every defect lives in the statistics-presentation and claim layers, not in the
computation.** One exception exists and this verdict names it rather than smoothing it
into the headline: `lower()` and `route.insert_mediators` accept `NaN`/`inf` without
validation and emit a "successfully compiled" model carrying them (R14 N-1/N-2). No
computed result is wrong for any input reachable through the front door today, but that
is an unguarded ingress in `src/`, not a presentation defect, and the summary sentence
"every defect is in the presentation layer" is *very nearly* true rather than true.

**There are underclaims as well as overclaims -- at least three.** For a project whose
entire discipline was built to prevent overclaiming, the correction pressure has
overshot in places, and that is a finding in its own right:

- `|J| <= 6.0` is an **Extropic-documented fact** (Fig. 12's cap-sweep axis annotates the
  value `6` as `"(Z1)"`, independently re-extracted from the raw PDF by R9) and is
  currently disclaimed as an unsourced project assumption at **six** independently-typed
  display sites (A3 P-3, R9 I-9a).
- `bipartite_after` renders `"unavailable: field absent from receipt"` for unmediated
  overlays when the true, measured `nx.is_bipartite` answer is sitting in the **same
  receipt's** `metrics.json` (R8). We say we do not know something we do in fact know.
- `tests/test_target.py::test_coupling_cap_is_marked_assumed_not_sourced` is a **test that
  pins an underclaim** -- it asserts, and enforces on every CI run, a statement about the
  `|J|` cap that primary-source evidence has now superseded (A3 P-2). Related: the five
  `"F-14"`-style fact ids backing `TargetProfile` are uncitable in-repo, yet every one
  that was traced turned out to be an accurate verbatim quote -- a citability gap, not a
  fabrication (A3 P-2), and `node_budget = 250,000` is likewise real and citable, not
  wrong (controller's own claim to the contrary was overturned by evidence).

**The Critical scientific defect survived 560 tests for a diagnosable reason, and the
diagnosis dictates the fix.** R19's test-potency exercise found **1 of 3** chosen
assertions would actually fail if the behaviour it guards regressed. The one that *is*
potent -- `test_estimator_discriminates_iid_from_strongly_correlated` -- is a genuinely
well-built discriminative control that would catch a broken estimator immediately. But
**no test anywhere in the suite calls `render_acf_plot`**, and every test in
`test_ess.py` constructs its own correctly-shaped `(n_chains, n_samples)` input via
`ar1_chains`. The estimator has excellent unit tests; the **caller/contract boundary has
none**, and that boundary is exactly where the defect lives. More estimator tests would
be useless. This shapes the fix queue.

---

## 1. Verdict table

One row per area. Severity is the audit's own severity of the *worst* finding in that
area. "Queue #n" refers to section 4.

| # | Area | Status | Evidence | Severity | Required action |
|---|---|---|---|---|---|
| 1 | Energy semantics & sign conventions | **PASS** | R3: IR / `energy_of_draw` / independent oracle agree to `4.441e-16` across all 8 states of an asymmetric 3-spin instance; 4th cross-check vs thrml's own softmax to `5.551e-17`; single sign flip located at `lower.py:318-320` and shown to compose correctly with thrml's internal negation | -- | None |
| 2 | Sampled distribution vs exact Boltzmann | **PASS** | R15: pre-registered chi-square, `alpha=0.01`, `p = 0.176471`, `TV = 0.0122`, `N = 10,000`, against `thrml_backend.sample` at the app's own `n_warmup=300`/`steps_per_sample=4`; criterion fixed before the run, not retuned | -- | None |
| 3 | Representation: codewords, decode, mediators | **PASS** | R7: full codeword-space enumeration k=2..6, zero misclassifications; mediator marginal preserved to `8.327e-17` (triangle, 4 configs) and `1.110e-16` (K4, 2 mediators); `assert_beta_consistent` fires correctly | -- | None |
| 4 | Independent oracle infrastructure | **PASS** | A4: oracle written from the physics, not from `src/`; Wolfram cross-check to 11 significant figures; watched failure recorded verbatim including its divergence from the plan's predicted message | -- | None (note: oracle overflows past `beta ~ 1e5`; needs a max-shift before reuse there -- R14, correctly attributed to the oracle, not to LATTICE) |
| 5 | Reproducibility given receipt + seed | **PASS** | R17: byte-identical draws across two freshly-launched processes; `demo/world.png` regenerates to an identical SHA256 | -- | None. Compile-time reproducibility is `UNAVAILABLE` (cannot be tested without `tsuc compile`); three sibling figures `UNAVAILABLE`, not confirmed |
| 6 | UI must not change the mathematics | **PASS** | R12: `SPEED_LEVELS` is four dict literals structurally incapable of carrying `n_warmup`/`steps_per_sample`; all four real assignment sites read bare module constants; A1's live trace corroborates at `speed_idx=0` vs Full; Pause truncates the *push loop*, never the `simulate()` call | MINOR | Queue #19 (caller-side test) |
| 7 | thrml usage | **PASS** | R2: block Gibbs over chromatic colour blocks via thrml's own `sample_states`, never a hand-rolled loop; target topology (degree 16, bipartite, the four offset rules) independently matches Z1's published connection rules | -- | None |
| 8 | torx / Thermalizers correspondence | **CONCERN** | R1/R2: torx fires **0** times in 1.2M call events, confirmed twice by structurally different methods; Thermalizers has no released library; the app's own "what is this" material never mentions either | MODERATE (filed Important; downgraded here -- see §5) | Queue #12 |
| 9 | Live sampler statistics (tau / ACF / ESS) | **FAIL** | A2 C-1 + R4 F-R4: Sokal's estimator run over a chain-major-flattened concatenation of independent restarts; `effective_sample_size`'s reliability gate is **unreachable by construction** from `render_acf_plot`; `tau = 0.9868` flattened vs `0.9976` correctly shaped on the *same* draws | **CRITICAL** | Queue #1 |
| 10 | Trace rendering continuity | **FAIL** | R4 I-2 + R19 F1: one unbroken polyline through restart-interleaved draws, x-axis labelled `"sweep"` (energy) and the identical rendering for magnetization | IMPORTANT | Queue #3 |
| 11 | Displayed precision | **CONCERN** | R19 F2: `task_validity` displayed `0.247031` (6 s.f.) against a binomial SE of `0.0054` at n=6400; `codeword_violation_rate` `0.0132812` against SE `0.0014` -- ~3 orders of magnitude of unearned precision | IMPORTANT | Queue #8 |
| 12 | Evidentiary integrity of receipts | **FAIL** | R17 RP-1: `demo/receipts/small/simulation.json` is git-tracked and silently overwritten by ordinary clamped use; this audit's own runs overwrote it twice; no `.gitignore` entry; nothing distinguishes it from genuinely frozen compile evidence beside it | **CRITICAL** (process/evidentiary -- see §5) | Queue #2 |
| 13 | Hardware-claim provenance (the caps) | **CONCERN** | A3 P-3 + R9 I-9a: `|J| <= 6.0` is documented (Fig. 12 `"6 (Z1)"`, re-extracted independently) yet disclaimed identically to the genuinely unsourced `|b|` cap at six sites; `TargetProfile` has one `Sourced` field feeding both gates | IMPORTANT (**underclaim**) | Queue #4 |
| 14 | Canonical-source discipline in `demo/` | **CONCERN** | R16: `demo/layers.py:56` hardcodes `FIELD_CAP = 6.0` as a literal while `frontier.py` reads `gates.json` live, as the project's own stated norm requires; its test pins the constant against itself | IMPORTANT | Queue #5 (**strictly after #4**) |
| 15 | Numerical input validation (`src/`) | **CONCERN** | R14 N-1/N-2: `lower()` validates neither term weights nor `EnergyModel.beta`; `insert_mediators`' `acosh` gadget propagates `NaN`/`inf` into weights and `offset` with a normal-looking `MediationReport` | IMPORTANT | Queue #6 |
| 16 | Onsager / `beta_c` / regime | **CONCERN** | R10: docstring contradicts its own cited identity (`sinh(2*0.8814) = 2.828`, not 1) while the code is correct; ratio moves 2.85x (6.46 vs 2.27) on mediator inclusion alone; **degree 16 vs Onsager's degree 4** -- a coordination-number mismatch the caveat never names | IMPORTANT | Queue #7 |
| 17 | Topology / bipartite reporting | **CONCERN** | R8: `bipartite_after` renders `"unavailable: field absent from receipt"` when the measured answer is in the same receipt, and the stated reason is itself wrong (the key is present with value `null`) | IMPORTANT (**underclaim**) | Queue #11 |
| 18 | Layers / overlays / composite | **CONCERN** | R13: `layers.py`'s own **"MANDATORY CAVEAT"** -- that stacking is a directed/ancestral factorization, never a joint Boltzmann sample -- appears in a source docstring and one code comment, and reaches the screen nowhere | IMPORTANT | Queue #9 |
| 19 | Zero-valid / infeasibility language | **CONCERN** | R6: two independently-typed wordings for one event; the more commonly hit one makes the **pins** the grammatical subject ("no valid world satisfies these pins"), the other makes the **batch** ("0/180 draws valid"); both wear the same absolute `"INFEASIBLE:"` prefix | IMPORTANT (borderline MODERATE -- see §5) | Queue #10 |
| 20 | Conditioning / contract claims | **PASS** | R5: the zero-rule `elev_band` contract is disclosed live (`n_rules` measured every call, never cached); base-clamp-vs-overlay-nudge distinction is on screen with a named regression hook | MINOR | Queue #21 |
| 21 | Local-field / sigmoid honesty | **PASS** | R11: post-hoc reconstruction confirmed from the arithmetic *and* from thrml's installed source (the true field never leaves `compute_parameters`); disclosed at three on-screen sites, including that reconstruction artifact and genuine defect are indistinguishable | MINOR | Queue #22 |
| 22 | Test coverage at the caller/contract boundary | **FAIL** | R19: 1 of 3 assertions potent; no test calls `render_acf_plot`; the `SPEED_LEVELS` and `batch_feasibility` tests inspect a static dict shape and bare substrings, not the behaviour they appear to guard | IMPORTANT | Tests accompany #1; plus #19, #20 |
| 23 | Performance / cost compromises | **CONCERN** | R18: no cost-forced scientific compromise found -- the one plausible story (`_VERIFY_SAMPLE_PARAMS` sized for cost) was **ruled out numerically**: ~4.2 s more on a ~204 s compile would have made this receipt's `ess` reliable. But `N_WARMUP=300` is sized by a UI-tick budget, not by any convergence diagnostic | MODERATE | Queue #18 |

**Row counts by status: PASS 9 · CONCERN 10 · FAIL 4 (23 rows).**
**By severity: CRITICAL 2 · IMPORTANT 10 · MODERATE 2 · MINOR 3 · no-severity (clean PASS) 6.**

---

## 2. Verdicts A-E

### A. Scientific correctness -- **SOUND**

The mathematics LATTICE performs is correct, and the audit tried hard to prove otherwise.
The energy convention has exactly one sign flip, located at `lower.py:318-320`, and it
composes correctly with thrml's own separate internal negation -- verified numerically, not
argued (R3). The falsification test the plan mandated was constructed so it *could* fail
(an asymmetric instance with no relabelling or sign symmetry to hide a cancelling error)
and it produced a `TV` of 0.276 with a genuine rank inversion, so the agreement result is
not vacuous. The distributional check was pre-registered at `alpha = 0.01` against the
real sampler and passed at `p = 0.176` on its first and only run. The oracle backing all
of this shares no code with `src/` and was itself cross-checked against a third party to
11 significant figures. Representation round-trips exactly over the *complete* codeword
space, and mediation preserves the marginal to float64 noise including multi-mediator
`offset_correction` accumulation.

The one qualification: correctness was established **for reachable inputs**. `lower()`
and `insert_mediators` accept `NaN`/`inf` and emit a model that looks compiled (R14
N-1/N-2). No live path currently supplies such a value -- `validate_coefficient_scale`
guards `beta` transitively -- but term weights have no equivalent front-door guard at
all, and the failure mode is a `verdict=COMPILED` receipt over a garbage model with no
diagnostic pointing back at the defective term. That is the single defect in this audit
that is not in the presentation layer, and it should not be filed away with the
presentation findings.

### B. Sampler semantics -- **CORRECT AT THE SAMPLER, WRONG AT THE READOUT**

The sampler itself is a pure function of its arguments (verified empirically, bit-for-bit,
twice: same process and two fresh processes), draws from a uniform-random init with a
fresh `jax.random.key(seed)` per call, and holds no hidden chain state (R4, R17). Nothing
about the *drawing* is wrong.

What is wrong is what the app then says about the sequence it accumulates. Each unclamped
tick is an independent restart with a fresh 300-step warmup; each clamped batch is six
mutually independent chains flattened chain-major. These are appended into one flat
`energy_trace` with no chain-boundary metadata, and Sokal's single-chain estimator is run
over the result. Worse than "an unreliable number": `render_acf_plot` imports
`integrated_autocorrelation_time` **directly**, so `effective_sample_size`'s reliability
gate -- the mechanism that exists precisely to refuse an untrustworthy answer -- is not
merely bypassed for this instance, it is unreachable from this code path by construction
(R4). And the gate would not have caught it anyway: `N/tau >= 5000` asks whether a valid
chain is long enough, a question orthogonal to whether the input is a chain at all. The
correct pattern exists in this same repository (`demo/ess_run.py`) and the live panel does
not use it. The defect's blast radius was checked honestly and is confined among
*statistics* to the ACF/tau computation -- magnetization, the energy histogram, and
per-cell occupancy are order-invariant and were confirmed so by an actual shuffle test,
not by assertion (R4). Its blast radius among *renderings* is larger, and Wave 4 found the
second instance (R19 F1) that Wave 2 did not go looking for.

### C. Torx / THRML / Thermalizers -- **HONEST AT THE SUBSTRATE, SILENT ABOUT THE REST**

thrml is used exactly as documented: `IsingEBM` built from the compiler's own model,
`Block`s from its own chromatic colour classes, `SamplingSchedule`, `sample_states` --
never a hand-written Gibbs loop, and the one bypass (`_edgeless_*`) is a documented
workaround for a specific thrml vendor `IndexError`, not a general avoidance (R2). The
topology targeted -- degree 16, bipartite, the `(1,0),(2,1),(2,3),(4,1)` offset rules --
independently matches Z1's own published connection rules.

Torx does not execute. Zero calls in 1.2M traced events, confirmed twice by structurally
different instrumentation, including monkeypatched sentinels on ten of the real `torx`
package's own public constructors (A1, R1). It exists in exactly one file as a
compile-time-only, single-bond oracle that the live app never reaches. Thermalizers has
no released library to be compatible with. And the embedding techniques are genuinely
different: mediators subdivide an **edge** to fix a bipartite-parity conflict; chain
embedding subdivides a **node** to amplify effective degree. These must never be described
as one project implementing the other's method.

R2's framing is the defensible claim and this verdict adopts it: **"convergent,
independent design on a shared target, not shared implementation."** Nothing in any
future whitepaper may exceed it. The concern is that none of this reaches a viewer --
`grep -in "torx\|thermalizer" demo/*.py` returns zero hits -- so the app's own material
explains the hardware target and says nothing about the software correspondence in either
direction.

### D. Representation / energy -- **PASS, WITH THE PROVENANCE OF THE CAPS INVERTED**

The representation layer is the cleanest result in the audit: exact over the full codeword
space, exact marginal preservation under mediation, `assert_beta_consistent` correctly
refusing a resample at a beta the mediator gadget was not built for. Energy agrees three
ways plus thrml's own route.

The defect adjacent to it is not in the representation but in what the app says about the
numbers bounding it. `TargetProfile` carries exactly one `Sourced` field where two are
needed, so `gates.py` reads one `cap`/`cap_assumed`/`cap_source` triple and reuses it
verbatim for both the coupling gate and the field gate. The consequence is six display
sites telling a viewer that a citable Extropic figure is an unsourced project assumption.
This cannot be fixed with a text edit -- it is a schema change, and it must land before
the `demo/layers.py` `FIELD_CAP` literal is touched, or that literal gets re-hardcoded
against a field that is about to be split in two.

### E. UI honesty -- **MIXED, AND THIS IS WHERE THE WORK IS**

The good is genuinely good and should not be lost in the list of defects. The PIPELINE
panel's own header says `"(compiled once, at load)"` and every one of its nine rows is
frozen -- R1 checked two stages beyond A1's list and found the panel's honest claim is
*stronger* than the finding suggested (R1). The zero-rule contract is disclosed live, with
`n_rules` measured on every call rather than cached (R5). The overlay-pin-vs-base-clamp
distinction is on screen with a named regression hook so a test can pin it (R5). The
sigmoid plot's caption states the reconstruction, states *why*, states the expected
consequence, and -- unusually -- states that a reconstruction artifact and a genuine
sampler defect cannot currently be told apart, rather than attributing the deviation away
(R11). The composite's base-staleness residual is disclosed in real on-screen text (R13).
The speed control's on-screen claim that it changes display rate only is *true*, and R12
proved it structurally rather than observationally.

The bad is concentrated and consistent in kind: **numbers and pictures that assert more
than their data supports.** A `tau` that is not a `tau`. Two connected polylines implying
continuous dynamics across hard restart boundaries. Six significant figures on a
proportion known to two. A composite whose own source calls a caveat "mandatory" and never
shows it. A caveat on `beta/beta_c` that lists three qualitative departures without any of
the three quantitative magnitudes that matter (2.85x mediator sensitivity; degree 16 vs 4;
which `j_max` is in play). And, cutting the other way, two places that hide a fact the
system already knows.

---

## 3. F -- The decision

# GO-WITH-CONDITIONS

Theme and visual work may resume, subject to the five conditions below. Every condition is
checkable by an explicit command or an explicit diff inspection; none is aspirational.

**Why not NO-GO.** The core is sound and independently verified. The defects are not in
the mathematics, and no plausible theme change can reach the mathematics -- R12 established
this structurally, not by sampling a few controls: `SPEED_LEVELS` cannot express
`n_warmup`, all four assignment sites read bare constants, and every passive UI interaction
(resize, detach, pause, layer switch) was traced and reaches no sampler call. A NO-GO would
be a stronger reaction than the evidence warrants.

**Why not plain GO.** The defects live in exactly the layer theme work touches. Three
concrete hazards make an unconditional GO unwise:

1. Restyling a plot whose *data semantics* are wrong (queue #1, #3) bakes the wrong
   semantics into a new visual language and makes the later fix a re-restyle.
2. The PIPELINE panel's honesty currently rests on a caption rendered at `font=(MONO, 8)`
   *below* a nine-row `[x]` checklist that borrows the installer idiom for "this just ran"
   (R1, Minor). A theme pass that raises the checklist's prominence or dims the caption
   converts a Minor into a live overclaim without a single line of logic changing. This is
   the clearest case in the audit of a defect that theme work can *create*.
3. Theme work means running the app repeatedly, and every clamped run silently overwrites
   a git-tracked receipt file (queue #2). The controller reverted this file about a dozen
   times across this project and never asked why. Resuming visual work without fixing it
   first guarantees more of the same contamination.

### The conditions

**C1 -- Fix queue #2 (RP-1) lands before any further app runs.** `demo/receipts/small/simulation.json`
must stop being a tracked file that ordinary sampling mutates.
*Checkable:* run the app's clamped path once; `git status --short` must come back clean.

**C2 -- No theme work touches `render_acf_plot`, `_draw_trace`, or `render_line_plot`
until queue #1 and #3 land.** These three renderers are the ones whose data semantics are
under repair.
*Checkable:* `git diff` for any theme commit must not touch those three functions; a diff
that does must cite finding `C-1`/`F-R4`, `I-2`/`F-R6`, or `F1`/`F-R10` in its message.

**C3 -- Theme work is styling only: no caption, label, unit, or number may change wording,
precision, or provenance framing.** Any text change belongs to the fix queue and must cite
a finding ID.
*Checkable:* every string literal changed by a theme diff must appear in a queue entry.

**C4 -- The PIPELINE panel's `"(compiled once, at load)"` header and its explanatory
caption must not lose prominence relative to the nine `[x]` rows.** No theme pass may make
the caption smaller, dimmer, or lower in the visual hierarchy than it is at `1b13405`.
*Checkable:* compare font size, colour token, and DOM/widget order for those two elements
before and after; both must be >= their current weight.

**C5 -- Queue #4 (the `TargetProfile` schema split) lands before any restyle of the six
caps-disclaimer sites, and before queue #5.**
*Checkable:* `grep` for the six disclaimer strings; a theme diff touching any of them
before `max_abs_bias` exists on `TargetProfile` is a violation.

**Standing:** all 560 tests plus every Phase-F test pass at each task boundary, and no
assertion is ever weakened (unchanged from the plan). The three-D lattice view remains
blocked -- it is a new surface, not a restyle, and nothing in this audit examined it.

---

## 4. G -- The fix queue, ordered by scientific importance

Ordering criterion: **how badly does this corrupt a claim a viewer would act on?** Not
ease, not file count, not risk of merge conflict.

| # | Finding | What must change | Layer | Sequencing constraint |
|---|---|---|---|---|
| 1 | **C-1 / F-A1 / F-R4** (CRITICAL) | Either collect `tau` from a genuine `(n_chains, n_samples)` buffer gathered within one `sample_chains` call -- the pattern `demo/ess_run.py` already implements -- and route it through `effective_sample_size` so the reliability gate is reachable; or drop the `tau` claim and relabel the plot to state exactly what it measures. **The accompanying test must be at the caller/contract boundary** (call `render_acf_plot` with app-realistic chain-concatenated data), *not* another estimator test: R19 proved the estimator's own tests are already potent and still gave zero coverage here | `demo/` (+ `tests/`; `src/tsu_compiler/ess.py` is correct and should not change) | First. Must precede #3 -- if the fix introduces chain-boundary metadata, #3's polyline break consumes it, and doing #3 first means inventing that metadata twice |
| 2 | **RP-1 / F-R1** (CRITICAL, evidentiary) | Stop `simulate()` writing a git-tracked file inside the frozen-evidence directory: relocate the output, or `.gitignore` it, or make the write opt-in. Nothing in a receipt directory may be mutable session state | `src/tsu_compiler/simulate.py` + repo config (`.gitignore`) | **Must land before any other queue item that runs the app**, and before theme work resumes (condition C1). Ranked #2 on scientific importance, first on wall-clock order |
| 3 | **I-2 / F-R6** + **F1 / F-R10** (IMPORTANT) | Break the connected polyline at recorded chain/restart boundaries (or use scatter marks), and rename the energy trace's x-axis away from `"sweep"`. **Fix both call sites in one change** -- `_draw_trace` and `render_line_plot` -- since fixing one and not the other is literally how F-R10 came to exist | `demo/` | After #1 |
| 4 | **P-3 / F-A5** + **I-9a / F-R7** (IMPORTANT, underclaim) | Add `max_abs_bias: Sourced` to `TargetProfile`, distinct from `max_abs_coupling`; give `gates.py` a second read site so `field_cap` stops borrowing the coupling cap's provenance; correct the five prose strings plus the VERIFICATION panel's shared `assumed` boolean. Also retire `test_target.py::test_coupling_cap_is_marked_assumed_not_sourced`, which currently enforces the superseded claim | `src/` (`target.py`, `gates.py`) + `demo/` (six sites) + `tests/` | **Must precede #5.** A schema change; the plan permits `src/` only when the root cause is genuinely there, and here it is |
| 5 | **F-R2** (IMPORTANT) | `demo/layers.py:56` must read the canonical value (`gates.json` / `tsu.target`) the way `frontier.py` already does, instead of the literal `6.0`. Its test must pin it against the canonical source, not against itself | `demo/` (+ `tests/`) | **Strictly after #4.** Fixing this first re-hardcodes the literal against a field that is about to be split in two -- the known constraint |
| 6 | **N-1 / N-2 / F-R5** (IMPORTANT) | Validate term weights and `EnergyModel.beta` in `lower()`, and `J` in `insert_mediators`' gadget, with the same discipline `validate_coefficient_scale` already applies to `coefficient_scale`. A non-finite value must fail loudly, never emit a `COMPILED` model | `src/` (`lower.py`, `route.py`) | Independent. The only compute-layer defect in the queue -- do not let its position below three presentation items imply it is a presentation issue |
| 7 | **R10 pre-finding + R10** (IMPORTANT) | Correct `onsager_betac`'s docstring (`2*Kc = arcsinh(1)`, so `Kc = ln(1+sqrt(2))/2`) and add a test asserting `sinh(2 * onsager_betac(1.0)) ~= 1.0`, which pins the identity and would fail under the docstring's version. Then strengthen the on-screen caveat with the magnitudes that matter: display the workload-only ratio (2.27) beside the mediator-inclusive one (6.46), and name the coordination mismatch explicitly (degree 16 vs Onsager's degree 4). **Keep the number** -- it is honestly computed and clearly labelled orienting | `demo/` + `tests/` | Independent. The docstring half is a latent 2x trap and should not wait for the caveat half |
| 8 | **F2 / F-R11** (IMPORTANT) | Stop applying blanket `.6g` to sampling-measured proportions. Either display an explicit uncertainty (the `n` is already known where these are computed) or cap significant figures at what `n` supports. `beta`, `j_max`, `onsager_betac` are derived/exact and may keep their current precision -- the fix is the category distinction, not a global format change | `demo/` (possibly the receipt schema, to carry `n` to the display site) | Independent. Latent beyond the audited receipt: `energy_tv`/`execution_tv` hit the same issue on any exactly-enumerable receipt |
| 9 | **R13 Important / F-R12** (IMPORTANT) | Put `layers.py`'s own MANDATORY CAVEAT on the screen where a composite is shown: each layer is its own sample conditioned on the layer below, never one joint draw. Reuse `layers.py`'s wording rather than inventing new phrasing that can drift from it | `demo/` | Independent |
| 10 | **R6 Important / F-R8** (IMPORTANT) | Reword `batch_feasibility`'s reason so the **batch**, not the pins, is the grammatical subject, matching the band path's wording, and give both paths one shared helper so they cannot drift apart again. The existing substring test would pass against a strictly *worse* string -- it must be strengthened, not reused | `demo/` + `tests/` | Independent |
| 11 | **R8 Important / F-R9** (IMPORTANT, underclaim) | Thread the measured `report.bipartite` (or a distinct "no mediation needed" sentinel) into the temperature-control readout so it can state the answer it already has, and fix the `"field absent from receipt"` reason, which is itself wrong -- the key is present with value `null` | `demo/` | Independent |
| 12 | **T-1 / R2** (MODERATE) | State in the app's own "what is this" material what the Extropic-stack correspondence is and is not: thrml used as documented; torx never executes live; Thermalizers has no released library; mediators are not chain embedding. Use R2's sentence verbatim | `demo/explainer.py` (docs) | Independent |
| 13 | **RP-2 / R17** (IMPORTANT as filed; MINOR in effect) | Log the session's initial `seed_base`, symmetrically with the two reseed paths that already log | `demo/` | Independent |
| 14 | **RP-3 / R17** (IMPORTANT) | Add a test that a committed world figure still matches a fresh render of its generating script | `tests/` | **Do this before, not after, theme work touches any rendering constant** -- a palette change in `render_world.py` currently desyncs `demo/world.png` silently |
| 15 | **P-2 / F-A4** (IMPORTANT) | Create the fact registry the `"F-14"`-style ids imply, resolving each to document, page, and quote. Every id traced resolved to an accurate verbatim quote, so this is citability, not correction | docs (+ `src/tsu_compiler/target.py` comments) | Independent; pairs naturally with #4 |
| 16 | **I-1 / F-A3** (IMPORTANT) | Remove the double codeword-test/decode/contract-validate on the clamped path -- `simulate()`'s own `got` is discarded and re-derived by `classify_draw`. Cost plus drift risk, not a present error | `demo/` (+ `src/tsu_compiler/simulate.py` if the return contract changes) | Independent |
| 17 | **P-4 / F-A6** (MINOR) | Either display `regime.json`'s `precision_limited` verdict (disclaimed as resting on the assumed `coupling_bits`) or stop loading it. A computed, honest finding is currently discarded | `demo/` | Independent |
| 18 | **R18 Moderate** (MODERATE) | Cross-check `N_WARMUP`/`CLAMP_N_WARMUP` against the receipt's own measured `tau` where available, surfaced as a gate rather than left implicit. At `tau ~ 5.58` the current 300 is ~54x and very likely adequate -- the defect is that nothing would notice if a slower-mixing receipt were swapped in | `demo/` (+ receipt schema to expose `tau`) | After #1, which is where a trustworthy live `tau` would come from |
| 19 | **R12 Minor / R19 assertion 1** (MINOR) | Add a test constructing a `SampleWorker` at two `speed_idx` values and asserting the pushed `sampler_params["n_warmup"]` is identical. The existing tests inspect a static dict and would pass against a caller-side regression | `tests/` | Independent |
| 20 | **R16 Minor** (MINOR) | Cross-check `energy_of_draw`, `search._from_ising`, and `ess_run._energy_of` against each other on the same live draw. Three independently-typed copies of one formula, with the only coverage anchored to one of them against a hand value | `tests/` | Independent |
| 21 | **R5 Minor** (MINOR) | Either measure and report the overlay-pin honour rate (the data is already in `_on_band_regenerated`'s hands, and `monotonicity_violations` is the precedent) or soften `overlay_pin_patch`'s unmeasured "rarely" | `demo/` | Independent |
| 22 | **R11 Minor / R14 N-3 / R19 Minor / R6 Minor / RP-4 / R1 Minor** (MINOR, batch) | Pin the sigmoid caption to any future canvas export; make `ess.py`'s `unavailable` reason distinguish NaN-contaminated from constant; reword the relaxation strip's "sweeps" button label to match its own caption; split `HARDWARE_INFEASIBLE` so a search-budget timeout is not named like a proven violation; add `if __name__ == "__main__":` guards to the three world scripts; correct `dependency_map.md`'s prose off-by-one (`contract-fail: 11`, not 12) | mixed `demo/` / `src/` / docs | Independent; batch at the end |

---

## 5. What this section judged differently from the reviewers who filed it

Recorded because a verdict that rubber-stamps twenty sections is worth as little as one
that rejects them.

**Overstated:**

- **RP-1's CRITICAL severity, as a *scientific* severity.** R17 explicitly raised it above
  its own other findings because it compromises the audit's evidentiary record, and that
  reasoning is sound. But nothing displayed is wrong, no result is invalidated, and R17's
  own TEST A/B independently showed sampling reproducibility is byte-identical across
  processes. It is a Critical **process** defect and an Important scientific one. It keeps
  its priority in the queue (#2, and first in wall-clock order) for reasons of evidentiary
  hygiene, not because it is the second-most-severe scientific defect. Filing it beside
  C-1 under one word flattens two different kinds of failure.
- **R2's T-1, filed Important.** The app makes *no* claim about the Extropic software
  stack; `FOOTER_TEXT` disclaims by negation. The failure scenario is a viewer inferring
  something from silence. That is a real communication gap and worth fixing, but the
  honest absence of a claim is the correct default, and the sharper risk is a future
  whitepaper, not the shipped UI. Recorded as MODERATE here.
- **R6's Important, borderline.** The behaviour is correct and provably transient (the
  flag resets on the first contrary draw, and R6 verified the reset paths), and the draw
  count sits in the same sentence. The finding is about the grammatical subject of one
  clause. R6's own verdict already hedges ("narrower than the brief's worst case"). It
  stays Important because it is precisely the class this project exists to prevent, but it
  is the weakest Important in the queue.

**Understated:**

- **R14 N-1/N-2, and the audit's own headline framing around it.** "Every defect lives in
  the presentation layer" is the audit's most quotable result and it has exactly one
  exception, which is this. An unguarded `NaN`/`inf` ingress producing a `COMPILED`
  receipt over a garbage model is a `src/` defect in the compute layer. R14 correctly
  notes `beta` is unreachable through the front door -- but term weights have *no*
  front-door guard, and R14's own reachability caveat is doing more work to soften the
  finding than the evidence supports.
- **R19's F2 (`fmt_value` precision).** Filed Important, and correctly, but its scope
  reads narrower than it is. The fix is structural -- the system currently has no way to
  express "this float is a sampling-measured proportion" -- and R19 itself notes
  `energy_tv`/`execution_tv` would hit the identical problem on any exactly-enumerable
  receipt. It is latent blast radius, not a one-receipt cosmetic issue, and it is on
  screen constantly.
- **R1's PIPELINE checklist Minor, *in the context of resuming theme work*.** As filed
  against `1b13405` it is correctly Minor: the caption is present and correctly worded.
  But it is the one finding in the audit that a theme pass can promote to a live overclaim
  without touching a line of logic, which is why it became condition C4 rather than a
  queue entry.
- **R17's RP-3 (no figure/script sync test), same reason.** Filed Important on general
  grounds; it is more urgent than that specifically because the next phase of work
  restyles rendering constants, which is exactly the change that would desync a committed
  PNG silently.

---

## 6. The audit's own reliability

The controller made two substantive claims during this audit and **evidence overturned
both**. It hypothesised tied couplings to explain the ~9.988:1 ratio between Z1's 215,904
published coupling parameters and its ~2.16M implied edges; R2/A3 tested it against the
Thermalizers paper and it failed -- every "tied weight" hit there refers to kernels reused
across time steps in a stochastic *program*, not to physical couplings. And it recorded
that "our node budget is wrong; the die is 269,568"; A3 traced 250,000 to a verbatim quote
in `2608.01615v1` p.5 (*"the entire chip has ~250,000 nodes"*), so the value is documented,
not fabricated, and the "wrong" verdict was itself wrong.

**What that implies, in the direction that favours the audit.** Both were caught, both by
the *same* mechanism -- the plan's rule that findings are re-verified from implementation
and primary sources rather than inherited -- and both were caught by agents auditing the
controller's own claims, which is the hardest direction for a correction mechanism to work
in. Both were recorded in the ledger in full, including the controller's own admission of
having reverted `simulation.json` about a dozen times without asking why, and of having
failed to put the blast-radius question about the polyline to itself after putting it to
Wave 2 about the `tau` series. An audit that surfaces its own controller's misses is
functioning. This same discipline produced the audit's best moments elsewhere: an agent
that hit three crashes at extreme `beta` and correctly blamed **its own oracle** rather
than filing them against `src/` (R14); an agent that recorded a watched test failure
verbatim when it differed from the plan's predicted message and diagnosed why, instead of
forcing a match (A4); an agent that had its source-mutation potency experiment blocked by
the harness and *said so* rather than claiming a run it did not perform (R19).

**What that implies, in the direction that does not.** Two for two is not a rate -- the
sample is two -- but it is two out of two, and it says plainly that the controller's
*unverified priors* did not survive contact with evidence in this audit. The controller is
also the adjudicator. So: any claim in this audit that originated with the controller and
was **not** independently re-verified by a wave agent should be treated as unverified. That
category includes the adjudication rulings themselves and the severity labels -- which is
part of why §5 above exists. It does not touch the core-physics results, which are the
audit's strongest evidence class precisely because they were produced by wave agents
against independent oracles under a pre-registered criterion, with no controller claim
upstream of them.

**Declared coverage boundaries, so they are not mistaken for clean results.** Compile-time
reproducibility is `UNAVAILABLE` (untestable without `tsuc compile`, which the plan
forbids). Three of the six committed world figures were not re-rendered and hash-checked;
two more pass a `seed` variable never traced to its origin -- all five are `UNAVAILABLE`,
not confirmed. One-hot encoding was read but not full-space tested. The coupling-parameter
ratio is parked, unresolved, with no claim resting on it. R19's potency exercise was
completed by literal reading of assertions, not by mutation. The sampler was never
observed at the Gibbs-step level -- both traces were bounded honestly at the JIT boundary
and crossings tallied by package rather than fabricated.

---

*R20 complete. Phase F may begin, in the order given in §4, under
`superpowers:systematic-debugging` and `superpowers:test-driven-development` per the plan.
No behaviour was changed by this section.*
