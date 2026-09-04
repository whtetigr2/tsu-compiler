# Task 6 -- The emergence experiment

Plan: `2026-09-04-lattice-rule-taxonomy.md`, Task 6. The question that
motivated the entire plan: **do behavioural rules produce recognisable
geography, or just noise?**

`specs/emergence_8x8.yaml` carries the full account of what is built and,
just as load-bearing, what is deliberately excluded and why (a class the
matrix did not clear, or a mechanism this task could not verify safe at
this scale/composition, does not get built in). This report does not repeat
that reasoning; it records the experiment's own honesty trail: prediction
before compiling, the compile verdict exactly as it came back, and an
honest judgement against the prediction.

---

## Step 1 -- the spec

Four rules, four of Task 3's EXACT, gate-fitting classes, on an 8x8 binary
grid ("raised" (1) / not (0) -- the same elevation-indicator semantics
`lattice_elev_band_8x8.yaml` already uses):

1. **GRADIENT** -- `product_over_edges`, same-value adjacency preference,
   weight -0.4 (the identical proven-safe shape and weight
   `lattice_elev_band_8x8.yaml` already measured clean).
2. **NEIGHBOURHOOD** -- `neighbourhood_count` (Task 5's new template),
   value=1, target=2, weight=0.1 -- deliberately weaker than GRADIENT, a
   second-order counter-force against solid blobs.
3. **PAIRWISE** -- a hand-written `product` term between two named cells
   (g0_0, g7_7), weight -0.5.
4. **GAMEPLAY** -- a hand-written `product` term, exactly-one-of-3 centre
   candidates (g3_3, g3_4, g4_3), weight 0.5, with the checkable half
   (at-most-one) wired into `contract.validate` via three `forbid_both`
   rules over the same three cells.

Excluded: `morphology`, `ecological` (DISTORTED), `statistical`
(EXACT-but-degree-63), `boundary` (INEXPRESSIBLE), `climate` (measured, via
`analyse()`, to collide with NEIGHBOURHOOD on a shared categorical grid:
max_degree=18 > 16), `cluster`/`topological`/`hydrological` (cheap alone,
but a version that visibly influences the rendered terrain needs a
capacity gate only ever verified at 3-cell scale -- left out rather than
included as inert, terrain-disjoint decoration). Five classes were judged
buildable and safe at this scale and composition; this spec builds four of
them as four *distinct* rules (fewer than ten, per the brief's own
instruction not to invent a mechanism to hit a number) -- PAIRWISE and
GAMEPLAY are each a single hand-written term, not swept over the grid, so
they read as smaller in scope than GRADIENT/NEIGHBOURHOOD but are no less
honest instances of their own matrix rows.

**Structural pre-check** (via `tsu.passes.analyse.analyse` on the real
lowered model -- NOT `tsu compile`, which this task may run only once):
n_nodes=64, n_edges=307, max_degree=12 (Z1 cap 16, margin 4), bipartite=
False, max_abs_J=0.35 (cap 6.0), max_abs_b=0.90 (cap 6.0). Confirmed
directly against the committed spec file by both `tsu inspect
specs/emergence_8x8.yaml` and a standalone `encode`/`lower`/`analyse` call
-- both agree exactly.

---

## Step 2 -- prediction, written BEFORE compiling

Written before `tsu compile` was run. This is the pre-registration; Step 5
judges the actual result against this, not the other way around.

**P1 (GRADIENT produces contiguous patches, not noise).** Raised cells in
sampled worlds should show positive spatial autocorrelation -- visually
contiguous patches, and a largest-connected-component size and
component-count distribution measurably different from an i.i.d. Bernoulli
grid sampled at the same density. *Falsifier:* if the raised-cell pattern
is statistically indistinguishable from independent per-cell noise at the
same density (largest component size close to the i.i.d. baseline), GRADIENT
had no measurable effect at the sampled beta/weight.

**P2 (NEIGHBOURHOOD keeps density moderate, not saturated).** Overall
raised-cell density should sit at a moderate level (not near 0% or 100%
across samples), and should not freeze to a single constant world.
*Falsifier:* if every sample is (near-)identical -- a single dominant phase
covering nearly the whole grid, the classic freezing failure mode this
project has hit before at stronger weights -- NEIGHBOURHOOD (and thermal
noise generally) failed to counteract GRADIENT's clumping.

**P3 (PAIRWISE leaves a localised signature).** g0_0 and g7_7 should be
raised together more often than chance and more often than an arbitrary
same-distance pair elsewhere on the grid. *Falsifier:* no detectable
elevated co-occurrence between g0_0 and g7_7 specifically.

**P4 (GAMEPLAY's exactly-one holds more than chance).** Among VALID samples
(task_valid=True, i.e. satisfying the at-most-one contract), exactly one of
{g3_3, g3_4, g4_3} being raised should occur more often than the "at most
one" contract alone would produce by chance, and `task_validity` overall
should be measurably above what three independent fair coins would give
(P(at most one of 3) = 1/2 at p=0.5, lower at higher p). *Falsifier:* if
task_validity is at or below the naive independent-coin baseline, the soft
preference (weight 0.5) added nothing beyond the hard-ish contract check.

**P5 (the actual bar -- Paul's).** *"If you look at the map and go 'holy
shit, I didn't tell it to draw a mountain range there' -- you've reached
the thing."* Rendered worlds should show a recognisable macro-structure --
a handful of clearly bounded, irregularly-shaped regions -- as opposed to
either uniform static noise or a single degenerate blob covering the whole
grid. This is the qualitative judgement Step 5 makes honestly, not talked
around: if it looks like noise, this report says so.

---

## Step 3 -- compile

```
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
    -m tsu.cli compile specs/emergence_8x8.yaml --out audit/receipts/emergence_8x8 \
    --placement-restarts 12 --placement-iters 200000
```

**Verdict: `COMPILED`.** Run exactly once, per the task's budget. Receipt at
`audit/receipts/emergence_8x8/`.

**Full gate table** (`gates.json`, target `z1`):

| gate | passed | measured | limit | assumed |
|---|---|---|---|---|
| degree | true | 12 | 16 | false |
| coupling_cap (\|J\|) | true | 0.35 | 6.0 | false |
| field_cap (\|b\|) | true | 0.9 | 6.0 | true |
| node_budget | true | 64 | 250000 | false |
| colouring | true | -- | distinct | false |

Every gate the structural pre-check predicted (Step 1) held under the real
compile, exactly: degree 12, \|J\|max 0.35, \|b\|max 0.90 -- no drift
between the cheap `analyse()`-only pre-check and the real gate evaluation.

**What the pre-check could NOT predict, because it only runs after
`place()`:** encoding selection picked `domain_wall` over `one_hot`
(`candidates.json`) -- both reduce to the same 64-variable, all-binary
model here, so the choice is immaterial to this spec; more informatively,
**embedding this NON-bipartite 64-node logical model onto Z1's bipartite
substrate required 116 mediator spins** (`mediation.mediator_count`,
`mediation_method: bfs_depth_parity_with_greedy_local_search`), bringing
the PHYSICAL p-bit count to 180 -- comfortably inside the 250,000
node_budget, but nearly 3x the logical spin count, and a cost this task's
own `analyse()`-only pre-check (Step 1) genuinely could not see:
`GraphReport.mediators` reports `-1` ("not computed") once a graph exceeds
20 nodes (`MAXCUT_EXACT_LIMIT`), which ours does at 64. This is worth
naming plainly: gate-fitting (degree/\|J\|/\|b\|/node_budget) says nothing
about mediation cost, and the real placement step found a real, substantial
one that the cheap pre-check had no way to surface. It happened not to
matter here (node_budget has enormous headroom), but a denser or larger
emergence spec could hit it.

**Pass timings** (`passes.json`): `place` dominates at 222.55 s (the
`--placement-restarts 12 --placement-iters 200000` search); every other
pass (encode/lower/analyse/gate_checks/route/build_program/regime/verify)
totals under 3 s combined. Consistent with `lattice_elev_band_8x8.yaml`'s
own documented ~200 s compile time.

**Workload summary** (`workload.json`): 64 variables, 290 logical
interactions (term count, matching `len(s.terms)` measured directly against
the spec), state_cardinality=2 (binary throughout), constraint_classes=1
(the contract's three `forbid_both` rules are one KIND of check, per
`constraint_classes`' own documented meaning -- not a raw rule count).

**Regime caveat** (`regime.json`), reported honestly though it does not
affect the SOFTWARE sampling this report's Step 4 relies on:
`regime: "precision_limited"`, `basis: "two distinct |J| values are closer
together than one quantisation step and may become indistinguishable"`.
This is a caveat about a REAL Z1 chip's coupling-value resolution, not
about the exact floating-point `IsingModel` `simulate()` samples in Step 4
-- flagged here because it is a genuine, freshly-measured fact about this
spec's hardware realism the receipt itself surfaced, not because it changes
anything below.

---

## Step 4 -- sampling and terrain mix

Sampled via `tsu.simulate.simulate` against the already-compiled receipt
(never re-running `compile_spec`/`place`) -- `audit/emergence_sample.py`,
committed alongside this report and reproducible with the pinned
interpreter (`PYTHONIOENCODING=utf-8 ".../python.exe" audit/emergence_sample.py`).
32 chains x 200 samples, 400 warmup steps, seed=1: **6400 total draws**.
Every number below is computed directly from those draws (or, for the
i.i.d. baselines, from 2000 freshly generated synthetic grids at the SAME
measured density) -- nothing here is asserted without being computed
first.

**Validity.** `codeword_violations: 0` (expected -- an all-binary spec has
no domain-wall chain to violate). `task_validity: 0.2009` (1286/6400),
consistent with the receipt's OWN compile-time verification pass
(`verification.json`: `task_validity: 0.21375`, a different seed/sample
budget -- the two independent measurements agree to within their own
sampling noise, a cheap but real cross-check of determinism).

**Terrain mix (density).** Over the 1286 valid samples: mean raised-cell
fraction **0.834** (std 0.047), range [0.672, 0.938]. **1286 of 1286 valid
samples are pairwise-distinct grids** -- not a frozen/degenerate single
world repeated.

**P1 -- clustering.** Largest-4-connected-component size averages 53.03
cells (of 64) against an i.i.d.-Bernoulli(0.834) baseline of 52.93 -- a
ratio of 1.002, i.e. **uninformative**: at 83% density, ANY i.i.d. random
field is deep in the supercritical percolation regime (site-percolation
threshold on this lattice is ~0.593) and nearly always has one giant
component regardless of any coupling, so this metric saturates for both
the sampled and the null-hypothesis case and cannot discriminate. A
sharper statistic, added after seeing this: **boundary length** (count of
4-adjacent cell pairs with DIFFERING values -- the direct complement of
what GRADIENT's `(1,1)`-only term rewards). Sampled mean: **27.53** edges
(of 112 possible); i.i.d. baseline at the same density: **30.93**. Ratio
**0.890** -- sampled worlds have measurably FEWER value-disagreeing
adjacent pairs than chance at the same density (11% fewer), a real,
positive, if modest, clumping signal.

**P2 -- density.** Mean density 0.834 is NOT "moderate" under a strict
reading of the pre-registered prediction; see Step 5.

**P3 -- PAIRWISE (g0_0, g7_7).** P(both raised) = 0.7154 vs. the
independent-marginals product 0.7033 -- a 1.7% elevation. The named
baseline pair (g0_7, g7_0 -- same Chebyshev distance, opposite corners,
touched by no rule) shows P(both raised) = 0.6143 vs. independent-product
0.6076 -- a 1.1% elevation. The named pair's excess (1.7%) is only
marginally larger than the untouched pair's (1.1%); see Step 5.

**P4 -- GAMEPLAY (g3_3, g3_4, g4_3).** Among valid samples: 0 raised in
5.60% (72/1286), exactly 1 raised in **94.40%** (1214/1286), 2+ raised in
0.00% (contract-enforced, by construction). `task_validity` (0.2009) is
**2.7x** the naive independent-Bernoulli(0.834) baseline for "at most 1 of
3 iid cells" (0.0735) -- and that baseline is itself GENEROUS to the null
hypothesis: two of the three candidate cells (g3_3-g3_4, g3_3-g4_3) are
grid-adjacent, so GRADIENT's own clumping term actively pulls them toward
MATCHING (mostly both-raised, given the 83% ambient density), which would
make "at most 1 of 3" *harder* than the independent baseline suggests, not
easier -- so the true "no GAMEPLAY term" baseline is lower still, and the
measured 0.2009 is a real, attributable effect of GAMEPLAY's own weight-0.5
term (helped by the hard-ish `forbid_both` contract check) overriding both
ambient density and local GRADIENT pull specifically at those 3 cells. The
"exactly 1 vs. 0" split conditional on validity (94.4% vs. 5.6%), however,
is close to what conditioning on ambient density (83.4%) predicts alone
(93.8%, computed from `3p(1-p)^2 / [3p(1-p)^2+(1-p)^3]` at p=0.834) -- see
Step 5 for what this nuance means for P4.

**A finding neither predicted nor searched for: a strong, precisely
localised suppression at the GAMEPLAY cells.** The per-cell raised-
probability, averaged over all 1286 valid samples (`per_cell_probability.png`):
grid-wide values sit in a tight 0.77-0.90 band EXCEPT at exactly the three
GAMEPLAY candidate cells -- `g3_3: 0.182`, `g3_4: 0.375`, `g4_3: 0.387` --
against a grid-wide mean of 0.834. A first look at six individually
rendered worlds (`worlds.png`) suggested a recurring light patch near the
grid's edges; tested as a real hypothesis (boundary/corner cells have
lower grid-graph degree, hence weaker GRADIENT coupling, hence maybe more
likely to be a "hole") against all 1286 valid samples, not just six, and
found NOT supported: mean P(raised) by position is corner=0.809,
edge=0.848, interior=0.827 -- no meaningful monotonic trend, differences of
2-4 points, nothing like the GAMEPLAY cells' 40-60-point drop. The
recurring patches visible by eye in the six-world panel were, on this
check, mostly small-sample coincidence; the one real, strongly localised,
fully-explained structural feature in this experiment is GAMEPLAY's own.

**Renders.** `audit/receipts/emergence_8x8/worlds.png` (6 valid samples,
light=unraised/green=raised) and `audit/receipts/emergence_8x8/per_cell_probability.png`
(the aggregate heatmap above) are committed alongside this report.

---

## Step 5 -- honest judgement

Judged against Step 2's own prediction, in order:

**P1 (contiguous patches, not noise): CONFIRMED, modestly.** The metric I
pre-registered (largest-component size) turned out to be the wrong tool --
it saturates once density is deep in the percolating regime, which this
system is. That is a real methodological miss in the prediction, not a
finding about the rules, and I am naming it rather than quietly swapping in
the boundary-length statistic and pretending it was the plan all along.
Boundary length, computed after seeing the saturation, DOES show a real
effect: 11% fewer differing-value adjacent pairs than an i.i.d. field at
the same density. GRADIENT is doing real, measurable, directionally-correct
work. It is a modest effect, not a dramatic one -- consistent with a
weight (-0.4) deliberately chosen to match a proven NON-freezing regime
rather than to maximise visible structure.

**P2 (moderate density, no freezing): PARTIALLY CONFIRMED.** The
"doesn't freeze" half held cleanly -- 1286 of 1286 valid samples were
pairwise-distinct, nowhere near the single-frozen-world failure mode this
project has hit before at stronger weights. The "moderate" half did not:
mean density 83.4% is a real, predictable skew, not noise -- and, in
hindsight, predictable BEFORE compiling from the spec's own design, not
just after: GRADIENT's `product_over_edges` term only rewards `(1,1)`
adjacency (mirroring `lattice_elev_band_8x8.yaml`'s own one-sided design),
with no matching `(0,0)` term, so there is a built-in asymmetric pull
toward "raised" that NEIGHBOURHOOD's weaker, genuinely symmetric pull
(toward a target density, not toward either extreme) only partially
offsets. I should have derived this from the spec BEFORE writing "moderate"
into the prediction; I did not, and P2 is graded against what I actually
wrote, not against what I now realise I should have anticipated.

**P3 (PAIRWISE leaves a localised signature): NOT CONFIRMED --
inconclusive.** The named pair (g0_0, g7_7) does show slightly more
co-occurrence above its own independence baseline (1.7%) than an arbitrary
untouched pair at the same distance (1.1%), which is directionally what
P3 predicted -- but the gap between them (0.6 percentage points, on 1286
samples) is small enough that I cannot honestly call it distinguished from
sampling noise or from the generic positive correlation ANY two cells pick
up from shared global density fluctuations across chains. A weight of -0.5
touching only 2 of 64 cells, inside a system where 290 other terms are
also pulling on the same cells (directly or via chained correlation), may
simply be too weak relative to the rest of the model to leave a detectable
individual signature at this sample budget. This is the plan's own
"negative result is worth more than a tuned positive one" principle in
practice: I am not rounding a 0.6-point gap up to a confirmation.

**P4 (GAMEPLAY's exactly-one holds more than chance): CONFIRMED, with a
real nuance stated plainly.** `task_validity` (0.2009) is 2.7x a naive
independent-density baseline (0.0735) that, if anything, UNDERSTATES the
difficulty GAMEPLAY had to overcome (GRADIENT's own adjacency pull works
AGAINST validity for two of the three candidate-cell pairs, at this
density) -- so the at-most-one constraint is holding well above chance, a
real and attributable effect of the weight-0.5 term plus its contract
check. The nuance: WITHIN the valid subset, the exactly-1-vs-0 split
(94.4%/5.6%) is close to what conditioning on ambient density alone would
predict (93.8%/6.2%) -- meaning GAMEPLAY's own soft pull mostly buys
"at-most-one holds at all" rather than an ADDITIONAL preference for
"exactly one" over "none" specifically, on top of what density already
implies. Both halves are true and both are reported.

**An unpredicted finding, investigated rather than just noted:** the
GAMEPLAY cells show a strong, precisely localised probability suppression
(0.18-0.39 vs. a grid-wide 0.83) -- not something Step 2 predicted in this
form (P4 predicted validity elevation, not a visible per-cell "hole"). A
visual hypothesis about a DIFFERENT recurring pattern (boundary cells) was
formed from six rendered worlds, then TESTED against all 1286 valid
samples and found not supported (corner/edge/interior means differ by only
2-4 points) -- reported as a tested-and-rejected hypothesis, not quietly
dropped once it didn't pan out.

**P5 -- Paul's own bar.** Looking at `worlds.png` and
`per_cell_probability.png` honestly: this is **not** a "holy shit, I didn't
tell it to draw a mountain range" moment. The single clearest, strongest
structural feature in this experiment -- the GAMEPLAY hole -- is exactly
what was asked for, precisely where it was asked for, which is a genuine
and non-trivial demonstration of COMPOSABILITY (a small, narrowly-scoped
rule holding its own shape against 289 other terms pulling on the same
cells) but is not the kind of unprompted surprise the bar names. The
diffuse, grid-wide structure (GRADIENT's clumping, boundary length 11%
below chance) is real but modest, and reads more as "a mostly-uniform
raised field with a few scattered, moderately-irregular holes" than as
richly varied geography. **Called plainly: this is a real but modest
positive result, not a dramatic one.**

**What an 8x8 grid can and cannot show, stated explicitly.** Eight cells
across is a small canvas, and this bounds what P1/P5 can show regardless of
weights. A "coastline" or a "ridge" is a claim about an EXTENDED, elongated
boundary -- something that reads as a line or a curve, not a blob's edge.
At 8x8, the largest raised patch already averages 53 of 64 cells; there is
barely enough unraised area (about 10 cells on average) to form more than a
couple of small, roughly convex holes, and 10 cells split across a couple
of holes cannot trace anything long enough to look like a coastline even on
the most generous reading -- there simply is not enough substrate for that
shape to exist at this size, independent of whether GRADIENT/NEIGHBOURHOOD
are "working" in the sense this report measured. The boundary-length result
(11% below chance) is real and is the right kind of evidence at this scale
-- a STATISTICAL signal computed over many samples -- but it does not
license a claim about visible large-scale shape, and this report is not
making one. A grid an order of magnitude larger (something like 40x40 or
64x64, matching the scale several already-committed production specs use)
is what a "does it draw a mountain range" test actually needs to be a fair
test; an 8x8 run can show whether the underlying couplings push in the
right direction (which they measurably do) but cannot show what a person
would recognise as terrain. That is a property of the CANVAS SIZE chosen
for this task, not a further failure of the rules themselves, and it is
recorded here as a finding about what this experiment can and cannot
license, not folded silently into the P1/P5 verdicts above.

**Diagnosis of why, stated rather than left implicit.** GRADIENT's
`(1,1)`-only asymmetric coupling is the dominant force on density and
pushes it high; NEIGHBOURHOOD's counter-pull (weight 0.1, a fifth of
GRADIENT's 0.4) is too weak relative to GRADIENT to produce the kind of
back-and-forth tension (ridges, coastlines, two comparably-sized phases)
that would make for visually rich, surprising terrain -- it measurably
softens GRADIENT's blobs (the boundary-length result) without seriously
contesting the density skew. A spec built instead around a SYMMETRIC
self-affinity design -- rewarding BOTH `(0,0)` and `(1,1)` adjacency, the
way `lattice_small_8x8_k3.yaml`'s three same-value terms do together, so
neither phase has a built-in energetic advantage -- would very likely
produce a more balanced, two-phase landscape and is the natural next
experiment; this task's own gate (build only from what Task 3 already
cleared, verify structurally before the one real compile, never invent a
new mechanism to chase a nicer picture) is exactly why that revision is
future work and not something quietly substituted into this run after
seeing the result.

---

## Concerns

- The Step 2 prediction's P1 metric (largest-component size) was the wrong
  tool for this density regime (percolation saturation) -- a genuine
  planning miss, corrected in-flight and reported as such rather than
  silently replaced.
- P3 (PAIRWISE) is genuinely inconclusive at this sample budget (1286 valid
  draws); a larger sample or a stronger weight would be needed to say more,
  and this report does not claim more than the data supports.
- `climate` and `cluster`/`topological`/`hydrological` were excluded on
  reasoning grounded in fresh `analyse()` measurements (climate) and the
  matrix's own small-scale-only verification (the flow classes) --
  `specs/emergence_8x8.yaml`'s own description carries the full argument;
  neither exclusion required inventing or extending a mechanism, per this
  task's own constraint.
- Placement (`place`, 222.55 s) found 116 mediator spins were needed to
  embed this non-bipartite 64-node model onto Z1's bipartite substrate --
  a real hardware cost this task's cheap `analyse()`-only pre-check could
  not see (`GraphReport.mediators` reports `-1`/"not computed" above 20
  nodes). It did not threaten the node_budget gate here, but a larger or
  denser emergence spec built the same way should expect a similar,
  unmeasurable-in-advance mediation cost.
