# Coarse-to-fine cascade report

Plan: `SPR/docs/superpowers/plans/2026-09-07-coarse-to-fine-cascade.md`.
Branch `lattice-rules`. This file is the deliverable Task 2's brief points
at ("report into `audit/cascade_report.md`") and the one Task 4 completes.

Per the plan's own demand (its Global Constraints, and the reason it exists
at all -- "the previous emergence run's credibility was weakened because
prediction and outcome landed in one commit and the ordering could not be
proven"): **the section below is written and committed BEFORE Step 4/5's
official measurement script (`demo/cascade_world.py`) is run.** The
measured results are appended in a SEPARATE, LATER commit, so the ordering
is provable from git history rather than asserted in prose.

## Task 2, Step 4 — pre-registered acceptable band (written before measuring)

**The measurement:** for each 8x8 coarse cell, upsampled to a 2x2 block of
16x16 fine cells and conditioned there via `cascade_patch`/`bias_patch`,
what fraction of its four children keep the coarse cell's inherited value,
pooled over multiple independent 8x8 backdrops and every valid fine-layer
draw of each?

**Why a band, and why pre-registered separately from the run:** this
number can fail in two opposite directions, and both are easy to narrate as
success after the fact --

- **near-chance** (children statistically independent of their parent) means
  the patch did nothing: the cascade is an illusion, biases too small or a
  wiring bug silently no-ops the conditioning.
- **near-100%** (every child equals its parent) means pure blocky
  upsampling: the fine layer adds no detail at all, so nothing was gained by
  sampling a second, more expensive level.

**On the plan's own illustrative "near 25%" floor:** the plan states "near
25%" for the no-relationship case. Taken literally that is the naive
uniform-over-4-outcomes number, but this domain has k=3 categorical values
(water/rock/grass), not 4, and its true chance-matching floor is
`sum_v p_v^2` over the ACTUAL marginal frequency of each value in the
8x8 base model's decode -- exactly 1/3 (~0.333) only if the three values are
equally frequent, and higher than that if the model's marginals are skewed
toward one value. The base spec's own weights (`specs/lattice_small_8x8_k3.yaml`:
water/rock/grass self-attraction all at the same -0.4, water-rock forbidden
at the same 1.0) are symmetric across the three values, so no STRUCTURAL
reason favours one value's marginal over another's -- but symmetry of the
weights does not guarantee a measured-uniform marginal on any one finite
sample, and this report does not assume one. The strength=0.0 point in the
Step 5 sweep below measures the true floor directly; the band below is
anchored to a principled estimate of it (~0.33-0.40 for a mildly skewed
3-way domain), not to the plan's illustrative "25%" verbatim.

**PRE-REGISTERED BAND for the headline measurement (strength=1.0, chosen
because it is the standard order-of-magnitude value this project already
uses for a fresh `bias_patch` call -- `demo/elevation.py`'s own calibrated
STRENGTH=1.0, "the same order of magnitude as this receipt's own biases"):**

> **Pooled inheritance fraction in `[0.40, 0.90]` counts as the cascade
> working.**
>
> - **Floor 0.40**: comfortably above the ~0.33 naive-uniform chance level
>   and above a plausible moderately-skewed chance level (~0.35-0.40) for a
>   3-way categorical variable whose defining weights are symmetric across
>   its three values. A measured fraction at or below this is
>   statistically indistinguishable from "the patch did nothing" without
>   further argument.
> - **Ceiling 0.90**: comfortably below 1.0, leaving room for genuine new
>   fine-scale variation (>=10% of children diverging from their parent)
>   to exist. Above this, at most 1 in 10 children ever differs from its
>   parent -- that reads as near-total freezing to the parent's block
>   structure, not refinement.
> - A result **inside** `[0.40, 0.90]` is read as: the patch transmits real
>   structure downward without collapsing the fine layer to a frozen copy
>   of the coarse one -- the cascade mechanism works as designed, at least
>   for one level of refinement.
> - A result **at or below the measured strength=0 floor** falsifies the
>   mechanism outright: conditioning added nothing beyond chance.
> - A result **above ~0.90** is a working conditioning signal that is
>   simply too strong at this strength -- not proof the mechanism is dead,
>   but a sign `strength=1.0` should be lowered. Step 5's sweep is what
>   distinguishes "the mechanism is dead" from "this strength is too high":
>   if inheritance falls inside or near the band at LOWER strengths and
>   only saturates near 1.0 as strength rises, the mechanism is judged
>   working and the sweep's saturation point is reported instead of a
>   single pass/fail verdict at strength=1.0 alone.

**Disclosed methodological note, in the interest of not laundering a clean
story:** before writing this pre-registration, an ad hoc engineering smoke
test (outside this repo, in a scratch scaffolding file, never committed)
was run once to confirm the compile/patch/sample plumbing worked at all --
one parent (seed=7), one fine-layer draw batch, strength=1.0 -- and
incidentally printed a pooled inheritance of ~0.577 for that single
uncommitted run. That number is disclosed here rather than hidden. The
band above (`[0.40, 0.90]`) was derived from the chance-floor and
near-freezing-ceiling reasoning stated above, independently of that
figure -- the same band would have been written had the smoke test printed
a different number, since neither endpoint references 0.577 or was moved
toward it. The smoke test result is NOT used anywhere below as a
substitute for the official, committed, multi-parent pooled measurement
that follows in a later commit.

---

## Task 2, Step 4/5 — measured (this section committed AFTER the band above)

Ran `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" -u demo/cascade_world.py` in the foreground, repo root. `demo/receipts/small` was sampled via
`tsu.simulate.simulate`, never recompiled (confirmed: `git status --short
demo/receipts/small` shows no changes after the run). `analyse(...).bipartite`
was `True` for the 16x16 fine layer, confirmed before `place()` was ever
called (`_compile_fine_layer`'s own assertion) -- `place()` took 0.0047s,
0 mediators, matching the plan's own 16x16-overlay-class expectation of
near-instant placement.

**Non-degeneracy of this instance:** five INDEPENDENT 8x8 backdrops (seeds
7, 71, 137, 211, 389), not one. Each strength's pooled fraction is built
from ~609,000-613,000 (coarse cell, fine draw) pairs (5 parents x
~476-479 valid fine draws each x 256 children per draw) -- not a
single-draw measurement. Per-parent breakdown at the headline strength=1.0
(`demo/cascade_runs/inheritance_strength_1.0.json`): fractions of
0.5755, 0.5766, 0.5755, 0.5760, 0.5782 across the five independent parents
-- a spread of 0.003, not one outlier parent carrying the pooled number.
Codeword validity stayed at 476-480 of 480 draws (>=99%) at every strength
tested, so "valid draws" was never a thin, easily-biased subset.

### Pre-registered band vs. measured result

| | value |
|---|---|
| Pre-registered band (strength=1.0) | **[0.40, 0.90]** |
| Measured strength=0.0 floor (no patch at all) | **0.3326** (203856/612864) |
| Measured pooled fraction at strength=1.0 (headline) | **0.5764** (352346/611328) |
| Verdict | **Inside the band.** The cascade transmits real coarse structure to the fine level without freezing it into a blocky copy. |

The naive uniform-3-way chance estimate (~0.333) predicted in the
pre-registration essentially exactly matches the measured strength=0.0
floor (0.3326) -- the base spec's symmetric weights do in fact produce a
close-to-uniform marginal over water/rock/grass, confirming the reasoning
behind the pre-registered floor rather than merely coinciding with it by
luck.

### Full sweep (5 parents pooled per point, `demo/cascade_runs/inheritance_sweep.json`)

| strength | pooled fraction | delta from previous | per-unit-strength rate |
|---|---|---|---|
| 0.0 | 0.3326 | -- | -- |
| 0.25 | 0.3906 | +0.0580 | 0.232/unit |
| 0.5 | 0.4519 | +0.0613 | 0.245/unit |
| 0.75 | 0.5145 | +0.0625 | 0.250/unit |
| 1.0 | 0.5764 | +0.0619 | 0.248/unit |
| 1.5 | 0.6921 | +0.1157 | 0.231/unit |
| 2.0 | 0.7872 | +0.0952 | 0.190/unit |
| 2.5 | 0.8589 | +0.0717 | 0.143/unit |
| 3.0 | 0.9097 | +0.0508 | 0.102/unit |
| 3.5 | 0.9433 | +0.0335 | 0.067/unit |
| 4.0 | 0.9648 | +0.0215 | 0.043/unit |
| 5.0 | 0.9867 | +0.0219 | 0.022/unit |

**Where it saturates:** the marginal rate is essentially CONSTANT
(~0.23-0.25 pooled-fraction points per unit of `strength`) from 0.0 through
about 1.5 -- conditioning is "linear" in this range, not yet fighting
diminishing returns. Past strength~1.5-2.0 the rate visibly declines
(0.19/unit at 2.0, 0.10/unit at 3.0), and by strength=3.0 the pooled
fraction has already crossed the pre-registered 0.90 ceiling (0.9097) --
i.e. **strength=3.0 is already outside the "working, not frozen" band**,
even though strength=1.0 sits comfortably inside it. By strength=4.0-5.0
the curve is within 2-4 points of its 1.0 ceiling and each further unit of
strength buys under 2.5 points -- practically saturated. No FieldCapExceeded
was raised anywhere in this sweep (0.0 through 5.0); `|b|max` had enough
headroom (base 2.5 against a 6.0 cap) that the field cap was never the
thing that stopped the sweep -- diminishing returns from the domain-wall
representation itself is.

**Reading for the cascade design:** `strength` is a real, tunable
dose-response knob over the full range this project cares about --
strength~1.0 gives "the coarse world visibly guides the fine world, but the
fine world still has room to add its own detail" (57.6% inherited, 42.4%
free); strength~3.0+ gives "the fine world is basically a magnified copy of
the coarse one" (>=91% inherited). Task 3/4 should pick a strength inside
the pre-registered working band (1.0 is a defensible default; the plan does
not ask this task to also optimize it) rather than defaulting to the
highest strength that avoids FieldCapExceeded, since the ceiling end of
this curve is exactly the "no new detail" failure mode Task 2 pre-registered
against.

### Concerns

1. Only ONE level transition (8->16) was measured. Whether the SAME
   strength value produces comparable inheritance at 16->32 or 32->64 (where
   each "coarse" cell is itself already a conditioned, not independently
   sampled, layer) is Task 3's open question -- nothing here extrapolates
   to those transitions.
2. The fine layer carries literally zero self-rule terms (see this
   module's design note above) -- inheritance here measures conditioning
   in isolation from any competing fine-scale clumping preference. A fine
   layer WITH its own self-rule (if one can be found that stays bipartite)
   might inherit less at the same strength, since it would have its own
   pull working against the patch; that configuration was not tested here
   and is not claimed to behave identically.
3. The pre-registered band's exact numbers (0.40/0.90) are a stated
   judgment call, not a derived optimum -- reasonable alternate choices
   (e.g. 0.35/0.85) would not have changed today's verdict (0.5764 clears
   either), but a reader should not treat 0.40/0.90 as anything more
   precise than "meaningfully above chance, meaningfully below frozen."
4. `run_cascade`'s own `[8, 16]` demonstration (Steps 1-3, seed=7,
   strength=1.0) produces exactly ONE representative world per level, by
   design (ancestral sampling propagates one realized draw forward) -- it
   is not itself the measurement; `measure_inheritance`'s five-parent pool
   is.

