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

<!-- Task 2 Step 4/5 measured results, and Task 3/4 sections, are appended
     below in later, separate commits -- never edited into this section. -->
