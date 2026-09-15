# Adversarial verification dispatch — for Grok, on Paul's machine

**From:** Claude (owns `src/`, `tests/`, `audit/`, `out/`, README, LICENSE, NOTICE, pyproject)
**To:** Grok (owns `demo/extropic-pack/**`, `demo/gibbs-observatory/**`)
**Date:** 2026-09-14
**Why now:** Paul is about to approach Extropic directly. The bar is "no
reviewer gets to ask *'wait, you're doing IID sampling?'*" — and equally, no
reviewer gets to say *"that's a long-winded way of saying 1 + 1 = 2."*

---

## The one rule that makes this worth doing

**Every check below names, in advance, the result that would be BAD NEWS for
us. A check that cannot produce bad news is not a check — it is decoration.**

If a check comes back bad, that is a **success of the process**, and it gets
written down exactly as measured. Do not repair a number. Do not soften a
sentence. Do not drop a row. If something cannot be measured on this machine,
say `unavailable` and name the reason — that is an honest result and it is what
this project does everywhere else.

Two failure modes to avoid in equal measure:

| Failure | What it looks like | How to avoid it |
|---|---|---|
| **The IID trap** | Reporting a statistic that cannot detect the problem it implies it ruled out. We just shipped one of these — see below. | State what the statistic is blind to, then measure that thing directly. |
| **The 1+1=2 trap** | Elaborate verification of something that could not have been otherwise. | If you can predict the answer with certainty before running it, it is not a check. Skip it and say why. |

---

## Path split — unchanged, do not break it

Write results **only** under `demo/extropic-pack/**` or
`demo/gibbs-observatory/**`. Do not edit `src/`, `tests/`, `audit/`, `out/`,
README, LICENSE, NOTICE or pyproject — those are Claude's, and a silent
cross-tree edit destroys the only thing making this two-agent split trustworthy.
Anything you need changed in Claude's tree, raise in Model Chat.

**Do not vendor DTM code.** `pschilliOrange/dtm-replication` carries no licence.
Install and run it if useful; publish only our own measurements.

**No secret values** in any file or commit message. **No hardware claims** —
everything here is CPU simulation via thrml. **No watts, no joules, no MHz.**

---

## Context: what Claude just found, so you don't redo it

A claims-vs-evidence audit of `audit/codon_sampleability.py` found the headline
sampling result rested on a statistic that was **structurally incapable** of
detecting a stuck chain. Full write-up: `audit/findings/R20.md`.

The harness judged sampleability from one scalar — the mean magnetisation —
folded by `abs()`. Two problems:

1. Folding by `abs()` is legitimate for a **Z2-symmetric** model. These models
   carry a nonzero bias on **every** spin (31/31, 266/266, 481/481, 3147/3147),
   so there is no symmetry to fold. Folding inflated ESS by 5.9%–41%.
2. Far worse: **a scalar summary of any kind can be perfectly healthy while the
   chains are catastrophically stuck.**

`audit/diagnostic_control.py` demonstrates (2) on a model with a known exact
answer — a 16×16 zero-field Ising lattice, Onsager Kc = 0.440687:

| coupling | βJ | regime | R̂ on \|m\| | R̂ on signed m | max R̂ per spin |
|---|---|---|---|---|---|
| ferromagnetic | 0.80 | ordered | **1.0000** ✗ | **160.90** ✓ | 12.53 ✓ |
| antiferromagnetic | 0.80 | ordered | **1.0000** ✗ | **1.0000** ✗ | 11.39 ✓ |
| either | 0.20 | disordered | 1.000 | 1.000 | 1.000 (correctly quiet) |

The antiferromagnetic row is the important one: both ground states are
checkerboards with net magnetisation ≈ 0, so **both** scalars report perfect
health on chains frozen in different configurations. Only the per-spin
diagnostic sees it.

The harness now uses a signed order parameter over logical spins only, and the
verdict requires **every** logical spin to mix: max R̂ within threshold, zero
frozen spins, and zero spins left uncertified. Guarded by
`tests/test_diagnostic_can_report_failure.py`.

**Corrected headline** (`out/codon-sampleability/`): the previously published
`ESS 6837 / R-hat 1.002` was a folded scalar. Use the per-spin figures from the
regenerated receipt. **`demo/extropic-pack/HERO_PROGRAMS.md:7` and
`demo/extropic-pack/OBSERVATORY_DEMO.md:7` both quote the old numbers and are
yours to correct** — that is the first action item.

---

## Check 1 — Is the seed=0 result representative?

**Question.** Every number in the receipt came from `seed=0`. One seed is an
anecdote.

**Method.** Run `audit/codon_sampleability.py`'s measurement at ≥8 distinct
seeds on `codon_spike_full`. Report the distribution of max R̂ per spin, min ESS
per spin, frozen count, and uncertified count.

**BAD NEWS would be:** the verdict flips on any seed; max R̂ exceeds threshold
on any seed; min ESS varies by more than ~2×; any seed produces a frozen spin.
Any of these means the published single-seed number is not a property of the
model, and the receipt must say so.

**Not 1+1=2 because:** we genuinely do not know the seed-to-seed spread. It has
never been measured.

---

## Check 2 — Where does this workload STOP being easy? (the real IID question)

**This is the most important check in the dispatch.** Read it twice.

**Question.** `codon_spike_full` — 3,147 logical spins, 5,025 physical after
mediation — samples on a CPU in about 13 seconds with every spin mixing. A
serious reviewer will immediately ask the obvious follow-up:

> *If CPU block Gibbs handles your flagship workload in thirteen seconds, what
> is the thermodynamic hardware for?*

We do not currently have an answer grounded in measurement, and **we must not
approach Extropic without one.** The honest answer requires knowing where this
family of models stops being easy.

**Method.** Take `codon_spike_full` and scale β over a ladder (suggest 0.5, 1.0,
1.5, 2.0, 3.0, 4.0, 6.0 — β=1.0 is Extropic's published model). At each β
report: max R̂ per spin, min ESS per spin, frozen count, wall time, and the draw
count needed for every spin to certify. Escalate draws until certified or the
budget is exhausted, and say which happened.

Label clearly: **β ≠ 1.0 is a hardness probe of our own construction, not
Extropic's model.** Never present a scaled-β result as their workload.

**BAD NEWS would be:** every spin certifies easily at every β we can reach. That
would mean this workload family is simply not hard for CPU block Gibbs, and the
contribution we can honestly claim is **the compilation and the preflight
verdict — not a sampling advantage.** That is a genuinely uncomfortable result
and it is also completely fine to report; what is *not* fine is going to
Extropic implying a sampling advantage we never measured.

**Also bad news, differently:** if it becomes hard at some β, report the β and
the draw count. Do not extrapolate to what hardware would do. We have no
hardware and no basis for a speedup claim.

**Not 1+1=2 because:** either outcome changes what we are allowed to say to
Extropic. That is the definition of a check that matters.

---

## Check 3 — Are the draws actually correlated, or effectively independent?

**Question.** This is the literal "you're doing IID sampling?" question, asked
directly instead of being defended against rhetorically.

**Method.** On the same draws from `codon_spike_full`:
1. Compute per-spin τ as the harness does.
2. Randomly permute each chain **along the sample axis** and recompute τ.
   Shuffling destroys time ordering, so shuffled τ must collapse toward 1.
3. Report both distributions side by side.

**BAD NEWS would be:** τ_shuffled ≈ τ_original. That would mean our τ estimate
is reading noise rather than autocorrelation, and every ESS in the project is
unfounded. (`tests/test_ess.py` already cross-checks the estimator against
statsmodels, arviz, an analytic AR(1) table and Geyer IPS — so this would be
surprising. Surprising is exactly why it is worth running: the existing tests
validate the estimator on **synthetic** series, never on our real draws.)

**Also report:** the per-spin τ distribution itself. If median per-spin τ is
≈ 1.0 at Extropic's own β = 1.0, then block Gibbs is producing near-independent
draws on this model, and **we should say so plainly** rather than let a reviewer
discover it. That is not a defect in our code — it is a fact about the workload,
and it feeds directly into Check 2.

---

## Check 4 — Does mediation preserve the marginal AT SCALE?

**Question.** `insert_mediators` is proven to preserve the exact marginal over
the original spins — verified against a brute-force oracle on small models
(finding R7, `tests/test_mediator_insertion.py`). The full spike adds **1,878
mediator spins** to 3,147 logical ones. The proof is algebraic and should hold
at any size, but it has never been checked at scale, and 1,878 gadgets each
introducing a small numerical error is exactly where an algebraic argument meets
floating point.

**Method.** Two samplers, one model:
- **A:** mediate `codon_spike_full`, sample with block Gibbs, marginalise onto
  the 3,147 logical spins.
- **B:** sample the **unmediated** model directly. It is non-bipartite, so block
  Gibbs does not apply — use single-site Gibbs / Metropolis over the original
  graph. A slow reference is fine; correctness is the point, not speed.

Compare per-spin means and a large sample of pairwise correlations. State the
comparison as a **z-score against the Monte Carlo standard error of both
estimates**, not as a raw difference — a raw difference has no threshold and
invites choosing one after seeing the data.

**BAD NEWS would be:** any systematic offset — per-spin means differing by more
than a few standard errors, especially with a consistent sign, or correlations
involving mediated edges shifted relative to un-mediated ones. That would mean
the Fabric Tax buys a **distorted** distribution, which would invalidate the
central claim of the compiler.

**Not 1+1=2 because:** the algebra is proven; the *numerics at 1,878 gadgets*
are not. Those are different claims.

---

## Check 5 — Independent reimplementation of the per-spin diagnostic

**Question.** The per-spin max R̂ / min ESS is now load-bearing for every
sampling verdict. It is Claude's code, reviewed by Claude, tested by Claude.

**Method.** Recompute max R̂ and min ESS across all 3,147 logical spins using
`arviz` directly (`az.rhat`, `az.ess`) on the same draw array. Do not import
`tsu_compiler.ess` or `tsu_compiler.preflight.diagnostics`.

**BAD NEWS would be:** arviz disagrees materially. Note arviz defaults to
**rank-normalised split-R̂**, which is a different and generally stricter
estimator than the BDA3 form we use — so a small systematic difference is
expected and is **not** a defect. The bad news is a *large* disagreement, or
arviz flagging spins ours passes. If arviz's split-R̂ is stricter and still
passes, that is a stronger result than ours and worth reporting as such.

---

## Check 6 — Verify the DTM hop counts are really a lower bound

**Question.** `audit/dtm_preflight.py` reports chain-embedding costs from BFS
over Z1's 16 offsets, claimed as a **lower bound**. The claim is stated but
never demonstrated.

**Method.** For the short jumps, brute-force-verify BFS optimality by exhaustive
search over offset combinations. For the long jumps (reach 13–24), confirm
the BFS result is reproducible and that no shorter route exists within the
search window. Separately, attempt one real embedding of a small preset and
report the **actual** spin cost against the BFS bound.

**BAD NEWS would be:** a real embedding costs *less* than our stated lower
bound — which would mean the bound is wrong, not conservative. Report the gap
either way; if a real embedding costs 3× the bound, that number belongs in the
receipt, because "lower bound" without a measured gap is not very informative.

---

## What to produce

Under `demo/extropic-pack/` (your tree):

1. **Corrected hero numbers** — `HERO_PROGRAMS.md:7` and
   `OBSERVATORY_DEMO.md:7` currently quote `ESS 6837 / R-hat 1.002`, a folded
   scalar. Replace with the per-spin figures from the regenerated
   `out/codon-sampleability/codon_sampleability.json`.
2. **One verification note** carrying, per check: the question, the method, the
   measurement, and **what would have been bad news** — even where the result
   was clean. A reader must be able to see the check could have failed.
3. Each check gets the project's standard protocol header (SYSTEM DEFINITION,
   STATE VARIABLES, TRANSITION RULES, ALLOWED / FORBIDDEN OPERATIONS,
   ASSUMPTIONS, INVARIANTS, MEASUREMENTS, NULL HYPOTHESES, SUCCESS / FAILURE
   CRITERIA, PROVENANCE, SCOPE OF VALIDITY) and a plain "what this is" table at
   the top. See `audit/codon_sampleability.py` or `audit/diagnostic_control.py`
   for the house style.

Report anything needing a change in Claude's tree via Model Chat.

## Environment

```
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe"
```
Working directory `C:\Users\whtet\Documents\tsu-compiler`. Never commit to any
other repository. Stage by explicit path — never `git add -A`. Do not touch
`receipts/EXP-G8-D2/`, `receipts/EXP-G8-D3/`, `figures/z1_lab_screenshot.png`,
`demo/cascade.py`, `demo/cascade_runs/`. Do not commit Observatory or Model-Chat
session dumps.

## Standing principle

Extropic has silicon. If a measurement of ours disagrees with something they
published, the first hypothesis is that **we** are wrong, or that they observed
hardware behaviour that does not surface in THRML or simulation. Nothing in this
dispatch is an error report about their work, and nothing produced from it
should read as one.
