# Verification: `tsu` against official Extropic workloads

**Date:** 2026-09-13  
**Compiler:** local `tsu-compiler` on PaulPC  
**Official workload:** [extropic-ai/codon_opt](https://github.com/extropic-ai/codon_opt) (paper [arXiv:2606.17327](https://arxiv.org/abs/2606.17327))  
**Also:** THRML docs 5-spin Ising chain  
**Claim level:** hardware-**fit** audit of Extropic’s exported Ising graphs. Not a re-proof of domain-wall energy equivalence (that is Extropic’s).

## Method

1. Build Extropic `CodonProblem` → `prepare_ising_statics` → export constraint + inter-position edges with folded `(β, P)` weights (Extropic’s convention).
2. Load as `tsu` edge-list Ising (`beta=1` with folded weights).
3. Run `analyse` + `gate_checks` against `TargetProfile` Z1.
4. Where useful: `insert_mediators` → re-gate; light `place` on small instances.

Artifacts: `Documents\tsu-compiler\out\extropic-verify\`

## Results — codon_opt Ising/DWC at P=10, β=1

| Workload | spins | edges | deg | bipartite | colours | Z1 gates | Notes |
|---|---:|---:|---:|---|---:|---|---|
| tiny 10 aa | 31 | 118 | 12 | **no** | 4 | **all PASS** | matches paper “deg ~12” |
| CLI default ~100 aa | 266 | 875 | 12 | no | 4 | **all PASS** | |
| spike 200 aa | 481 | 1460 | 12 | no | 4 | **all PASS** | |
| **spike full 1273 aa** | **3147** | 9557 | **12** | no | 4 | **all PASS** | paper: 3147 spins, deg 12 — **exact match** |
| THRML docs chain | 5 | 4 | 2 | yes | 2 | **PASS** + `grid_embed` | sanity |

Gates checked: degree≤16, \|J\|≤6 (sourced), \|b\|≤6 (assumed), node budget 269568, colouring integrity.

## What `tsu` adds on top of Extropic’s own sampler

Extropic codon Ising is **4-colour block Gibbs** on a **non-bipartite** interaction graph. Z1’s published fabric is **2-colour / bipartite**.

After `tsu.passes.route.insert_mediators`:

| Workload | mediators | spins after | bipartite | deg | \|J\|max | Z1 gates | place |
|---|---:|---:|---|---:|---:|---|---|
| tiny | 22 | 53 | **yes** | 12 | 2.85 | **all PASS** | **OK** (140 edges realized) |
| ~100 aa | 168 | 434 | yes | 12 | 2.85 | **all PASS** | (not run) |
| **full spike** | **1878** | **5025** | **yes** | 12 | 2.85 | **all PASS** | (gates only) |

So: Extropic’s published spike Ising is Z1-cap-legal at P=10, but **not Z1-chromatic-legal** until mediated. `tsu` makes that precise and keeps the mediated program under the same caps.

## P-ramp stress (tiny) — Extropic anneals P

| P | \|J\|max | \|b\|max | coupling_cap | field_cap |
|---:|---:|---:|---|---|
| 1 | 0.40 | 0.89 | PASS | PASS |
| 5 | 1.25 | 1.74 | PASS | PASS |
| 10 | 2.50 | 2.99 | PASS | PASS |
| 20 | 5.00 | 5.49 | PASS | PASS |
| **40** | **10.0** | **10.5** | **FAIL** | **FAIL** |

At high constraint penalty, Extropic’s own schedule exceeds Z1’s sourced \|J\|≤6. `tsu` refuses that schedule before anyone flashes silicon.

## Honest limits (put these in the PR)

- Fit audit ≠ DWC correctness proof.
- Full-spike geometric placement after mediation not fully searched here (5025 spins); gates + bipartiteness + tiny place are the verified claims.
- \|b\|≤6 remains **assumed** in `tsu` until Extropic cites a numeric h_max.
- Torx / Thermalizers end-to-end not publicly runnable; codon_opt + THRML are the strongest official OSS workloads today.

## Suggested Extropic-facing narrative

> We took Extropic’s open codon_opt Ising (DWC) for the SARS-CoV-2 spike — 3147 spins, degree 12, matching the paper — and ran it through an independent Z1 fit compiler. Caps pass at P=10. The graph is 4-colourable, not bipartite; after mediation (1878 mediators → 5025 spins) it is bipartite, still under degree/\|J\|/\|b\|/budget, and Z1-chromatic. P-ramp to 40 is rejected by the sourced coupling cap. This is the Thermalizers-shaped developer contract: **fit / mediate / place / refuse illegal schedules**.

## Next steps for you

1. Write-up using this report + `EXTROPIC-NOTE.md` + mediator math (TV ~1e-16).
2. Public GitHub repo (`tsu` + `out/extropic-verify` summaries, not secrets).
3. Branch / PR or discussion to Extropic linking codon_opt + this fit audit.

---

## B5 — post-mediation re-gate (added 2026-09-14)

The method note above (step 4, "`insert_mediators` → re-gate") was right, and the
**compiler was not doing it**. Gates ran on the pre-mediation model; `place()`
then mediated; nothing re-checked. That is now fixed, and `preflight` predicts
the result before placement runs.

**Why it matters: mediation always raises the coupling it replaces.**

    A = acosh(exp(2·β·|J|)) / (2·β)     and     A > |J|  for every J ≠ 0

because `acosh(e^{2x}) > 2x` iff `e^{2x} > cosh(2x)`, which holds for all x > 0.

**The exposed window.** Solving `A ≤ cap`:

    |J| ≤ ln(cosh(2·β·cap)) / (2·β)

At β = 1 against the sourced cap of 6, that is `ln(cosh(12))/2 = 5.6534`. **The
12 is 2·β·cap, not a degree** — at cap 4 the bound is `ln(cosh(8))/2 = 3.6534`,
and at β = 2 it is `ln(cosh(24))/4 = 5.8267`. The codon family also having degree
12 is a numerical coincidence and nothing more.

So any model with `5.6534 < |J| ≤ 6.0` **passes the coupling gate and then needs a
coupling Z1 cannot hold.** Reproduced on a triangle at |J| = 6.0: zero gate
failures, then `|J|max = 6.3466` after mediation, with 2 of 4 couplings
unprogrammable. A COMPILED receipt certifying a model the hardware cannot
represent.

**This is against a sourced figure, not an assumption.** The post-mediation
failure returns `assumed = False` — it is Extropic's documented Fig. 12 cap, not
this project's working value.

**Found independently by two external reviewers**, from different evidence: one
from the P=10/P=40 behaviour in this very document, one from the gadget algebra.
Five internal reviews, a whole-branch review and 800+ tests had missed it,
because `search.py`'s own comment documented the gate ordering as a decision and
every internal reader took it as one.

**Bearing on the numbers above.** The spike's measured post-mediation
`|J|max = 2.8466` sits far below 5.6534, so nothing in this document's results
changes. The window matters for schedules at higher P: the P-ramp table's own
P=20 row (`|J|max = 5.00`) is already inside one mediation step of the bound.

---

## Sampleability vs placeability (measured 2026-09-14)

The "Honest limits" note above says the full-spike **geometric placement** after
mediation was not fully searched at 5,025 spins. That is accurate and it stands.
It has since been read as *"the full spike is blocked"*, which is **wrong** —
placement and sampling are different passes with different costs, and conflating
them understates what this compiler can already do.

Measured directly (`audit/codon_sampleability.py`, receipt in
`out/codon-sampleability/`), CPU only, 16 chains:

| workload | logical → physical | draws | tau | max R-hat **per spin** | min ESS **per spin** | frozen | uncertified | seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `codon_tiny_10aa` | 31 → 53 | 64,000 | 6.92 | 1.0013 | 7,580 | 0 | 0 | 0.8 |
| `codon_default_prefix` | 266 → 434 | 128,000 | 5.81 | 1.0029 | 6,218 | 0 | 0 | 10.7 |
| `codon_spike_200aa` | 481 → 764 | 128,000 | 5.34 | 1.0009 | 6,367 | 0 | 0 | 8.2 |
| **`codon_spike_full`** | **3,147 → 5,025** | 128,000 | 5.46 | **1.0018** | **5,991** | **0** | **0** | **44.4** |

**Every published codon model, including the full spike, samples on this
machine — and the claim rests on every individual spin, not on their average.
For the full spike that is all 3,147 logical spins: none frozen, none left
uncertified, the worst of them at R-hat 1.0018 with an effective sample size of
5,991.**

### These numbers replace an earlier, weaker table — and the correction matters

An earlier version of this section reported the full spike at `tau 4.68 / ESS
6,837 / R-hat 1.002` in 10.8 seconds. Those figures came from a single scalar
order parameter, the mean magnetisation, folded by `abs()`. A claims-vs-evidence
audit found two things wrong with that, recorded in full at
`audit/findings/R20.md`:

- Folding by `abs()` is the right move for a **Z2-symmetric** model. These
  models carry a nonzero bias on *every* spin — 3,147 of 3,147 on the full
  spike — so there is no symmetry to fold, and folding inflated ESS by between
  5.9% and 41% depending on the workload.
- More seriously, **a scalar summary cannot detect a stuck chain at all** if the
  thing it is stuck in averages out. `audit/diagnostic_control.py` demonstrates
  this on a model with a known exact answer: on an ordered antiferromagnet whose
  chains are frozen in different configurations, both the folded *and* the
  signed scalar report R-hat 1.0000, while the maximum over individual spins is
  11.39.

The verdict now requires every logical spin to mix, and the harness escalates
the draw count until every one of them certifies rather than stopping when the
mean looks healthy. That is why the full spike now reads 128,000 draws and 44.4
seconds instead of 32,000 and 10.8 — **the work got four times more expensive
because it is now actually checking what it claimed to check.** The earlier
"under eleven seconds" was measuring less.

The ESS figures are trustworthy in a specific sense: `tsu_compiler.ess` refuses
to return a number below its AR(1)-validated floor of `N/tau >= 5000` and
returns `unavailable` with a reason instead. That refusal applies per spin as
well as to the scalar, so a spin whose own ESS falls below the floor blocks the
verdict — it does not quietly drop out of the minimum.

### Are these draws actually correlated, or effectively independent?

Worth answering directly rather than leaving for a reader to wonder about, since
an effective sample size means nothing if the underlying chain is producing
independent draws — at that point the MCMC machinery is decoration.

Per-spin integrated autocorrelation time, derived from the same receipt
(`tau = draws / ESS`, so a value of 1.0 means the draws are effectively
independent):

| workload | draws | tau at the **median** spin | tau at the **worst** spin |
|---|---:|---:|---:|
| `codon_tiny_10aa` | 64,000 | 5.52 | 8.44 |
| `codon_default_prefix` | 128,000 | 3.39 | 20.59 |
| `codon_spike_200aa` | 128,000 | 3.08 | 20.10 |
| **`codon_spike_full`** | 128,000 | **2.82** | **21.37** |

The draws are genuinely correlated: the typical spin decorrelates over about
three recorded draws and the worst spin over roughly twenty-one. That spread is
the whole reason the full spike needs 128,000 draws rather than the 32,000 the
scalar was satisfied by — a handful of slow spins set the cost, and averaging
over 3,147 of them hides exactly that.

Note also that `steps_per_sample` is 4, so one recorded draw is four Gibbs
sweeps; in sweeps the worst spin decorrelates over roughly 85.

**What this does NOT establish.** That these workloads are *hard*. They are not,
on this machine — the full spike certifies in 44 seconds on a CPU. Nothing here
measures a sampling advantage for thermodynamic hardware, and no such claim is
made anywhere in this pack. What this compiler demonstrably provides is the
compilation and the preflight verdict; where this model family stops being
tractable for CPU block Gibbs is an open question we have scoped but not yet
answered (`audit/GROK_VERIFICATION_DISPATCH.md`, Check 2).

**What is still not measured:** geometric placement of the 5,025-spin mediated
model. That limit is unchanged, is a property of the placement search rather than
of the sampler, and is the one a larger machine would address.
