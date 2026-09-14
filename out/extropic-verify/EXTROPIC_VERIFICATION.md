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
