# The Fabric Tax — what Z1's connectivity costs, measured

**Produced by** `audit/connectivity_cost.py` · **receipt** `connectivity_cost.json`
**No hardware. No sampling.** This is a structural measurement over compiler passes.

## Why this is not a residual measurement

arXiv:2608.01615 App. J states that a spin which is simultaneously a pairwise
neighbour and a three-body partner *"needs both an affine edge to the output and
a bilinear edge to a hidden spin, and on a bipartite graph it cannot keep both"*,
leaving *"an irreducible per-site residual"*. The paper records that this residual
is **not simulated**, and names connectivity-aware compilation as future work.

That residual is real for an approach that **fits** a conditional onto the
lattice. This compiler does not fit. `place.py` raises `CompileError` with
`limit=0` the moment one edge is unrealized, so it never ships a model with a
dropped coupling: **the structural residual is zero on every successful compile.**

> Read that as the conditional it is. This script measures the mediation tax and
> **does not run `place()`**, so it establishes the antecedent for none of the
> models below, and `structural_residual` is recorded as unmeasured rather than
> as a number. Nor does every model here compile: `codon_spike_full` exhausts
> its placement budget for structural reasons (`audit/findings/R33.md`). An
> earlier version of this pack recorded that field as `0.0` with the reasoning
> above attached, which made a fabricated measurement out of a sound argument.

The cost does not vanish — it moves. Preserving the exact marginal on a bipartite
substrate requires mediator spins, so the price is paid in **spins** rather than
in fidelity. This measures that price.

**Two designs, two currencies, not commensurable:** they trade fidelity for
connectivity; this trades spins for connectivity. Neither number converts into
the other, and this report makes no claim about which is preferable.

## Measured

| workload | logical | mediators | physical | **tax** | \|J\|max before → after | bipartite |
|---|---:|---:|---:|---:|---|---|
| `thrml_docs_chain` **(control)** | 5 | **0** | 5 | **1.00×** | 0.5000 → 0.5000 | yes → yes |
| `codon_tiny_10aa` | 31 | 22 | 53 | 1.71× | 2.5000 → 2.8466 | no → yes |
| `codon_default_prefix` | 266 | 168 | 434 | 1.63× | 2.5000 → 2.8466 | no → yes |
| `codon_spike_200aa` | 481 | 283 | 764 | 1.59× | 2.5000 → 2.8466 | no → yes |
| `codon_spike_full` | 3,147 | 1,878 | 5,025 | **1.60×** | 2.5000 → 2.8466 | no → yes |

Workloads are Extropic's own published `codon_opt` Ising models, already in
`out/extropic-verify/`. `codon_spike_full` is the SARS-CoV-2 spike at 3,147 spins
and degree 12 — the figures the paper itself reports.

## The control is the point

`thrml_docs_chain` is **already bipartite**, so it must need zero mediators and
pay exactly 1.00×. It does.

That is what makes every other row meaningful. Had a bipartite graph also paid
tax, the numbers would be measuring this project's mediation code rather than
Z1's lattice constraint, and the table would be worthless. The script exits
non-zero if that control ever breaks.

## What the numbers say

**The tax converges.** 1.71× at 31 spins, 1.63× at 266, 1.59× at 481, 1.60× at
3,147. Across two orders of magnitude it settles near **1.6×** rather than
growing — so for this graph family the tax is approximately a constant factor,
which is a more useful statement than any single row.

**Coupling inflation is exact and predictable.** Mediation raises every coupling
it touches to `A = acosh(exp(2β|J|)) / (2β)`, and `A > |J|` for every `J ≠ 0`.
Measured 2.5000 → 2.8466 on every codon workload, matching the closed form to
within 1e-9. This is why gates are re-run after mediation: at β=1 a model passes
`|J| ≤ 6` and still needs `A > 6` for any `|J| > ln(cosh(12))/2 = 5.6534`.

**Independent reproduction.** These figures reproduce `out/extropic-verify/`'s
own 3147 → 1878 → 5025 exactly, from a separate script over the same inputs.

## Scope of validity

- **Every workload here is a degree-12 biochemistry graph.** 1.6× is a property
  of *this family* against a bipartite target. A different interaction structure
  will pay differently, and nothing here licenses quoting 1.6× as a Z1 constant.
- **Mediation count depends on the 2-partition chosen.** `insert_mediators`
  subdivides every edge lying within one side of a BFS-parity cut; a different
  cut yields a different count. These are this placer's numbers, not a lower
  bound on what any compiler would pay.
- **This is not placement.** The mediated spike is gated and verified bipartite;
  full geometric placement of 5,025 spins was not searched, consistent with the
  limitation `out/extropic-verify/EXTROPIC_VERIFICATION.md` already states.
- **`|b| ≤ 6.0` remains assumed**, not a sourced Extropic figure.

## What this does not claim

**Extropic has not measured this.** Their paper leaves the connectivity residual
explicitly unsimulated. This is a measurement on *this* placer, of a cost *this*
design chooses to pay, offered as complementary to their open question — not as
a comparison, a benchmark, or a correction to anything they published.

## Gates after mediation (rubric D1/D2)

The mediated model must still clear every Z1 gate, and the receipt records per
gate whether its limit is a **sourced** Extropic figure or this project's own
**assumption** — because a pass against an assumed cap is weaker evidence than a
pass against a documented one, and a receipt that does not distinguish them
invites the reader to treat both the same.

`codon_spike_full`, after mediation, at 5,025 spins:

| gate | measured | limit | result | limit provenance |
|---|---:|---:|---|---|
| `degree` | 12.0 | 16.0 | PASS | sourced |
| `coupling_cap` | 2.8466 | 6.0 | PASS | sourced |
| **`field_cap`** | **4.0503** | **6.0** | PASS | **ASSUMED** |
| `node_budget` | 5,025 | 269,568 | PASS | sourced |
| `colouring` | — | distinct | PASS | sourced |

**The exposure this makes visible.** `|b|` reaches 4.05 against a cap of 6.0 that
this project assumed rather than sourced. Extropic has published no numeric
`h_max` that we have found. If their real bias cap is below 4.05, this model does
not fit and the PASS above is wrong — not because the measurement is wrong, but
because the limit it was measured against is ours. That is the single most
load-bearing open question for this workload, and it is one number from Extropic
away from being settled.

Every other gate on this model clears a **sourced** figure with real headroom.
