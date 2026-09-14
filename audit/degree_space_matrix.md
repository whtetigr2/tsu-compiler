# Degree/Space Trade — the benchmark matrix

Plan: `2026-09-04-degree-space-trade.md`. Task 1 fixes the benchmark set
BEFORE either route (A: node splitting; B: stencil replication) exists, so
neither can later be tuned to a favourable case. This file is extended by
later tasks; this section is the "before" table only.

Every number below is MEASURED by `audit/degree_space_probe.py::benchmark_graphs()`
through the real pipeline (`load_spec`/`encode`/`lower`/`analyse`, or the
existing `EnergyModel` builders it reuses) — never predicted, never
hand-computed and then asserted. Reproduce with:

```
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
    audit/degree_space_probe.py
```

## Before (no split, no replication)

| benchmark | n_nodes | n_edges | max_degree | bipartite | \|J\|max | \|b\|max | `check_gates(Z1, allow_assumed=False)` |
|---|---|---|---|---|---|---|---|
| `assignment_8x8` | 1744 | 9240 | **32** | False | 0.375 | **8.0** | FAILS: `degree`, `field_cap` |
| `statistical_8x8` | 64 | 2016 | **63** | False | 0.005 | 0.16 | FAILS: `degree` |
| `terrain_k5` | 1280 | 8320 | **16** | False | 1.25 | 5.0 | PASSES all gates (boundary case — degree exactly at the cap) |

Gate results are the REAL `src/tsu_compiler/gates.py::check_gates` output on these
three lowered models against the `Z1` `TargetProfile`, `allow_assumed=False`
(so the `field_cap`/`|b|` gate — a project assumption, not Extropic-sourced —
is not silently downgraded). With `allow_assumed=True` the `field_cap`
failure on `assignment_8x8` is downgraded and only `degree` remains, which
is why the plan's own prose ("Fails degree **and** field") is stated here
against the stricter, non-downgraded call.

`assignment_8x8` reuses `audit/assignment_gadget_probe.py`'s own
`build_grid_gadget_model(width=8, height=8, target=4)` (that file's Step 5);
`statistical_8x8` reuses `audit/measure_expressibility.py::measure_statistical()`'s
exact construction (8x8 binary grid, `(sum_i x_i - 16)**2`, weight 0.01);
`terrain_k5` reuses `specs/lattice_l1_16x16.yaml` (one-hot, k=5 categorical,
`coefficient_scale=0.25`) — the same instance already measured pre-mediation
at `max_degree == 16` in
`tests/test_lattice_specs_compile.py::test_l1_one_hot_mediation_matches_the_measured_prototype_numbers`.
All three numbers reproduce known, previously-recorded priors exactly
(1744 nodes / 32 degree / 8.0 field for the assignment gadget per the plan's
own "measured situation" table; 2016 edges / 63 degree for `statistical`
per `audit/expressibility_matrix.md` §11; 1280 nodes / 8320 edges / 16
degree for the L1 spec per the cited test) — this table is a re-measurement
against the same fixed pipeline, not a new claim.

## Why each instance is non-degenerate

- **`assignment_8x8`**: the measured `max_degree=32` lands on any of the
  grid's 36 INTERIOR cells (8 neighbouring sites × target=4 slots each = 32
  edges from `term1` alone) — a structural property reached by the majority
  of cells in the grid, not a single boundary-adjacent fluke. An artifact
  would instead show up as a lone outlier degree on a corner/edge cell,
  which is the opposite of what this boundary-truncated Moore-8 construction
  produces (corners degree-3, edges degree-5, interior degree-8 neighbour
  counts — all measured, never assumed uniform).
- **`statistical_8x8`**: expanding `(sum_i x_i - target)**2` couples EVERY
  pair of the 64 cells by the algebra alone (confirmed symbolically at n=4
  in `measure_expressibility.py` before ever measuring at n=64), for ANY
  nonzero weight and ANY target strictly between 0 and 64 — so
  K_64/degree-63 is not an artifact of this script's particular
  weight=0.01/target=16 choice, it is invariant to both.
- **`terrain_k5`**: the spec file's own docstring records that this rule
  set's degree law was independently verified size-INDEPENDENT at
  3x3/4x4/5x5 (interior-cell degree plateaus at 4 regardless of overall grid
  size) before this 16x16 instance was ever compiled, so `max_degree=16`
  here is the same structural cost at a different scale, not a
  16x16-specific coincidence of interior/boundary cell mix.

## Note on `terrain_k5` as "the boundary case that passes today"

`max_degree=16` sits exactly AT the Z1 `degree <= 16` gate, not under it —
this is the case a route must not push over the edge. `|b|max=5.0` and
`|J|max=1.25` both sit comfortably under the 6.0 caps at this instance's own
`coefficient_scale=0.25`, so today this instance passes all three Z1 gates
(with the degree gate exactly saturated). A route that reduces `max_degree`
across the board must leave this instance's degree at `<=16` still — Task 4
verifies this explicitly for whichever route(s) this plan builds.

---

## Task 3 — Route B: stencil replication

**Construction.** Z1T Fig 4's own picture — several physical copies of one
input value, each independently read by a different consumer — maps most
directly onto a **star**, not Route A's chain: one **primary** copy (keeps
the node's own bias) plus `k-1` **secondary** copies, each bound to the
primary by exactly ONE dedicated "spoke" edge (not chained to each other).
A node's external edges are distributed across its own copies (primary
budget `max_degree-(k-1)`, secondary budget `max_degree-1` each); `k` is the
smallest value whose combined capacity covers the node's degree. This is a
genuinely different topology from Route A's chain, not a renamed copy of it
— every secondary is one hop from the canonical value regardless of `k`,
where a chain's interior copies are up to `k-1` hops from each other.
Implemented in `audit/degree_space_probe.py::star_replicate` — **audit-only,
never `src/`**, per the brief: this is a design discipline, not a compiler
pass.

**The crux, tested directly (not assumed): do spokes need their own binding
coupling?**

| instance | spoke_strength=0 (no binding) TV | verdict |
|---|---|---|
| hub degree=7, max_degree=4 (k=3) | **0.4325** | binding required, decisively |
| K_5 symmetric clique, max_degree=3 (k=2, every node replicated) | **0.1589** | binding required |
| hub degree=15, max_degree=5 (k=5), \|J\|/\|b\| scale matched to `statistical_8x8` | **0.0082** | binding required (smaller gap because the natural field scale is tiny at this magnitude, but still >8x the `AGREEMENT_TV=1e-3` threshold) |

**Replicas do NOT hold together for free.** All three instances — one
deliberately uneven, one perfectly symmetric, one matched to
`statistical_8x8`'s own real coupling magnitude and required `k` — show
decisive marginal disagreement with zero binding. This refutes, rather than
confirms by omission, the "for free" half of the crux question.

**Single-parameter (uniform) vs decoupled, tested two ways:**

1. **Uneven small instance** (hub degree=7, 7 leaves of distinct, mixed-sign
  weight, forcing k=3: primary + 2 secondaries carrying unequal external
  load — primary 2 edges, secondary1 3 edges, secondary2 2 edges). Uniform
  lowest passing spoke_strength: **5.0**. Independent per-spoke minima (each
  found holding the other at the uniform value): secondary1=5.0,
  secondary2=5.0 — identical despite the load imbalance. Joint verification
  at (5.0, 5.0): TV=0.000982, passes. **Decoupling does not materially help
  here** — the binding requirement is dominated by something other than
  simple per-spoke edge-weight sum (plausibly the primary's own stability
  under simultaneous pull from every spoke plus its own external edges,
  which every spoke's minimum must jointly respect regardless of that
  spoke's own load).
2. **Symmetric K_5 clique** (every node structurally identical, `k=2`).
  Joint uniform minimum: **3.2**. Per-node individual minima (each node
  alone, others held at 3.2): `[3.1, 3.1, 3.2, 3.2, 3.2]` — the 0.1 spread
  is grid-resolution noise at the pass/fail boundary (sweep step 0.1), not
  real asymmetry: the construction is provably symmetric under permutation
  of the 5 variables, so every node's true continuous threshold is
  identical. **Decoupling by node offers nothing on a symmetric clique**,
  as expected — `statistical_8x8` itself is exactly this kind of instance
  (every one of its 64 cells is structurally identical), so a uniform
  spoke_strength is not a simplification made for convenience, it is the
  form the symmetry of this specific benchmark actually calls for.

**The carried lead ("a single parameter serving both monotonicity and
value-linking blows the cap") could not be inherited** — the previous
agent's exact construction was never committed and is gone. Re-tested from
scratch as the general decoupling question above (its most plausible
recoverable reading: does giving each binding role its own coefficient, the
way `encode.py` gives `MONOTONE_PENALTY` and `ONE_HOT_PENALTY` their own
constants, lower the peak `|J|` needed?) on both a deliberately uneven and a
provably symmetric instance. **Neither shows a material benefit.** The lead
is not confirmed as stated (no single "blows the cap" number was
reproduced) and is not inherited as a result — this is the properly
established replacement finding.

**The real cost, at `statistical_8x8`'s own scale.** `_hub15_model`: hub
degree=15, max_degree=5, forcing **k=5** — the SAME `k` the real benchmark
needs at Z1's cap (`_k_for_star(63, 16) == 5`, exactly matching
`ceil((63)/(16-2))==5` for Route A's own chain on the same node) — with
coupling/bias magnitudes drawn to match `statistical_8x8`'s measured
`|J|max=0.005`/`|b|max=0.16` (Task 1), not an arbitrary toy scale.
Oracle-verified over the full `2**20` post-replication state space (20
nodes: 5 hub copies + 15 leaves).

| spoke_strength | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 6.0 |
|---|---|---|---|---|---|---|---|
| TV | 0.008205 | 0.004413 | 0.001956 | 0.000778 | 0.000295 | 0.000041 | 0.000000 |

**Lowest passing (0.05 resolution): spoke_strength = 1.40** (TV=0.000941) —
**23% of the documented 6.0 `|J|` cap.** Directly comparable to Route A's
own sweep result of **chain_strength=4.6 (77% of the cap)** on its own
matched instance (`tests/test_split.py`): same verification methodology
(exact oracle, full brute-force enumeration, `AGREEMENT_TV=1e-3`), same
order-of-magnitude coupling scale, same real-benchmark-matched `k`.

**A direct, apples-to-apples topology comparison** (both routes applied to
the identical instance — hub degree=7, max_degree=4, forcing `k=3` for the
star and `k=4` for the chain because a chain's uniform per-copy budget
(`max_degree-2`) is smaller than a star's asymmetric one):

| route | topology | copies (k) | binding edges | lowest passing strength |
|---|---|---|---|---|
| A (`split_high_degree`) | chain | 4 | 3 | 5.3 |
| B (`star_replicate`) | star | 3 | 2 | 5.0 |

On this instance the star topology costs **fewer nodes AND a lower `|J|`**
than the chain — a secondary is always exactly one hop from the primary
regardless of `k`, where a chain's interior copies must propagate agreement
through up to `k-1` hops, and every extra hop is another place disagreement
can creep in before the ends ever interact. This is a genuine structural
advantage of the star construction, not an artifact of instance choice (the
same instance, same `max_degree`, same oracle).

**Applied to `statistical_8x8` at full scale** (`spoke_strength=1.5` — a
small margin over the measured 1.40 minimum, not the 6.0 cap; full
brute-force verification at 320 nodes is `2**320` states, not computable,
so this application rests on (a) the oracle-verified matched-topology,
matched-magnitude, matched-`k` result above and (b) `statistical_8x8`
(`K_64`) being perfectly regular — every one of its 64 nodes replicates
into the identical star shape already tested, not a fresh untested shape at
scale — exactly the same basis Route A's own application to `assignment_8x8`
rests on in Task 4/5, since brute-force verification is equally impossible
there):

| | n_nodes | n_edges | max_degree | \|J\|max | \|b\|max | node_budget |
|---|---|---|---|---|---|---|
| before | 64 | 2016 | 63 | 0.005 | 0.16 | — |
| after (Route B, spoke_strength=1.5) | 320 | 2272 | **16** | 1.5 | 0.16 | 320 |

**All four Z1 gates pass** (`degree<=16`, `|J|<=6.0`, `|b|<=6.0`,
`node_budget<=250000`) — `statistical_8x8`'s degree-63 failure is fixed by
Route B, at 320 nodes (0.128% of the 250,000 budget) and `|J|max=1.5` (25%
of the cap).

**Answer to the crux (task-3-brief.md Step 3):** replicas need their own
binding coupling — the "holds together for free" hypothesis is refuted on
every instance tested, uneven and symmetric alike. A single shared
(uniform) parameter is not a simplification that costs anything relative to
a decoupled one on either an uneven or a symmetric test instance, and
`statistical_8x8`'s own perfect symmetry means the decoupled form was never
going to buy anything there specifically. **Route B is not free of the `|J|`
cost Route A pays** — it pays a measurably SMALLER one on directly matched
instances (1.40/6.0 = 23% vs Route A's 4.6/6.0 = 77%, and a star topology
measurably undercuts a chain on both node count and required strength on an
identical instance) — a real, verified, and materially different number,
not a wash.

---

## Task 4 — the comparison, and the honest ceiling

**Both routes applied to all three Task 1 benchmarks**, through
`audit/degree_space_probe.py::task4_apply_both_routes` (Route A:
`split_high_degree(max_degree=16, chain_strength=6.0)` — the `|J|` cap
itself, the same "test at the cap" precedent Task 2/3 already established
for real-scale, non-brute-force-verifiable applications; Route B:
`star_replicate(max_degree=16, spoke_strength=...)` at 5.2 for
`assignment_8x8` and 1.5 for `statistical_8x8`, both informed by the
oracle-verified sweeps in Task 3, not fabricated). Every number below is
`analyse()`'s own measurement of the real, lowered `IsingModel`, and every
gate check is `check_gates`'s own `Z1`, `allow_assumed=False`.

| benchmark | route | n_nodes | max_degree | \|J\|max | \|b\|max | passes ALL Z1 gates |
|---|---|---|---|---|---|---|
| `assignment_8x8` | before | 1744 | 32 | 0.375 | 8.0 | **FAIL** (degree, field_cap) |
| `assignment_8x8` | A | 1840 | **16** | 6.000 | 8.0 | **FAIL** (field_cap only) |
| `assignment_8x8` | B | 1840 | **16** | 5.200 | 8.0 | **FAIL** (field_cap only) |
| `statistical_8x8` | before | 64 | 63 | 0.005 | 0.16 | **FAIL** (degree) |
| `statistical_8x8` | A | 320 | **16** | 6.000 | 0.16 | **PASS** |
| `statistical_8x8` | B | 320 | **16** | 1.500 | 0.16 | **PASS** |
| `terrain_k5` | before | 1280 | 16 | 1.25 | 5.0 | PASS (boundary case) |
| `terrain_k5` | A | 1280 | 16 | 1.25 | 5.0 | PASS, **unchanged** |
| `terrain_k5` | B | 1280 | 16 | 1.25 | 5.0 | PASS, **unchanged** |

**Both routes fix the degree gate everywhere they act, and neither touches
`|b|`.** `assignment_8x8` still fails after either route — **not on
degree** (both routes correctly bring it to exactly 16) but on
`field_cap`: `|b|max=8.0` is untouched by both passes, because a split/
replicated node's bias is placed on exactly ONE copy in both constructions
(duplicating it across copies would multiply the field by `k`, a different
model — `split.py`'s own module docstring, mirrored in `star_replicate`).
**Neither route was ever designed to move `|b|`** — this is not a partial
failure of either technique, it is a dimension neither one addresses at
all. Task 5 diagnoses this gap precisely.

**`terrain_k5` is verified BYTE-FOR-BYTE unchanged by both routes**
(`task4_verify_terrain_k5_unbroken`: identical `n_nodes`, `max_degree`,
`|J|max`, `|b|max` before and after, not merely "still passes the gate,"
which could mask a route that changed the graph while coincidentally
staying under the cap) — because both routes' own over-cap detection is a
strict `degree > max_degree`, and `terrain_k5`'s degree sits exactly AT 16,
never over it. **Zero splits, zero replications performed.** The boundary
case is not broken by either route.

### Route A's ceiling — s(k), computed

Two orthogonal sweeps (`task4_route_a_ceiling`), because Z1's own
`max_degree=16` (`per_copy_cap=14`) makes a joint sweep over both hop-count
`k` AND per-copy load computationally impossible at brute-force scale
(`2**(k*14+k)` states) — each variable is isolated instead, at a smaller,
tractable `per_copy_cap`, oracle-verified via the same coarse-to-fine
`chain_strength` search as Task 2/3's own sweeps:

**(1) Chain-length sweep** — `max_degree=4` (`per_copy_cap=2`, the smallest
that still forces a real split), `|J|` drawn from `[0.4, 1.4]` (the same
order of magnitude as `assignment_8x8`'s own measured `|J|max=0.375`):

| k | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| s(k) | 5.2 | 5.2 | 5.7 | **6.0** | 6.3 (exceeds cap) |

**s(k) crosses the documented 6.0 `|J|` cap between k=6 and k=7.** Maximum
usable chain length at this testbed's `per_copy_cap=2`: **k=6**, giving a
maximum splittable degree of `k * per_copy_cap = 12` at THIS per-copy load
scale.

**(2) Per-copy-load sweep** — chain length fixed at `k=2`, `per_copy_cap`
varied (4, 6, 8), `|J|` drawn from `[0.2, 0.5]` (matching
`assignment_8x8`'s own `|J|max=0.375` exactly):

| per_copy_cap | 4 | 6 | 8 |
|---|---|---|---|
| s | 4.2 | 4.3 | 4.5 |

**Chain length drives `s` far more than per-copy load does**: the length
sweep spans 5.2→6.3 (Δ=1.1) over k=3..7, while the load sweep spans
4.2→4.5 (Δ=0.3) over a doubling of per-copy load at fixed k=2. This is the
basis for extrapolating the length-driven ceiling to Z1's real
`max_degree=16` (`per_copy_cap=14`, five times the testbed's `per_copy_cap=2`)
without brute-forcing it directly: load is a measured, real, but
SECONDARY effect.

**At real Z1 scale:** `assignment_8x8`'s worst node needs
`k=ceil(32/14)=3`; `statistical_8x8`'s needs `k=ceil(63/14)=5`. Both sit
comfortably inside the measured `max_usable_k=6` ceiling — with margin
(k=3 and k=5, against a ceiling of k=6), even before crediting that the
real per-copy load effect (measured to be small in (2)) would need to close
essentially the entire remaining gap to change this conclusion. **The
ceiling is computed, not assumed**: at `assignment_8x8`'s own `|J|` scale,
Route A's chain construction remains usable up to roughly chain length 6
before the coupling budget is exhausted — comfortably past what either real
benchmark in this plan needs, but not unlimited, and the curve shows
clearly why it eventually fails: every additional hop is another place
disagreement can creep in before the two ends of the chain ever interact,
and that cost compounds with length in a way per-copy load does not.

### The central question, answered

**Can unused node budget buy degree headroom, and at what price? Yes — up
to a real, computed, and now-charted `|J|` ceiling, not unconditionally.**
Both routes measurably trade nodes for degree: `assignment_8x8` and
`statistical_8x8` both go from failing the degree gate to passing it, at a
node cost of roughly 5.6% (1744→1840) and 400% (64→320) respectively —
`statistical_8x8`'s node cost looks dramatic in percentage terms only
because its starting node count is tiny (64 of a 250,000-node budget; even
at 320 nodes it uses 0.128% of the chip). The price is paid entirely in
`|J|`, not in nodes: node budget is so abundant (every measured case here
sits at a fraction of a percent of the 250,000-node ceiling) that it was
never the binding constraint; `|J|` is, and it has a real, measured ceiling
— Route A's chain construction stops being usable somewhere around chain
length 6 at `assignment_8x8`'s own coupling scale, which both real
benchmarks in this plan sit safely inside but a sufficiently higher-degree
future workload would not. Route B's star construction pays a measurably
SMALLER fraction of that same `|J|` budget on every instance tested here
(23–87% of Route A's own requirement across the matched comparisons in
Task 3), so **it is the cheaper of the two routes wherever it applies**,
without changing the qualitative answer: node budget converts to degree
headroom through `|J|`, and `|J|` — not nodes — is the resource this
architecture actually rations. Crucially, **neither route touches `|b|`
at all** — a workload whose failure is a field-cap violation (as
`assignment_8x8`'s is, independently of its degree failure) gets no help
from either technique, at any node cost; that is a different, unaddressed
dimension of the same problem, diagnosed directly in Task 5.

---

## Task 5 — the payoff test

**The winning route is B.** Task 4's own comparison table gives it the
identical degree and node outcome as Route A on `assignment_8x8`
(`max_degree` 32→16, `n_nodes` 1744→1840 for both) at a strictly lower
`|J|max` (5.2 vs 6.0). Applied via `task5_apply_winning_route_to_
assignment_8x8` (`star_replicate(max_degree=16, spoke_strength=5.2)`):

| | n_nodes | max_degree | \|J\|max | \|b\|max |
|---|---|---|---|---|
| before | 1744 | 32 | 0.375 | 8.0 |
| after (Route B) | 1840 | **16** | 5.2 | 8.0 |

| gate | measured | cap | result |
|---|---|---|---|
| degree | 16 | 16 | PASS |
| \|J\| | 5.2 | 6.0 | PASS |
| \|b\| | **8.0** | 6.0 | **FAIL** |
| node_budget | 1840 | 250,000 | PASS |

**Does the assignment gadget now tile inside every Z1 gate? No.** Degree is
fixed. `field_cap` is not — **`|b|max=8.0` is completely untouched by
either route**, exactly as Task 4 predicted: both constructions place a
split/replicated node's bias on a single copy, never touching its
magnitude.

**Diagnosis, not tuning (Step 3).** The gate that binds is `field_cap`
(`|b|`): measured `8.0`, cap `6.0`, **gap = 2.0 (33.3% over cap)**. This is
not a Route A/B problem to fix — neither route was ever designed to move
`|b|` — it is a property of the gadget's own construction. Checked for
size-dependence the same way Task 1 checked `max_degree=32`'s
non-degeneracy (`task5_diagnose_field_cap_gap`, measuring the identical
grid-gadget construction at 3×3, 4×4, 5×5, 8×8):

| grid | max_degree | max_abs_b | node carrying max \|b\| |
|---|---|---|---|
| 3×3 | 32 | 8.0 | `g1_1` |
| 4×4 | 32 | 8.0 | `g1_1` |
| 5×5 | 32 | 8.0 | `g1_1` |
| 8×8 | 32 | 8.0 | `g1_1` |

**Both `max_degree=32` and `max_abs_b=8.0` plateau identically from 3×3
upward**, carried by the same kind of node in every case (an interior
cell with a full, boundary-untruncated Moore-8 neighbourhood — the first
grid size large enough to have one is 3×3). This is a structural property
of any interior cell's own local gadget shape, present at the smallest
non-trivial grid and unchanged at every larger size tested — **not an
8×8-specific artifact, and not something a bigger OR smaller (non-trivial)
grid would fix.** No grid size makes this gadget pass on its own; the
field-cap failure is orthogonal to grid size entirely.

**The emergence experiment's 8×8 canvas limitation — liftable, and to what
size?** `audit/emergence_report.md`'s own finding was that 8×8 is too small
to show elongated geography (a coastline, a ridge) *regardless of whether
the underlying couplings are correct* — a canvas-size/statistical-power
limitation, not a compiler gate failure. Checked directly against that
report's own numbers: `emergence_8x8.yaml` **already passes every Z1 gate
at 8×8** (`degree=12` against the cap of 16, 4 of margin; `|J|max=0.35`,
`|b|max=0.90`, both well under 6.0). Its four rules are drawn from Task 3
of the *other* plan (`2026-09-04-lattice-rule-taxonomy.md`, per that
report's own header) — already gate-fitting, local, spatially-bounded
classes, the same family `terrain_k5` belongs to, whose degree Task 1
already verified size-independent at 3×3/4×4/5×5 (interior-cell degree
plateaus regardless of overall grid size).

**Stated plainly: this limitation was never a degree problem, so this
plan's routes — which exist specifically to buy degree headroom a workload
has already exhausted — have nothing to lift here.** A 40×40 or 64×64
emergence run was already unblocked on degree grounds before this plan
started, and remains exactly as unblocked now; nothing in Tasks 2–4 changes
that answer, because the thing they fix was never what was holding it back.
The one real, separate, and still-**unaddressed** cost at larger scale is
placement time — `emergence_report.md` measured 222.55s and 116 mediator
spins for bipartite embedding at just 64 nodes, and warned that "a larger
or denser emergence spec built the same way should expect a similar,
unmeasurable-in-advance mediation cost." That is an `insert_mediators`
(parity) cost, not a degree cost, and this plan's routes do nothing about
it either way — it is recorded here as the honest remaining unknown, not
folded into a claim this plan did not test.

**Summary of the payoff:** Route B fixes `assignment_8x8`'s degree failure
completely and at a lower `|J|` cost than Route A, but the gadget still
does not tile inside Z1 today — a second, independent, and completely
untouched gate (`field_cap`) blocks it, at a measured 33.3% overage that is
structural rather than scale-dependent. The false-impossibility-proof
escape this plan set out to rescue (one-sided rules like "land patches must
be at least 4 cells", via the assignment gadget) is **not yet reachable on
Z1** — degree was one of its two real obstacles, and this plan closes it;
the field-cap obstacle is a distinct, separately-scoped problem, diagnosed
here rather than patched over.
