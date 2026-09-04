# Expressibility Matrix

Plan: `2026-09-04-lattice-rule-taxonomy.md`, Task 3 (Task 4, the auxiliary-
variable question, extends this document separately). This is the plan's
**primary output** — knowledge, not a feature. It answers, per rule class,
"can the pairwise IR (`Linear`/`Product` only, per `src/tsu/ir.py`) express
this rule's own stated meaning, and at what cost?"

**Method.** For each of the twelve `tsu.rules.RULE_CLASSES`, the smallest
concrete instance of that class's characteristic mathematical shape was
built, lowered through the real compiler pipeline (`tsu.spec.load_spec` /
`tsu.passes.encode.encode` / `tsu.passes.lower.lower`), and measured via
`tsu.passes.analyse.analyse`. Every number below was produced by
`audit/measure_expressibility.py` — re-run it to reproduce every figure in
this document; none was predicted and left unconfirmed. Where a prediction
was made ahead of measurement (as the brief specifically requires for
`statistical`), both the prediction and the measured confirmation are shown.

**Result up front, stated plainly:** of the twelve classes, **9 are EXACT
in the IR** (no change of meaning going through `Linear`/`Product`), **2 are
DISTORTED** (a real meaning change, named), and **1 is INEXPRESSIBLE**
(confirmed by the compiler's own `ThreeBodyError`, not merely argued).
**Zero are `UNRESOLVED`** — every class was settled by direct construction
and measurement. This is a better outcome than the plan's own self-review
flagged as a live risk ("Task 3 may find that most of the twelve classes are
DISTORTED or INEXPRESSIBLE") — but see the "IR-exact vs. hardware-exact"
distinction below: three of the nine EXACT classes fail the Z1 **hardware**
gates at the measured instance's scale, which is a separate axis from
IR-expressibility and must not be conflated with it.

**A distinction load-bearing for the whole table.** "EXACT" means the IR
form changes no *meaning*. It does **not** mean the resulting `IsingModel`
fits Z1's `degree <= 16` / `|J| <= 6.0` / `|b| <= 6.0` gates (`src/tsu/gates.py`,
`src/tsu/target.py`) at whatever scale a workload happens to use. Three rows
below are EXACT-but-gate-failing at the measured scale (`statistical`
outright; `ecological` marginally). Do not read "EXACT" as "will compile
clean at any size" — it means "no rewriting was needed to preserve meaning."

---

## Summary table

| # | Class | Verdict | n_nodes | n_edges | max_degree | bipartite | \|J\|max | \|b\|max | vs Z1 gates (16 / 6.0 / 6.0) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `pairwise` | **EXACT** | 2 | 1 | 1 | True | 0.25 | 0.25 | pass |
| 2 | `neighbourhood` | **EXACT** | 9 | 6 | 3 | False | 0.25 | 0.0 | pass |
| 3 | `morphology` | **DISTORTED** | 9 | 28 | 7 | False | 1.25 | 0.0 | pass |
| 4 | `cluster` | **EXACT** | 288 | 584 | 6 | False | 0.5 | 1.5 | pass |
| 5 | `gradient` | **EXACT** | 64 | 112 | 4 | True | 0.2 | 0.8 | pass |
| 6 | `topological` | **EXACT** | 5 | 3 | 2 | False | 4.0 | 3.0 | pass |
| 7 | `hydrological` | **EXACT** | 3 | 0 | 0 | True | 0.0 | 8.0\* | **fails \|b\| at this weight**\* |
| 8 | `climate` | **EXACT** | 4 | 4 | 2 | True | 2.5 | 3.25 | pass |
| 9 | `ecological` | **DISTORTED** | 18 | 121 | 15 | False | 3.25 | **7.0** | **FAILS field_cap** |
| 10 | `gameplay` | **EXACT** | 3 | 3 | 2 | False | 2.5 | 2.5 | pass |
| 11 | `statistical` | **EXACT (IR)** | 64 | 2016 | **63** | False | 0.005 | 0.16 | **FAILS degree gate** |
| 12 | `boundary` | **INEXPRESSIBLE** | — | — | — | — | — | — | n/a (rejected before lowering) |

\* `hydrological`'s `|b|max`=8.0 is measured at the demonstration weight
(8.0, chosen to match the pre-existing `tests/test_conserve_over_edges.py`
fixture, reused rather than re-invented) and genuinely **does** exceed the
6.0 field cap **at that specific weight** — this is not glossed over. But
unlike `statistical`/`ecological`, this is a **weight choice**, not a
structural cost the rule's *shape* forces: the term is a single-variable
squared form (`n_edges=0`, no coupling at all), so `|b|` scales linearly in
weight with no other structural driver, and any weight `<= 6.0` fits
comfortably (measured directly: weight=6.0 gives `|b|max=6.0` exactly).
Contrast `statistical`, whose degree blow-up cannot be fixed by choosing a
smaller weight at all — that is the real distinction this footnote exists
to draw.

All twelve rows are direct output of `audit/measure_expressibility.py`
(`python audit/measure_expressibility.py`, from repo root, pinned
interpreter). Nothing in this table was hand-adjusted after measurement.

---

## 1. `pairwise` — EXACT

**Instance:** 2 cells, 1 edge; `Product(VarRef(x0), VarRef(x1), -1.0)`
(encourage both to 1).

**Arithmetic:** `Product(a, b, weight)` **is** the IR's own native pairwise
shape — a rule that says "these two cells interact" needs no rewriting
whatsoever. Nothing to prove beyond "this term kind exists."

**Measured:** `n_nodes=2, n_edges=1, max_degree=1, bipartite=True, |J|max=0.25, |b|max=0.25`.

---

## 2. `neighbourhood` — EXACT

**Instance:** 3×3 grid, centre cell's 4-neighbour count `N`, target=2:
`Product(L, L, 0.5)` where `L = (Σ 4 neighbours) - 2`.

**Arithmetic:** the rule's own *intent* is "as close to target as
possible" — symmetric by definition. `(N-2)^2` is therefore not a
substitution for a different rule; it **is** the honest form of this one.
Measured table: `N=0..4 -> [4, 1, 0, 1, 4]` (symmetric, as the rule itself
wants).

**Measured:** `n_nodes=9, n_edges=6, max_degree=3, bipartite=False,
mediators=2, |J|max=0.25, |b|max=0.0`. The 4-neighbour clique (K4 among the
4 neighbour spins) forces 2 mediators — the honest non-bipartite cost of a
degree-4 clique, unrelated to any distortion.

**Contrast with `morphology` below** is the entire point of this row:
identical IR shape (`Product(L,L,w)`), opposite honesty verdict, because the
*rule's own stated intent* differs (symmetric vs one-sided).

---

## 3. `morphology` — DISTORTED

**Instance:** the plan's own worked example, `E(i) = λ·max(0, 4-N)` over the
full Moore-8 neighbourhood (3×3 grid, centre cell), target=4 — chosen at
this exact size specifically so the measured table reproduces the plan's
own N=0..8 table.

**Arithmetic (why no polynomial form exists):** `max(0, 4-N)` has a kink at
`N=4`. For `N >= 4` the function is identically 0; for `N < 4` it is
`4-N`, non-constant. No single polynomial `p` can be identically zero on
`{4,5,6,7,8}` (infinitely many roots would force `p ≡ 0`) while also
matching `4-N` on `{0,1,2,3}` — so no polynomial of any degree equals
`max(0,4-N)` on the integers 0..8. It is genuinely not expressible as a
`LinearForm` power. The **nearest pairwise form**, `(4-N)^2` (a squared
`LinearForm`, hence `Product(L,L,w)`), is measured directly:

| N | 0 | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| `(4-N)^2` | 16 | 9 | 4 | 1 | **0** | 1 | 4 | 9 | 16 |

**Distortion, named:** N=6 is penalised exactly as hard as N=2 — "at least
4" became "exactly 4," forbidding N=5,6,7,8 which the original rule
permitted for free. `src/tsu/passes/expressibility.py`'s
`_one_sided_threshold` already carries this exact reasoning (Task 2); this
row is the from-scratch re-derivation at the plan's own worked scale.

**Measured:** `n_nodes=9, n_edges=28, max_degree=7, bipartite=False,
mediators=12, |J|max=1.25, |b|max=0.0`. The 8-clique (Moore-8 neighbours all
mutually coupled) costs degree 7 per node (well under 16) but needs 12
mediator spins — non-bipartite, as any k>2-clique squared-form is.

---

## 4. `cluster` — EXACT (cost scales in node count, not degree)

**Instance:** "these designated cells must all belong to one connected
component" — a genuinely global, non-local predicate; the known escape is
one flow commodity (`conserve_over_edges`) per additional member, proving
reachability from a shared root. Measured on the full 8×8 grid with 1 vs 2
commodities (`g0_0 -> g7_7`, then also `g0_0 -> g7_0`).

**Ground-state correctness** for the underlying primitive (conservation +
capacity gate) is verified by brute force under `topological` (§6) —
`cluster` and `topological` share the identical mechanism; this row's job
is specifically the **cost of composing multiple members**, which is what
distinguishes "one pair" from "a whole cluster."

**Measured (freshly, under the current split `max_abs_coupling`/`max_abs_bias`):**

| commodities | n_nodes | n_edges | max_degree | \|J\|max | \|b\|max |
|---|---|---|---|---|---|
| 1 | 176 | 292 | 6 | 0.5 | 0.5 |
| 2 | 288 | 584 | 6 | 0.5 | 1.5 |

**The structural finding:** each commodity's flow spins are disjoint from
every other commodity's (separate `flow_prefix`), so adding a cluster
member grows **node count** linearly (176 → 288) but leaves **max_degree
unchanged** (6 → 6). Cluster size is therefore not a degree-gate risk on
this substrate — it is a node-budget risk (Z1's `node_budget` gate,
250,000, per `src/tsu/target.py`), which a modest cluster is nowhere near.

**Re: the brief's "collided with the penalty-dominance guard, capped at
weight ≤0.2 vs. a 4.0 adjacency penalty."** This specific collision was
**not reproduced** by the constructions measured here or in §6 — see §6's
own note for the numbers (conservation weight can reach 12.0, not 0.2,
before hitting the 6.0 cap, for a single commodity on this grid). Whether
the originally-reported collision depended on a construction this task did
not reproduce (e.g. a denser competing term, or measurement taken before
`max_abs_coupling`/`max_abs_bias` were split into separate `Sourced`
fields — see `src/tsu/target.py`'s own docstring on that split) is
**UNRESOLVED as a *historical reconciliation*** — what is NOT unresolved is
the class's own current expressibility and cost, which are measured above
and are unambiguously favourable.

---

## 5. `gradient` — EXACT, cheap (confirmed, not re-derived)

**Instance:** the committed production spec, `specs/lattice_elev_band_8x8.yaml`
(binary overlay, `product_over_edges` weight -0.4), loaded and lowered
verbatim — not reconstructed.

**Measured:** `n_nodes=64, n_edges=112, max_degree=4, bipartite=True,
mediators=0, |J|max=0.2, |b|max=0.8`.

This is an **exact match** to the spec file's own 2026-09-01 documented
measurement (64 spins, degree 4, `|J|` 0.20, `|b|` 0.80, bipartite). The
brief's prediction — "should be cheap, because it is the layer stack
already built" — is **confirmed**, not merely repeated: bipartite means 0
mediators, and at k=2 (binary) the representation penalty
`P*(k-2)/2 = 0`, so an overlay costs nothing on the encoding-penalty axis
that binds richer (k>2) layers.

---

## 6. `topological` — EXACT, cheap under current caps

**Instance (smallest, exact-enumeration-checkable):** a 3-cell corridor.
Flow conservation (`conserve_over_edges`, source `g0_0`, sink `g2_0`,
weight 8.0) plus a capacity `Product` gating each flow edge through the
middle cell's own occupancy (`f * (1 - open)`, weight 6.0 each) — reused,
not modified, from `tests/test_conserve_over_edges.py`'s own corridor
fixture.

**Verified by brute force over the full 2^5=32-state space:** ground state
is `{g0_0:0, g1_0:1, g2_0:0, f__g0_0__g1_0:1, f__g1_0__g2_0:1}`, `E=0.0` —
the corridor cell is forced OPEN, matching "a path exists" exactly, even
though "closed" is locally free and "open" costs nothing extra to prefer
here (the full competing-incentive version of this claim is in
`tests/test_conserve_over_edges.py::test_corridor_ground_state_keeps_the_path_open_under_a_competing_incentive`,
independently re-confirmed by this task, not just cited).

**Measured:** `n_nodes=5, n_edges=3, max_degree=2, bipartite=False,
mediators=1, |J|max=4.0, |b|max=3.0`.

**Cost re-measurement on the full 8×8 grid (single s-t pair, conservation
weight alone), under the CURRENT (post `max_abs_coupling`/`max_abs_bias`
split) caps** — this is the brief's specifically-requested re-measurement:

| weight | \|J\|max | \|b\|max | max_degree |
|---|---|---|---|
| 1.0 | 0.5 | 0.5 | 6 |
| 6.0 | 3.0 | 3.0 | 6 |
| 12.0 | 6.0 | 6.0 | 6 |
| 12.5 | 6.25 | 6.25 | 6 |

`max_degree` is flat at 6 (well under the 16 cap) across the entire weight
sweep — weight scales `|J|`/`|b|` **linearly** and only reaches the 6.0 cap
at weight=12.0. **This is a materially different result from the brief's
own recorded prior finding** ("capped at weight ≤0.2, which cannot outrank
a 4.0 adjacency penalty"): under the current caps, a single reachability
pair on the full 8×8 grid is cheap, with roughly 24× more weight headroom
than the prior figure implies. Reported honestly as a fresh, favourable
measurement — see §4's closing note on why the discrepancy with the prior
figure is not chased further here.

---

## 7. `hydrological` — EXACT, cheap (pure conservation, no inequality)

**Instance:** 2-cell edge, source `g0_0` (1 unit), sink `g1_0` (1 unit),
weight 8.0 — the smallest possible conservation instance.

**Verified over the full 2^3=8-state space:** `E=0.0` exactly when the flow
variable satisfies conservation (`f__g0_0__g1_0=1`), `E=16.0` otherwise
(`8*(1)^2` unmet at each of the two endpoints), for **every** combination of
`g0_0`/`g1_0` — the term genuinely doesn't reference them, exactly as
declared.

**Why this is the easy case, and `topological`/`cluster` are not:**
`inflow - outflow - net == 0` is an **equality**, and squaring a linear
form keeps an equality pairwise (`Product(L,L,w)`) with **zero** change of
meaning — there is no kink to distort, unlike the inequality/existence
predicate (`does a path exist`) that `topological`/`cluster` build on top of
the *same* flow-variable substrate. The one-sidedness only enters once you
ask an *existence* question of the flow, not when you merely conserve it.

**Measured:** `n_nodes=3, n_edges=0, max_degree=0, bipartite=True,
mediators=0, |J|max=0.0, |b|max=8.0`. Zero edges: with only one flow
variable, the conservation term's `L` has a single coefficient, so its
squared form produces no coupling at all — a pure bias term.

---

## 8. `climate` — EXACT (categorical variant of the gradient-smoothness shape)

**Instance:** a smooth categorical gradient — "cold(0) must not sit next to
warm(2)" — k=3 domain-wall categorical, 2 cells, one edge:
`product_over_edges(a_value=0, b_value=2, weight=3.0)`.

**Verified over all 9 (v0,v1) combinations:** `E=3.0` iff `{v0,v1}={0,2}`,
`E=0.0` for all 7 other combinations (cold-cold, cold-mild, mild-anything,
warm-warm) — exact match, no approximation.

**Measured:** `n_nodes=4, n_edges=4, max_degree=2, bipartite=True,
mediators=0, |J|max=2.5, |b|max=3.25`.

**Why this differs meaningfully from `gradient` (§5), not just relabels
it:** `gradient` is the binary-overlay case (k=2, no chain, no
representation penalty). `climate` exercises the **categorical
domain-wall encoding's own structural cost** — `MONOTONE_PENALTY`-scaled
terms enforce chain monotonicity, contributing the measured
`|J|max=2.5`/`|b|max=3.25` even though the workload's own declared term
weight is only 3.0. Bipartite still holds (domain-wall never mediates), so
0 mediators — a genuinely different, informative cost profile from binary
overlays despite being the same "value-pair-over-an-edge" IR shape.

---

## 9. `ecological` — DISTORTED (same shape as morphology; markedly worse cost)

**Instance:** ">=3 forest(value=1) neighbours" over the Moore-8
neighbourhood, k=3 categorical (0=water/1=forest/2=rock) domain-wall
neighbours, target=3.

**Arithmetic:** mathematically identical to `morphology` — `max(0, 3-N)`
has the same kink, same non-polynomial argument, same nearest pairwise form
`(3-N)^2`:

| N | 0 | 1 | 2 | **3** | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| `(3-N)^2` | 9 | 4 | 1 | **0** | 1 | 4 | 9 | 16 | 25 |

**Distortion, named:** "at least 3 forest neighbours" becomes "exactly 3" —
N=5 (surplus of 2) penalised as hard as N=1 (deficit of 2).

**Measured:** `n_nodes=18, n_edges=121, max_degree=15, bipartite=False,
mediators=56, |J|max=3.25, |b|max=7.0`.

**Why this row earns its own place rather than being folded into
`morphology`:** the categorical (domain-wall) encoding's own structural
cost **compounds** with the one-sided distortion's clique cost.
`max_degree=15` sits 1 below the Z1 cap (16) — the closest of any measured
row. More importantly, **`|b|max=7.0` at this weight (1.5) already exceeds
the Z1 `field_cap` of 6.0** — this is a real, freshly-measured gate
failure at a weight that is not unusually large, not a contrived edge case.
`morphology`'s equivalent binary construction never got close to either
cap. This is the clearest concrete evidence in this matrix that
one-sided-threshold rules over **categorical** (not binary) neighbourhoods
are the more urgent cost risk of the two DISTORTED classes.

---

## 10. `gameplay` — EXACT

**Instance:** exactly one of 3 candidate binary "spawn" cells chosen:
`Product(L, L, 5.0)`, `L = (spawn0+spawn1+spawn2) - 1`.

**Verified over the full 2^3=8-state space:** `E == 5*(count-1)^2` exactly,
for every one of the 8 combinations (E=5 at count=0, E=0 at count=1, E=5 at
count=2, E=20 at count=3).

**Arithmetic:** "exactly one" **is** symmetric by definition (unlike
morphology's "at least") — `(sum-1)^2` therefore changes nothing; this is
the same honesty argument as `neighbourhood` (§2), applied to the
compiler's own one-hot exactly-one penalty shape.

**Measured:** `n_nodes=3, n_edges=3, max_degree=2, bipartite=False,
mediators=1, |J|max=2.5, |b|max=2.5`. 3+ mutually-exclusive candidates form
a clique (K3 here) — non-bipartite, 1 mediator, the identical structural
cost `src/tsu/passes/encode.py`'s own one-hot encoder already documents and
pays for a k=3 categorical.

---

## 11. `statistical` — EXACT (IR), FAILS the Z1 degree gate at n=64

**Prediction (written down before measuring, per the brief's own
instruction):** `(Σ_i x_i - target)^2` is pairwise, but because it sums
over **every** cell, expanding it couples every cell to every other cell.
Predicted: on the full 8×8 grid (n=64), this produces a complete graph
`K_64` — `n_edges = C(64,2) = 2016`, `max_degree = 63` — which blows the
Z1 `degree <= 16` gate by 47.

**Arithmetic (sympy, confirming the shape before measuring at scale):**
for binary occupancy (`x_i^2 = x_i` — idempotent), expanding
`(Σx_i - t)^2` gives a linear part plus `2·Σ_{i<j} x_i·x_j` over **every**
pair `i<j` — by definition the edge set of `K_n`. Checked symbolically at
n=4:

```
(x0+x1+x2+x3-t)^2 = t**2 - 2*t*x0 - 2*t*x1 - 2*t*x2 - 2*t*x3
                     + x0**2 + 2*x0*x1 + 2*x0*x2 + 2*x0*x3
                     + x1**2 + 2*x1*x2 + 2*x1*x3 + x2**2 + 2*x2*x3 + x3**2
```
— every cross term `x_i*x_j` present, confirming the complete-graph shape
generically (not just at n=64).

**Measured (target=16, i.e. 25% of 64 cells, weight=0.01):**
`n_nodes=64, n_edges=2016, max_degree=63, bipartite=False, mediators=-1
(too large for exact max-cut), |J|max=0.005, |b|max=0.16`.

**Prediction confirmed exactly: `n_edges=2016=C(64,2)` and `max_degree=63`.**

**The IR-exact / hardware-exact distinction, stated precisely for this
row:** `(Σx_i-target)^2` changes **no meaning** — "prefer a count near
target" *is* symmetric-in-count by definition, so this is honestly
IR-EXACT, not a distortion. But it is a complete graph by construction, for
**any** nonzero weight and **any** target strictly between 0 and 64 — this
is not a tunable-away property of this specific instance. It **fails the
Z1 degree gate** (63 ≫ 16) as a hardware-realizability matter, entirely
separate from the expressibility question this matrix otherwise measures.
No decomposition (windowed/partial sums, hierarchical aggregation
auxiliaries) that might relieve this was attempted here — **flagged as
open, not chased**, per "record failures instead of repairing them."

---

## 12. `boundary` — INEXPRESSIBLE (confirmed by the compiler's own `ThreeBodyError`)

**Instance attempted:** "at most K value-crossing edges" — an
isoperimetric-style global count. Crossing indicator over an edge (u,v):
`x_u + x_v - 2·x_u·x_v` (XOR) — **already degree 2** in cell occupancies,
i.e. not itself a `LinearForm`. `N_cross` sums this over a path's edges;
the rule squares a target deviation of `N_cross`.

**⚠ Documented near-miss trap (2 edges, 3 cells, target=1):** built and
lowered first, and it **succeeded** — `n_nodes=3, n_edges=1, max_degree=1,
bipartite=True`, collapsing to a single `Product` between the two endpoint
cells. This is **not** evidence of general expressibility: with only 2
edges, `N_cross ∈ {0,1,2}` and target=1 sits exactly on the parity
boundary, so `(1-N_cross)^2` happens to collapse to `XOR(x0,x2)` by
coincidence. This is precisely the plan's own warned-against trap ("the
near-miss is worse than the miss, because it looks like it worked") —
recorded here explicitly so it is never later mistaken for a general proof.

**Smallest GENERIC instance (3 edges, 4 cells, target=1):** built via a
hand-rolled `sympy_expr` term (the same duck-typing extension point
`tsu.passes.lower._accumulate_symbolic` already supports, used only to
probe — not to modify `src/tsu`). `lower()` — the compiler's **own**
pairwise-degree checker — raises:

```
ThreeBodyError: order-4 term s_x0 * s_x1 * s_x2 * s_x3 survived expansion
with coefficient -0.5; the model is not pairwise and cannot be placed on a
pairwise target
```

Independently confirmed via sympy: expanding `(1 - N_cross)^2` over 4 raw
occupancy symbols (pre-idempotent-reduction) has monomials up to total
degree 4, and — unlike `boundary`'s 2-edge degenerate case — this does
**not** fully cancel down to degree ≤2 once binary idempotency (`x_i^2=x_i`,
i.e. `s_i^2=1`) is correctly applied (the compiler's own reduction, which
this matrix trusts because Task 3's other 11 rows independently confirm it
against brute-force truth tables). A genuine 3-distinct-variable coupling
(e.g. `x0·x1²·x2 -> x0·x1·x2`) survives reduction, and at 4 cells a genuine
4-way coupling survives too.

**Reason:** `N_cross` is already a degree-2 polynomial in cell occupancies
— it cannot be the `L` half of `Product(L,L,w)` at all, because `L` must be
a `LinearForm`. Squaring a target deviation of an already-quadratic
quantity is a degree-4 operation in general; only a numerically degenerate
small instance avoids it. **INEXPRESSIBLE**, confirmed by the compiler's
own machinery, not merely argued by hand.

---

## Cross-cutting findings

1. **The three "most likely to bite" items, resolved:**
   - `statistical`: predicted `K_64` (degree 63); **measured exactly
     confirmed** (`n_edges=2016=C(64,2)`, `max_degree=63`). Blows the
     degree-16 gate by 47, for any nonzero weight/target — an IR-exact,
     hardware-inexpressible rule at this scale.
   - `cluster`/`topological`: re-measured under the current
     (`max_abs_coupling`/`max_abs_bias` split) caps. The specific prior
     collision ("weight capped at ≤0.2 vs. a 4.0 adjacency penalty") was
     **not reproduced** — single-pair reachability on the full 8×8 grid is
     cheap (weight headroom to 12.0, not 0.2); the real cost driver for
     multi-member clusters is **node count**, not degree (flat at 6
     regardless of commodity count, measured at 1 and 2 commodities).
   - `gradient`: confirmed cheap, matching the committed spec's own prior
     measurement exactly (64 spins, degree 4, bipartite, 0 mediators).

2. **Several of the twelve class *labels* collapse to a small number of
   underlying mathematical shapes** once domain vocabulary is stripped —
   `morphology` and `ecological` are the same one-sided-threshold shape
   (binary vs. categorical neighbourhoods); `cluster` and `topological`
   share one flow-conservation-plus-capacity-gate mechanism at different
   scale; `gradient` and `climate` share one value-pair-over-an-edge
   mechanism (binary vs. categorical). This is not a flaw in the taxonomy —
   `src/tsu/rules.py`'s own docstring is explicit that class labels are
   data, for human classification, and the pairwise substrate "does not
   care what a rule is called, only what its measurement computes"
   (`expressibility.py`'s docstring). It is, however, a genuine finding:
   the twelve-class taxonomy has roughly 6-7 distinct *cost profiles*
   underneath it, and Task 5 (generic term templates) should be scoped to
   those shapes, not to twelve separate implementations.

3. **`INEXPRESSIBLE` classes are rejected by the compiler's own machinery,
   not merely argued.** `boundary`'s `ThreeBodyError` came from
   `tsu.passes.lower.lower` itself — the same guard that protects every
   production spec. This matrix's one negative-IR verdict is exercised by
   the actual guard rail, not a parallel argument that could drift from it.

4. **`UNRESOLVED` count: 0 of 12 classes.** The two DISTORTED classes
   (`morphology`, `ecological`) are the natural candidates for Task 4's
   auxiliary-variable question, appended to this document separately.

---

## Reproduction

```
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/measure_expressibility.py
```

The script is standalone (not a pytest file; `testpaths = ["tests"]` does
not collect it) and prints every number transcribed into this document.
