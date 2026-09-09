# Expressibility Matrix

Plan: `2026-09-04-lattice-rule-taxonomy.md`, Tasks 3-4. This is the plan's
**primary output** — knowledge, not a feature. It answers, per rule class,
"can the pairwise IR (`Linear`/`Product` only, per `src/tsu/ir.py`) express
this rule's own stated meaning, and at what cost?" — and, for the classes
that cannot, whether an auxiliary variable rescues them.

**Method.** For each of the twelve `tsu.rules.RULE_CLASSES`, the smallest
concrete instance of that class's characteristic mathematical shape was
built, lowered through the real compiler pipeline (`tsu.spec.load_spec` /
`tsu.passes.encode.encode` / `tsu.passes.lower.lower`), and measured via
`tsu.passes.analyse.analyse`. Every number below was produced by
`audit/measure_expressibility.py` (Task 3), `audit/aux_variable_probe.py`
(Task 4), `audit/assignment_gadget_probe.py` and
`audit/boundary_rosenberg_probe.py` (code-review corrections, see below) —
re-run them to reproduce every figure in this document; none was predicted
and left unconfirmed. Where a prediction was made ahead of measurement (as
the brief specifically requires for `statistical`), both the prediction and
the measured confirmation are shown.

**Result up front, stated plainly, corrected after code review
(2026-09-04-lattice-rule-taxonomy, findings C1/C2/C3):** of the twelve
classes, **10 are EXACT in the IR** (no change of meaning going through
`Linear`/`Product`, counting `boundary` via the Rosenberg-quadratized
auxiliary construction added below) and **2 are DISTORTED** (a real meaning
change, named). **Zero are INEXPRESSIBLE and zero are `UNRESOLVED`** — this
matrix originally reported `boundary` as INEXPRESSIBLE and a one-sided
auxiliary rescue as structurally impossible for `morphology`/`ecological`;
both claims were **wrong**, and are corrected in place below (§12 and the
Task 4 section) rather than quietly revised — the original reasoning and
its refutation are both kept so the mistake stays visible. This is a
better outcome than the plan's own self-review flagged as a live risk
("Task 3 may find that most of the twelve classes are DISTORTED or
INEXPRESSIBLE") — but see the "IR-exact vs. hardware-exact" distinction
below: some of the EXACT classes fail the Z1 **hardware** gates at the
measured instance's scale (a separate axis from IR-expressibility, and,
per the grid-scale measurement in the Task 4 section, a separate axis from
whether an auxiliary *construction* is affordable once tiled across a real
workload rather than measured in isolation).

**A distinction load-bearing for the whole table.** "EXACT" means the IR
form changes no *meaning*. It does **not** mean the resulting `IsingModel`
fits Z1's `degree <= 16` / `|J| <= 6.0` / `|b| <= 6.0` gates (`src/tsu/gates.py`,
`src/tsu/target.py`) at whatever scale a workload happens to use, and it
does **not** mean an auxiliary construction that is EXACT and gate-passing
in ISOLATION stays gate-passing once the same construction is applied at
every site of a real grid (see the Task 4 section's grid-scale
measurement: expressible and affordable-at-scale are different claims).
Two rows below are EXACT-but-gate-failing at the measured single-instance
scale (`statistical` outright, structurally; `hydrological` at its
demonstration weight only, not structurally — see that row's own
footnote). `ecological` also fails a gate at its measured scale, but its
own verdict is DISTORTED, not EXACT — an earlier version of this
paragraph miscounted it as a third EXACT-but-failing row (I5, code
review); it is a separate finding about a DISTORTED class, not another
instance of the same EXACT-but-failing pattern. Do not read "EXACT" as
"will compile clean at any size, in any composition" — it means "no
rewriting was needed to preserve meaning," which is necessary but not
sufficient for either claim.

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
| 12 | `boundary` | **EXACT (via aux)**\*\* | 7 | 21 | 6 | False | 4.5 | 4.5 | pass |

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

\*\* `boundary`'s row was corrected under code review (C3): the ORIGINAL
verdict, INEXPRESSIBLE, ruled out only a **direct, auxiliary-free**
construction, while this same matrix accepts auxiliary flow variables as
legitimate for rows #4, #6 and #7 — an inconsistent standard. §12 below
carries the corrected verdict and its arithmetic; the measured numbers
here (7 nodes = 4 cells + 3 Rosenberg auxiliaries, degree 6, both caps at
4.5) are from `audit/boundary_rosenberg_probe.py`, verified against the
matrix's own 4-cell/3-edge instance by brute force (all 16 states) and
through the real `lower()`/`analyse()` pipeline, not measure_expressibility.py.

Rows 1–11 are direct output of `audit/measure_expressibility.py`
(`python audit/measure_expressibility.py`, from repo root, pinned
interpreter); row 12 is `audit/boundary_rosenberg_probe.py`. Nothing in
this table was hand-adjusted after measurement.

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

**Capacity-gate caveat, attached here to `cluster`'s OWN verdict (I7, code
review) rather than left implicit behind the §6 cross-reference above:**
the brute-force check that establishes the capacity gate actually behaves
correctly (flow only counts through a cell that is itself "open," not
merely reachable) is performed ONLY at §6's 3-cell corridor scale. The
176/288-node measurements in the table below do **not themselves include a
capacity gate at all** — `measure_cluster()` builds pure
`conserve_over_edges` commodities with no capacity `Product` term, so what
is measured there is the cost of proving *reachability between a fixed
source and sink*, not of proving *reachability that stays within open
cells*, which is what "these designated cells belong to one connected
component" would need in a realistic composition. The capacity gate's own
correctness (§6) and cluster's own cost-at-scale (this section) are each
individually measured, but their **composition** — a capacity-gated flow
per member, at grid scale — is not, and should not be read as implied by
citing §6 alone.

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
weight ≤0.2 vs. a 4.0 adjacency penalty" (corrected under I6, code
review):** conservation **ALONE** reaches weight 12.0, not 0.2, before the
6.0 cap — that part stands, and is what the table above measures. But an
earlier draft of this section additionally claimed the brief's *contested*
finding (conservation actually competing against the 4.0 adjacency
penalty) was "not reproduced," without ever actually measuring that
contested setup. `audit/contested_conservation_probe.py` measures it
directly: the adjacency term alone already produces `|b|max=8.0` on this
grid, over the 6.0 cap before conservation enters at all, and sweeping
conservation weight from 0.1 to 12.0 leaves `|b|max` at 8.0 throughout —
every swept weight fails the field cap. **The contested setup was not
measured before; when measured, it fails, confirming rather than
contradicting the prior finding.** This is not a cluster-specific result
(the measurement above uses a single reachability pair, not a multi-member
cluster), but it corrects this section's own prior claim about it, and see
§6 for the identical correction and the fuller arithmetic.

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
`tests/test_conserve_over_edges.py::test_corridor_ground_state_keeps_the_path_open_under_a_competing_incentive`
-- cited as existing, pre-committed coverage; I9, code review: an earlier
draft of this row additionally claimed that test was "independently
re-confirmed by this task," which is unsubstantiated -- nothing in this
task's own scripts re-runs or re-derives that specific test's claim, so the
claim is dropped rather than left standing on nothing).

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
at weight=12.0. **Conservation ALONE is cheap on the full 8×8 grid** at
these weights.

**I6 correction (code review):** an earlier draft of this row compared that
solo measurement against the brief's own prior finding ("capped at
weight ≤0.2, which cannot outrank a 4.0 adjacency penalty") and called the
gap "roughly 24× more weight headroom" — but the prior finding describes a
**CONTESTED** setup (conservation competing against a separate adjacency
penalty for the same `|b|` budget), which this matrix had never actually
measured; comparing a solo number against a contested one is not a
comparison at all. `audit/contested_conservation_probe.py` measures the
contested setup directly: a `product_over_edges` adjacency term
(weight=4.0, `symmetric: true` doubling it to an effective 8.0 per edge)
on the same 8×8 grid already produces `|b|max=8.0` **by itself, with zero
conservation weight present** — already over the 6.0 field cap before
conservation enters at all. Sweeping conservation weight from 0.1 to 12.0
alongside it leaves `|b|max` at 8.0 throughout (conservation's own peak
bias, measured above, never exceeds 8.0 in this range), so the model fails
the field cap at **every** swept weight. **The prior finding is therefore
more likely correct, not less** — once the contested setup is actually
measured, it fails, exactly as originally reported. Reframed accordingly:
the contested setup was not measured in the original draft of this row;
when measured, it fails. §4's own closing note (`cluster`) draws the
matching distinction for the multi-commodity case.

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

## 12. `boundary` — EXACT via a Rosenberg-quadratized auxiliary (re-verdicted, C3)

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

**Reason the DIRECT, auxiliary-free construction fails:** `N_cross` is
already a degree-2 polynomial in cell occupancies — it cannot be the `L`
half of `Product(L,L,w)` at all, because `L` must be a `LinearForm`.
Squaring a target deviation of an already-quadratic quantity is a degree-4
operation in general; only a numerically degenerate small instance avoids
it. This much of the original finding **stands**, confirmed by the
compiler's own machinery, not merely argued by hand.

**C3 (code review) — the INEXPRESSIBLE verdict does not follow from this.**
The reasoning above rules out only a *direct, auxiliary-free* construction
— but this same matrix accepts auxiliary FLOW variables as legitimate,
EXACT escapes for rows #4 (`cluster`), #6 (`topological`) and #7
(`hydrological`), all of which also fail a direct, auxiliary-free
construction of their own predicate. Calling `boundary` INEXPRESSIBLE on
"no auxiliary-free form exists" while calling those three rows EXACT on
"no auxiliary-free form exists, but an auxiliary one does" is not the same
standard applied twice — it is two different standards, and `boundary` got
the less charitable one for no stated reason. Nothing in the original
finding actually tried an auxiliary-variable escape here before declaring
the class settled.

**The escape, tried:** Rosenberg's standard quadratization (Rosenberg 1975
/ Boros–Hammer) for reducing a pseudo-Boolean polynomial's degree by one
per introduced auxiliary. For each edge `(x_i, x_j)`, introduce an
auxiliary `w` constrained by penalty (not by construction) to equal
`x_i·x_j`:

```
penalty(w, x_i, x_j, P) = P * (x_i*x_j - 2*x_i*w - 2*x_j*w + 3*w)
```

which is exactly 0 when `w == x_i AND x_j`, and `>= P` for every other
combination of `(x_i, x_j, w)` — verified directly over all 8 combinations
in `audit/boundary_rosenberg_probe.py`. Substituting `w` for every
`x_i·x_j` crossing product turns `N_cross` **linear**:

```
N_cross = x0 + 2*x1 + 2*x2 + x3 - 2*(w0 + w1 + w2)      (this 4-cell/3-edge path)
```

so `(target - N_cross)^2` becomes `Product(L, L, weight)` over `{x0..x3,
w0..w2}` — genuinely pairwise — plus three Rosenberg penalty gadgets, each
built from `Linear`/`Product` terms over at most 2 variables. `P` must be
chosen large enough to dominate the OUTER term's own incentive to misuse
`w`, not just large enough to dominate the penalty gadget in isolation:
checked by brute force rather than assumed (`P=4.0` is the smallest value
found, by grid search, to work on this instance; `P=5.0` used below for
margin). This is instance-dependent — re-checked at (5 cells, target=2)
and (6 cells, target=3): `P=4.0`–`6.0` sufficed at 5 cells, but 6 cells
needed `P` between 6 and 8 — the same kind of dominance bound
`src/tsu/passes/encode.py`'s own `MONOTONE_PENALTY`/`ONE_HOT_PENALTY`
guards already compute for their own structural penalties, not a fixed
constant that trivially generalises to every scale.

**Verified on the matrix's own 4-cell/3-edge/target=1 instance**
(`audit/boundary_rosenberg_probe.py`):
- Brute force over all `2^4 * 2^3 = 128` `(x, w)` states: `min_w(energy)`
  agrees with `(1 - N_cross)^2` **exactly on all 16 `x`-states**.
- Through the REAL `tsu.passes.lower.lower` / `tsu.passes.analyse.analyse`
  pipeline (not hand-rolled): `lower()` does **not** raise `ThreeBodyError`
  — `n_nodes=7` (4 cells + 3 Rosenberg auxiliaries), `n_edges=21,
  max_degree=6, bipartite=False, mediators=9, max_abs_J=4.5, max_abs_b=4.5`
  — **passes all three Z1 gates** at this instance's scale. The
  `lower()`-derived `(J, b, offset)` was cross-checked against
  `model.energy()` over all 128 states before trusting the measurement.

**Corrected verdict: EXACT, via an auxiliary construction — not
INEXPRESSIBLE.** The precise, honest statement (per the brief's own
distinction): "no auxiliary-free pairwise form exists" is TRUE and
confirmed by `ThreeBodyError`, exactly as originally found. "No pairwise
form exists at all" is FALSE — a Rosenberg-quadratized auxiliary form
exists, reproduces `(target - N_cross)^2` exactly (zero distortion beyond
whatever `(target-N_cross)^2` itself already is relative to `boundary`'s
own "at most K" wording — a separate question this row, like the original,
does not re-litigate), and fits Z1 at the matrix's own instance scale. As
with `cluster`/`topological` (§4/§6), this says nothing yet about whether
the SAME construction, tiled across every edge of a realistic grid rather
than one 3-edge path, still fits Z1 — that grid-scale question was not
measured here and is flagged open, the same caveat the Task 4 section
below draws explicitly for the one-sided-threshold assignment gadget (C1).

---

## Task 4 — the auxiliary-variable question

**Probe (as specified):** an auxiliary binary `z` per neighbourhood.
`z·(target-N)` is `Product(VarRef(z), L, weight)` — pairwise, no
`ThreeBodyError` concern at all. The open question: can `z` be
*constrained*, using more pairwise terms, to actually indicate `N < target`
— without needing another non-pairwise rule to build that constraint?

**C2 correction (code review) — the original instance was degenerate.**
This section originally used a 2-cell neighbourhood (`N=x0+x1`, `target=1`)
as its "smallest exact-enumeration instance." That instance is degenerate:
`max(0, 1-N)` at `n=2, target=1` is **already** exactly
`(1-x0)*(1-x1)` — a plain `Product(L, L, 1.0)` with **zero** auxiliaries
needed at all (its exact multilinear polynomial has total degree 2, which
a bare pairwise form can already express directly — confirmed in
`audit/aux_variable_probe.py`'s `step0`). Two of the original verdict's
three evidence lines (the ground-state table and the distribution
comparison) therefore came from an instance where the very mechanism under
test was never exercised. **This is the exact mirror image of the
near-miss trap §12 documents for `boundary`** — there, a degenerate 2-edge
instance happened to collapse to a working pairwise form and was nearly
mistaken for a general proof of expressibility (a false positive); here, a
degenerate 2-cell instance happened to make the auxiliary unnecessary and
was mistaken for a general test of the mechanism (a false negative in the
opposite direction). Same lesson, told twice in this document: check
whether "the smallest instance" is smallest-and-representative or
smallest-and-accidentally-special before trusting what it shows.

**Corrected smallest NON-DEGENERATE instance:** 3-cell neighbourhood,
`N=x0+x1+x2`, `target=1` — confirmed non-degenerate first, not assumed: its
exact multilinear polynomial (`audit/aux_variable_probe.py`'s
`_multilinear_reference_polynomial`) is
`-x0*x1*x2 + x0*x1 + x0*x2 - x0 + x1*x2 - x1 - x2 + 1`, total degree 3 —
genuinely not expressible as any zero-auxiliary pairwise form (max degree
2), so this instance actually exercises the mechanism the probe exists to
test. Intended `max(0, 1-N)`: `{N=0: 1.0, N=1: 0.0, N=2: 0.0, N=3: 0.0}`.

**Step 1 — ground-state search.** The most general pairwise extension of
the probe's own stated shape — `Product(z, target-N, weight)` plus the only
other way to touch `z` without introducing a non-pairwise term,
`Linear(z, mu)` — was searched over a 156-pair `(weight, mu)` grid.
**No pair reproduces `max(0, target-N)` exactly.** Representative case
(`weight=1.0, mu=0.0`):

| x0 | x1 | x2 | N | want | got | z* | |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 1.0 | 0.0 | 0 | **DISAGREE** |
| 0 | 0 | 1 | 1 | 0.0 | 0.0 | 0 | match |
| 0 | 1 | 0 | 1 | 0.0 | 0.0 | 0 | match |
| 0 | 1 | 1 | 2 | 0.0 | -1.0 | 1 | **DISAGREE** (wrong *sign* — a reward, not a cost) |
| 1 | 0 | 0 | 1 | 0.0 | 0.0 | 0 | match |
| 1 | 0 | 1 | 2 | 0.0 | -1.0 | 1 | **DISAGREE** (wrong sign) |
| 1 | 1 | 0 | 2 | 0.0 | -1.0 | 1 | **DISAGREE** (wrong sign) |
| 1 | 1 | 1 | 3 | 0.0 | -2.0 | 1 | **DISAGREE** (wrong sign) |

**Step 3 — why this fails structurally, not just on this grid.** Every term
the probe is allowed to add contains `z` as a factor:
`Product(z, target-N, weight)` and `Linear(z, mu)` **both** evaluate to 0
at `z=0`, for **any** `(x0,x1)` and **any** `(weight, mu)`. `z=0` is always
a legal choice, so:

```
min_over_z(energy) <= energy(z=0) == 0        for every configuration
```

But `max(0, target-N)` needs to reach values up to `target` (here, 1.0) —
**strictly positive** — at maximum deficit. A quantity that is always ≤0
cannot equal a quantity that is sometimes >0. This is not a search-
resolution artefact; it is a structural property of "every added term
contains `z` as a factor," and it holds for **any** `(weight, mu)`.

**⚠ WITHDRAWN CLAIM (C1, code review) — this generalisation is FALSE, and
is kept here, struck through in spirit but not in fact, so the mistake
stays visible rather than being quietly edited away.** The original text
read: *"Replace `z` with any finite set `z_1..z_k`, each entering only as a
factor in a product term. The all-zero point `z_1=...=z_k=0` still zeroes
every such term, so the same `min_over_z <= 0` argument applies verbatim.
Escaping it requires a term that is NOT purely auxiliary-multiplicative...
this IS the one-sided-threshold problem this probe set out to solve."*

**The flaw:** this compares ABSOLUTE energies. An energy model is defined
only up to an additive constant — `IsingModel.offset` carries exactly this,
and `p(x) ~ exp(-beta*E(x))` cancels it, so two energy functions differing
by a constant induce the IDENTICAL distribution. "`min_over_z(energy) <= 0`
for every configuration" therefore constrains **nothing** about whether a
one-sided rule is expressible up to the additive constant every energy
model already carries — the constant can simply sit at `-target`. The
argument correctly shows `z=0` is always available and zeroes every
z-containing term; it does NOT show that the resulting family of functions
can never equal `max(0, target-N)` **up to a constant shift**, which is the
only sense in which two energy models are ever required to agree.

**The counterexample — an assignment/matching gadget.** Let `z[s][i]` mean
"slot `s` is assigned to cell `i`," for `s` in `0..target-1` and `i` over
the `n` neighbourhood cells:

```
term1:  sum_{s,i}          z[s][i] * (-x_i)          (reward a slot landing on a raised cell)
term2:  P * sum_s          sum_{i<j} z[s][i]*z[s][j]  (each slot used at most once)
term3:  P * sum_i          sum_{s<s'} z[s][i]*z[s'][i] (each cell used at most once)
```

**Every term still contains an auxiliary as a factor** — precisely the
shape the withdrawn claim said was impossible. With `P > 1` (the smallest
safe choice; see below), the global minimum over `z` is a genuine partial
matching of slots to raised cells (any deviation trades a marginal reward
of at most 1 for a collision penalty of at least `P`), and since every
`(s,i)` pair is a legal edge (a complete bipartite graph between slots and
cells), a matching of size `min(N, target)` using only raised cells always
exists and is optimal. Hence:

```
min_z(energy) = -min(N, target) = max(0, target - N) - target      EXACTLY
```

— the intended one-sided function, reproduced up to the additive `-target`
constant that `p(x) ~ exp(-beta*E)` makes irrelevant.

**Verified (`audit/assignment_gadget_probe.py`), by method:**
- **Full brute force over `(x, z)` jointly** at `(target, n)` in
  `{(1,2), (2,3), (3,4)}` (≤ 2^16 states): `min_z(energy) + target` matches
  `max(0, target-N)` **exactly, on every state**, at all three scales.
- **`(target, n) = (4, 6)`** (2^30 joint states, not brute-forceable in
  reasonable time): verified instead via exhaustive search over every
  VALID matching (a few hundred candidates) plus 20,000 random
  non-matching `z` per `x` as a broad corroborating sample — all 64
  `x`-configurations match exactly; the exchange argument above is what
  actually carries the claim at this scale, not the sampling.
- **The plan's own morphology scale** (`target=4`, Moore-8, `n=8`), through
  the REAL `tsu.passes.lower.lower` / `tsu.passes.analyse.analyse`
  pipeline, not hand-rolled: **32 auxiliary spins, `n_nodes=40,
  max_degree=11, |J|max=0.375, |b|max=3.5`** — passes all three Z1 gates
  (degree 11 ≤ 16, both magnitude caps ≤ 6.0) at this single-neighbourhood
  scale.
- **Finite-`beta` distribution comparison** (`target=2, n=3`, `beta=1.0`,
  against an independent oracle): total variation distance between the
  INTENDED distribution (built directly from `max(0,target-N)`, no
  pairwise form at all) and the assignment gadget's own marginal is
  **0.1210**, versus **0.1307** for the `(target-N)^2` two-sided
  reference at the identical scale — the gadget is measurably closer to
  the intended one-sided rule than the honest DISTORTED alternative,
  confirmed by direct computation, not asserted.

**The withdrawn argument's technical observation is not even wrong on its
own terms — its CONCLUSION is.** Setting every `z[s][i] = 0` really does
zero every term of the gadget too (checked directly: `energy(x, z=0) = 0`
for every `x`), so `min_z(energy) <= 0` for every configuration, exactly as
the withdrawn argument predicted, and the gadget does not somehow "escape"
that property. What the withdrawn argument got wrong is treating this as a
contradiction with `max(0, target-N)` needing to reach positive values —
it isn't one, because `min_z(energy)` only needs to reproduce
`max(0,target-N)` **up to the additive constant** every energy model
already carries. Here `min_z(energy)` ranges over exactly `[-target, 0]`
as `N` ranges from 0 to `target` or beyond, landing on
`-min(N,target) = max(0,target-N) - target` at every `N` — an exact match,
once the (irrelevant, cancels in `p(x)~exp(-beta E)`) `-target` shift is
accounted for. The withdrawn argument never considered that shift; that
omission, not any property of `z=0`, is the entire flaw.

**"Expressible" and "affordable" are different claims — the real prize,
and the honest ceiling.** The gadget costs `target * n` auxiliary spins
**per neighbourhood** — 32 at the morphology scale above. Tiling it across
every cell of a realistic grid, rather than measuring one neighbourhood in
isolation, is a DIFFERENT question, measured here for the first time
(`audit/assignment_gadget_probe.py`'s `step5_grid_scale_ceiling`): one
independent gadget per cell of a full **8x8 grid**, `target=4`, Moore-8
(boundary-truncated, not toroidal) neighbourhoods:

```
1680 auxiliary spins total (not the naive 32*64=2048 -- boundary cells
    have fewer than 8 neighbours, so fewer aux; interior cells still cost
    32 each)
n_nodes=1744, n_edges=9240, max_degree=32, max_abs_J=0.375, max_abs_b=8.0
```

**This FAILS the Z1 gates** — `max_degree=32` is DOUBLE the 16 cap, and
`max_abs_b=8.0` exceeds the 6.0 field cap. The node-budget gate
(1744 ≪ 250,000) is nowhere near binding, and aux-spin COUNT alone (1680)
looked entirely affordable — **the actual binding constraint is neither of
those**: an interior cell is a Moore-8 neighbour of up to 8 other cells,
and each of those cells' own gadgets references it via `target=4` slots
(`term1`), so a single interior cell's occupancy variable accumulates
degree `8 * 4 = 32` and bias magnitude `8 * 4 * 0.25 = 8.0` from the
overlapping neighbourhoods that all want it as one of their own inputs.
This is a genuine structural cost of tiling a *neighbourhood-local*
gadget across a grid where neighbourhoods overlap, not a fixable weight
choice (contrast `hydrological`'s footnote in the summary table, where the
gate failure WAS a tunable weight) and not primarily a node-count problem
(contrast `cluster`, §4). **No decomposition that might relieve this
(e.g. sharing slots across overlapping sites, or a global rather than
per-site assignment structure) was attempted here — flagged as open, per
"record failures instead of repairing them."**

**Both claims, stated separately, as the brief requires:** *"One-sided
rules are expressible via a pairwise auxiliary construction"* is TRUE —
proven above, at four scales, through the real compiler pipeline, closer
to the intended distribution than the two-sided alternative. *"One-sided
rules are affordable at grid scale via this construction"* is FALSE, as
measured — the naive per-cell tiling of the SAME construction fails both
the degree and field-cap gates at just 8x8 scale. Neither claim implies
the other, and this matrix's own earlier draft over-generalised in the
opposite direction (claiming inexpressibility) before this measurement
existed; the corrected position is not "therefore it is free to use
everywhere" either.

**What this means for `morphology`/`ecological` (§3, §9) and Task 5's own
implementation choice, corrected:** the original close of this section (see
cross-cutting finding #5, also corrected) instructed Task 5 to implement
the `(target-N)^2` nearest-pairwise form because no auxiliary escape was
believed to exist. That reasoning was wrong — an escape exists and is
proven above. Task 5's actual choice (implement `(target-N)^2`, documented
DISTORTED) is still the right call, but for a DIFFERENT, narrower reason:
the escape that exists is not yet affordable at the grid scale this
compiler's own workloads actually need, as just measured. If a future task
finds a decomposition that fixes the degree/field-cap blowup above, this
conclusion should be revisited — it is a cost finding, not an
impossibility, and cost findings can be overturned by a better
construction in a way impossibility findings cannot.

**Step 4 — distribution-level cross-check against the independent oracle**
(`audit/oracles/exact.py`'s `exact_boltzmann`/`exact_energy`, which does
not import `src/tsu`). Both models' `(J,b)` were built via
`tsu.passes.lower.lower` — already independently validated exact by Task
3's brute-force checks (§§1,7,8,10) — and cross-checked against
`model.energy()` before being handed to the oracle, so nothing here depends
on trusting `lower()` blindly. (An earlier version of this check
hand-derived `(J,b)` directly and silently mis-collected a same-spin square
`s_i^2` as a linear bias term — exactly the "second, weaker computation"
class of bug `src/tsu/passes/encode.py`'s own docstrings warn against. The
bug was caught by the same-style sanity check now baked into the script,
not by inspection, and the script has been corrected accordingly.)

At `beta=1.0`, marginal over the 3-cell neighbourhood (aux `z` summed out):

| | (0,0,0) | (0,0,1) | (0,1,0) | (0,1,1) | (1,0,0) | (1,0,1) | (1,1,0) | (1,1,1) |
|---|---|---|---|---|---|---|---|---|
| aux-model marginal | 0.0508 | 0.0743 | 0.0743 | 0.1382 | 0.0743 | 0.1382 | 0.1382 | **0.3117** |
| `(target-N)^2` reference | 0.0819 | 0.2227 | 0.2227 | 0.0819 | 0.2227 | 0.0819 | 0.0819 | 0.0041 |

By `N` (the only thing either model can be tracking): aux-model
`P(N=0..3) = [0.0508, 0.223, 0.4145, 0.3117]`; reference
`P(N=0..3) = [0.0819, 0.6682, 0.2458, 0.0041]`; total variation distance
between the two `P(N)` distributions is **0.4763**. The reference
distribution matches the already-known `morphology` distortion: `N=0`
(deficit) and `N=2` (surplus, the "exactly" side-effect) are roughly
symmetric around `N=1`, as Task 3 measured. The aux-model marginal is
neither this nor the intended one-sided rule: it most favours `N=3` (where
the intended penalty is **zero**), the opposite of tracking a deficit.
`z`'s free escape to 0 does not merely blur the deficit signal — it makes
the fully-raised neighbourhood look most attractive of all eight states.
(Both distributions are wrong relative to the intended rule; this
quantifies how differently they are wrong, not which one is closer — see
the Task 4 "assignment gadget" subsection below for a SEPARATE
construction that actually is measurably closer to the intended
distribution than the two-sided reference.)

### Verdict: the single/finite-auxiliary `z*(target-N)` probe FAILS

This probe (the specific mechanism the plan names — one auxiliary,
entering only as `Product(z, target-N, weight)` plus an optional
`Linear(z, mu)` bias) does not reproduce `max(0, target-N)` on the
corrected non-degenerate instance — not at the ground-state level (exact
enumeration, all 8 states checked, 5 of 8 disagree at the representative
weight, several with the wrong sign), not at the full-distribution level
(independent oracle, marginal favours the state the intended rule
penalises least), and not for any tested `(weight, mu)` (156-pair grid,
zero matches) — and Step 3's structural argument explains precisely why no
choice of `(weight, mu)` for THIS SPECIFIC one-auxiliary shape ever could.

**This remains a genuine, valuable negative result about the specific
mechanism tested** (a single auxiliary, entering only via one `Product`
and one `Linear` term) — it does **not** generalise to "no finite
auxiliary construction of any shape can rescue a one-sided threshold,"
which the ORIGINAL version of this section claimed and which the Task 4
"assignment gadget" subsection below (C1, code review) shows is FALSE by
explicit counterexample. The general question this probe's own closing
note originally flagged — "does SOME auxiliary mechanism exist for
one-sided thresholds" — is now answered: **yes**, by a genuinely different
mechanism (an assignment/matching gadget, not a multi-bit slack variable
as originally speculated, though the slack-variable idea remains untried
and is not needed now that a working mechanism is known). What is NOT yet
answered, and is flagged open below, is whether that working mechanism is
*affordable* once tiled across a realistic grid — measured, and found
wanting, in the same subsection.

---

## Cross-cutting findings

1. **The three "most likely to bite" items, resolved (revised, I6):**
   - `statistical`: predicted `K_64` (degree 63); **measured exactly
     confirmed** (`n_edges=2016=C(64,2)`, `max_degree=63`). Blows the
     degree-16 gate by 47, for any nonzero weight/target — an IR-exact,
     hardware-inexpressible rule at this scale.
   - `cluster`/`topological`: re-measured under the current
     (`max_abs_coupling`/`max_abs_bias` split) caps. Conservation **ALONE**
     is cheap on the full 8×8 grid (weight headroom to 12.0). The prior
     finding's actual claim — a collision with a competing 4.0 adjacency
     penalty — was NOT measured in this matrix's first pass; corrected
     under I6 (code review): once actually measured
     (`audit/contested_conservation_probe.py`), the contested setup DOES
     collide, at every conservation weight from 0.1 to 12.0
     (`|b|max=8.0` throughout, driven by the adjacency term alone, against
     the 6.0 cap) — the prior finding stands. The real cost driver for
     multi-member clusters (composing several commodities with no
     competing term) is **node count**, not degree (flat at 6 regardless
     of commodity count) — that part of the finding is unaffected.
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

3. **`INEXPRESSIBLE` count, corrected under C3 (code review): ZERO of 12
   classes.** This finding originally read "`INEXPRESSIBLE` classes are
   rejected by the compiler's own machinery, not merely argued," citing
   `boundary`'s `ThreeBodyError`. That `ThreeBodyError` is real and still
   correctly rejects the DIRECT, auxiliary-free construction — but §12 is
   re-verdicted EXACT via a Rosenberg-quadratized auxiliary that does not
   trigger it, verified against the matrix's own instance both by brute
   force and through the real compiler pipeline. No class in this matrix
   is rated INEXPRESSIBLE any longer; the twelve classes are now 10 EXACT,
   2 DISTORTED.

4. **`UNRESOLVED` count: 0 of 12 classes.** Task 4's own closing
   sub-question ("does some auxiliary mechanism rescue a one-sided
   threshold, if not the single-multiplicative-indicator one tested") is
   no longer open: the Task 4 section's "assignment gadget" subsection
   (C1, code review) answers it — yes, by a matching/assignment
   construction, verified at four scales and through the real pipeline.
   What remains open is narrower and stated explicitly where it arises:
   whether that construction is *affordable* once tiled across a full
   grid (measured and found NOT to fit, at 8x8 scale, in the Task 4
   section), and whether a smarter decomposition could fix that.

5. **Corrected under C1 (code review) — the auxiliary-variable escape
   DOES exist for morphology/ecological's shape, but is not yet affordable
   at grid scale.** This finding originally claimed Task 4's probe "fails
   structurally, not just on the tested grid: ANY auxiliary that enters
   only as a factor in product terms has a free, zero-energy escape... so
   `morphology`/`ecological` therefore stay DISTORTED." **The generalisation
   in that claim is FALSE** — see the Task 4 section's "assignment gadget"
   subsection for the counterexample and its full arithmetic; the "free,
   zero-energy escape at z=0" observation is TRUE but does not preclude a
   construction whose energies land in `[-target, 0]`, matching the
   intended one-sided function up to the additive constant every energy
   model is defined only up to. `morphology`/`ecological` DO still stay
   DISTORTED in Task 5's actual implementation — but for a cost reason,
   not an impossibility one: the assignment gadget that WOULD express them
   exactly costs `target * n` auxiliaries per neighbourhood and, tiled
   across a realistic 8x8 grid, fails the Z1 degree gate (32 vs. 16 cap)
   and field cap (8.0 vs. 6.0) on cells that are Moore-8 neighbours of
   several overlapping sites at once. "One-sided rules are expressible"
   and "one-sided rules are affordable at grid scale" are different
   claims; the first is now TRUE (reversing this finding's original
   position) and the second is FALSE as measured (supporting Task 5's
   actual implementation choice, on different grounds than originally
   given).

---

## Reproduction

```
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/measure_expressibility.py
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/aux_variable_probe.py
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/assignment_gadget_probe.py
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/boundary_rosenberg_probe.py
PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" audit/contested_conservation_probe.py
```

The first two scripts are Task 3/4's original output; the last three were
added under code review (2026-09-04-lattice-rule-taxonomy) to produce the
C1 (assignment-gadget counterexample and grid-scale ceiling), C3
(`boundary` Rosenberg re-verdict) and I6 (contested-setup remeasurement)
corrections respectively. `aux_variable_probe.py` was also corrected
in place under C2 (non-degenerate instance) rather than superseded — it
still demonstrates the single-auxiliary probe's genuine failure, just on
an instance that actually exercises the mechanism under test. All five
scripts are standalone (not pytest files; `testpaths = ["tests"]` does not
collect them) and print every number transcribed into this document.
