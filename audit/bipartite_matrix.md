# Bipartite matrix — Task 1 (all-bipartite-world plan)

Measures how far the bipartite binary overlay's 0.0s placement curve extends
past the point `audit/placement_curve.py` already established (8x8 through
20x20), against `audit/bipartite_routes.py::bipartite_at(n, "overlay")`.

Instance (identical shape to `audit/placement_curve.py`'s `OVL`, only `n`
varies): one binary variable per grid cell, single self-rule
`product_over_edges(a_value=1, b_value=1, weight=-0.4)`, `domain_wall`
encoding (a no-op for binary variables — they pass through `encode` as
`Binary()` directly; see `src/tsu/passes/encode.py` line 242).

**Non-degeneracy.** The rule weight (-0.4) is nonzero, `a_value == b_value`
is a genuine self-rule (not a trivially-satisfied or trivially-violated
condition), and the grid's 4-neighbour adjacency produces real edges at
every size (measured `n_edges` below is never 0, and `max_abs_J`/`max_abs_b`
are never 0 either — both would be, on a degenerate all-zero-weight or
edgeless instance). The `PLACED` verdict is not a fluke of the annealer
getting lucky on a small/sparse instance: `code_path` below was confirmed by
calling `_try_grid_embed` directly on the exact same graph `place()` builds,
and that function's own result is verified against every edge before it is
ever returned (see `src/tsu/passes/place.py`'s docstring) — a returned
grid embedding is provably correct, not heuristic.

## Full curve

Rows 8×8–20×20 are `audit/placement_curve.json` as originally written (not
re-run by this task); rows 16×16–64×64 are this task's new measurement
(`audit/bipartite_routes.py` → `audit/bipartite_matrix.json`). The original
sweep did not record `max_abs_J`/`max_abs_b`, so those two cells for the
8–20 overlay rows read `unavailable: not recorded by placement_curve.py`
rather than being guessed — per this project's "verification never
fabricates" rule, a value this task did not itself measure is never
back-filled from a same-shaped-but-different run.

| grid | n_nodes | n_edges | max_degree | bipartite | \|J\|max | \|b\|max | place wall time | mediators | result | code path |
|---|---|---|---|---|---|---|---|---|---|---|
| 8×8 base k=3 | 128 | 512 | 9 | False | unavailable | unavailable | 216.9 s | 64 | PLACED | annealer (mediated) |
| 10×10 base k=3 | 200 | 820 | 9 | False | unavailable | unavailable | 388.4 s | — | placement_effort_exhausted | annealer (mediated) |
| 12×12 base k=3 | 288 | 1200 | 9 | False | unavailable | unavailable | 583.6 s | — | placement_effort_exhausted | annealer (mediated) |
| 14×14 base k=3 | 392 | 1652 | 9 | False | unavailable | unavailable | 960.7 s | — | placement_effort_exhausted | annealer (mediated) |
| 16×16 base k=3 | 512 | 2176 | 9 | False | unavailable | unavailable | 1520.5 s | — | placement_effort_exhausted | annealer (mediated) |
| 20×20 base k=3 | 800 | 3440 | 9 | False | unavailable | unavailable | 1949.3 s | — | placement_effort_exhausted | annealer (mediated) |
| 8×8 overlay | 64 | 112 | 4 | **True** | unavailable | unavailable | 0.0 s | 0 | PLACED | grid_embed (not directly confirmed by this task) |
| 10×10 overlay | 100 | 180 | 4 | **True** | unavailable | unavailable | 0.0 s | 0 | PLACED | grid_embed (not directly confirmed by this task) |
| 12×12 overlay | 144 | 264 | 4 | **True** | unavailable | unavailable | 0.0 s | 0 | PLACED | grid_embed (not directly confirmed by this task) |
| 14×14 overlay | 196 | 364 | 4 | **True** | unavailable | unavailable | 0.0 s | 0 | PLACED | grid_embed (not directly confirmed by this task) |
| 20×20 overlay | 400 | 760 | 4 | **True** | unavailable | unavailable | 0.0 s | 0 | PLACED | grid_embed (not directly confirmed by this task) |
| **16×16 overlay** | 256 | 480 | 4 | **True** | **0.2** | **0.7999999999999999** | **0.0017 s** | 0 | PLACED | **grid_embed (confirmed)** |
| **24×24 overlay** | 576 | 1104 | 4 | **True** | **0.2** | **0.7999999999999999** | **0.0038 s** | 0 | PLACED | **grid_embed (confirmed)** |
| **32×32 overlay** | 1024 | 1984 | 4 | **True** | **0.2** | **0.7999999999999999** | **0.0075 s** | 0 | PLACED | **grid_embed (confirmed)** |
| **40×40 overlay** | 1600 | 3120 | 4 | **True** | **0.2** | **0.7999999999999999** | **0.0111 s** | 0 | PLACED | **grid_embed (confirmed)** |
| **64×64 overlay** | 4096 | 8064 | 4 | **True** | **0.2** | **0.7999999999999999** | **0.0286 s** | 0 | PLACED | **grid_embed (confirmed)** |

Note the 8×8–20×20 overlay rows' `code_path` is annotated "not directly
confirmed by this task": `placement_curve.py` never called `_try_grid_embed`
itself to check, it only recorded `place_s=0.0` and `mediators=0`, which is
*consistent* with the grid-embed path but was not independently verified at
the time. This task's own 16×16 remeasurement (bold row) used the identical
instance shape and got `place_s=0.0017s`, `mediators=0`, `code_path=
grid_embed`, confirmed directly — good evidence the un-confirmed rows took
the same path, but stated as inference from a same-shape remeasurement, not
as a re-verified fact about the original run's specific process.

## Findings

**It never stops placing instantly.** Every size from 16×16 through 64×64
placed via the deterministic `_try_grid_embed` path, confirmed independently
(not inferred from a fast `place_s`) by calling `_try_grid_embed` on the
exact graph `place()` builds and checking its own return value. **64×64
(4,096 cells) placed in 0.0286 seconds** — under 29 milliseconds, with 0
mediators. Wall time grows slightly with `n` (1.7 ms → 3.8 ms → 7.5 ms →
11.1 ms → 28.6 ms across 256 → 4096 nodes) — consistent with the
near-linear BFS-order backtracking `_try_grid_embed` performs, not with any
search or annealing cost. There is no discontinuity, no fallback to the
annealer, and no failure anywhere in the swept range. **This is a plain
statement, not a hedge: a 4,096-cell world places in under 30 milliseconds.**

**Nothing drifted.** `max_degree` (4), `max_abs_J` (0.2), and `max_abs_b`
(0.7999999999999999 ≈ 0.8) are bit-for-bit identical across all five newly
measured sizes (16×16 → 64×64). This is a structural fact, not a
coincidence: every interior cell of a 4-neighbour grid has degree 4
regardless of grid size, and this term's coefficients are a fixed function
of a node's own degree and the fixed weight −0.4 — so the maximum, achieved
at any interior node, cannot change with `n`. Both gates sit far under the
6.0 cap (`|J|`: 0.2/6.0 = 3.3%; `|b|`: 0.8/6.0 = 13.3%), with no trend
toward it as `n` grows — the drift this task was asked to rule out does not
exist for this instance, for the sizes actually measured (16×16–64×64; the
8–20 rows were not re-measured for these two fields, see above).

**Node budget.** At the largest size measured, 64×64, the overlay uses
**4,096 nodes against the Z1 node budget of 250,000** (`src/tsu/target.py`,
`Sourced(250_000, "F-15", ...)`) — **1.64% of budget consumed, 98.36%
headroom** (245,904 nodes unused, measured not extrapolated). Separately,
as a labeled *extrapolation* (not a measurement): the same one-spin-per-cell
structure would need a roughly 500×500 grid (250,000 cells) before a single
overlay layer alone exhausted the budget outright; multiple stacked
bipartite layers (as Task 2's binary-stack route proposes) divide that
headroom by the number of layers. Nothing at that scale was run.

## Concerns

1. **This measures one overlay instance (a single clumping self-rule),
   not the full workload.** A real world composes multiple layers; whether
   several bipartite layers placed independently (or one combined bipartite
   graph) still takes the `grid_embed` path at 64×64 is Task 2/3/5's
   question, not this one's.
2. **The 8×8–20×20 rows are carried over from `audit/placement_curve.json`
   verbatim** (grid/n_nodes/n_edges/degree/bipartite/place_s/mediators/
   result only) **and are not re-measured by this task.** Their `|J|max`,
   `|b|max`, and confirmed `code_path` were never recorded by that original
   sweep and are marked `unavailable`/"not directly confirmed" above rather
   than filled in — this task ran `bipartite_at` only at 16, 24, 32, 40, 64
   as instructed, not at 8, 10, 12, 14, 20.
3. **64×64 is the largest size this task was asked to measure; it is not
   necessarily where the curve "stops being interesting."** Given zero
   drift and sub-30ms times, there is no measured evidence of an upcoming
   ceiling before the node budget itself — but budget-adjacent sizes
   (hundreds of thousands of nodes) were not measured and are only
   extrapolated above, explicitly labeled as such.

---

# Task 2 — Route A: the all-binary stack

Three independent binary layer specs (`specs/binary_stack_l0.yaml`,
`_l1.yaml`, `_l2.yaml`), one spin per grid cell each, **self-rules only**
(`product_over_edges` with `a_value == b_value == 1`, a clumping term) —
domain story: bit 0 "damp ground", bit 1 "loose scree", bit 2 "plant
cover". `demo/binary_world.py::compose_state` (Task 4) reads the three
decoded layers as `bits[0] + 2*bits[1] + 4*bits[2]`, giving 8 distinguishable
composed states per cell against the k=3 base layer's 3.

**Non-degeneracy.** Each layer's weight is nonzero and distinct (-0.4 /
-0.35 / -0.3 — chosen so the three layers are not silent copies of one
instance; weight magnitude changes `|J|max`/`|b|max` but never the graph
topology or its bipartiteness, which are functions of which cells are
coupled, not of the coupling's sign or size), the rule is a genuine
self-rule (`a_value == b_value`, not trivially satisfied/violated), and the
grid's 4-neighbour adjacency produces real, nonzero edges at both measured
sizes (`n_edges` and both gate maxima are never 0). Each row was measured
via `audit/bipartite_routes.py::bipartite_at`, with **THE SAFETY RULE**
(`analyse(...).bipartite is True` checked before every `place()` call)
enforced by that same function, not re-implemented here.

## Measured

| grid | layer | weight | n_nodes | n_edges | max_degree | bipartite | \|J\|max | \|b\|max | place wall time | mediators | result | code path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8×8 | binary_stack_l0 | -0.4 | 64 | 112 | 4 | **True** | 0.2 | 0.7999999999999999 | 0.0005 s | 0 | PLACED | grid_embed |
| 64×64 | binary_stack_l0 | -0.4 | 4096 | 8064 | 4 | **True** | 0.2 | 0.7999999999999999 | 0.028 s | 0 | PLACED | grid_embed |
| 8×8 | binary_stack_l1 | -0.35 | 64 | 112 | 4 | **True** | 0.175 | 0.7000000000000001 | 0.0004 s | 0 | PLACED | grid_embed |
| 64×64 | binary_stack_l1 | -0.35 | 4096 | 8064 | 4 | **True** | 0.175 | 0.7000000000000001 | 0.0352 s | 0 | PLACED | grid_embed |
| 8×8 | binary_stack_l2 | -0.3 | 64 | 112 | 4 | **True** | 0.15 | 0.6 | 0.0004 s | 0 | PLACED | grid_embed |
| 64×64 | binary_stack_l2 | -0.3 | 4096 | 8064 | 4 | **True** | 0.15 | 0.6 | 0.028 s | 0 | PLACED | grid_embed |

Raw rows: `audit/bipartite_matrix_task2.json`.

**Finding.** All three layers are bipartite at both sizes, place via the
deterministic `grid_embed` path (confirmed, not inferred from a fast
`place_s` — `bipartite_at` calls `_try_grid_embed` on the exact graph
`place()` builds, same as Task 1), take **0 mediators**, and place in under
36 ms even at 64×64 (4,096 cells). This is the identical shape and result
Task 1 already measured for its single "overlay" instance, now confirmed
independently for three distinctly-weighted layers — three separately
compilable binary specs is not a regression on any axis Task 1 measured.

## Step 3 — what cannot be expressed this way, named concretely

The base layer (`specs/lattice_small_8x8_k3.yaml`) has four terms:

1. `{a_value: 0, b_value: 1, weight: 1.0}` — **the cross-rule**: water (0)
   directly adjacent to rock (1) is penalized. This is the rule tied to the
   spec's own `contract.validate` check (`forbid_value_pair_over_edges`,
   a_value=0, b_value=1) — the one hard, compile-time-enforced guarantee
   the whole plan is about relocating.
2. `{a_value: 0, b_value: 0, weight: -0.4}` — self-rule: water clumps with
   water.
3. `{a_value: 1, b_value: 1, weight: -0.4}` — self-rule: rock clumps with
   rock.
4. `{a_value: 2, b_value: 2, weight: -0.4}` — self-rule: grass clumps with
   grass.

Rules 2, 3, 4 map directly onto the three binary layers' own self-rules
(exactly what `binary_stack_l0/l1/l2.yaml` implement) — **these are not
lost.**

**Rule 1 is lost, and the reason is more specific than "cross-rules break
bipartiteness."** That framing is the base layer's own diagnosis for *why
the base layer itself* is non-bipartite (a k=3 domain-wall chain, and — see
Task 3 below — even some of its self-rules turn out to break bipartiteness
too, for a reason that has nothing to do with being a cross-rule). For the
**binary stack specifically**, the obstruction was checked directly rather
than inherited: a same-layer CROSS-rule on a plain binary (k=2) variable
was built and measured (`audit/bipartite_routes.py`-style probe, `a_value:
0, b_value: 1, weight: 1.0` on an 8×8 binary grid) and came back
`n_nodes=64, n_edges=112, max_degree=4, bipartite=True, max_abs_J=0.5` —
**identical topology to the self-rule case, still bipartite.** This checks
out from the encoding itself: `tsu.spec._value_indicator` for a Binary
variable always returns a `LinearForm` over the *same single* `VarRef` (the
cell's own spin) regardless of which value (0 or 1) it indicates — a
domain-wall chain, which is what lets a cross-rule introduce
different-chain-position couplings for k≥3, does not exist for k=2 at all
(`encode.py`'s `_build_categorical`: a `Binary` variable is appended
straight into `variables`, never chained). **So "only self-rules survive
within a layer" is not literally true for a plain binary layer** — a
same-layer 0-vs-1 cross-rule would stay bipartite too, if one were written.

**What is actually lost is architectural, not rule-shape-specific.** Rule
1 forbids a *composed* state 0 (water) from touching a composed state 1
(rock) — and under `compose_state`'s convention, composed state 0 is bits
`(0,0,0)` and composed state 1 is bits `(1,0,0)`: they differ *only* in
layer 0's bit, **conditioned on layers 1 and 2 both being 0 at both
cells.** No single one of the three independently-compiled binary specs
has any visibility into the other two layers' bits — each is its own
separate `IsingModel`, placed and sampled on its own — so no
`product_over_edges` term (self or cross) written inside `binary_stack_l0`
alone can ever condition on `binary_stack_l1`/`_l2`'s values. **A hard,
hardware-representable hard energy term for "composed 0 never touches
composed 1" cannot be written in this architecture at all**, regardless of
which per-layer rule shape is chosen — not because the rule is a
cross-rule, but because it needs joint visibility across layers that three
independently-compiled bipartite specs structurally do not have. This is
exactly why Task 4 relocates it to a **cross-layer bias patch** (applied at
sample time, after the other layers are already decoded) instead of a
compiled term — and why that relocation is necessarily a soft nudge, not a
guarantee.

---

# Task 3 — Route B: categorical base, self-rules only

`specs/selfrule_base.yaml`: the identical k=3 categorical shape as
`specs/lattice_small_8x8_k3.yaml`, with the one cross-rule
(`{a_value: 0, b_value: 1, weight: 1.0}`, water-adjacent-to-rock) dropped —
only the three self-rules remain (`a_value == b_value`, one per value,
weight -0.4 each, unchanged from the base layer's own).

**Non-degeneracy.** All three weights are the base layer's own real values
(not invented for this task), the rules are genuine self-rules on a real
3-value domain (not a trivial k=1/k=2 stand-in), and the grid's 4-neighbour
adjacency produces real, nonzero edges (`n_edges=512` below is never 0, nor
are the gate maxima).

## Step 1–2 — measured

**THE SAFETY RULE applied literally**: `bipartite_at(8, "selfrule_k3")` was
called and its `analyse(...).bipartite` result read *before* any `place()`
was attempted, exactly as Task 1/2 did — `place()` was never called for
this kind, at any size.

| grid | n_nodes | n_edges | max_degree | bipartite | \|J\|max | \|b\|max | result |
|---|---|---|---|---|---|---|---|
| 8×8 selfrule_base | 128 | 512 | 9 | **False** | 2.5 | 2.1 | SKIPPED_non_bipartite |

Raw row: `audit/bipartite_matrix_task3_probe.json`.

## Step 3 — the finding: Route B is not bipartite, even at 8×8

**This is the finding, per Task 3's own instruction: report it, mark Route
B dead, do not attempt to force it.** No placement was run at 64×64 (or at
any size) — the 8×8 result already answers the question this task asked,
and Task 3 explicitly says not to force it.

This directly **contradicts** the plan's inherited assumption ("earlier
measurement found domain-wall self-rules stay bipartite") — which is
exactly why the plan itself asked this to be *confirmed with a real rule
set rather than inherited*. Confirmed, and the confirmation is negative.

**Root cause, isolated by direct measurement (exploratory diagnostic, not
part of Route B's own deliverable — labeled as such per this project's
epistemic discipline: two additional `analyse()`-only probes, no
`place()`, same 8×8 grid, not committed as a spec file):**

| instance | terms | bipartite | n_edges | max_degree |
|---|---|---|---|---|
| extremes-only | `{0,0,-0.4}`, `{2,2,-0.4}` | **True** | 288 | 5 |
| middle-only | `{1,1,-0.4}` | **False** | 512 | 9 |
| all three self-rules (= selfrule_base.yaml) | `{0,0}`,`{1,1}`,`{2,2}` | **False** | 512 | 9 |

The middle-value self-rule (`a_value: 1, b_value: 1`) **by itself** is
already non-bipartite, with the identical `n_edges`/`max_degree` (512 / 9)
as the full base layer. The mechanism: domain-wall's value-1 indicator for
k=3 is `occ(chain[0]) - occ(chain[1])` — it spans *both* of a cell's chain
spins with opposite sign, so `Product(indicator_u(1), indicator_v(1),
weight)` across a grid edge expands into four couplings
(`u_chain0-v_chain0`, `u_chain0-v_chain1`, `u_chain1-v_chain0`,
`u_chain1-v_chain1`), not just the two "aligned same chain-position"
couplings a boundary value's single-spin indicator produces. The diagonal
(`u_chain0-v_chain1`, `u_chain1-v_chain0`) couplings are what introduce odd
cycles. This has nothing to do with the term being a *cross*-rule
(`a_value != b_value`) — it is a genuine **self**-rule (`a_value ==
b_value == 1`) that breaks bipartiteness anyway, because the value it
self-agrees on happens to be an interior (non-boundary) value of a k≥3
domain-wall chain. "Self-rules stay bipartite" is true only for
domain-wall's two boundary values (0 and k-1, each touching a single chain
spin); it is false for any interior value, and k=3 has exactly one interior
value (1) — which is precisely the value the base layer's own third rule
(rock clumps with rock) already used.

**Route B is dead as specified.** No attempt was made to "fix" it by
dropping the middle self-rule too (extremes-only measured bipartite above
is reported as a diagnostic data point explaining the mechanism, **not**
as a proposed alternative Route B — the task instruction is to report the
non-bipartite finding and stop, not to search for a bipartite subset).

---

# Task 4 — what the trade actually costs

`demo/binary_world.py` composes Task 2's three binary layers into one of 8
states (`compose_state`) and measures the cost of relocating the base
layer's hard rule ("value 0 never adjacent to value 1") from a compiled
energy term into a cross-layer bias patch (`demo/layers.py::bias_patch`).

## Steps 1–3 — TDD `compose_state`

`tests/test_binary_stack.py::test_three_binary_layers_compose_to_eight_states`
(the plan's own snippet, plus two additional tests covering all 8 bit
combinations and input validation) was written first and run before any
implementation existed:

```
ModuleNotFoundError: No module named 'binary_world'
```

— the exact failure the plan predicted. `compose_state` (`state = bits[0]
+ 2*bits[1] + 4*bits[2]`) was then implemented in `demo/binary_world.py`;
all three tests pass. **Production-change check, done for real (not
asserted):** the bit-order was temporarily reversed
(`bits[2] + 2*bits[1] + 4*bits[0]`) — 2 of 3 tests failed exactly as
expected (`assert 4 == 1`, etc.); separately, the two input-validation
`raise ValueError` statements were temporarily removed — the rejection
test failed with an uncaught `IndexError` instead of the expected
`ValueError`. Both changes were reverted and all three tests confirmed
passing again before proceeding.

## Step 4 — the relocated rule as a cross-layer bias patch

`compose_state`'s own convention makes composed state 0 = bits `(0,0,0)`
and composed state 1 = bits `(1,0,0)` — they differ **only** in layer 0's
bit, and both require layers 1 and 2 to read 0. So "0 never adjacent to 1"
is entirely a statement about layer 0's bit, conditioned on layers 1/2
already being 0 at both cells of an edge. `build_relocation_patch` builds
a `{(cell, 1): weight}` patch for layer 0 from two **already-decoded**
layer-1/2 grids (a fixed "backdrop"): a cell is *eligible* to read as
composed 0 or 1 iff both backdrop layers are 0 there; for each eligible
cell, its eligible neighbours' own (backdrop) bit-0 values push it toward
agreement (avoiding a 0-vs-1 collision), scaled by `strength`. This is
folded into layer 0's own compiled program via `demo/layers.py::bias_patch`
— no hand-rolled patching mechanism, no recompile.

## Step 5 — measured violation rate, and a degenerate first attempt caught before being reported

**First attempt (rejected, not reported as a result):** a single fixed
backdrop (one draw each of layers 1/2, mirroring `demo/stacked_world.py`'s
own `grid1 = grids1[0]` precedent) produced **0% violations at every
strength from 0.0 to 0.35, including zero nudge** — which looked like a
clean win but was checked before being believed: only **2 of 64 cells**
were eligible (both layers 0) in that one backdrop. Measured directly
across ten additional backdrop draws per layer: layer 1 and layer 2's own
self-rules reward *only* the 1-1 agreeing pair, never 0-0 (an asymmetric
clumping term, not a symmetric one) — so each layer's own marginal
`P(bit=0)` sits well under 50% (measured 10.9% for layer 1, 15.1% for
layer 2 over the full ensemble below), making "eligible" cells rare and an
**adjacent pair** of eligible cells (the only configuration the rule can
ever fire on) rarer still. A single backdrop is therefore a **degenerate
instance for this specific measurement**: most contain zero
adjacent-eligible pairs, so "0%" was a structurally empty measurement, not
evidence the patch works — exactly the false-negative failure mode this
project's Global Constraints name explicitly. This was caught by checking
the eligible-cell count before trusting the rate, not discovered later.

**Fix: pool over many independent backdrop realizations, not one.** 240
independent, unconditioned draws of each of layers 0/1/2 were taken (one
`thrml_sample` call per layer, `n_chains=8 * n_samples=30 = 240` draws
each — comfortably past the plan's "≥200 draws" floor). Measured over the
full 240-backdrop ensemble:

- `P(bit=0)`: layer 1 = 10.9%, layer 2 = 15.1%.
- Mean eligible cells/backdrop: 1.20/64. Mean adjacent-eligible
  pairs/backdrop: 0.083. **18/240 backdrops (7.5%) have any adjacent-eligible
  pair at all**; 20 such pair-"slots" total across the ensemble.

Every backdrop's eligible-pair slots were checked against the full
240-draw unpatched layer-0 batch for the **zero-nudge** rate (any
`(layer0 draw, backdrop)` pairing is a genuine sample of the unconditioned
joint `p(l0)*p(l1)*p(l2)`, so this reuses already-drawn data, no extra
sampling); for each **patched** strength, only the 18 backdrops with a
real eligible pair needed a fresh patched draw (the other 222 backdrops'
patch is the empty dict, and `bias_patch` on an empty patch is a proven
no-op — `tests/test_layers.py::test_empty_patch_is_a_noop` — so their
contribution is provably identical to the zero-nudge case without
re-sampling them).

**Two statistics are reported, not one, because the overall rate alone
would hide the effect:**

| strength | \|b\|max | overall (all adjacent pairs, all backdrops) | conditional (eligible pairs only) |
|---|---|---|---|
| 0.0 (zero nudge) | 0.800 | 871/6,451,200 (**0.0135%**) | 871/4,800 (**18.15%**) |
| 0.02 | 0.810 | 0.0138% | 18.60% |
| 0.05 | 0.825 | 0.0133% | 17.94% |
| 0.1 | 0.850 | 0.0142% | 19.08% |
| 0.2 | 0.900 | 0.0121% | 16.25% |
| 0.35 | 0.975 | 0.0110% | 14.79% |
| 0.5 | 1.100 | 0.0103% | 13.85% |
| 0.75 | 1.350 | 0.0095% | 12.75% |
| 1.0 | 1.600 | 0.0083% | 11.15% |
| **1.5** | 2.100 | **0.0072%** | **9.69% (minimum)** |
| 2.0 | 2.600 | 0.0076% | 10.25% |
| 3.0 | 3.600 | 0.0084% | 11.25% |
| 5.0 | 5.600 | 0.0107% | 14.37% |

**The "overall" column is dominated by a fact that has nothing to do with
the patch**: composed states 0 and 1 are a rare corner of this instance's
state space (both require the minority phase of two independently
ferromagnetic layers), so the vast majority of adjacent pairs can never
violate this rule regardless of any patch, at any strength — reporting
only "0.01%" would be true but misleading, the same shape of false
negative the single-backdrop draft produced. **The conditional column
(among pairs that actually could violate) is the honest measure of what
the patch does**, and it is what "the price of this plan" should be read
from.

## Comparison — the number the plan asked for

**Hard energy term, current 8×8 base
(`specs/lattice_small_8x8_k3.yaml`): 0% by construction** (contract-validated
via `forbid_value_pair_over_edges`, never sampled — it cannot occur).

**Relocated as a bias patch: 18.15% at zero nudge, falling to a measured
minimum of 9.69% at strength 1.5 — never 0%, at any strength tried.** The
difference between baseline and the minimum (871/4800 vs 465/4800) is
~15 standard errors apart (SE ≈ 0.56 percentage points at n=4800, p≈0.18)
— a real, not noise-level, effect of roughly halving the violation rate,
matching the qualitative shape of the project's earlier elevation-band
measurement (8.32% → 4.69%, also "roughly halves without eliminating").
**That gap — 0% by construction vs. a floor around 9.7% no patch strength
in this sweep got below — is the number this plan asked to be reported,
not tuned away.**

## Step 6 — the sweep, and the degenerate zone

The rate is **not monotone in strength**: it falls from 18.15% (strength
0) to a minimum of 9.69% at strength 1.5, then **rises back up** — 10.25%
at 2.0, 11.25% at 3.0, 14.37% at 5.0, heading back toward the unpatched
baseline. This is the degenerate zone the plan asked to locate: at high
enough strength the patch stops improving the rule it was relocated for
and starts working against it (`|b|max` at strength 5.0 is 5.6, still
under the 6.0 field cap, so this is not a field-cap artifact — it is the
patch overwhelming layer 0's own base self-rule, exactly the mechanism the
plan names).

**This refutes "~0.2" as a universal threshold, while confirming a
degenerate zone exists.** For this configuration the zone begins between
strength 1.5 and 2.0 (`|b|max` ≈ 2.1–2.6) — roughly an order of magnitude
higher, in raw strength units, than the ~0.2 figure measured earlier for
`demo/stacked_world.py`'s distance-weighted ROCK patch. The two numbers
are not directly comparable: `strength` here scales a per-cell sum of up
to 4 neighbours' ±1 contributions (max magnitude `4 × strength`), while
`stacked_world.py`'s `ALPHA` scales a per-cell *distance-to-water* value
(routinely larger than 4) — the same word ("strength"/"alpha") multiplies
a differently-scaled quantity in each mechanism, so the raw number where
degeneracy begins is instance-specific, not a portable constant, and this
task's own instance is measured, not inherited.

## Concerns

1. **Only 20 adjacent-eligible-pair "slots" exist across the entire
   240-backdrop ensemble**, and only 18 of 240 backdrops contain any at
   all. The conditional statistic's ~15-SE significance is real, but it
   rests on a genuinely small number of *distinct spatial configurations*
   (18), each evaluated across many layer-0 resamples — not 18 independent
   spatial layouts times independent everything. A different random seed
   for the layer-1/2 batches would very likely realize a different set of
   18 backdrops (the underlying marginals — 10.9%/15.1% — are the stable,
   reproducible fact; which specific 18 backdrops realize an
   adjacent-eligible pair is seed noise on top of that).
2. **This measures one specific rule relocation (composed 0 vs 1 via
   layer 0's bit) under one specific patch design** (a linear function of
   eligible neighbours' backdrop bit-0 values) — not a general claim about
   every possible relocated rule or every possible patch construction.
   Task 4's own design note (top of `demo/binary_world.py`) states this
   explicitly: a rule between two composed states differing in layer 1 or
   2's bit would need a different layer patched, and was not measured.
3. **Route B (Task 3) has no equivalent measurement in this task** — Task
   4's own file list scopes the violation-rate deliverable to
   `demo/binary_world.py`/`tests/test_binary_stack.py`, i.e. Route A only;
   Route B was already found dead in Task 3 and no relocation was
   attempted for it.
