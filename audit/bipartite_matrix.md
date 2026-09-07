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
