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

Gate results are the REAL `src/tsu/gates.py::check_gates` output on these
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
