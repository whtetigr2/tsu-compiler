# Thermodynamic Programs — sitrep for Paul

**Date:** 2026-09-14  
**Shelf:** `/workspace/gibbs-observatory/programs/`  
**Receipts:** `/workspace/gibbs-observatory/receipts/prog_*`  
**Target:** z1 · **Honesty:** software compile + THRML sim — **no silicon**

## ★ Original flagship gift — Alloy Distribution Lab

| Program | nodes | mediators | Domain |
|---------|------:|----------:|--------|
| **alloy_ordering_8x8** | **64** | **0** | **metallurgy / Bragg–Williams ordering alloy** |

- YAML: `programs/alloy_ordering_8x8.yaml` → receipt `prog_alloy_ordering_8x8` (**COMPILED**, degree 4)
- UI: left nav **Alloy Lab** · decode API `/api/lab/alloy/*` · heroes `hero/alloy/`
- Docs: [`../ALLOY_DISTRIBUTION_LAB.md`](../ALLOY_DISTRIBUTION_LAB.md)
- **Codon demoted** from flagship (still available for Fabric Tax)

---

## What we stocked

**17 / 17 COMPILED** (16 prior + alloy) onto Z1 (ideal-first `tsu` compile via Observatory `program_service`).

### Promoted from sweeps (were already COMPILED)

| Program | nodes | mediators | Domain |
|---------|------:|----------:|--------|
| codon_opt_tiny | 18 | 6 | Extropic SEQ / codon opt |
| placement_8x8_exclusion | 64 | 0 | packing / keep-out |
| ilt_mask_adjacency_2x2 | 8 | 0 | ILT / mask rules |
| roster_shift_conflicts | 5 | 1 | scheduling |
| maxcut_cycle7 | 8 | 1 | Max-Cut / networks |
| mimo_detect_4 | 6 | 2 | MIMO detection |
| floorplan_4zone | 12 | 4 | facilities layout |
| mrf_denoise_8x8 | 64 | 0 | vision MRF |
| terrain_layout_6x6 | 108 | 36 | GIS landcover |

### New real programs (authored + compiled this pass)

| Program | nodes | mediators | Domain |
|---------|------:|----------:|--------|
| number_partition | 12 | 6 | finance / scheduling sibling |
| knapsack_tiny | 9 | 4 | resource allocation |
| jobshop_tiny | 8 | 2 | 3×2 job-shop lite |
| ecology_lotka_lite | 16 | 0 | competitive ecology spins |
| market_binary_factors | 14 | 4 | sparse/banded market factors |
| seq_design_longer | 32 | 8 | 8-position codon scale-up |
| sat_3tiny | 11 | 4 | 3-SAT via Rosenberg auxiliaries |

`seq_design_longer` (8 positions) **did COMPILE** (~6s) — no need to stop or truncate further. 10-position not required tonight.

## What runs

Every `prog_*` receipt has `program.json` + gates and is **Apply / Run THRML** ready.
Open via **File → Open receipt…** and pick `prog_<name>`, or paste YAML in the notepad and Compile → Apply → Run.

## Torx-only for later (not this compiler shelf)

Current tsu Ising YAML = binary/categorical variables + linear / product / product_over_edges / conserve_over_edges / neighbourhood_count.

**Cannot express here (Torx-native / continuous):**

- Continuous Lotka–Volterra ODEs and reaction–diffusion fields
- Torx diffusion / continuous latent dynamics
- Full floating-point portfolio SDEs, PDE ILT physics, etc.

Those belong on a Torx backend shelf later — not inventable as Z1 Ising receipts today. `ecology_lotka_lite` is the **discrete** competitive-exclusion stand-in (honest Extropic vibe without pretending to be continuous LV).

## Stress fails (still under sweeps, not promoted)

- `dense_qubo_clique20` — degree 19 > 16 → HARDWARE  
- `highk_biome_infeasible` — degree/field over budget → HARDWARE  

Useful as negative controls; not on the runnable programs shelf.

## 3D n≤16 reminder

State-space **3D PCA only when n ≤ 16**. Prefer `prog_number_partition` (12), `prog_ecology_lotka_lite` (16), `prog_jobshop_tiny` (8), `prog_roster_shift_conflicts` (5) for the 3D view tonight. Big grids stay 2D cloud.

## Best 3 tonight

1. **`prog_alloy_ordering_8x8`** — **original gift** · Alloy Lab occupancy + order metrics  
2. **`prog_number_partition`** — classic Ising; 3D-ready  
3. **`prog_codon_opt_tiny`** — Fabric Tax / mediator story (demoted from flagship)  

Then optionally: `prog_market_binary_factors` (sparse markets vs failed K20) and `prog_seq_design_longer` (codon scale-up that still fits).
