# Z1 Workload Compile Sweep — Discovery Sitrep

**Date:** 2026-09-14  
**Audience:** Paul (novice-friendly)  
**Target:** Extropic Z1 profile via `tsu` ideal-first compile  
**Honesty:** Software compile + (later) JAX/THRML simulation only — **not Extropic silicon**.

---

## What is a “workload”?

A workload is a **Thermodynamic Program**: a YAML that names variables (binary or categorical), energy terms (what is preferred or forbidden), and optional validation rules. Think of it as the real problem statement — codon choices, who works which shift, which mask features may touch — written so the `tsu-compiler` can turn it into an Ising model Z1’s architecture can host.

## What does “lower onto Z1” mean?

Lowering means **compiling** that program into a **degree ≤ 16, bipartite, chromatic-block-Gibbs Ising** that fits Z1’s published ports, colouring, and coupling/field caps. The compiler tries encodings (here `domain_wall` usually wins), may insert **mediator** spins so the coupling graph becomes 2-colorable, then checks gates. A `COMPILED` verdict is a software receipt you can Apply into Observatory’s THRML sampler — still a CPU/GPU sim, not a claim that chips ran.

---

## Ladder results

| id | real problem | verdict | encoding | nodes | mediators | max deg | notes |
|----|--------------|---------|----------|------:|----------:|--------:|-------|
| `codon_opt_tiny` | Tiny codon design (usage + adjacent repeats) | **COMPILED** | domain_wall | 18 | **6** | 4 | Extropic/THRML codon class, k=3 truncated synonyms |
| `placement_8x8_exclusion` | Hard-core keep-out / agent packing | **COMPILED** | domain_wall | 64 | 0 | 4 | Bipartite grid; no mediators |
| `ilt_mask_adjacency_2x2` | ILT/mask forbidden feature abut | **COMPILED** | domain_wall | 8 | 0 | 3 | Tiny mask-rule surrogate |
| `roster_shift_conflicts` | Nurse/machine double-booking | **COMPILED** | domain_wall | 5 | **1** | 3 | Conflict triangle → 1 mediator |
| `maxcut_cycle7` | Network Max-Cut on odd cycle | **COMPILED** | domain_wall | 8 | **1** | 2 | C7 needs 1 mediator for bipartite fabric |
| `mimo_detect_4` | 4-bit MIMO ML-detection QUBO | **COMPILED** | domain_wall | 6 | **2** | 3 | Dense interference → 2 mediators |
| `floorplan_4zone` | Facility zone adjacency | **COMPILED** | domain_wall | 12 | **4** | 5 | Categorical rooms; mediation tax |
| `mrf_denoise_8x8` | Image MRF smoothness prior | **COMPILED** | domain_wall | 64 | 0 | 4 | Classic vision Ising; bipartite |
| `terrain_layout_6x6` | Landcover layout 6×6 k=3 | **COMPILED** | domain_wall | 108 | **36** | 9 | Works; heavy mediator tax |
| `dense_qubo_clique20` | Dense all-pairs portfolio/QUBO | **HARDWARE** | — | (20 at gate) | — | **19** | degree 19 > Z1’s 16 |
| `highk_biome_infeasible` | k=5 16×16 biome attempt | **HARDWARE** | — | (1024 at gate) | — | **18** | degree 18 **and** field 8.0 > 6.0 |

**Score: 9 COMPILED / 2 HARDWARE refusals / 0 crashes.**

Receipts: `/workspace/gibbs-observatory/receipts/sweep_<id>/`  
Machine table: `/workspace/gibbs-observatory/sweeps/SWEEP_RESULTS.json`  
YAML sources: `/workspace/gibbs-observatory/sweeps/workloads/*.yaml`

---

## Concrete discoveries

1. **Codon optimization lowers onto Z1 at tiny scale.**  
   `codon_opt_tiny` (6 positions × k=3 truncated synonyms, usage + adjacent-pair terms, Extropic/THRML paper class) **COMPILED** with domain_wall → 12 logical DW spins + **6 mediators** = 18 nodes, degree 4. A Z1 owner’s first codon experiment should start here, not at full spike-protein length.

2. **Mediators appear when the logical coupling graph is not bipartite.**  
   Odd cycle Max-Cut (`maxcut_cycle7`) → 1 mediator; roster conflict triangle → 1; dense MIMO K4-ish → 2; codon chain after DW encoding → 6; 6×6 terrain → **36** mediators. Grid-only binary smoothness (`mrf_denoise_8x8`, `placement_8x8_exclusion`) stays mediator-free because the grid is already bipartite.

3. **Dense all-pairs hits the degree wall cleanly.**  
   `dense_qubo_clique20` (20-asset fully connected QUBO) fails with `degree: measured=19 limit=16` — IDEAL-logic fine, Z1 ports refuse. Thin the graph (banded correlations, factor model) rather than “hoping” placement fixes degree.

4. **High-k landcover fails two gates at once.**  
   `highk_biome_infeasible` (k=5, extra hard adjacency partners) fails **degree 18>16** and **field_cap 8.0>6.0** (field gate is assumed-sourced). Matches the lattice_small docstring lesson: k=5 one-hot structural fields blow the cap; k=3 + mild weights is the working regime.

5. **Freeze-safe weights matter as much as topology.**  
   Terrain/mask siblings use 1.0 / −0.4 (or exclusion +1.0) so `|b|` stays sampler-friendly. The ILT 2×2 with weight 4.0 compiles but shows `|b|=4.5` — closer to freeze; fine for exact tiny checks, risky as a sampling demo.

---

## What Paul should try next in Observatory notepad

Paste this file into the Thermodynamic Program notepad, then **Preflight → Compile**:

**`/workspace/gibbs-observatory/sweeps/workloads/codon_opt_tiny.yaml`**

That is the highest-value real app that already COMPILED. After Apply, sample in THRML and read decoded synonym indices.  
Follow-ups: lengthen the peptide one AA at a time; or open `placement_8x8_exclusion.yaml` for a mediator-free packing demo; avoid pasting `dense_qubo_clique20.yaml` / `highk_biome_infeasible.yaml` unless you want to *watch* a gate refusal.

---

## Standing honesty

All of the above is **CPU-side `tsu` compile** (and Observatory’s **JAX/THRML** sampler if you Apply). It does **not** mean a Z1 die executed these programs. Label every demo: *JAX/THRML simulation — not Extropic silicon.*
