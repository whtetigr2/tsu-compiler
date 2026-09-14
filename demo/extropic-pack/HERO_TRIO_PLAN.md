# Hero Trio plan — THRML-sim real workloads for Extropic

**Status:** GO from Paul 2026-09-14 — dispatched to Model Chat; H3 amended (denoise lead)  
**Date:** 2026-09-14  
**Goal:** Preemptively show Extropic three *deep* real-world Thermodynamic Programs under THRML/SIM (not more toy YAMLs), with decode metrics + honest receipts.

## Hard path split (do not break)

| Writer | May edit |
|--------|----------|
| **Grok** | `demo/extropic-pack/**`, `demo/gibbs-observatory/**` only |
| **Claude** | `src/**`, `tests/**`, `audit/**`, `out/**`, README, LICENSE, NOTICE, pyproject |
| Cross-tree asks | Model Chat only — never silent edits |

## What already exists (do not rebuild)

- Pack one-pager: `demo/extropic-pack/README.md` (h_max ask, Fabric Tax, B5, SIM honesty)
- Observatory shelf: 18 COMPILED programs; receipts for codon_opt_tiny, placement_8x8_exclusion, ebm_bars_stripes, …
- Claude verify pack: `out/extropic-verify/` (codon_tiny → spike_full Fabric Tax ~1.6×)
- Known wall: full-spike geometric placement after mediation (5,025 spins) unfinished on 32 GB; document, don't fake

## Hero trio (flagship narrative — not the whole shelf)

| # | Program | Why Extropic cares | Depth target (this push) |
|---|---------|--------------------|--------------------------|
| **H1** | **Codon family** | Their published language | Tiny live in Observatory + cite spike_200 / spike_full verify numbers; sample+decode on the largest codon that fits RAM honestly |
| **H2** | **Placement / exclusion 8×8** | Chip-adjacent packing | THRML sample gallery + decode (% valid hard-core / exclusion violations) |
| **H3** | **MRF / small denoiser (lead)** + bars-and-stripes (baseline) | Their AI / Z1T bet | **Lead:** `mrf_denoise_8x8` (or thin RBM denoise) with before/after decode. **Baseline:** bars-and-stripes purity 0.9375 vs 0.90 proves train→compile→sample; not the Extropic walkthrough hero |

**Keep as foil (not hero):** one HARDWARE-class fail (e.g. dense clique / high-k) already in sweeps — pack links it so preflight looks like engineering.

**Demote from Extropic lead story:** alloy flagship identity, maxcut/knapsack/sat toys, Sudoku-class CSP.

---

## Grok will do (after go)

1. **Observatory hero path** — click script in `OBSERVATORY_DEMO.md` + pack README section: H1→H2→H3 order, World|Fabric Tax for codon, decode panels for placement + RBM.
2. **Decode + sample artifacts under `demo/gibbs-observatory/`**  
   - H2: sample grids / occupancy stats + short decode note  
   - H3: refresh EBM Lab purity evidence if stale; no silicon claims  
   - H1: if Claude unlocks a larger codon sample path, wire receipt into shelf; else deep-link verify pack only
3. **Pack narrative** — `HERO_PROGRAMS.md` (or README §) with: problem statement, compile verdict, Fabric Tax, decode metric, RAM/honest limits, pointers to Claude's `out/extropic-verify/`
4. **Claim hygiene pass** — every hero page: SIM/THRML banner; ASSUMED gates named; no watts/joules
5. **Do not** touch `src/`, tests, or invent coupler-sharing maps

## Claude needs to do (after go)

1. **Codon scale-up path** — confirm which codon sizes are sampleable on PaulPC (32 GB): tiny / 200aa / full. Document wall in `EXTROPIC_VERIFICATION.md` if placement search still blocked at 5,025.
2. **THRML sample harness for heroes** — reproducible scripts under `audit/` or `out/` that emit ESS/τ/Gelman–Rubin *only when honest*; else `unavailable` + reason (existing receipt culture).
3. **H1 verify freshness** — spike_full / spike_200 summaries still match published Extropic figures; coupler_parameters SOURCED + per_edge_independent_J ASSUMED stay correct.
4. **Compiler support if decode needs it** — e.g. receipt fields for decode hooks / workload metadata; no silent physics changes without Model Chat note.
5. **Optional:** thin `examples/` pointer in root README to three heroes (Claude owns README) — Grok supplies bullets via chat.
6. **Do not** commit Observatory/Model-Chat session dumps; leave `EXTROPIC-NOTE.md` cleanup as separate hygiene (already flagged)

## Joint / Paul

| Item | Owner |
|------|--------|
| Go / no-go on dispatch | Paul |
| Final Extropic send decision | Paul |
| Model Chat facilitation | Paul + both agents |
| GPU / RAM for larger samples | Paul (JC / funding path) — plan assumes CPU 32 GB |

## Success criteria (definition of done)

- [x] Three hero programs each have: problem blurb, COMPILED receipt id, Fabric Tax (or N/A if bipartite 1.00×), **one decode metric**, SIM honesty line *(H1 codon sampleability still Claude placeholder)*
- [x] Pack README links all three + verify pack + one intentional fail *(foil noted in HERO_PROGRAMS)*
- [x] Observatory demo script walks the trio in <5 minutes (H1→H2→H3)
- [x] No new silicon / energy claims; h_max + coupler-mapping asks unchanged
- [x] Path split respected (edits only under `demo/extropic-pack` + `demo/gibbs-observatory`)

## Order of operations (after go)

1. Dispatch this plan into Model Chat (Grok posts; Claude acks / amends)
2. Claude: codon sampleability matrix + any harness gaps (blocking for H1 depth)
3. Grok: H2 + H3 decode/sample polish in parallel (unblocked)
4. Grok: pack `HERO_PROGRAMS.md` + Observatory demo update
5. Joint: claim hygiene read-through; Paul reviews before any Extropic send


## Amendment — H3 denoise vs bars-and-stripes (Paul 2026-09-14)

Bars-and-stripes matters as a **pre-registered distribution baseline** (easy purity metric; we already beat 0.90). It is *not* the most persuasive Extropic demo.

**Extropic-facing H3 lead:** small **denoising** model (`prog_mrf_denoise_8x8` already COMPILED on shelf) — before/after grids, residual energy / PSNR-style or Hamming decode. That reads as a real AI workload.

**Bars-and-stripes:** keep as appendix receipt proving the EBM train→compile loop; do not lead the walkthrough with it.

Codon "find something new": treat as a hoped-for byproduct of deeper spike/P-ramp / Fabric Tax / gate work — not a promise. Document surprises in verify + pack; never invent them.

## Out of scope this push

- Binding Model Chat to non-localhost / remote share  
- Alloy as Extropic identity  
- Inventing Z1 coupler-sharing structure  
- Full-die 269k-pbit sample on current RAM  
- Repo hygiene (`EXTROPIC-NOTE.md`) unless Paul adds it to scope

