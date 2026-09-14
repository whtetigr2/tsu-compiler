# Gibbs Observatory — build plan (from Lattice)

**Status:** APPROVED 2026-09-13 — full web UI (Vite/React) from the start. Paul: go full Observatory, ok waiting.  
**Date:** 2026-09-13  
**Author:** Grok Bot  
**Source app:** `demo/lattice_app.py` + siblings  
**Compiler:** `src/tsu/`  
**Product name:** **Gibbs Observatory** (Nsight-for-thermo-sampling)  
**Not the product name:** World Studio / world generator (demoted)

---

## 0. One-sentence mission

**Inspect a compiled thermodynamic sampling program** — lattice, spins, couplings, schedule, residuals, THRML live draws — with Extropic public workloads as known-good drop-ins and a notepad for a user **Thermodynamic Program** (Gill’s term: a stochastic program aimed at a sampler, not “thermodynamic programming” as a research slogan).

---

## 1. What’s already real (keep; re-label)

| Lattice today | Observatory role |
|---|---|
| Pipeline strip + COMPILED verdict (`passes.json`) | **Compile receipt header** |
| Frontier deg/\|J\| headroom | **Capacity / gate HUD** |
| Live lattice + mediator strip | **Physical spin view** |
| Decoded world (valid-only) | **Decode skin** (optional view, not hero) |
| Pins / clamps | **Conditioning controls** |
| Scope: M, energy hist, local field, heatmap | **Scope rack** (fix C-1 τ before starring ACF) |
| Verification PASS chips | **Gate ledger** |
| THRML chromatic block Gibbs, CPU footnote | **Honest runtime** |
| World Studio Generate / 64×64 terrain | **Demote** → hamburger / “Worlds” drawer |

Evidence this is not fake: live session on box showed draw #1620, ~24.7% valid, 192 spins (128+64 mediators), gates PASS, β fixed.

---

## 2. Information architecture (overload on purpose)

```
┌─ Menu (☰) ─────────────────────────────────────────────────────────┐
│ File: Open receipt… · Open Extropic example… · Save snapshot       │
│ View: Language Sci|Prog · Theme · Layout density                   │
│ Worlds: World Studio… (demoted) · Saved worlds…                    │
│ Help: What is this? · Claim hygiene · Standing prohibitions        │
└────────────────────────────────────────────────────────────────────┘

┌─ PROGRAM BAR ──────────────────────────────────────────────────────┐
│ receipt path · encoding · β (FIXED if mediated) · kernel · verdict │
│ Extropic example chip OR “custom Thermodynamic Program”            │
└────────────────────────────────────────────────────────────────────┘

┌─ LEFT RAIL (nav) ─┬─ MAIN STAGE (tabbed) ──────────┬─ RIGHT RAIL ──┐
│ Overview          │ [active view]                  │ Gate ledger   │
│ Spins             │                                │ Sampler stats │
│ Couplings         │                                │ Claim badges  │
│ Connectivity      │                                │ Language tip  │
│ Schedule          │                                │               │
│ State space       │                                │               │
│ Residuals         │                                │               │
│ Scope             │                                │               │
│ Program notepad   │                                │               │
│ Worlds (drawer)   │                                │               │
└───────────────────┴────────────────────────────────┴───────────────┘

FOOTER: Pause · Seed · Step · Speed · Save snapshot · CPU/thrml · no silicon
```

### Main stage views (each is visual AF)

1. **Overview** — compact “mission control”: live decode OR spin chessboard + 5 sparkline strips + gate traffic lights.  
2. **Spins** — physical lattice: world vs mediator layers, hover field/bias, click pin. Chromatic V1/V2 pulse while sampling.  
3. **Couplings** — edge graph overlaid on lattice; stroke weight/color = |J|; sign ferro/antiferro; cap proximity glow.  
4. **Connectivity** — logical graph (pre-mediate) vs physical (post-mediate); Fabric Tax: click same-parity edge → show mediator. Bipartite before/after.  
5. **Schedule** — 2-colour block timeline; which spins update when; halo/Markov-blanket callout (SPR N-033).  
6. **State space (3D)** — for *small* exact models (n≤12–16): 3D embedding of configs (PCA/MDS on spin vectors or energy landscape surface over 2 order params). For large models: **refuse with reason** + show 2D projection of recent samples only (no fake full 2^N).  
7. **Residuals** — connectivity residual heatmap (handoff §E); TV/current if transport program (N-026).  
8. **Scope** — existing plots, C-1 fixed, ESS only when shaped.  
9. **Program notepad** — editable Thermodynamic Program (YAML/`tsu` spec or constrained DSL), **Compile preview** (preflight only), **Load into sampler** only after ideal-first compile succeeds. Label UI copy carefully: “Thermodynamic Program” + footnote citing Gill / Extropic usage, not “thermodynamic programming.”  
10. **Worlds** (hamburger) — existing World Studio / save-load, unchanged physics, off the hero path.

### Language toggle

| Mode | Labels |
|------|--------|
| **Scientific** | β, J, b, χ, ESS, TV, Onsager β_c (orienting), mediators, chromatic blocks |
| **Programmer** | temperature, coupling, bias, colour count, effective samples, total variation, schedule phase, hidden spins |

Same numbers; glossary tooltips both ways.

---

## 3. Extropic known-good examples (drop-in shelf)

Ship as curated receipts + specs (not live download required):

| Example | Source | Why |
|---------|--------|-----|
| Toy grid (current `receipts/small`) | local compile | Fast default |
| `codon_opt` tiny | `Documents\codon_opt` + `out\extropic-verify` | Paper workload, small |
| `codon_opt` full spike (read-only place viz if too heavy to live-sample) | EXTROPIC_VERIFICATION | 3147→5025 mediated story |
| Occupancy / exclusion (Placement Lab kernel) | if packaged as spec | Fitted class |
| THRML docs `grid_embed` chain | if in verify pack | Vendor docs path |

UI: **Examples → Extropic public…** loads receipt + locks claim badges to “software / documented Z1 caps / no silicon.”

---

## 4. Thermodynamic Program notepad (spec)

- Buffer: YAML workload (`tsu.spec`) or side-by-side “intent notes” markdown (not executed).  
- Actions: **Preflight** · **Compile (ideal-first)** · **Apply** · **Revert**.  
- Output panel: gate table, mediator count, bipartite, errors — same as CLI `tsuc preflight` / `compile`.  
- Forbidden: silent apply without receipt; β slider on mediated models; silicon language.

---

## 5. Phased execution (approve before code)

### Phase 0 — Hygiene (web shell) (½–1 day)
- Identity: Gibbs Observatory (already on MVP); lock standing-prohibitions banner.
- Shell IA: ☰ menu, program bar, left rail, right rail; Worlds demoted to menu drawer (placeholder OK).
- Ship curated receipt shelf pointing at Lattice `receipts/small` (+ stubs for Extropic examples).
- Fix VERDICT **C-1** (ACF/τ) before Scope is featured; RP-1 sim out of git.
**Exit:** browser opens as Observatory against a real receipt; Worlds not on hero path.

**Done (2026-09-13):** Phase 0+1 implemented in `/workspace/gibbs-observatory` — see `PHASE01_REPORT.md` / `STATUS.md`.

### Phase 1 — Hero stage (web) (2–3 days)
- Receipt-backed THRML: load `program.json` / formulation / gates from receipt dir (not only random presets).
- Spins view: chessboard + mediator strip + chromatic V1/V2 pulse while streaming.
- Couplings overlay (|J| stroke, ferro/antiferro).
- Connectivity / Fabric Tax: logical vs physical bipartite; mediator callout.
- Sci/Prog language toggle (same numbers).
**Exit:** demo compile-receipt → live spins → mediators without Worlds.

**Done (2026-09-13):** receipt `small` COMPILED → live THRML 192 spins (128+64 mediators); Worlds drawer only.

### Phase 2 — Overload rack (2–3 days)
- Schedule view.  
- Residuals heatmap MVP (even if metric v0).  
- Scope rack polish + heatmaps.  
- Extropic examples shelf (codon_opt tiny + toy).  
**Exit:** information-dense default layout; examples one click.

**Done (2026-09-13):** Schedule + Residuals + Fabric Tax click + examples shelf (`small`, `elev_band`, `codon_opt` stub); WS harden — see `PHASE02_REPORT.md`.

### Phase 3 — Program notepad + 3D (2–4 days)
- Notepad → preflight/compile/apply against `src/tsu`.  
- State-space 3D for small exact models only; large = projected sample cloud + honest unavailable.  
**Exit:** user can type/edit a Thermodynamic Program and watch it sample, or get a clear compile failure.

**Done (2026-09-13):** Notepad wired to `/workspace/tsu-compiler-review/tsu`; State space honest 3D/2D — see `PHASE03_REPORT.md`.

### Phase 4 — Polish / Extropic pack (1–2 days)
- Snapshot export (PNG + JSON receipt slice).  
- Claim hygiene panel.  
- Optional: Fabric Tax as default Overview.  
**Exit:** recordable demo for Extropic without world-gen in the first 10 seconds.

**Done (2026-09-13):** Snapshot + claim hygiene + Fabric Tax Overview accent + demo script — see `PHASE04_REPORT.md`. **v1 Observatory complete** for plan phases.

---

## 6. Stack decision

| Option | Pros | Cons |
|--------|------|------|
| **A. Evolve Tk `lattice_app.py`** | Fastest; already live THRML | Harder 3D / dense layout; Tk UA pain |
| **B. New web UI (Vite/React) + FastAPI sampling** like Gibbs Observatory MVP on box earlier | Visual AF, 3D (three.js), overload-friendly | Bigger lift; wire to `tsu` receipts |
| **C. Hybrid** — keep Tk sampler core; Observatory UI in browser talking to local sampler process | Best visuals + keep THRML path | Two processes |

**APPROVED stack: B — full Vite/React + FastAPI from the start** (Paul 2026-09-13).  
Existing box MVP (`/workspace/gibbs-observatory`) is the seed: keep THRML engine, rebuild UI to Lattice-receipt IA (program bar, rails, Fabric Tax, Worlds demoted). Tk Lattice remains reference + receipt source, not the product shell. No paid Origin — build locally, sync to PaulPC `demo/gibbs-observatory`.

---

## 7. Standing prohibitions (UI copy)

Never show as measured: silicon execution, energy savings, THRML timing = TSU timing, “Thermalizers-complete.”  
β mediated = FIXED. Invalid draws never rendered as worlds. ESS unavailable when contract broken.

---

## 8. Success demo script (for Extropic)

1. Open Observatory → Extropic example `codon_opt` tiny.  
2. Overview: gates PASS, mediators visible.  
3. Connectivity: 4-colour logical vs bipartite physical.  
4. Spins: live THRML draws, V1/V2 pulse.  
5. Notepad: tweak one bias → preflight → apply.  
6. Worlds *not* opened unless asked.

---

## 9. Out of scope for v1

- Real TSU silicon  
- Full Thermalizers clone  
- DOOM / Living World as default  
- Claiming 3D full state space for 192-spin models  

---

*End plan. Stack approved: full web. Execute Phases 0→1→2→3→4 in Vite/React.*
