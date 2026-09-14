# Flagship Thermodynamic Program — Paul’s edge vs Extropic public

**Honesty badge:** JAX/THRML simulation — **not** Extropic silicon.  
**Date:** 2026-09-14  
**Shelf:** `programs/` · **Receipts:** `receipts/prog_*`

## The gap Extropic doesn’t publicly ship

| Stack | What you get | What you don’t |
|-------|--------------|----------------|
| **Extropic public** | THRML codon notebook, Torx examples, sparse-transformers / Z1T | **Sampling / training** demos — not a compiled Z1 receipt inspector with live mediators |
| **Paul’s toolkit** | `tsu` compile → **receipt** → **Gibbs Observatory** (gates, **Fabric Tax**, live mediators, Sci/Prog) → notepad iterate | Silicon claims (we don’t make them) |

Public Extropic material shows how to **sample** and **train**. Observatory shows how a **compiled Z1 program** looks after mediation: bipartite fabric, gate ledger, world vs Fabric Tax mediators, chromatic block-Gibbs, live THRML on the receipt.

That compile-receipt → Fabric Tax → live THRML loop is the flagship demo.

## Hero receipt (run tonight)

**Primary:** `prog_codon_opt_tiny`  
- Source YAML: `programs/codon_opt_tiny.yaml`  
- Extropic SEQ / codon-opt class; domain_wall; **mediators present** (Fabric Tax story)  
- Already on the programs shelf and walkthrough path

**Alternate (scale-up):** `prog_seq_design_longer`  
- Source: `programs/seq_design_longer.yaml`  
- Longer codon design; still COMPILED; more world + mediators to show dual-panel viz

No separate `flagship_codon_story.yaml` — the existing codon receipts are the flagship. Re-authoring a near-duplicate would dilute the shelf.

See also: `programs/FLAGSHIP_README.md`, `programs/THERMODYNAMIC_PROGRAMS_SITREP.md`, `MEDIATOR_VIZ_NOTES.md`.

## Demo script (what Paul runs tonight)

### 1. Open the receipt
1. Start Observatory (launcher, or backend + `frontend` vite/build).
2. **☰ → File → Open receipt…**
3. Pick **`prog_codon_opt_tiny`** (or `prog_seq_design_longer`).
4. Confirm honesty badge: **JAX/THRML simulation – not Extropic silicon**.
5. Glance Overview: gate traffic lights, world vs mediator census, Fabric Tax accent.

### 2. Connectivity — click mediator callout (Fabric Tax)
1. Left nav → **Connectivity**.
2. Compare **Logical (pre-mediate)** vs **Physical (post-mediate)**.
3. Click a mediated pair or a physical edge that touches a mediator.
4. Read the **Fabric Tax** callout: which world pair was split by which auxiliary spin.

### 3. Couplings / Spins — mediators live with world
1. **Couplings** — dual-panel canvas: **World** (left) and **Fabric Tax · mediators** (right).  
   - Stroke ∝ |J|; green ferro / red antiferro.  
   - **Dashed** edges = world–mediator couplings (Fabric Tax bridges).  
   - Caption mentions Fabric Tax when mediators > 0.  
   - Never a “cold / hidden” footer strip.
2. **Spins** — same dual-panel dignity; phosphor mediator nodes; chromatic pulse on active blocks; click a mediator to select (right rail spin detail).

### 4. Run THRML sampler
1. Bottom bar → **Run** (or Step).
2. Watch chromatic pulse alternate blocks; energy / magnetization update.
3. Status stays **CPU / thrml no silicon**.

### 5. Optional: notepad tweak → recompile
1. Left nav → **Program notepad**.
2. Load / paste `programs/codon_opt_tiny.yaml` (or tweak a weight / length carefully).
3. **Preflight** → **Compile** (target z1, ideal-first).
4. On **COMPILED** → **Apply** → **Run THRML** again.
5. Re-check Connectivity Fabric Tax and Couplings dual-panel.

## Why this is the flagship

- **Compile receipt** you can open, not only a notebook cell.
- **Fabric Tax** is visible and clickable (Connectivity + Couplings/Spins).
- **Live mediators** share the canvas with world spins — equal visual dignity.
- **Sci / Prog** language toggle and gate ledger stay honest about software sim.

Extropic’s public codon / Torx / Z1T materials remain excellent for sampling and training. Observatory is the missing **inspect-the-compiled-fabric** layer on top of `tsu`.
