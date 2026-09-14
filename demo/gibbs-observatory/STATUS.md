# Gibbs Observatory — STATUS

**Date:** 2026-09-14  
**Phase:** 0 + 1 + 2 + 3 + 4 complete — **v1 Observatory complete** (plan phases)  
**Version:** 0.6.0 — **visual v0.6 ThermoLith-class shell**

## What works

- **Visual v0.6 ThermoLith-class shell** — workflow stepper, Experiment left rail, hero stage, Sampler state right rail; phosphor tokens (`#0a0b0c` / `#7ec8c0`). See `VISUAL_RESTYLE_NOTES.md`.
- Real **THRML 0.1.4** + **JAX** sampling with chromatic 2-color blocks.
- **Receipt loader** + `GET /api/receipts`, `GET /api/receipts/{id}`, `GET /api/examples`, `GET /api/receipts/{id}/spec`.
- **Receipt-backed THRML** from `program.json`; β FIXED when mediated.
- Observatory **IA shell**: menu, program bar, left/right rails, footer; Worlds demoted.
- Views: Overview (Fabric Tax accent), Spins, Couplings, Connectivity (Fabric Tax), Schedule, Residuals, Scope, **State space**, **Program notepad**.
- **Thermodynamic Program** notepad → `tsu` preflight / ideal-first compile / apply (`TSU_ROOT` or sibling `tsu-compiler` / `tsu-compiler-review`).
- State space: 3D PCA for n≤16; large n → refuse + 2D sample cloud.
- Examples shelf: `small`, `elev_band` ready; `codon_opt` stub.
- Sci | Prog language toggle.
- **Claim hygiene** panel (Help + right rail) + standing prohibitions; live badges.
- **Snapshot export** (PNG stage + JSON receipt slice); `POST /api/snapshot`, `GET /api/claim-hygiene`.
- **Guided walkthrough** (first-visit + Help restart) + Help tabs (About / Walkthrough / UI map / Receipts / Claim hygiene).
- WebSocket reconnect; pause clears sticky transport errors.
- Honest label: **JAX/THRML simulation — not Extropic silicon**.

## Verification

```
PYTHONPATH=. pytest backend/tests -q
# optional notepad: PYTHONPATH=.:../tsu-compiler pytest backend/tests -q
cd frontend && npm run build
curl /api/health /api/claim-hygiene /api/receipts /api/examples
```

See `WALKTHROUGH.md`, `HELP_WALKTHROUGH_NOTES.md`.

## Beyond plan v1 (known leftover)

- `codon_opt` stub until a real receipt is packaged
- Active-block remains step-parity visual cue (not a per-sweep THRML observer)
- No stored logical edge list in receipts — Fabric Tax pairs are **derived**
- Notepad compile is validated on toy-class specs; large mediated lattices may be slow/fail (errors shown)
- Worlds save/load still demoted stubs; Theme locked dark

## Thermodynamic Programs shelf (2026-09-14)

- Curated runnable YAML at `programs/` (**16 COMPILED** → `receipts/prog_*`).
- New: number_partition, knapsack_tiny, jobshop_tiny, ecology_lotka_lite, market_binary_factors, seq_design_longer, sat_3tiny.
- Promoted: codon_opt_tiny, placement, ILT, roster, maxcut, MIMO, floorplan, MRF, terrain.
- See `programs/README.md`, `CATALOG.json`, `THERMODYNAMIC_PROGRAMS_SITREP.md`.
- Examples shelf lists `prog_*` curated entries; File→Open receipt sees them.

## Z1 workload compile sweeps (2026-09-14)

- Real-world ladder (codon_opt_tiny, placement exclusion, ILT mask, roster, Max-Cut, MIMO, floorplan, MRF, terrain, +2 stress fails): **9 COMPILED / 2 HARDWARE** onto Z1 via `tsu` — see `sweeps/DISCOVERY_SITREP.md` and `sweeps/SWEEP_RESULTS.json`.

## Launcher (2026-09-14)

- Windows `GibbsObservatory.exe` one-click setup+run (PyInstaller). See `GITHUB_ARTIFACT.md`, `launcher/`.
- Binary gitignored; ship as GitHub Release asset. Source launcher in git.
