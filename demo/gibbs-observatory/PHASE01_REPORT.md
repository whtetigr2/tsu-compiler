# Phase 0 + Phase 1 report — Gibbs Observatory

**Date:** 2026-09-13  
**Tree:** `/workspace/gibbs-observatory`

## Outcome

MVP evolved into an Nsight-style **compiled-program inspector**. Hero path is receipt inspect + live THRML. Worlds demoted to drawer/menu.

## What works

### Backend
- `backend/app/receipt_loader.py` — loads receipt dirs; normalized payload (verdict, encoding, β/FIXED, kernel, gates, spins world/mediators, edges/weights, frontier, connectivity, ESS honesty). Does not invent fields.
- `GET /api/receipts`, `GET /api/receipts/{id}`
- Existing health / presets / sample / ws/stream kept
- `reset` accepts `receipt_id`; engine samples from receipt `program.json` Ising (nodes, edges, weights, biases, chromatic blocks)
- If a receipt cannot drive THRML, falls back to `lattice2d` with banner `sampling fallback: preset (receipt graph not yet wired)`

### Frontend
- Shell: ☰ File/View/Worlds/Help · program bar · left rail · main stage · right rail (gates, sampler stats, claim badges, language tip) · footer Run/Pause/Step/Seed/Speed + CPU/thrml + no silicon
- Sci | Prog glossary toggle
- Real views: Overview, Spins (world + mediator strip + chromatic pulse), Couplings (|J| stroke, ferro/antiferro), Connectivity / Fabric Tax, Scope
- Stubs: Schedule, Residuals, State space, Program notepad; Worlds drawer placeholder

### Receipt `small`
- Verdict **COMPILED**, encoding `domain_wall`, β **1.0 FIXED**, kernel `chromatic_block_gibbs`
- Gates: degree, coupling_cap, field_cap (assumed), node_budget, colouring — all passed
- 192 spins (128 world + 64 mediators), 576 edges — **THRML wired for real** (not fallback)

## Stubbed (Phase 2/3)

| Item | Phase |
|------|-------|
| Schedule block timeline | 2 |
| Residuals heatmap | 2 |
| Extropic examples shelf (codon_opt, …) | 2 |
| Interactive Fabric Tax edge→mediator | 2 |
| Program notepad compile/apply | 3 |
| State space 3D (small-n only) | 3 |
| Snapshot export | 4 |

## Schema gaps found in receipts

1. **`metrics.json` `mediators: 0`** while `program.json` / `passes.mediation` list **64** mediators. Loader prefers `program.mediator_nodes` and notes the mismatch in connectivity notes.
2. **No logical edge list** — only `workload.json` counts (`variables`, `logical_interactions`). Connectivity view shows counts + honest note.
3. **`formulation.json`** is construct summaries, not a graph IR usable for sampling.
4. **Placement coords** are shared/overlapping for domain-wall bit pairs (same cell → same xy); UI uses them for world layout + separate mediator strip.
5. **ESS / TV / energy_scale** often string `"unavailable: …"` — surfaced honestly, never as fake numbers.
6. Some gate limits (`field_cap`) marked `assumed: true` — shown with an “assumed” badge.

## Exact run commands

```bash
cd /workspace/gibbs-observatory
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# other terminal
cd /workspace/gibbs-observatory/frontend && npm run dev
# open http://127.0.0.1:5173
```

```bash
PYTHONPATH=. pytest backend/tests -q   # 15 passed
cd frontend && npm run build           # ok
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/receipts
curl -s http://127.0.0.1:8000/api/receipts/small | head -c 200
```

## Quality bar

- `pytest backend/tests -q` → **15 passed**
- `npm run build` → **success**
- Live curl: health ok; receipts lists `small`; `small` inspect returns COMPILED + gates; receipt reset samples 192 spins without fallback banner

## Files touched (summary)

**New:** `backend/app/receipt_loader.py`, `backend/tests/test_receipt_loader.py`, frontend shell/view components under `frontend/src/components/`, `frontend/src/lib/glossary.ts`, `PHASE01_REPORT.md`

**Updated:** `backend/app/main.py`, `sampler_engine.py`, `graph_presets.py`, `frontend/src/App.tsx`, `types.ts`, `hooks/useGibbsSocket.ts`, `SpinField.tsx`, `index.css`, `README.md`, `STATUS.md`
