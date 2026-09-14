# Phase 4 report — Gibbs Observatory

**Date:** 2026-09-13  
**Tree:** `/workspace/gibbs-observatory`  
**Version:** 0.5.0

## Outcome

Plan Phase 4 polish / Extropic pack is complete. Snapshot export (PNG + JSON receipt slice), claim hygiene panel (Help + right rail), light Fabric Tax Overview accent, and demo/README script cover the full Phase 0–4 stack. **v1 Observatory complete** for plan phases.

## Deliverables

### 1. Snapshot export

- **File → Save snapshot** and footer **Save snapshot** capture:
  - PNG of the main-stage canvas (SpinField / state-space), or a fallback placard if no canvas
  - JSON receipt slice (verdict, gates, spins counts, β, encoding, kernel, timestamp, claim badges)
- Browser download only — **no sim dumps** written into `receipts/` or git
- Optional backend: `POST /api/snapshot` returns enriched metadata (`schema: gibbs-observatory.snapshot.v1`)
- Client util: `frontend/src/lib/snapshot.ts`
- `.gitignore` ignores `*.snapshot.json` and `gibbs-observatory-*.png`

### 2. Claim hygiene panel

- Help → **Claim hygiene** tab + right-rail **Claim hygiene** section
- Standing prohibitions (no silicon, no energy, THRML≠TSU timing, never Thermalizers-complete, mediated β FIXED, ESS when contract broken, invalid draws ≠ worlds, no fake 2^N)
- Live badges from receipt + honesty flags
- API: `GET /api/claim-hygiene?receipt_id=…`
- Component: `ClaimHygienePanel.tsx`

### 3. Fabric Tax Overview accent

- When connectivity / fabric_tax data exists, Overview shows a small **Fabric Tax** callout (pair count or logical/physical note) — light touch, does not replace mission control

### 4. Demo polish

- Boot still loads curated `small` on Overview (Worlds demoted) — Extropic-facing first ~10s need no Worlds
- README **60–90s demo script** updated for Phases 0–4 (examples → Overview → Fabric Tax → Spins → snapshot → notepad optional)

### 5. Version / docs

- Backend + health + frontend `package.json` → **0.5.0**
- `STATUS.md` marks v1 Observatory complete for plan phases
- This report: `PHASE04_REPORT.md`

## Quality

```
PYTHONPATH=/workspace/gibbs-observatory:/workspace/tsu-compiler-review pytest backend/tests -q
# 47 passed

cd frontend && npm run build
# success (v0.5.0)
```

New tests: `backend/tests/test_phase04.py` (suite: **47 passed**) (health version, claim-hygiene, snapshot slice honesty, API snapshot).

## Standing prohibitions (kept)

No silicon / energy / Thermalizers-complete language; mediated β FIXED; ESS honesty; no sim dumps in git receipts.

## How to run

```bash
cd /workspace/gibbs-observatory
source .venv/bin/activate
export PYTHONPATH=/workspace/gibbs-observatory:/workspace/tsu-compiler-review
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# other terminal
cd frontend && npm run dev
# open http://127.0.0.1:5173
```

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s 'http://127.0.0.1:8000/api/claim-hygiene?receipt_id=small' | head -c 400
curl -s -X POST http://127.0.0.1:8000/api/snapshot \
  -H 'content-type: application/json' \
  -d '{"receipt_id":"small","view":"overview","step":1}'
```

## Files touched (summary)

**New:** `backend/app/snapshot.py`, `backend/tests/test_phase04.py`, `frontend/src/lib/snapshot.ts`, `frontend/src/components/ClaimHygienePanel.tsx`, `PHASE04_REPORT.md`

**Updated:** `backend/app/main.py` (0.5.0 + routes), `backend/app/receipt_loader.py` (badges), `frontend/src/App.tsx`, `TopMenu.tsx`, `FooterBar.tsx`, `HelpModal.tsx`, `RightRail.tsx`, `OverviewView.tsx`, `index.css`, `package.json`, `.gitignore`, `README.md`, `STATUS.md`, `GIBBS_OBSERVATORY_PLAN.md`

## Leftover beyond plan v1

- `codon_opt` remains a stub until a real receipt is packaged
- Active-block remains step-parity visual cue (not a per-sweep THRML observer)
- Fabric Tax pairs stay **derived** (no stored logical edge list in receipts)
- Notepad compile validated on toy-class specs; large mediated lattices may be slow/fail
- Optional richer Scope heatmaps; Worlds save/load still demoted stubs
