# Phase 2 report — Gibbs Observatory

**Date:** 2026-09-13  
**Tree:** `/workspace/gibbs-observatory`

## Outcome

Overload rack delivered: Schedule + Residuals views, Extropic/curated examples shelf, interactive Fabric Tax mediator callouts, and WebSocket reconnect / sticky-error hardening. Receipt-backed THRML for `small` and `elev_band` unchanged in spirit (still real sampling).

## Deliverables

### 1. Schedule view (real)
- `ScheduleView` reads receipt `schedule` / `blocks` (colour 0 & 1 sizes + index lists).
- Timeline cards pulse the live `active_block` from WS batches while streaming.
- Honest note: active block is step-parity from the engine, not a per-sweep THRML observer.

### 2. Residuals view (MVP)
- No residual matrix in receipts → `has_receipt_residual_matrix: false`.
- **Derived** v0 bars/heatmap cells from:
  - `physical_edges − logical_interactions`
  - `physical_nodes − logical_variables` (+ mediator count when present)
  - gate headroom (`limit − measured`) for degree / coupling_cap / field_cap / node_budget
- Verification TV strings surfaced as **unavailable** (never invented numbers).
- UI labels derived metrics with a `derived` chip.

### 3. Extropic / curated examples shelf
| Id | Status | Notes |
|----|--------|-------|
| `small` | ready | Default mediated toy (192 spins) |
| `elev_band` | ready | Copied read-only from Lattice `demo/receipts/elev_band` (64 spins, THRML-ready) |
| `codon_opt` | stub | `receipts/codon_opt/README.md` only — “receipt not packaged yet” |

- Menu **File → Open Extropic example…** and ReceiptPicker shelf section.
- `GET /api/examples` returns catalog + packaged/stub status.
- `l0` / `l1` not copied (empty `program.json` / HARDWARE, not THRML-ready).

### 4. Interactive Fabric Tax
- Backend `derive_fabric_tax`: degree-2 mediators → world pair map from `program.json` edges (`derived: true`).
- Connectivity view: click mediated pair or physical edge touching a mediator → mediator callout.
- `small`: 64 derived pairs; `elev_band`: 0 (unmediated bipartite).

### 5. WebSocket harden (`useGibbsSocket` + `/ws/stream`)
- Auto-reconnect with exponential backoff (cap 8s).
- Status: `connecting` | `open` | `reconnecting` | `closed`.
- Avoid sticky vague “WebSocket error” (browsers fire `onerror` before `onclose` on HMR/proxy blips).
- Pause / reset / successful graph message clears transport sticky text; dismiss (×) control in stage header.
- Backend sends a short `status` hello on accept; sampler errors prefixed `sampler:`.

## Standing prohibitions (kept)
- No silicon claims; mediated β FIXED; no invented receipt residual fields; ESS/TV honesty preserved.

## Quality

```
PYTHONPATH=. pytest backend/tests -q   # 25 passed
cd frontend && npm run build           # success
```

## How to run

```bash
cd /workspace/gibbs-observatory
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# other terminal
cd /workspace/gibbs-observatory/frontend && npm run dev
# http://127.0.0.1:5173
```

```bash
curl -s http://127.0.0.1:8000/api/examples
curl -s http://127.0.0.1:8000/api/receipts/elev_band | head -c 300
```

## Files touched (summary)

**New:** `frontend/src/components/views/ScheduleView.tsx`, `ResidualsView.tsx`, `backend/tests/test_phase02.py`, `receipts/elev_band/**`, `receipts/codon_opt/README.md`, `PHASE02_REPORT.md`

**Updated:** `backend/app/receipt_loader.py` (fabric_tax, residuals, schedule, examples_shelf), `backend/app/main.py` (`/api/examples`, WS hello, v0.3.0), `frontend/src/hooks/useGibbsSocket.ts`, `App.tsx`, `types.ts`, `ConnectivityView.tsx`, `ReceiptPicker.tsx`, `TopMenu.tsx`, `LeftNav.tsx`, `index.css`, `README.md`, `STATUS.md`, `GIBBS_OBSERVATORY_PLAN.md`

## Phase 3 gaps (remaining)
- Program notepad → preflight/compile/apply against `tsu`
- State-space 3D for small-n only; large = projected sample cloud + honest refuse
- Snapshot export / claim-hygiene polish (Phase 4)
- Optional: richer Scope heatmaps; wire logical edge list if compiler starts emitting it
