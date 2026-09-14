# Phase 3 report — Gibbs Observatory

**Date:** 2026-09-13  
**Tree:** `/workspace/gibbs-observatory`

## Outcome

Thermodynamic Program notepad is real (preflight · ideal-first compile · apply · revert) wired to the importable `tsu` package at `/workspace/tsu-compiler-review/tsu`. State space view is honest: 3D PCA embedding only for small n; large models refuse 3D with reason and show a 2D sample cloud only.

## Deliverables

### 1. Thermodynamic Program notepad (real)

- UI label: **Thermodynamic Program** with Gill / Extropic footnote — **not** “thermodynamic programming”.
- Actions: **Preflight** · **Compile (ideal-first)** · **Apply** · **Revert**.
- Buffer: YAML / `tsu.spec` style; seeded from the loaded receipt’s `spec.yaml` (fallback: curated toy / small).
- Backend module: `backend/app/program_service.py`
  - Discovers `tsu` at `/workspace/tsu-compiler-review` (package dir `tsu/`, **not** empty `src/`).
  - Adds that root to `sys.path` (or use `PYTHONPATH=/workspace/tsu-compiler-review`).
  - Requires `networkx` + `sympy` in Observatory `.venv` (added to `requirements.txt`).

| Endpoint | Role |
|----------|------|
| `GET /api/program/status` | tsu discoverability |
| `POST /api/program/preflight` | YAML → gate table / bipartite / mediators (no receipt) |
| `POST /api/program/compile` | Ideal-first `compile_spec` → writes `receipts/notepad` when COMPILED |
| `POST /api/program/apply` | **Only** with existing COMPILED THRML-ready receipt id |
| `GET /api/receipts/{id}/spec` | Raw `spec.yaml` for seeding |

**Forbidden (enforced):** silent apply without receipt; no β slider on mediated models in this path; silicon language kept to the standing badge.

**Apply path:** compile must produce `receipts/notepad` with verdict `COMPILED` → Apply → frontend loads that receipt exactly like curated `small` (`selectReceipt`).

### Compile capabilities & limits

| Case | Behavior |
|------|----------|
| Tiny specs (`toy`, `adjacency_2x2_k3`) | Preflight ~ms; compile ~2–3s → COMPILED receipt |
| Preflight always | Full gate table + bipartite + mediator estimate |
| Large / non-bipartite / 8×8 mediated | May be slow or fail placement; errors / candidate reasons shown — **not** invented success |
| `extro-torx` | Optional; cross-check may be unavailable without it (compile still works) |
| Failed compile | Clear errors; Apply stays disabled / refused |

Documented in API `limits` field and this report: do not expect every Lattice-scale edit to compile quickly in this environment; notepad is validated on toy-class programs.

### 2. State space 3D (honest)

- Nav **State space** is real (`StateSpaceView`).
- **n ≤ 16:** rotatable 3D PCA embedding of recent sample vectors (canvas, lightweight — no fake full 2^N claim).
- **n > 16** (e.g. 192-spin `small`): **refuse with reason** + **2D PCA projection of recent sample cloud only**.
- PCA util: `frontend/src/lib/pca.ts`. Threshold: `STATE_SPACE_3D_MAX_SPINS = 16`.

### 3. Quality

```
PYTHONPATH=/workspace/gibbs-observatory:/workspace/tsu-compiler-review pytest backend/tests -q
# 39 passed

cd frontend && npm run build
# success
```

New tests: `backend/tests/test_phase03.py` (notepad API, apply refusal, compile/apply roundtrip, spec seed, health/tsu).

## Standing prohibitions (kept)

No silicon claims; mediated β FIXED; no silent apply; no fake full state-space for large n; ESS/TV honesty unchanged.

## How to run

```bash
cd /workspace/gibbs-observatory
source .venv/bin/activate
pip install -r requirements.txt   # includes networkx sympy
export PYTHONPATH=/workspace/gibbs-observatory:/workspace/tsu-compiler-review
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# other terminal
cd frontend && npm run dev
```

```bash
curl -s http://127.0.0.1:8000/api/program/status
curl -s -X POST http://127.0.0.1:8000/api/program/preflight \
  -H 'content-type: application/json' \
  -d '{"yaml":"name: t\nvariables:\n  a: {domain: binary}\nterms:\n  - {kind: linear, form: {a: 1.0}, weight: -0.1}\n"}'
```

## Files touched (summary)

**New:** `backend/app/program_service.py`, `backend/tests/test_phase03.py`, `frontend/src/components/views/NotepadView.tsx`, `frontend/src/components/views/StateSpaceView.tsx`, `frontend/src/lib/pca.ts`, `PHASE03_REPORT.md`

**Updated:** `backend/app/main.py` (v0.4.0 + program routes), `backend/app/receipt_loader.py` (`spec_yaml`), `frontend/src/App.tsx`, `LeftNav.tsx`, `types.ts`, `index.css`, `requirements.txt`, `.gitignore` (`receipts/notepad/`), `README.md`, `STATUS.md`, `GIBBS_OBSERVATORY_PLAN.md`

## Phase 4 leftover

- Snapshot export (PNG + JSON receipt slice)
- Claim hygiene panel polish
- Optional: Fabric Tax as default Overview
- Optional: richer Scope heatmaps; logical edge list if compiler emits it
