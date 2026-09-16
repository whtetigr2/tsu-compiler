# Gibbs Observatory

Nsight-style **compiled-program inspector** for Extropic-style thermodynamic / block-Gibbs sampling, driven by real **[THRML](https://docs.thrml.ai)** (+ JAX).

> **JAX/THRML simulation — not Extropic silicon**

Hero path: open a compile receipt → inspect gates / spins / couplings / connectivity → stream live THRML draws. Worlds / world-gen are demoted to a menu drawer (Phase 2+).

**New here?** Start the in-app tour (**Help → Start walkthrough…**) or read [WALKTHROUGH.md](WALKTHROUGH.md).

## Features (Phase 0–4 + help)

- **Receipt shelf:** curated receipts under `receipts/` (ships with `small`)
- **Inspect payload:** verdict, encoding, β (FIXED when mediated), kernel, gate ledger, world vs mediator spins, edges/weights, Fabric Tax logical vs physical
- **Live THRML:** receipt-backed Ising from `program.json` (nodes/edges/weights/biases/blocks); presets remain as fallback
- **UI shell:** ☰ menu · program bar · left rail · main stage · right rail · footer transport
- **Views:** Overview, Spins, Couplings, Connectivity, Schedule, Residuals, Scope, **State space**, **Program notepad**; Worlds drawer
- **Guided walkthrough** + **Help** user guide (About · Walkthrough · UI map · Receipts & sampling · Claim hygiene)
- **Examples shelf:** `small`, `elev_band` packaged; `codon_opt` stub (“receipt not packaged yet”)
- **Fabric Tax:** click mediated pair / physical edge → mediator callout (derived from program.json)
- **WebSocket:** auto-reconnect; pause clears sticky transport errors
- **Thermodynamic Program notepad:** Preflight · Compile (ideal-first) · Apply · Revert via optional sibling `tsu` package
- **State space:** 3D PCA for n≤16; large models refuse 3D + 2D sample cloud only
- **Sci | Prog** language toggle (same numbers)
- **Standing prohibitions** in Help + claim badges (no silicon, no energy claims, mediated β FIXED, ESS only when honest)
- **Claim hygiene** panel (Help tab + right rail) with live honesty flags
- **Snapshot export** — File / footer → PNG of stage + JSON receipt slice (`POST /api/snapshot`)
- Overview **Fabric Tax** accent when connectivity data exists



## Windows: one-click launcher

For a GitHub clone on Windows, double-click **`GibbsObservatory.exe`** in the repo root (build it with `launcher\build_windows_exe.ps1`, or download the release artifact).

It will:

1. Check for Python 3.10+ and Node.js/npm  
2. Create `.venv` and install `requirements.txt`  
3. `npm install` the frontend  
4. Start the API + UI and open the browser  

See [launcher/README.md](launcher/README.md) and [WALKTHROUGH.md](WALKTHROUGH.md).

## Requirements

- Python 3.10+ (tested on 3.13)
- Node.js 18+ / npm
- CPU is fine (JAX CPU)

## Setup

From the **repo root** (wherever you cloned it):

```bash
cd gibbs-observatory   # or: cd path/to/gibbs-observatory

python3 -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (cmd):  .venv\Scripts\activate.bat
# Windows (PowerShell):  .venv\Scripts\Activate.ps1

pip install -r requirements.txt

cd frontend && npm install && cd ..
```

## Run

**Terminal 1 — API (port 8088):**

```bash
cd gibbs-observatory
source .venv/bin/activate   # Windows: .venv\Scripts\activate
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8088 --reload
```

**Notepad compile:** the notepad needs an importable `tsu_compiler` package. The
package was renamed from `tsu` in tsu-compiler commit `8a8e941`; point at the
directory that *contains* `tsu_compiler/`, which for the src layout is
`.../tsu-compiler/src`. Either:

```bash
# src layout: ../../src/tsu_compiler/…
PYTHONPATH=.:../../src uvicorn backend.app.main:app --host 127.0.0.1 --port 8088 --reload
```

or set `TSU_ROOT` to that package root:

```bash
export TSU_ROOT=../tsu-compiler   # Windows: set TSU_ROOT=..\tsu-compiler
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8088 --reload
```

(A legacy checkout named `tsu-compiler-review` is also auto-discovered as a sibling.)

The port is **8088**, not 8000: the Vite proxy defaults to 8088 to avoid
colliding with other demos that use 8000. Starting the API on 8000 leaves the
UI unable to reach it.

**Terminal 2 — UI (port 5173, proxies `/api` and `/ws`):**

```bash
cd gibbs-observatory/frontend
npm run dev
```

Open **http://127.0.0.1:5173**

### Quick health check

```bash
curl -s http://127.0.0.1:8088/api/health
curl -s http://127.0.0.1:8088/api/receipts
curl -s http://127.0.0.1:8088/api/receipts/small | head -c 400
```

## Tests / build

```bash
source .venv/bin/activate
PYTHONPATH=. pytest backend/tests -q
# with notepad/tsu available:
# PYTHONPATH=.:../tsu-compiler pytest backend/tests -q

cd frontend && npm run build
```

## API

| Method | Path | Notes |
|--------|------|-------|
| GET | `/api/health` | service label |
| GET | `/api/presets` | lattice2d / chain1d / sparse |
| GET | `/api/receipts` | curated receipt list |
| GET | `/api/receipts/{id}` | normalized inspect payload |
| GET | `/api/examples` | curated examples shelf (incl. stubs) |
| GET | `/api/receipts/{id}/spec` | raw `spec.yaml` for notepad seed |
| GET | `/api/program/status` | tsu discovery |
| POST | `/api/program/preflight` | YAML → gates / bipartite / mediators |
| POST | `/api/program/compile` | ideal-first compile → `receipts/notepad` |
| POST | `/api/program/apply` | only with COMPILED receipt id |
| GET | `/api/graph` | current engine graph |
| POST | `/api/reset` | body may include `receipt_id` |
| POST | `/api/params` | live β/J/h (β locked when mediated) |
| POST | `/api/sample` | one batch |
| GET | `/api/claim-hygiene` | standing prohibitions + live badges (`?receipt_id=`) |
| POST | `/api/snapshot` | JSON receipt-slice metadata (no sim dumps / no disk write) |
| WS | `/ws/stream` | `reset` / `params` / `run` / `pause` / `step` (`receipt_id` on reset) |

## Demo script (60–90s, Extropic-facing — Phases 0–4)

Worlds is **not** required. Boot lands on curated receipt Overview. Prefer the in-app walkthrough on first visit.

1. Open **http://127.0.0.1:5173** — amber badge: *JAX/THRML simulation — not Extropic silicon*.
2. Program bar: `receipts/small`, encoding `domain_wall`, β **1.0 FIXED**, kernel `chromatic_block_gibbs`, verdict **COMPILED**.
3. **Overview** (~0–15s): gate traffic lights PASS; Fabric Tax accent if mediated; 192 spins (128 world + 64 mediators).
4. **Connectivity** (~15–30s): logical vs physical + click a mediated pair for mediator callout.
5. Hit **Run** → **Spins** (~30–50s): live THRML draws, chromatic V1/V2 pulse.
6. Right rail: gate ledger + **Claim hygiene** badges (software/THRML, no silicon, β FIXED, no energy claims).
7. **Save snapshot** (footer or File menu) → PNG + JSON receipt slice download (~50–60s).
8. Optional (~60–90s): **Program notepad** — Preflight on toy YAML; or **State space** on a small model / refuse+2D on `small`.
9. Worlds stays in ☰ / drawer — demoted; do not open unless asked.

## Receipts shelf

| Id | Status | Notes |
|----|--------|-------|
| `small` | packaged | Default mediated toy (192 spins) |
| `elev_band` | packaged | Bipartite band (64 spins), copied from Lattice demo |
| `codon_opt` | **stub** | Extropic paper workload — receipt not packaged yet |

Do not invent `program.json` for stubs. Copy real receipts read-only from Lattice when available.

## Docs

- [WALKTHROUGH.md](WALKTHROUGH.md) — clone-and-use tour (mirrors in-app walkthrough)
- [HELP_WALKTHROUGH_NOTES.md](HELP_WALKTHROUGH_NOTES.md) — what shipped for help / walkthrough
- [STATUS.md](STATUS.md) — version / verification

## Out of scope (v1)

Real TSU silicon, full Thermalizers clone, Worlds as hero, fake 3D state space for 192-spin models.

## License

MIT — see [LICENSE](LICENSE).
