# GitHub artifact: GibbsObservatory.exe

## What it is

Windows one-click launcher for a **cloned** Gibbs Observatory tree. It is **not** a full offline bundle of Python/Node/JAX — those stay on the machine PATH. The exe orchestrates setup + start.

## Clone and go

1. Clone this repository (or extract a release source zip).
2. Download **`GibbsObservatory.exe`** from the GitHub Release assets into the **repo root** (same folder as `backend/` and `frontend/`), *or* build it locally (below).
3. Double-click `GibbsObservatory.exe`.
4. First run: creates `.venv`, pip installs, npm installs (can take several minutes), starts API + UI, opens the browser.
5. Later runs skip installs when `.venv` / `node_modules` already exist.

Requires on the PC:

- Python 3.10+ on PATH (or `py -3`)
- Node.js LTS + npm on PATH

## Build (maintainers / CI)

```powershell
powershell -ExecutionPolicy Bypass -File launcher\build_windows_exe.ps1
```

Outputs:

- `dist\GibbsObservatory.exe` — attach this to the GitHub Release
- copy at repo root for local double-click (gitignored)

## Do not commit the binary

`GibbsObservatory.exe`, `dist/`, and `build/` are gitignored. Ship the exe as a **Release asset**, keep source (`launcher/`) in git.

## Ports (anti-collision)

This build opens **http://127.0.0.1:5188** (API **:8088**), not 5173/8000, so it will not land on a lithography / ThermoLith demo already bound to the common Vite port. Browser opens only after /api/health reports service=gibbs-observatory.
