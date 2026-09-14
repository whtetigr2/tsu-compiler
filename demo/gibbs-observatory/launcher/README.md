# Gibbs Observatory launcher

Windows one-click entry for clone-and-go:

1. Clone the repo.
2. Double-click **`GibbsObservatory.exe`** in the repo root (or build it below).
3. The launcher creates `.venv`, `pip install -r requirements.txt`, `npm install`, starts API + Vite, opens the browser.

## Requirements on the machine

- **Python 3.10+** on PATH (or `py -3`)
- **Node.js LTS** + npm on PATH

If missing, the launcher tells you how to install (`winget` / python.org / nodejs.org). It does not silently download runtimes.

## Build the `.exe` (maintainers)

On Windows, from the repo:

```powershell
powershell -ExecutionPolicy Bypass -File launcher\build_windows_exe.ps1
```

Produces `dist\GibbsObservatory.exe` and copies `GibbsObservatory.exe` to the repo root.

## Dev run without packaging

```powershell
python launcher\gibbs_observatory_launcher.py
```

## Notes

- Keep the exe next to `backend/` and `frontend/`.
- First run can take several minutes (JAX/THRML wheels + npm).
- Closing the launcher window stops the API and UI processes.
- JAX/THRML simulation — not Extropic silicon.

## Ports (important)

The launcher binds **API `:8088`** and **UI `:5188`** by default (not `8000`/`5173`), so a double-click does not open whatever lithography / ThermoLith / other TSU demo is already on the common Vite port.

It only opens the browser after `/api/health` reports `service=gibbs-observatory`.
