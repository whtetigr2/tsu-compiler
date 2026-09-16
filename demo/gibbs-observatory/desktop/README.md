# Desktop packaging

Builds the Thermodynamic Workbench into a folder you can hand to someone.

## What the build produces

A folder, `dist/ThermodynamicWorkbench/`, about **405 MB**, containing the
executable and an `_internal/` directory beside it. **Ship the folder.** The
executable alone does not run.

The target machine needs **nothing installed**: no Python, no Node, no network,
no Extropic account. Windows 11 already carries the WebView2 runtime the window
uses.

That is the point of this build. The previous `GibbsObservatory.exe` was a
bootstrapper: it required Python 3.10+ and Node.js already on PATH, then
created a virtual environment, ran `pip install` and `npm install`, started two
servers and opened a browser tab. Several minutes on first run, and not
something you could send to anyone.

## Build it

```powershell
powershell -ExecutionPolicy Bypass -File demo\gibbs-observatory\desktop\build.ps1
```

Roughly 90 seconds after the frontend build. The script builds the frontend,
freezes the application, and then checks three things before reporting success.

## Why the build script has gates

PyInstaller cannot be trusted to report its own failure. During development a
build failed outright — `--clean` could not delete the previous `dist/` because
a running copy of the application held
`_internal\clr_loader\ffi\dlls\amd64\ClrLoader.dll` open — **and still exited
0**, leaving the previous binary in place. That stale binary then passed its own
self-test, because it was a working build of older source.

So the script checks, and throws rather than warns:

1. **Nothing holds the output open.** Any running instance is stopped first, and
   the build refuses if one survives.
2. **The binary is newer than the source.** The newest `.py` and `.spec`
   timestamp is recorded before building and compared after. A build that
   silently did nothing cannot pass as fresh.
3. **The frozen binary passes its own self-test.** Not the source — the
   artifact.

## The self-test

```powershell
dist\ThermodynamicWorkbench\ThermodynamicWorkbench.exe --selftest
```

Exits 0 only if all four checks pass. It opens no window, so it runs on a
machine with no display and can gate a build or a release.

```
  [PASS] compiler: preflight verdict ok, placed=True, 16 spins on ecology_lotka_lite.yaml
  [PASS] service:  compiled ecology_lotka_lite.yaml through the API service, verdict ok
  [PASS] thrml:    sampled an 8-spin bipartite model, drew (2, 8)
  [PASS] frontend: built bundle at ...\_internal\frontend\dist
```

The compiler and service checks both compile **one of Extropic's published
workloads**, shipped in `programs/`. A single check then proves the compiler is
present, that it runs, and that the bundled data files reached the bundle.

`compiler` and `service` look redundant and are not. `compiler` imports the
compiler directly; `service` goes through `program_service`, which is how the
application reaches it. A frozen build once passed the first and failed the
second: the service discovered the compiler by hunting for a directory on disk,
and a bundle has none. The application opened, served every asset, and could
not compile anything. **A check that does not travel the path the product
travels can pass while the product is broken.**

## What went wrong on the way here

Recorded because each cost real time and each would otherwise be rediscovered.

| Symptom | Cause |
|---|---|
| Window never opened; 90 seconds of silence | The port probe set `SO_REUSEADDR`. On Windows that permits binding a port another process is actively listening on, so an occupied port was reported free and uvicorn then failed to bind. |
| App opened, served everything, reported the compiler unavailable | Compiler discovery required a directory on disk. A bundle has none, and neither does a `pip install`. The import is now tried first. |
| A build "succeeded" but shipped older code | PyInstaller exited 0 after failing to clear `dist/`, because a running instance held a DLL. Gates 1 and 2 above exist for this. |

## Verified

The build was copied out of the repository, to a directory with no `src/`, no
virtual environment and no repository above it, and run there. All four
self-test checks passed, the window opened, and the application compiled
`ecology_lotka_lite.yaml` over its own HTTP API with every gate passing.

Not yet verified on a machine that has never had Python or Node installed. That
is the strongest available check and it has not been performed.

## Notes

- **Python 3.14.2.** PyInstaller 6.22.3 and pywebview 6.2.1 both handle it.
- **`--onedir`, not `--onefile`.** A onefile bundle carrying JAX unpacks several
  hundred megabytes to a temporary directory on every launch. Blender ships as a
  folder for the same reason.
- **The executable is a console application on purpose.** `--selftest` has to
  reach real stdout so a script can read it; a windowed build discards that
  silently. `main.hide_console_window()` hides the window on a normal launch.
- **`workbench.spec` is hand-written source**, not generated output. The
  `*.spec` ignore rule is overridden for it specifically.
- Simulation only. This is not Extropic silicon, and no energy or performance
  claims follow from it.
