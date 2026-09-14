# demo — a worked example, and the receipts it produced

**The compiler is in [`src/tsu_compiler`](../src/tsu_compiler).** This directory is the workload
that was used to exercise it end to end, and the evidence that run produced.

## Why a world generator

The compiler needed a workload that was large, structured, and not chosen to
flatter it. A tile-based terrain generator qualifies: thousands of coupled binary
variables, real constraints between neighbours, and an output you can look at and
immediately tell whether it is wrong.

**It is not an argument for the hardware.** Perlin noise makes better terrain
faster on any CPU. The point was never that sampling is a good way to generate
terrain — it was that a compiler claiming to place arbitrary constrained problems
onto a fixed lattice should be made to do it, repeatedly, on something big enough
to break.

## What is here

| | |
|---|---|
| `receipts/` | Compile receipts from real runs — `gates.json`, `candidates.json`, `energy.json`, `verification.json`. This is what the top-level README means by "replayable receipts", and it is the thing worth opening first. |
| `world/` | The workload itself, as a spec-driven package. |
| `world_cli.py`, `world_walk.py` | Generate a world; walk around in one. |
| `lattice_app.py` | A desktop UI over the same pipeline, with live sampling. |
| `ess_run.py` | An honest effective-sample-size measurement, routed through `tsu.ess` so its reliability gate applies. Referenced by the live UI's ACF panel, which *refuses* to report a number rather than computing one on data the estimator's contract rules out. |
| `cascade.py`, `cascade_runs/` | A multi-scale experiment. Frozen — see `audit/` for its findings. |

The remaining `*_world.py` scripts are variants from that development: binary,
elevation, stacked, cascade. They are kept because they ran and their receipts
are real, not because each is a curated demo.

## Rendered outputs

The `.png` outputs are generated, not tracked — they were 14 MB of a 16 MB
repository. Regenerate any of them by running the script that produces it.

## Honest status

Everything here works and nothing here claims a hardware result. Every sampled
number came from `thrml` on CPU. The world generator is a worked example that
proves the compiler runs; it is not the product, and it never was.
