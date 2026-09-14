# tsu — a compiler for Z1-shaped thermodynamic hardware

**Paul Shaver · 2026-09-13 · host-side only. No silicon claim.**

Extropic ships **Torx** (stochastic programs) and **THRML** (near-metal block Gibbs). The public compiler that answers *will this fit Z1, and if not why* is still a whitepaper (Thermalizers). This repo is that missing middle, running today on CPU/JAX.

## What it does

Describe a constrained problem as a spec. The compiler:

1. Runs an **`ideal` control first** — hardware verdicts are only issued if the spec is logically sound.
2. **Searches encodings** (domain-wall vs one-hot), keeps rejected candidates as evidence.
3. Lowers to pairwise Ising, **inserts mediators** when the graph is not bipartite (`A = arccosh(exp(2 β |J|)) / (2β)`), refuses β mismatch.
4. **Places** on Z1's published degree-16 offsets (rotations of (1,0), (2,1), (2,3), (4,1)) — grid embed before heuristic; search-exhausted ≠ unplaceable.
5. Emits a chromatic `SamplingProgram` + a **replayable receipt** with sourced vs assumed caps (`|J|≤6` documented; `|b|≤6` marked assumed).

`tsu preflight` is the edit-loop tool: bipartite + degree + |J| + budget in milliseconds. Non-bipartite placement is the coffee-vs-afternoon line.

## Live run (2026-09-13)

```
python -m tsu inspect   specs/toy.yaml
python -m tsu preflight --spec specs/toy.yaml --out out/preflight-toy
python -m tsu preflight --spec specs/too_dense.yaml --out out/preflight-dense
python -m tsu compile   specs/toy.yaml --target z1 --out out/toy-z1
```

| run | result |
|---|---|
| inspect toy | 4 nodes, 3 edges, deg 2, bipartite, 0 mediators |
| preflight toy | **OK**, path `grid_embed`, 66 µs |
| preflight too_dense | **FAIL** — deg 19 > 16, not bipartite, remediations named |
| compile toy → Z1 | **COMPILED** in ~18 s including THRML verify |

Receipt `out/toy-z1/`:

- encodings: **domain_wall SELECTED** (4 p-bits); **one_hot VIABLE_NOT_SELECTED** (6 p-bits, 1 mediator)
- verification: `energy_tv = 0`, `codeword_violation_rate = 0`, `execution_tv = 0.010` (noise floor 0.015), task validity 0.90
- torx cross-check: honestly **unavailable** (PISING is exact only for a single bond)

## What is already proven (internal audit, 872 tests)

Energy IR / oracle / THRML agree to ~1e-16. Pre-registered χ² vs the real sampler (TV 0.012, N=10k). Mediator marginals ~1e-16. Domain-wall decode exact over full bitspace k=2..6. Independent oracle vs Wolfram to 11 sig figs.

Live ACF no longer fabricates τ on restart-flattened traces. Simulation writes do not overwrite git-tracked receipts.

## What this is not

- Not Thermalizers (no Torx→THRML variational compiler).
- Not a frozen-transformer energy win. That path is closed (~0.39% ceiling); see ThermoBridge 2026-08-05 memo.
- Not silicon. Host CPU + THRML/JAX only.

## Why Extropic should care

This is the developer contract Thermalizers will need: **fit / encode / mediate / place / verify / explain**, with receipts that refuse to invent unavailable fields. The visual skins (lattice worlds, Gibbs Observatory) sit on this spine. The science to send is the **tool**, not another softmax paper.

## 90-second demo

1. `preflight` on `toy.yaml` — green, grid_embed.
2. `preflight` on `too_dense.yaml` — red, degree + bipartite named, remediations.
3. `compile --target z1` on `toy.yaml` — open `candidates.json` + `verification.json`.
4. Optional: `demo/Run LATTICE.bat` for the world/clamp UI.

Requires an interpreter with `thrml` and `extro-torx` available; see the top-level README for install.
