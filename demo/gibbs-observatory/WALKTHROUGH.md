# Gibbs Observatory — walkthrough

Clone-and-use tour. The same steps run in-app as a spotlight coach-mark
(**Help → Start walkthrough…**). First browser visit auto-starts once
(`localStorage` key `gibbs-observatory-walkthrough-v1`).

> **JAX/THRML simulation — not Extropic silicon.** Mediated β is **FIXED**.
> No joule / energy product claims.

## 0. Boot

```bash
cd gibbs-observatory
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
# other terminal:
cd frontend && npm run dev
```

Open http://127.0.0.1:5173 — amber honesty badge in the top bar.

## 1. Mission

This app inspects a **compiled thermodynamic sampling program**, not a world
generator. Hero path: receipt → Overview → Connectivity (Fabric Tax) → Spins.
Worlds / World Studio stay demoted under ☰.

## 2. Program bar

Read the strip under the menu: receipt path, encoding, **β** (chip **FIXED**
when mediated), kernel, and verdict. **COMPILED** means gates passed and the
Ising program is THRML-ready — still software / CPU, not silicon.

## 3. Open a receipt / example

☰ → **File → Open receipt…** or **Open Extropic example…**.
Curated files live under `receipts/` (`small` default; `elev_band` packaged;
`codon_opt` stub until a real receipt ships — we do not invent `program.json`).

## 4. Left rail — Overview → Spins → Couplings → Connectivity → Schedule

| View | What to notice |
|------|----------------|
| **Overview** | Gate traffic lights, world vs mediator census, Fabric Tax accent |
| **Spins** | Live ±1 field; chromatic V1/V2 pulse while streaming |
| **Couplings** | Edge weights / biases (Sci\|Prog labels only) |
| **Connectivity** | Logical vs physical; click a mediated pair → mediator callout (Fabric Tax) |
| **Schedule** | Chromatic blocks; active-block is step-parity cue, not a per-sweep observer |

Also available: Residuals, Scope, State space (3D only for small n), Program notepad.

## 5. Run / Pause / Step

Footer transport: **Run** streams batches over the WebSocket, **Step** one draw,
**Pause** / **Reset** stop and clear sticky transport errors. Seed + speed tune
the sampler. Footer meta: `CPU / thrml` · `no silicon`.

## 6. Sci | Prog

Top-right toggle. Same numbers; glossary wording changes (β ↔ temperature scale,
J ↔ coupling, etc.).

## 7. Thermodynamic Program notepad

Left rail → **Program notepad**. Edit YAML → Preflight → ideal-first Compile →
**Apply** only when a **COMPILED** receipt exists on disk. Optional `tsu` package:

```bash
PYTHONPATH=.:../tsu-compiler   # or export TSU_ROOT=../tsu-compiler
```

## 8. Claim hygiene / no silicon

Help → **Claim hygiene…** (or right rail): standing prohibitions + live badges.
No silicon, no energy claims, mediated β FIXED, ESS only when the receipt allows.

## 9. Snapshot

☰ → **File → Save snapshot** or footer **Save snapshot** → PNG of the stage +
JSON receipt slice for demos.

## Restart the tour

- In-app: ☰ → Help → **Start walkthrough…**, or Help → Walkthrough tab → button
- Clear `localStorage` key `gibbs-observatory-walkthrough-v1` to auto-start again

## Keyboard

| Key | Action |
|-----|--------|
| Esc | Close walkthrough / drawers / Help |
| ← / → | Walkthrough Back / Next |
| Enter | Walkthrough Next / Finish |

See also [README.md](README.md) setup and [HELP_WALKTHROUGH_NOTES.md](HELP_WALKTHROUGH_NOTES.md).
