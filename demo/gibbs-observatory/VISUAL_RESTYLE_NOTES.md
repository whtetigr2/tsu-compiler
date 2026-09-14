# Visual restyle — ThermoLith-class shell (v0.6)

**Date:** 2026-09-14  
**Trigger:** Paul: “gibbs observatory this aint” vs ThermoLith screenshot  
**Scope:** Re-skin + re-layout under `/workspace/gibbs-observatory` only. No backend changes. No silicon / energy superiority claims.

## Design target

Match the **product-feeling** ThermoLith bar:

- Dark tokens closer to Placement Lab / ThermoLith: `bg #0a0b0c`, `surface #121416`, `elevated #1a1d21`, phosphor `#7ec8c0`
- Sparse layout, large hit targets, tight tracking, mono metrics
- Top brand + tagline + **THRML SIMULATOR / active|ready|offline** pill
- Workflow stepper: **Receipt → Inspect → Sample → Compare**
- Left **Experiment** rail (receipt, β FIXED, Run THRML sampler) + Inspect nav
- Center **hero stage** (compiled program graph in rounded panel)
- Right **Sampler state** card (READY/streaming, E / mag / sweep / active colour, selected-spin inspector)
- Honest device line: **CPU / THRML · silicon Unavailable**

Reference shot: `demo-shots/thermolith-reference.png`.

## What changed

| Area | Change |
|------|--------|
| `frontend/src/index.css` | Full token rewrite; new shell / workflow / experiment / hero / sampler classes; legacy view styles remapped to phosphor palette |
| `App.tsx` | Shell: TopMenu → WorkflowStrip → ProgramBar (quiet) → Experiment \| Hero \| Sampler → slim Footer |
| `TopMenu.tsx` | Brand mark + tagline; status pill replaces warn banner in the header |
| `WorkflowStrip.tsx` | **New** — headline + 4-step stepper mapped to Observatory flows |
| `LeftNav.tsx` | Experiment controls + Inspect view list + device card |
| `RightRail.tsx` | Sampler state primary; Gate ledger + Claim hygiene **collapsed** |
| `OverviewView.tsx` | Hero-first canvas; side mission strip only on narrow layouts |
| `SpinField.tsx` | Labeled ±1 nodes for `n ≤ 32` non-grid; phosphor edges; click-to-select; large-n heatmap/receipt layout kept readable |
| `FooterBar.tsx` | Slim mirror of transport + seed/speed; honest device meta |
| `Walkthrough.tsx` | Copy/targets updated for Experiment rail + hero framing |

## Kept working

- Receipts, WS sampling, all views (Overview / Spins / Couplings / Connectivity / Schedule / Residuals / Scope / StateSpace / Notepad)
- Walkthrough, Help, snapshots, claim hygiene
- Sci \| Prog, Worlds drawer (demoted), program notepad / tsu path unchanged

## How to run

```bash
cd /workspace/gibbs-observatory
# backend (example)
.venv/bin/uvicorn backend.app:app --reload --port 8000
# frontend
cd frontend && npm run dev
# or production build
cd frontend && npm run build && npm run preview
```

## Build

```
cd frontend && npm run build
# ✓ tsc -b && vite build (2026-09-14)
```

## Remaining gaps vs ThermoLith

1. **No tile / lithography metaphor** — Observatory is receipt-inspect, not inverse-litho tile placement; headline and stepper are mapped, not cloned.
2. **192-spin receipts** stay heatmap / world+mediator strip — labeled graph nodes only for small `n` (≤32).
3. **Selected-spin inspector** is basic (state + bias); no local-field / P(+1) physics model beyond receipt biases.
4. **View nav still long** in the left Inspect list — ThermoLith has almost no secondary views; we kept feature surface.
5. **Program meta strip** remains (quiet) for walkthrough + encoding/kernel; ThermoLith folds more into the hero chrome.
6. **Footer still present** as a slim transport mirror; ThermoLith puts primary CTA only on the left.
7. **Fonts** remain IBM Plex Sans/Mono (close enough; not Inter / custom ThermoLith faces).
8. **No animated graph layout** — positions come from receipt / backend; we do not re-layout like a force graph.

## Honesty

- Status pill: THRML simulator only  
- Device card + footer: CPU / THRML · silicon Unavailable  
- No joule / energy-product claims; energy metrics are simulation traces  
