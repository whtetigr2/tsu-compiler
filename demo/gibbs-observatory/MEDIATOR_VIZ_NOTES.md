# Mediator visualization notes

**File:** `frontend/src/components/SpinField.tsx` (receipt / mediated layout)  
**Also:** `CouplingsView.tsx`, `SpinsView.tsx`, glossary `mediators` tip

## What was wrong
- Mediators lived in a tiny footer strip labeled **“cold / hidden”**
- Coupling edges were filtered to world–world (`wset`) only
- Fabric Tax wasn’t taught in the Couplings/Spins canvas

## What we ship now
- **Dual-panel** canvas: **World** (left) and **Fabric Tax · mediators** (right), equal chrome dignity
- Copy: **“Fabric Tax · mediators (auxiliary spins for bipartite Z1 fabric)”** — never “cold / hidden”
- When `edgeMode === 'coupling'`: draw **all** edges that touch mediators; stroke by |J| / ferro–antiferro; **dashed** world–mediator bridges
- Mediator nodes: phosphor fill (`#7ec8c0` / `#1a2a2c`), chromatic pulse, clickable/selectable like world spins
- Captions/legend mention Fabric Tax when `mediators > 0`

## Non-goals
- No silicon claims
- THRML honesty badge unchanged
