# Extropic pack — packaging status

**Date:** 2026-09-14  
**Sync target (PaulPC):** `Documents\tsu-compiler\demo\extropic-pack\`  
**Box sources:** `/workspace/extropic-pack/` (+ Observatory at `/workspace/gibbs-observatory/`)

---

## Ownership split

| Owner | Owns | Does not own |
|-------|------|--------------|
| **Claude (VSCode / tsu-compiler)** | Compiler rubric, verify path, `out/extropic-verify/`, `EXTROPIC_VERIFICATION.md`, gate ledger / B5 spine on `tsu`, spike re-verify | Observatory UI polish, demo click script prose, this pack’s README narrative |
| **Grok (TSU Research Bot)** | Observatory demo script, Extropic-facing pack README, packaging checklist, Fabric Tax / h_max **story lock** in pack docs | Changing `tsu` compile physics or verify scripts |
| **Paul** | Final sync to PaulPC, Extropic send decision, any live Model Chat facilitation | — |

Lead story **locked** (Model Chat): h_max ask → Fabric Tax table → B5 closed form → SIM/THRML honesty.

---

## Deliverable checklist

### Pack folder (`/workspace/extropic-pack/` → PaulPC `demo\extropic-pack\`)

| Artifact | Owner | Status |
|----------|-------|--------|
| `README.md` — Extropic one-pager (h_max first, Fabric Tax, B5, Observatory, receipts) | Grok | **DONE** (2026-09-14) |
| `OBSERVATORY_DEMO.md` — click script **H1→H2→H3** | Grok | **DONE** (mirrored to `gibbs-observatory/OBSERVATORY_DEMO.md`) |
| `HERO_PROGRAMS.md` — H1/H2/H3 + Claude codon placeholders | Grok | **DONE** (2026-09-14) |
| `PACKAGING_STATUS.md` — this file | Grok | **DONE** |

### Compiler verify (Claude)

| Artifact | Owner | Status |
|----------|-------|--------|
| `tsu-compiler/out/extropic-verify/` | Claude | Assume maintained — **confirm path on PaulPC before send** |
| `EXTROPIC_VERIFICATION.md` (spike 3147→5025, gates, P=10/P=40) | Claude | Assume maintained — link from pack README |
| B5 post-mediate \|J\| re-gate wired / evidenced | Claude | Rubric item — cite in verify MD |
| Sourced vs assumed `field_cap` / h_max ask language | Claude + pack README | Pack asks Extropic; compiler must keep `assumed: true` until sourced |

### Observatory shelf (shared)

| Artifact | Status |
|----------|--------|
| `receipts/prog_codon_opt_tiny` | Packaged — **H1** Fabric Tax opener |
| Full spike receipt on Observatory shelf | Optional — open if present; else cite verify pack numbers only |
| `receipts/prog_placement_8x8_exclusion` + decode gallery | Packaged — **H2**; see `programs/placement_8x8_exclusion/DECODE.md` |
| `receipts/prog_mrf_denoise_8x8` + before/after | Packaged — **H3 AI lead**; see `programs/mrf_denoise_8x8/DECODE.md` |
| `receipts/prog_ebm_bars_stripes` | Packaged — **baseline appendix only** (pure_rate≈0.9375); not walkthrough lead |
| Honesty badge / claim hygiene | Shipped in Observatory UI |

---

## Claim hygiene (must stay green)

- [x] Pack README leads with **h_max ask**, not silicon energy
- [x] Fabric Tax framed as **codon-family ~1.6×**, control **1.00×** — not universal Z1
- [x] B5 bound stated as compile-spine closed form ≈ **5.6534**, not a device joule claim
- [x] Explicit **SIM / THRML — not silicon**
- [ ] Claude confirm verify artifacts path still `out/extropic-verify` on current `tsu-compiler` tree before Extropic send
- [x] H2/H3 decode artifacts + HERO_PROGRAMS + demo path H1→H2→H3 (Grok 2026-09-14)
- [ ] Paul sync box → PaulPC `demo\extropic-pack\` (+ observatory heroes)
- [ ] Optional: attach or symlink spike receipt / verify tarball beside pack

---

## Out of scope for this pack

- Thermalizers package parity / drop-in claims  
- Silicon execution or energy-product figures  
- Replacing Alloy Lab gift narrative inside Observatory product docs (codon is **demo path for Extropic pack**, not a rewrite of Alloy as Observatory gift)

---

*When Claude finishes verify polish, tick the confirm row above and bump the Date line.*

## Hard path split (anti-clobber) — 2026-09-14

Agreed in Model Chat with Claude:

| Writer | May edit |
|--------|----------|
| **Grok** | `demo/extropic-pack/**`, `demo/gibbs-observatory/**` only |
| **Claude** | `src/**`, `tests/**`, `audit/**`, `out/**`, README, LICENSE, NOTICE, pyproject |
| **Neither** | edits the other's tree; request cross-tree changes in Model Chat |

`out/extropic-verify` is **tracked in git** (was wholesale-gitignored under `out/` until this session — pack must not assume a clone already has it).
