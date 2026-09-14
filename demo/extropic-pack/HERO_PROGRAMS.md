# Hero programs — Extropic THRML-sim trio

**Date:** 2026-09-14  
**Honesty (every hero):** **SIM / THRML** (JAX + THRML block-Gibbs) — **not** Extropic silicon. **No** joule / watt / energy-product claims.  
**Path split:** Grok owns `demo/extropic-pack/**` + `demo/gibbs-observatory/**` only.

Walkthrough click path: [`OBSERVATORY_DEMO.md`](OBSERVATORY_DEMO.md)  
Pack one-pager: [`README.md`](README.md)

---

## Trio at a glance

| # | Program | Receipt | Fabric Tax | Decode metric | Depth |
|---|---------|---------|------------|---------------|-------|
| **H1** | Codon family | `prog_codon_opt_tiny` (+ verify spike) | ~**1.6×** on deg-12 family | Sampleability matrix *(Claude)* | Live tiny in Observatory; spike numbers from `out/extropic-verify/` |
| **H2** | Placement / exclusion 8×8 | `prog_placement_8x8_exclusion` | **1.00×** (0 mediators) | **% valid hard-core** / exclusion violation rate | Gallery + decode artifacts |
| **H3** | **MRF denoise 8×8 (AI lead)** | `prog_mrf_denoise_8x8` | **1.00×** | Hamming noisy→denoised gain; before/after grids | Hero before/after + gallery |
| Appendix | Bars-and-stripes RBM | `prog_ebm_bars_stripes` | **1.00×** | `pure_rate ≈ 0.9375` (gate 0.90) | Baseline train→compile only — **not** walkthrough lead |

**Intentional foil (not hero):** dense / high-k HARDWARE-class fail in sweeps — shows preflight is engineering, not cheerleading.

---

## H1 — Codon family (Fabric Tax opener)

**Problem.** Extropic’s published SEQ/codon workloads; mediators → bipartite fabric → **Fabric Tax**.

| Size | Role | Spins (logical → physical) | Fabric Tax | Live sample on PaulPC? |
|------|------|---------------------------:|-----------:|------------------------|
| `codon_tiny` / `prog_codon_opt_tiny` | Observatory live | **31 → 53** | **1.71×** | **Yes** (shelf receipt) |
| `codon_spike_200aa` | Verify pack | **481 → 764** | **1.59×** | *[PLACEHOLDER — Claude sampleability matrix]* |
| `codon_spike_full` | Verify pack flagship | **3147 → 5025** | **1.60×** | *[PLACEHOLDER — Claude; geometric place after mediation may wall on 32 GB]* |

**Cite (Claude-maintained, tracked):** `out/extropic-verify/` · `out/extropic-verify/EXTROPIC_VERIFICATION.md`  
Numbers above match pack README Fabric Tax table (control `thrml_docs_chain` = **1.00×**).

**Decode / gates.** Gate ledger on spike; only **ASSUMED** gate is `field_cap` / **h_max** — ask Extropic for numeric h_max. B5 post-mediate \|J\| bound ≈ 5.6534 at β=1, C=6.

**SIM honesty.** Observatory samples the **tiny** receipt live. Full spike may be inspect-only if RAM/placement blocks; never invent a live full-die sample.

*[Claude TODO: fill sampleability matrix — tiny / 200aa / full — ESS/τ/Gelman–Rubin or honest `unavailable` + reason.]*

---

## H2 — Placement 8×8 exclusion (chip-adjacent packing)

**Problem.** Hard-core keep-out on an 8×8 grid: no two occupants share an edge.

| | |
|--|--|
| YAML | `demo/gibbs-observatory/programs/placement_8x8_exclusion.yaml` |
| Receipt | `receipts/prog_placement_8x8_exclusion/` — **COMPILED** |
| Graph | 64 nodes · 112 edges · deg 4 · **0 mediators** · Fabric Tax **1.00×** |
| Decode doc | [`programs/placement_8x8_exclusion/DECODE.md`](../gibbs-observatory/programs/placement_8x8_exclusion/DECODE.md) |
| Script | `scripts/sample_placement_exclusion.py` |
| Gallery | `artifacts/placement_8x8_exclusion/gallery_thrml.png` · `hero/placement/` |

**Decode metric.** `valid_hardcore_rate` = fraction of THRML samples with zero occupied–occupied NN edges; `exclusion_violation_rate = 1 − valid_rate`.

**Example SIM numbers.** Live short batch ≈ **0.19** valid; shelf `verification.json` `task_validity ≈ 0.253` (longer chain). Quote the artifact you show.

---

## H3 — MRF denoise 8×8 (**AI hero lead**)

**Problem.** Binary image MRF: pairwise smoothness prior + **offline data unary** from a noisy observation → sample → recover structure.

| | |
|--|--|
| YAML | `demo/gibbs-observatory/programs/mrf_denoise_8x8.yaml` |
| Receipt | `receipts/prog_mrf_denoise_8x8/` — **COMPILED** |
| Graph | 64 nodes · ferro NN prior · **0 mediators** · Fabric Tax **1.00×** |
| Decode doc | [`programs/mrf_denoise_8x8/DECODE.md`](../gibbs-observatory/programs/mrf_denoise_8x8/DECODE.md) |
| Script | `scripts/sample_mrf_denoise.py` |
| Hero | `hero/denoise/hero_before_after.png` |

**Why lead over bars-and-stripes.** Denoise is observe→posterior→recover (AI / Z1T-shaped). Bars-and-stripes proves train→compile purity on a toy RBM — keep as **appendix**.

**Decode metric.** Hamming(clean, noisy) vs Hamming(clean, best denoised); **agreement gain** = noisy errors − denoised errors. Example SIM (blob, seed=11): **14 → 10** (gain **4**).

---

## Appendix — Bars-and-stripes (baseline only)

| | |
|--|--|
| Receipt | `prog_ebm_bars_stripes` |
| `pure_rate` | **≈ 0.9375** (gate **0.90**) |
| Doc | `demo/gibbs-observatory/EBM_BARS_STRIPES.md` |
| Artifacts | `artifacts/ebm_bars_stripes/` · EBM Lab UI |

**Do not** open this first in an Extropic walkthrough. Use when asked “do you have a train→compile AI loop?”

---

## Claim hygiene checklist

- [ ] SIM/THRML banner visible on every hero surface  
- [ ] ASSUMED gates named (`field_cap` / h_max)  
- [ ] No watts / joules / “ran on Z1”  
- [ ] Fabric Tax ~1.6× described as **deg-12 codon family**, not universal Z1 constant  
- [ ] H3 lead = denoise; bars-stripes = appendix  
- [ ] H1 spike sampleability filled by Claude or marked placeholder  

## Foil

Point at one sweep HARDWARE fail (e.g. dense clique) so preflight looks like engineering.
