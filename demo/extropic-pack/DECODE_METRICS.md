# Decode metrics — H2 / H3 (index)

**Honesty:** SIM/THRML only — not Extropic silicon. No energy/watt claims.

| Hero | Receipt | Metric | Doc / artifacts |
|------|---------|--------|-----------------|
| **H2** | `prog_placement_8x8_exclusion` | `valid_hardcore_rate` / `exclusion_violation_rate` | `../gibbs-observatory/programs/placement_8x8_exclusion/DECODE.md` · `artifacts/placement_8x8_exclusion/` |
| **H3** | `prog_mrf_denoise_8x8` | Hamming gain noisy→denoised; before/after grids | `../gibbs-observatory/programs/mrf_denoise_8x8/DECODE.md` · `hero/denoise/hero_before_after.png` |
| Appendix | `prog_ebm_bars_stripes` | `pure_rate ≈ 0.9375` (gate 0.90) | `../gibbs-observatory/EBM_BARS_STRIPES.md` |

Scripts (Observatory venv):

```text
scripts/sample_placement_exclusion.py
scripts/sample_mrf_denoise.py
```

Full trio narrative: [`HERO_PROGRAMS.md`](HERO_PROGRAMS.md).
