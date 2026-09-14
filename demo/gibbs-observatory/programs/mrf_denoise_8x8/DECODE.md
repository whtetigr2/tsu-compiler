# H3 decode — MRF denoise 8×8 (AI hero)

**Receipt:** `prog_mrf_denoise_8x8` (COMPILED) — **Extropic walkthrough lead for AI**  
**Honesty:** JAX/THRML software sim — **not** Extropic silicon. **No** energy / watt claims.

## Why this beats bars-and-stripes as the AI lead

| | **MRF denoise (lead)** | Bars-and-stripes (baseline appendix) |
|--|------------------------|-------------------------------------|
| Story | Observe → sample posterior → recover structure | Train RBM → purity on toy mixture |
| Extropic read | Z1T / vision-shaped workload | Proves train→compile loop |
| Decode | Hamming to clean; before/after grids | `pure_rate ≈ 0.9375` vs gate 0.90 |
| Receipt | `prog_mrf_denoise_8x8` | `prog_ebm_bars_stripes` |

Bars-and-stripes stays findable: `EBM_BARS_STRIPES.md`, `artifacts/ebm_bars_stripes/`,
EBM Lab UI. **Do not lead** the Extropic click path with it.

## Physics

Shelf YAML is the pairwise **smoothness prior** only (mild ferro NN, bipartite 8×8,
0 mediators). A real denoiser attaches a **data unary** from a noisy observation
**offline** — what `scripts/sample_mrf_denoise.py` does:

\[
\mathcal{E}(x) = E_{\mathrm{prior}}(x) + \text{local fields from noisy } y
\]

## Decode metrics

| Metric | Meaning |
|--------|---------|
| `hamming_noisy_vs_clean` | Bit errors in the observation |
| `hamming_best_denoised_vs_clean` | Best-of-N THRML sample vs clean |
| `agreement_gain_*` | Noisy errors − denoised errors (**positive = improved**) |
| `disagree_edge_frac_*` | NN disagreement rate (smoothness proxy) |

## How to reproduce

```powershell
cd Documents\tsu-compiler\demo\gibbs-observatory
.\.venv\Scripts\activate
python scripts\sample_mrf_denoise.py
# defaults: blob pattern, noise=0.22, data-strength=0.75, beta=2.0, n=48
```

## Artifacts

| Path | What |
|------|------|
| `before_after.png` | Clean \| Noisy \| THRML denoised |
| `before_after_grids.json` | Grids + per-sample Hamming |
| `decode_metrics.json` | Gains + compile note + baseline pointer |
| `gallery_thrml.png` | Ranked posterior samples |
| `../../hero/denoise/hero_before_after.png` | Slide hero |

### Example live decode (SIM, seed=11)

Hamming clean←noisy **14 → 10** (gain **4**); disagree-edge frac 0.36 → 0.27.
