# EBM Lab — bars-and-stripes RBM (first AI-shaped thermo demo)

**Honesty:** THRML/JAX SIM · trained **RBM** · **not** Extropic silicon · **not** Z1T.  
**No energy / watt advantage claims.**  
**Date:** 2026-09-14

## Diagnosis — pairwise Ising failed (locked)

Paul confirmed: after pairwise PCD train, `samples_after_train.png` showed **blobs**,
not bars/stripes. That is expected, not a bug in CD:

- Horizontal **bars** want strong **row** ferro + uncorrelated vertical neighbours.
- Vertical **stripes** want the opposite.
- A single pairwise grid averages those moments → mild ferro **everywhere** →
  Ising blobs / domains.
- CD moment L1 ≈ 0.33 looked like “success” but only matched wrong moments.
  **Moment match ≠ generative success** on this mixture.

## Fix — Restricted Boltzmann Machine

Classic model for this dataset: **RBM** with couplings **only** visible↔hidden.

| | Pairwise (failed) | RBM (this) |
|--|-------------------|------------|
| Visibles | 6×6 = 36 | **4×4 = 16** |
| Hiddens | 0 | **12** (tunable 8–12) |
| Edges | NN grid V–V | **only V↔H** |
| Graph | bipartite grid | bipartite V–H |
| Max degree | 4 | deg(v)=n_h≤12, deg(h)=16 |
| Mediators | 0 | **0 expected** |

Energy (±1 convention, matches THRML Ising export):

\[
E(v,h) = -\sum_{ij} W_{ij}\, v_i h_j - \sum_i a_i v_i - \sum_j b_j h_j.
\]

Exported as Ising on **all** spins `[v|h]` with edges only V–H. UI decode shows
**only the visible 4×4** image.

## Hard success criteria

After train, Gibbs-sample ≥32 visibles (sample/marginalize hiddens properly):

| Gate | Threshold |
|------|-----------|
| `pure_rate` = fraction `is_pure_bars_or_stripes` | **≥ 0.40** (aim ≥0.5) |
| Mean `bar_stripe_score` (row/col purity) | **≥ 0.75** |
| `samples_after_train.png` | Visually bars/stripes like `data_examples.png` |

Do **not** ship if the pure_rate gate fails — tune lr / n_hidden / epochs / CD
steps, or fall back to structured RBM (row+col hiddens).

## Z1 shape

Bipartite V–H, max degree 16 → under Z1 cap; expect **0 mediators**.
Locked compile: `prog_ebm_bars_stripes` → COMPILED (16v+12h nodes).

## How to run

```bash
cd /workspace/gibbs-observatory
.venv/bin/python scripts/train_bars_stripes_ebm.py --epochs 200 --compile
# API (optional lite retrain / sample)
# POST /api/lab/ebm/train  { "lite": true, "epochs": 12 }
# POST /api/lab/ebm/sample { "n": 16, "beta": 1.2, "warmup": 80 }
# GET  /api/lab/ebm/info
```

Frontend: left nav **EBM Lab**, or open receipt `prog_ebm_bars_stripes` (auto-opens).

## Artifacts

| Path | What |
|------|------|
| `artifacts/ebm_bars_stripes/` | `W.npy`, `a.npy`, `b.npy`, `J.npy`, `h.npy`, `train_log.json` (incl. `pure_rate`), PNGs |
| `programs/ebm_bars_stripes/` | mirrored checkpoint + sample grids |
| `programs/ebm_bars_stripes.yaml` | V↔H products + linear fields as tsu terms |
| `receipts/prog_ebm_bars_stripes/` | COMPILED receipt |
| `backend/app/ebm_bars_stripes.py` | data, `train_rbm_cd`, decode, YAML export |
| `frontend/.../EbmLabView.tsx` | EBM Lab UI (shows pure_rate) |

## Paul demo click path

1. Start Observatory (launcher or backend + `frontend` build/serve).
2. **☰ → File → Open receipt…** → **`prog_ebm_bars_stripes`**.
3. App auto-opens **EBM Lab**.
4. See train example tiles vs model sample tiles; **pure_rate** gauge; loss sparkline.
5. **Sample (THRML)** → decode gallery (4×4 visibles); optional **Retrain lite**.
6. Say out loud: *software sim, trained RBM, not silicon, not Z1T — pairwise failed, RBM fixed it*.

## SIM limits (say this)

> This proves **train → compile → inspect** on a classic generative toy.
> Pairwise Ising could not learn bars∪stripes; the RBM can. Sampling is
> THRML/JAX on CPU. It will not beat ImageNet.

## Quality note (2026-09-14)
First RBM ship was ~0.77 pure (crosses/L mixes). Locked sweep winner: dense PCD lr=0.08, 500 epochs, CD-15, sample β=2.0 → **pure_rate ≈ 0.94**. Gate raised to 0.90.

## Role in Extropic hero trio
**Baseline appendix only** — proves train→compile purity (`pure_rate ≈ 0.9375` vs gate 0.90).
**Do not lead** the Extropic walkthrough; AI lead is **`prog_mrf_denoise_8x8`** (see `HERO_PROGRAMS.md` / `programs/mrf_denoise_8x8/DECODE.md`).
