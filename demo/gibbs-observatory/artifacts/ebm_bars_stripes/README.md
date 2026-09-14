# EBM bars-and-stripes artifacts (RBM)

**Honesty:** THRML/JAX SIM · trained RBM · not Extropic silicon · not Z1T

## Diagnosis (pairwise failed)
Pairwise grid Ising cannot represent bars∪stripes: bars want row ferro +
uncorrelated vertical; stripes the opposite. Averaged moments → mild ferro
everywhere → blobs. Old CD L1≈0.33 was a false success.

## Dataset
Classic bars-and-stripes on **4×4** (16 visibles). Positive = pure
horizontal bars OR pure vertical stripes. Train set size: 90.

## Model
Restricted Boltzmann Machine: **16 visible +
12 hidden**, couplings **only V↔H** (bipartite).
Exported as Ising on all 28 spins.

## Train
CD/PCD with numpy block-Gibbs. Final CD moment L1 ≈ **0.4946**
in 0.541s (500 epochs).
**pure_rate = 0.938** (gate ≥ 0.9);
mean bar_stripe_score = 0.984 (gate ≥ 0.95).

## Files
- `W.npy` / `a.npy` / `b.npy` — RBM weights and biases
- `J.npy` / `h.npy` / `J_edge.npy` / `edges.json` — Ising export (vis then hid)
- `train_log.json` — loss curve + pure_rate
- `data_examples.png` / `samples_after_train.png`
- Program YAML: `../ebm_bars_stripes.yaml`
- Receipt: `../../receipts/prog_ebm_bars_stripes/`
