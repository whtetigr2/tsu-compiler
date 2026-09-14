# Alloy Distribution Lab — original Observatory gift

**Honesty:** JAX/THRML software simulation — **not** Extropic silicon.  
**No energy / watt advantage claims.**  
**Date:** 2026-09-14

## Why this gift (not Extropic codon)

Extropic’s public SEQ / codon notebooks already tell a biotech story. Paul’s
**original** flagship gift is metallurgy: Z1 samples Boltzmann \(P(x)\propto e^{-\beta E}\)
on a binary alloy lattice where **unlike neighbours are preferred** — Ising’s
home turf (antiferromagnetic NN → checkerboard Bragg–Williams order).

Decode must produce a **function** (order parameter, energy, best-of-N gallery),
not just raw spins. That is the Distribution Lab.

| | Codon (demoted) | **Alloy (flagship gift)** |
|--|-----------------|---------------------------|
| Domain | Extropic SEQ | Metallurgy / phase design |
| Graph | mediated codon factors | **8×8 NN lattice, 0 mediators** |
| Story | public Extropic vibe | **Paul’s original Observatory gift** |
| UI | Spin / Fabric Tax | **Alloy Lab occupancy + \|m_s\|** |

## Physics / encoding

- **Lattice:** 8×8 square, binary sites: `0 = A` (Cu-like), `1 = B` (Zn-like).
- **Coupling:** prefer unlike neighbours (ordering alloy):
  - `product_over_edges` AA weight `+0.45` (costly)
  - `product_over_edges` BB weight `+0.45` (costly)
  - `product_over_edges` AB weight `−0.20` (mild reward)
- Mild weights avoid freeze; bipartite NN → **0 mediators**; degree 4 ≪ 16 → **COMPILED**.

### Order metrics (decode)

| Symbol | Meaning |
|--------|---------|
| \(m_s\) | Staggered order: mean \((-1)^{x+y}\, s\) with \(s=+1\) for B, \(-1\) for A |
| \(\|m_s\|\) | Order strength (≈1 checkerboard, ≈0 disordered) |
| \(c_B\) | Concentration of B |
| SRO | Fraction of NN edges that are unlike |
| \(E_\mathrm{config}\) | Sum of YAML product_over_edges weights on the occupancy |

## Artifacts

| Deliverable | Path |
|-------------|------|
| Program YAML | `programs/alloy_ordering_8x8.yaml` |
| Receipt | `receipts/prog_alloy_ordering_8x8/` |
| Backend decode | `backend/app/distribution_lab.py` |
| API | `POST /api/lab/alloy/decode`, `POST /api/lab/alloy/batch`, `GET /api/lab/alloy/info` |
| UI | Left nav → **Alloy Lab** (`AlloyLabView.tsx`) |
| Heroes | `hero/alloy/*.png` |

### Compile stats (locked)

- **Verdict:** COMPILED (Z1 gates pass)
- **Nodes:** 64 · **Mediators:** 0 · **Max degree:** 4 · **Edges:** 112
- **Encoding:** domain_wall (SELECTED)

## How Paul demos tonight (click path)

1. Start Observatory (launcher, or backend + frontend build/serve).
2. **☰ → File → Open receipt…** → pick **`prog_alloy_ordering_8x8`**.
3. App auto-opens **Alloy Lab** (or click left-rail **Alloy Lab** / banner **Open in Alloy Lab**).
4. Hero shows copper/steel **occupancy grid**; side gauges show \|m_s\|, \(c_B\), unlike-edge %, \(E_\mathrm{config}\).
5. Click **Sample batch (THRML)** → best-of-N gallery + \(m_s\) histogram.
6. Caption always reads: *Distribution Lab · binary ordering alloy · THRML sim · not silicon*.
7. Optional: show hero PNGs under `hero/alloy/` for slides.

Secondary (Fabric Tax / codon) remains available as `prog_codon_opt_tiny` but is **not** the gift narrative tonight.

## SIM honesty (say this out loud)

> This is a **software** compile + THRML block-Gibbs sim on CPU/JAX.  
> It is **not** Extropic silicon. We make **no** joule or watt claims.  
> The value is configurational sampling for alloy order / phase design, decoded into functions Paul can demo.
