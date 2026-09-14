# H2 decode — hard-core placement 8×8

**Receipt:** `prog_placement_8x8_exclusion` (COMPILED)  
**Honesty:** JAX/THRML software sim — **not** Extropic silicon. **No** energy / watt claims.

## Problem

Place occupants on an 8×8 site grid so that **no two occupied cells share an edge**
(classical hard-core lattice gas / component keep-out). YAML encodes exclusion as a
positive `product_over_edges` weight on `(1,1)`.

## Decode metric

| Symbol | Definition |
|--------|------------|
| **valid hard-core layout** | Zero occupied–occupied nearest-neighbour edges |
| `valid_hardcore_rate` | Fraction of THRML samples that are valid |
| `exclusion_violation_rate` | `1 − valid_hardcore_rate` |
| `mean_edge_violations` | Mean count of forbidden occupied–occupied edges |
| `mean_occupancy` | Mean number of occupied sites (density × 64) |

Contract rule (compile-time): `forbid_value_pair_over_edges` on `(1,1)`.

## How to reproduce (PaulPC / Observatory venv)

```powershell
cd Documents\tsu-compiler\demo\gibbs-observatory
.\.venv\Scripts\activate
python scripts\sample_placement_exclusion.py
# optional: --n 64 --beta 1.0 --warmup 120 --seed 0
```

```bash
cd /workspace/gibbs-observatory   # box mirror
.venv/bin/python scripts/sample_placement_exclusion.py
```

## Artifacts

| Path | What |
|------|------|
| `decode_metrics.json` | Live-batch rates + compile note |
| `gallery_thrml.png` | Ranked sample strip (valid first; one VIOL contrast) |
| `sample_grids.json` | Raw 8×8 grids + per-sample flags |
| `../../hero/placement/` | Slide-ready hero frames |
| `../../artifacts/placement_8x8_exclusion/` | Canonical copy |
| `../../receipts/prog_placement_8x8_exclusion/` | COMPILED receipt |

### Typical numbers (SIM)

- **Live short batch** (this script, N=64, β=1): `valid_hardcore_rate ≈ 0.19`  
- **Shelf verification** (`verification.json` longer chain): `task_validity ≈ 0.253`  
- Fabric Tax **1.00×** (bipartite NN, 0 mediators, deg 4)

Both are software estimates — quote the file you opened; do not invent silicon rates.
