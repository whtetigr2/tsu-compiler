# Thermodynamic Programs shelf

Curated **runnable** YAML Thermodynamic Programs for Gibbs Observatory.
Each file is a real `tsu` workload spec. Successful ideal-first compiles land
under `receipts/prog_<id>/` (target **z1**).

**Honesty:** software compile + JAX/THRML simulation only — **not** Extropic silicon.

## How Paul opens one tonight

### Path A — File → Open receipt… (fastest)

1. Start Observatory (launcher or `uvicorn` + frontend).
2. Menu **☰ → File → Open receipt…**
3. Pick any `prog_*` id (e.g. `prog_codon_opt_tiny`, `prog_number_partition`, `prog_ecology_lotka_lite`).
4. Inspect gates / spins / connectivity, then **Run** (THRML sample).

Receipts are also listed under **☰ → File → Open Extropic example…** (examples shelf surfaces every on-disk receipt, including `prog_*`).

### Path B — Program notepad (edit → Compile → Apply → Run)

1. Open **Program notepad** (left nav).
2. Paste or load YAML from this folder, e.g.:
   - `programs/number_partition.yaml`
   - `programs/codon_opt_tiny.yaml`
   - `programs/ecology_lotka_lite.yaml`
3. **Preflight** (optional) → **Compile** (ideal-first, target z1).
4. On **COMPILED**, **Apply** loads `receipts/notepad` (or your chosen id) into the sampler.
5. **Run THRML**.

To force a stable shelf receipt id from Python:

```python
PYTHONPATH=/workspace/gibbs-observatory:/workspace/tsu-compiler-review
from pathlib import Path
from backend.app.program_service import compile_program
text = Path("programs/number_partition.yaml").read_text()
compile_program(text, target="z1", receipt_id="prog_number_partition")
```

Re-run the whole shelf: `programs/_compile_shelf.py`.

## Best 3 to try first

| Priority | Receipt | Why |
|----------|---------|-----|
| 1 | `prog_codon_opt_tiny` | Extropic SEQ class; mediated; iconic demo |
| 2 | `prog_number_partition` | Classic Ising; finance/scheduling sibling; n=12 → 3D state space OK |
| 3 | `prog_ecology_lotka_lite` | Extropic ecology vibe; 4×4; no mediators; clean graph |

Honorable mentions: `prog_jobshop_tiny`, `prog_market_binary_factors`, `prog_seq_design_longer` (8-codon scale-up, COMPILED).

## 3D state-space reminder

Observatory **State space** PCA-3D is for **n ≤ 16** physical nodes. Larger receipts (`placement_8x8`, `mrf_denoise_8x8`, `terrain_layout_6x6`, `seq_design_longer`) refuse 3D and show the 2D sample cloud instead — by design.

## Catalog

See `CATALOG.json` (id, title, domain, verdict, n_nodes, mediators, one-line why).
Plain-language sitrep: `THERMODYNAMIC_PROGRAMS_SITREP.md`.

## What's not on this shelf

- **Dense K20 / high-k biome** — HARDWARE (degree > 16); kept under `sweeps/workloads/` as stress fails.
- **Continuous Torx diffusion / ODEs / full Lotka–Volterra flows** — Torx-native, not expressible in current tsu Ising YAML. See SITREP.
