# Observatory demo script — Extropic-facing click path

**Honesty badge (must stay visible):** JAX/THRML simulation — **not** Extropic silicon. No energy / watt claims.  
**Hero path (this pack):** **H1 codon_tiny → H2 placement → H3 MRF denoise**  
**Not the lead:** bars-and-stripes (baseline appendix only). Alloy is the original Observatory gift — optional, not Extropic identity.

Mirror on box: `/workspace/gibbs-observatory/OBSERVATORY_DEMO.md`.  
Trio write-up: [`HERO_PROGRAMS.md`](HERO_PROGRAMS.md).

---

## 0. Start Observatory

**Windows (PaulPC):** double-click `GibbsObservatory.exe` if built, or:

```powershell
cd Documents\tsu-compiler\demo\gibbs-observatory
.\.venv\Scripts\activate
$env:PYTHONPATH="."
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
# other terminal:
cd frontend; npm run dev
```

**Box:**

```bash
cd /workspace/gibbs-observatory
source .venv/bin/activate
PYTHONPATH=. uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
# other terminal:
cd frontend && npm run dev
```

Open **http://127.0.0.1:5173** — confirm amber honesty badge.

Optional notepad compile: `PYTHONPATH=.:../tsu-compiler` or `TSU_ROOT` pointing at the `tsu` package root.

---

## 1. H1 — Codon tiny (Fabric Tax opener)

1. ☰ → **File → Open receipt…** (or **Open Extropic example…**).
2. Open **`prog_codon_opt_tiny`** — packaged SEQ/codon class; mediators → Fabric Tax.
3. If a full-spike receipt is on the shelf, you may inspect it and say: *same family; live sample may be tiny-only if spike is too large*.
4. Program bar: verdict **COMPILED**, kernel `chromatic_block_gibbs`, β **FIXED** when mediated.
5. **Overview** (~10s): gate traffic lights; world vs mediator census; Fabric Tax accent.

**Do not** invent `program.json` for the old `codon_opt` stub. Cite spike numbers from `out/extropic-verify/` (Claude).

### Connectivity — Fabric Tax

1. Left rail → **Connectivity**.
2. Compare **Logical (pre-mediate)** vs **Physical (post-mediate)**.
3. Click a mediated pair; read the **Fabric Tax** callout.
4. Talking point (lock): bipartite control (`thrml_docs_chain`) is **1.00×**; codon family sits ~**1.6×** (tiny **31→53 ≈ 1.71×** … spike **3147→5025 = 1.60×**) — **not** a universal Z1 constant.

### Couplings — dual-panel

1. Left rail → **Couplings**.
2. Dual-panel: **World** \| **Fabric Tax · mediators**.
3. Stroke ∝ \|J\|; green ferro / red antiferro; **dashed** = world–mediator bridges.

### Run THRML (tiny)

1. Footer → **Run** / **Step**; chromatic pulse; energy / mag update.
2. Footer meta: **CPU / thrml · no silicon**.
3. Optional: **Save snapshot**.

---

## 2. H2 — Placement 8×8 exclusion (packing / keep-out)

1. Open receipt **`prog_placement_8x8_exclusion`**.
2. Overview: **COMPILED**, 64 nodes, **0 mediators**, Fabric Tax **1.00×**, deg 4.
3. Run THRML; show occupancy-style spin view if available.
4. Decode story (say out loud): *valid hard-core layout = no two occupants share an edge*.
5. Point at gallery / metrics:
   - `artifacts/placement_8x8_exclusion/gallery_thrml.png`
   - `programs/placement_8x8_exclusion/DECODE.md`
   - Live batch `valid_hardcore_rate` / `exclusion_violation_rate` (or shelf `task_validity ≈ 0.25`)
6. Re-run offline if needed:

```powershell
python scripts\sample_placement_exclusion.py
```

---

## 3. H3 — MRF denoise 8×8 (**AI lead**)

1. Open receipt **`prog_mrf_denoise_8x8`**.
2. Overview: **COMPILED**, bipartite smoothness prior, **0 mediators**.
3. Show **before/after** hero (clean \| noisy \| THRML denoised):
   - `hero/denoise/hero_before_after.png`
   - `programs/mrf_denoise_8x8/DECODE.md`
4. Talking point: *shelf YAML is the prior; data unary from the noisy image is attached offline — then THRML samples the posterior.*
5. Decode metric: Hamming gain (example SIM: **14 → 10**, gain **4**).
6. **Why not bars-and-stripes here:** denoise is the AI/Z1T-shaped story; bars-stripes is the train→compile purity baseline only.

```powershell
python scripts\sample_mrf_denoise.py
```

---

## 4. Appendix only — EBM bars-and-stripes (baseline)

**Skip in the default Extropic path.** Use if asked “do you have a train→compile loop?”

1. Open **`prog_ebm_bars_stripes`** (or EBM Lab).
2. `pure_rate ≈ 0.9375` vs gate **0.90** — evidence in `artifacts/ebm_bars_stripes/` + `EBM_BARS_STRIPES.md`.
3. Still SIM/THRML — not silicon, not Z1T.

---

## 5. What to say / not say

| Say | Do not say |
|-----|------------|
| Compiled to Z1-**shaped** fabric; THRML samples the receipt | “Ran on Z1” / “Thermalizers-complete” |
| Only **ASSUMED** limit in the spike ledger is **h_max / field_cap** | Invent a sourced h_max |
| Fabric Tax ~1.6× on deg-12 codon family; 1.00× on bipartite control | “Z1 always costs 1.6× spins” |
| H3 denoise is the AI hero; bars-stripes is baseline purity | Lead with bars-and-stripes |
| B5: post-mediate \|J\| ≤ ln(cosh 12)/2 ≈ 5.6534 before `place()` | Silicon energy / joule product claims |

---

## Timing (≈3–5 minutes)

| t | Beat |
|---|------|
| 0–60s | **H1** Open `prog_codon_opt_tiny` → Overview → Connectivity Fabric Tax → Couplings dual-panel → short Run |
| 60–150s | **H2** Open `prog_placement_8x8_exclusion` → Run → decode gallery / valid-rate |
| 150–240s | **H3** Open `prog_mrf_denoise_8x8` → before/after hero → Hamming gain |
| optional | Bars-and-stripes appendix **or** alloy Distribution Lab (original gift, not Extropic identity) |

Full product walkthrough: `gibbs-observatory/WALKTHROUGH.md`. Packaging: [`PACKAGING_STATUS.md`](PACKAGING_STATUS.md). Heroes: [`HERO_PROGRAMS.md`](HERO_PROGRAMS.md).
