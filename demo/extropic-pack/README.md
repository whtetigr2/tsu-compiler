# Extropic pack — Z1-shaped compile + Observatory (one-pager)

**Audience:** Extropic (or Extropic-facing reviewers)  
**Honesty:** **SIM / THRML** — compiled programs sampled with JAX + [THRML](https://docs.thrml.ai). **Not** Extropic silicon. **No** joule / watt / energy-product claims.

---

## 1. Outreach opener — ask first

We compiled Extropic’s published **`codon_spike_full`** workload onto a **Z1-shaped** fabric (degree ≤ 16, bipartite, chromatic block-Gibbs) with an honest **gate ledger**.

| Gate | Result (spike, 5025 physical spins) | Provenance |
|------|-------------------------------------|------------|
| degree | PASS (measured deg **12** ≤ 16) | F-14 / `target.py` |
| coupling_cap \|J\| | PASS (**2.8466** / sourced cap **6**) | Thermalizers Fig. 12 `"6 (Z1)"` |
| node_budget | PASS (**5025** / **269568**) | published Z1 die figure |
| colouring / bipartite | PASS after mediation | Z1 2-colour schedule |
| **field_cap / h_max** | PASS vs **assumed** limit **6** (measured \|b\| ≈ **4.05**) | **only assumed gate** |

### Ask

**What is the numeric `h_max` (field / bias cap) for Z1?**  
Until that figure is published or confirmed, our `field_cap` gate must stay marked **ASSUMED**. We are happy to re-gate the spike receipt the day a sourced number exists.

Compiler verify artifacts (Claude-maintained on `tsu-compiler`, **tracked in git** — not merely assumed on Paul’s disk):

- `out/extropic-verify/`
- `out/extropic-verify/EXTROPIC_VERIFICATION.md`

---

## 2. Fabric Tax — methodological control first

**Fabric Tax** = physical spins / logical spins after mediator insertion (bipartite parity gadgets so the coupling graph is 2-colorable under Z1’s chromatic schedule).

**Methodological anchor:** `thrml_docs_chain` is already bipartite. It **must** pay **0 mediators** and exactly **1.00×**. If that control ever pays tax, the overhead is measuring our mediation code rather than Z1’s lattice constraint — and every other row would be worthless. The audit fails loudly if the control breaks.

| Workload | Logical → physical | Fabric Tax | Notes |
|----------|-------------------:|-----------:|-------|
| **`thrml_docs_chain` (bipartite CONTROL)** | **5 → 5** | **1.00×** | paid nothing — anchors the table |
| `codon_tiny_10aa` | **31 → 53** | **1.71×** | deg-12 biochem-shaped |
| `codon_default_prefix` | **266 → 434** | **1.63×** | |
| `codon_spike_200aa` | **481 → 764** | **1.59×** | |
| `codon_spike_full` | **3147 → 5025** | **1.60×** | 1878 mediators; reproduces verify pack |

**Takeaway:** on this **degree-12 codon / biochem family**, Fabric Tax **converges ~1.6×** across two orders of magnitude (1.71 → 1.63 → 1.59 → 1.60). That is a measured family property against a bipartite target — **not** a universal Z1 constant. Dense / high-k layouts can fail degree or field gates before tax is interesting.

Observatory visualizes the tax live: Connectivity (logical vs physical + click → mediator callout) and Couplings dual-panel (**World** | **Fabric Tax · mediators**).

---

## 3. B5 — post-mediate \|J\| closed form (before `place()`)

Mediator gadgets raise effective couplings. Rubric **B5** requires a **post-mediation re-gate** on \|J\| before geometry-binding `place()`.

**General bound** (any model), with \(A(J)=\mathrm{acosh}(e^{2\beta J})/(2\beta)\) and sourced coupling cap \(C\):

\[
|J| \le \frac{\ln(\cosh(2\beta C))}{2\beta}
\]

At **β = 1**, **C = 6** (sourced Z1 \|J\| cap):

\[
|J| \le \frac{\ln(\cosh 12)}{2} \approx 5.6534
\]

The **12** inside \(\cosh\) is \(2\cdot\beta\cdot C = 2\cdot1\cdot6\) — **not** codon degree-12. That numerical coincidence is a trap; do not read the formula as “deg-12 family magic.”

**Sanity strip:** cap=4 → \(\ln(\cosh 8)/2 \approx 3.6534\); β=2, C=6 → \(\ln(\cosh 24)/4 \approx 5.8267\); at the threshold, \(A(|J|)=C\) exactly.

**Known window (P-ramp):** spike at \|J\|_max ≈ 2.8466 is far clear of 5.6534; a **P=20** row at \|J\|_max = 5.00 yields \(A \approx 5.3466\) — within one mediation step of the cap. That is where the bound first bites.

Evidence trail: `EXTROPIC_VERIFICATION.md` (B5 section + P-ramp).

---

## 4. What Gibbs Observatory shows

Nsight-style **compiled-program inspector** for Z1-shaped Ising programs.

**Extropic hero click path:** **H1** `prog_codon_opt_tiny` → **H2** `prog_placement_8x8_exclusion` → **H3** `prog_mrf_denoise_8x8` (denoise is the AI lead). Details: [`HERO_PROGRAMS.md`](HERO_PROGRAMS.md) · [`OBSERVATORY_DEMO.md`](OBSERVATORY_DEMO.md).

1. Open **H1** receipt (`prog_codon_opt_tiny`, or spike if packaged).
2. **Overview** — gate traffic lights, world vs mediator census, Fabric Tax accent.
3. **Connectivity** — logical vs physical; click mediated pair → Fabric Tax callout.
4. **Couplings** — dual-panel World | Fabric Tax · mediators.
5. **Run THRML** — live chromatic block-Gibbs (CPU / `thrml` · no silicon).
6. **H2** placement — hard-core decode (`valid_hardcore_rate` / exclusion violations).
7. **H3** MRF denoise — before/after noisy→denoised grids (AI hero).

**Baseline appendix only (not lead):** **`prog_ebm_bars_stripes`** train→compile purity (`pure_rate ≈ 0.9375`).

Repo / box path: `gibbs-observatory/` (PaulPC sync: `Documents\tsu-compiler\demo\…`).

---

## 5. How to open receipts

| Path | What |
|------|------|
| Observatory UI | ☰ → **File → Open receipt…** → `prog_codon_opt_tiny` (or spike id if present) |
| On disk (Observatory) | `gibbs-observatory/receipts/<id>/` — `gates.json`, `program.json`, `metrics.json`, `target.json`, … |
| Compiler verify pack | `tsu-compiler/out/extropic-verify/` + `EXTROPIC_VERIFICATION.md` (**tracked**) |
| This pack | `Documents\tsu-compiler\demo\extropic-pack\` |

**Read first:** `gates.json` (assumed vs sourced), then Connectivity / Couplings in Observatory — do not invent `program.json` for stubs.

---

## Standing prohibitions

- No claim that a Z1 die executed these programs.
- No THRML wall-time as TSU timing; no energy / watt advantage figures.
- Mediated β is **FIXED**; ESS only when the receipt allows an honest estimate.
- `field_cap` / `h_max` remains **ASSUMED** until Extropic publishes a number.
- Do not confuse deg-12 with the \(\cosh(12)\) argument in the B5 bound.

---

*Heroes: [`HERO_PROGRAMS.md`](HERO_PROGRAMS.md). Ownership: [`PACKAGING_STATUS.md`](PACKAGING_STATUS.md). Click script: [`OBSERVATORY_DEMO.md`](OBSERVATORY_DEMO.md).*
