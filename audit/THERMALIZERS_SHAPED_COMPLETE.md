# Thermalizers-shaped complete — rubric scoring

Scored against `Grok bot Thermalizers path to completion handoff.md` (2026-09-13).
**A row is PASS only with evidence.** Where the evidence is weaker than the row
implies, it says so rather than rounding up.

**The goal is Thermalizers-SHAPED complete, not Thermalizers-complete.** Extropic's
`thermalizers` has no public release. Nothing here is byte-compatible with it,
derived from it, or a substitute for it.

## A — claim hygiene

| id | verdict | evidence | note |
|---|---|---|---|
| A1 | **PASS** | `README.md` "What this is, and what it is not" | States shaped-not-complete, and the front-end distinction: they compile Torx programs, this compiles declarative constraints or a raw edge list. |
| A2 | **PASS** | `src/tsu/target.py` | `\|J\|<=6` sourced to Fig. 12; `node_budget` 269,568 sourced to Fig. 05; `\|b\|<=6` **assumed** and flagged as such wherever it gates. |
| A3 | **PASS** | `src/tsu/passes/verify.py::to_dict` | Every field falls back to `unavailable` with a reason. An unmeasurable error bar prints why, never `0.000`. |
| A4 | **PASS** | grep over `src/` | No silicon, energy-saving, or GPU-superiority strings. |

## B — compile spine

| id | verdict | evidence | note |
|---|---|---|---|
| B1 | **PASS** | `search.py` module docstring | "Every compilation runs `ideal` FIRST." A fault in this compiler now reports `COMPILER_ERROR`, never as a hardware verdict about the user's model. |
| B2 | **PASS** | `README.md` "The pipeline" | encode → lower → analyse → gate → place → mediate → re-gate → program → verify, with what each step refuses. |
| B3 | **PASS** | `route.py::mediator_coupling`, `assert_beta_consistent` | Closed form kept; β mismatch refused. Mediators are bipartite-parity gadgets, explicitly not Torx chain embedding. |
| B4 | **PARKED** | `passes/split.py` docstring | Deliberately unwired, and the docstring names the four design questions wiring would require. A degree-exceeded compile rejects with remediations; it does not attempt a split. |
| B5 | **PASS** | `search.py::post_mediation_gate_checks`, `preflight`'s `mediated_coupling_cap` | Found independently by two external reviewers from different evidence. `A = acosh(exp(2β\|J\|))/(2β) > \|J\|` always, so a model can clear `\|J\|<=6` and then need `A>6` for any `\|J\| > ln(cosh(12))/2 = 5.6534`. Now predicted from the closed form before placement runs. |

## C — fidelity

| id | verdict | evidence | note |
|---|---|---|---|
| C1 | **PASS** | `search.py::_transport_note` + 2 mutation-proven tests | `conserve_over_edges` declares directed flow; the verification now states outright that TV does not certify a current (SPR N-026/N-027). Set only for flow workloads — the negative case is tested, so the disclosure cannot decay into boilerplate. |
| C2 | **PASS (vacuous)** | grep | Nothing in this repo claims Kawasaki or bond-Metropolis is Z1-native, because nothing mentions them. Clean by absence, not by argument. |
| C3 | **PASS** | `demo/lattice_app.py::render_acf_plot` | Refuses to report τ on a restart-flattened live trace rather than reshaping invalid data into a plausible number. |
| C4 | **PASS** | `backends/torx_backend.py` | Multi-edge returns `None` with EXP-TX4's reason; the independent control stays single-bond exact. |

## D — Extropic workload regression

| id | verdict | evidence | note |
|---|---|---|---|
| D1 | **PASS** | `audit/connectivity_cost.py`, `out/connectivity-cost/connectivity_cost.json` | Re-runs Extropic's published `codon_opt` models and re-gates each after mediation. Independently reproduces the existing pack's 3,147 → 1,878 → 5,025. |
| D2 | **PASS, with one caveat stated** | same | Every gate clears. Four clear **sourced** limits with headroom; `field_cap` clears an **assumed** one at `\|b\| = 4.05` of 6.0. If Extropic's real `h_max` is below 4.05 that PASS is wrong — the measurement is sound, the limit is ours. |
| D3 | **PASS** | `out/extropic-verify/EXTROPIC_VERIFICATION.md` | P=40 rejected by the sourced coupling cap, documented as expected behaviour under a sourced figure rather than as a defect report. |

## E — connectivity cost

| id | verdict | evidence | note |
|---|---|---|---|
| E1 | **PASS, metric reframed** | `audit/connectivity_cost.py` protocol block | This placer fails closed (`limit=0` on any unrealized edge), so its structural residual is zero on every successful compile and there is no residual to measure. The metric is what zero residual **costs**: mediator spins. |
| E2 | **PASS** | `out/connectivity-cost/REPORT.md` | 1.60× on the full spike; converges 1.71× → 1.60× across two orders of magnitude. The bipartite control pays exactly 1.00×, which is what makes the other rows measure Z1 rather than this codebase. |
| E3 | **PASS** | same report | States explicitly that Extropic has not measured this, that their residual and this tax are different currencies, and that no comparison is intended. |

## F — packaging

| id | verdict | evidence | note |
|---|---|---|---|
| F1 | **PASS** | `README.md`, `LICENSE`, `NOTICE` | Apache-2.0 with the patent grant; NOTICE states the independence explicitly. |
| F2 | **PASS** | `README.md` "For Extropic" | `EXTROPIC-NOTE.md` and both evidence packs linked — and, more to the point, actually **tracked**: `out/` was gitignored wholesale, so none of it was in the repository until this pass. |
| F3 | **PASS** | `EXTROPIC-NOTE.md` | ThermoBridge appears once, as a closed negative with its ~0.39% ceiling. |

## What is deliberately not claimed

- **No silicon.** Every sampled number comes from thrml on CPU. `provenance.json`
  records the JAX backend so an artifact cannot be mistaken for a hardware run.
- **No speed claim.** A TSU's advantage is throughput and energy, neither
  measurable without the device.
- **Not Thermalizers.** No Torx→THRML variational lowering, no Analytic
  Thermodynamic Kernels, no claim of package parity with code that is not public.
- **`\|b\| <= 6.0` is ours.** It is the one gate in the whole pack that a
  workload clears against an assumption, and the one number from Extropic that
  would settle it.
