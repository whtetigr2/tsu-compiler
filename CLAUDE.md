# tsu-compiler

Read this before writing any analysis code. **Most of what you are about to
hand-roll already exists here and is better than what you would write.** Every
entry below was built for this project and has been reviewed; a numpy
reimplementation is a regression, not a shortcut.

Interpreter: `PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe"`
Run from the repo root with `sys.path.insert(0, "src")`.

---

## Before you write numpy, check this table

| If you are about to work out… | Call this instead |
|---|---|
| whether a model fits the hardware | `preflight.check.preflight(ising, target)` |
| whether the contract makes variables impossible | `passes.presolve.presolve(spec)` |
| degree / |J| cap / |b| cap / colouring gates | `gates.gate_checks(ising, report, target)` |
| quantisation, precision, coupling collisions | `regime.analyse_regime(report, target, weights=…)` |
| graph shape: degree, bipartite, colour blocks | `passes.analyse.analyse(ising)` |
| R-hat | `preflight.diagnostics.r_hat(chains)` |
| ESS, autocorrelation, tau | `ess.effective_sample_size`, `ess.integrated_autocorrelation_time` |
| standard error from ESS | `preflight.diagnostics.stderr_from_ess` |
| the exact critical point | `preflight.sweep.onsager_betac(j_max)` |
| Binder cumulant / susceptibility | `preflight.sweep.binder`, `preflight.sweep.susceptibility` |
| ground truth on a small model | `audit/oracles/exact.py`: `exact_boltzmann`, `exact_energy` |
| exact distribution of a program | `backends.thrml_backend.exact_distribution` |
| sampling | `backends.thrml_backend.sample_chains` / `stream_chains` |
| an independent sampler cross-check | `backends.torx_backend.torx_cross_check` |
| mediator strength for a coupling | `passes.route.mediator_coupling(absJ, beta)` |
| beta consistency | `passes.route.assert_beta_consistent` |

`preflight(ising, target)` is the single front door. It returns gates **with
provenance** (a sourced Extropic fact and a project assumption render
differently — `max_abs_bias` is ASSUMED, not sourced), plus mediators,
placement, node budget, and the distinct-coupling count against the die's
programmable-parameter budget. It subsumes most one-off checks you would write.

Targets live in `target.PROFILES` (`z1`, and an ideal control). Build a new
`TargetProfile` for other hardware; every field is `Sourced(value, source,
quote)` and the source string is load-bearing — "assumed" must never be dressed
up as a citation.

## The compile pipeline

`spec.load_spec` → `passes.presolve.presolve` (analysis; reports what the
contract makes impossible, does not yet rewrite the model) → `passes.encode.encode` → `passes.lower.lower` →
`passes.analyse.analyse` → `passes.place.place` / `passes.route.route`
(`insert_mediators`, `split_high_degree`) → `passes.program.build_program` →
a backend. `passes.search.compile_spec` drives the whole thing.

Receipts: `receipt.write_receipt` / `load_receipt` / `replay`,
`report.render_report` / `render_explain`, `simulate.simulate`.
CLI: `tsuc compile | inspect | visualize | replay | report | explain | simulate | watch`.

## How work is judged here

**A control that cannot fail is not a control.** This is the project's most
expensive lesson, learned twice (`audit/findings/R23.md`, `R24.md`). Both times a
measurement was real, every check passed, and the check could not have detected
the defect. Before reporting a control as passed, ask what result would have
failed it.

- **Ablate.** Turn a term off. If the answer does not move, the term is not
  doing the work and the claim about it is wrong. R23 was a lattice whose
  couplings could be deleted with zero effect while the prose credited them.
- **Test where the term matters.** R24's first version tested quantisation on a
  clean input, where R23 had already established the couplings are inert — a
  true measurement of nothing.
- **Never weaken an assertion to make something pass.** Record the failure.
- **Verification never fabricates.** If a quantity could not be measured, it
  reads "unmeasured", never a guess.
- **Look at the artifact.** Render the page, print the bits, view the image.
  Several defects here were invisible in the diff and obvious on screen.

Findings live in `audit/findings/R*.md`, retractions filed next to results.

## Repo rules

- **Stage commits by explicit path. Never `git add -A`.**
- Do not touch: `receipts/EXP-G8-D2/`, `receipts/EXP-G8-D3/`,
  `figures/z1_lab_screenshot.png`, `demo/cascade.py`, `demo/cascade_runs/`.
- `demo/extropic-pack/**` belongs to Grok. Do not write there.
- `demo/gibbs-observatory/**` was Grok's too. That reservation is **suspended**
  at the owner's explicit request while Grok is unavailable: the Observatory is
  being re-shelled into the Thermodynamic Workbench, the project's main
  deliverable. See `docs/superpowers/specs/2026-09-16-thermodynamic-workbench-design.md`.
  Restore the reservation if Grok returns to that directory.
- `specs/emergence_8x8.yaml` and `pyproject-review.toml` are QWEN leftovers.
  Leave them alone.
- `out/` is gitignored with a per-directory allowlist in `.gitignore`. Add
  `!out/<dir>/` when a new evidence pack should be kept.
- The DTM repo has no licence: may run, **must not vendor**. Same instinct for
  any third-party code, permissive licence or not — this project's work should
  be its own.
- No secret values in any file or commit message.
- Extropic has real hardware. Do not dismiss what they publish as error.
- **No hardware claims.** Everything here is simulation until someone else
  checks it on silicon.
