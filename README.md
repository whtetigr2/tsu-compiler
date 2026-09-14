# tsu — a compiler for thermodynamic sampling units

Describe a constrained computational problem, and this compiles it into a sampling
program, then checks that program against a documented hardware target — or tells
you exactly why it does not fit, and which representation to try next.

**No hardware claim.** The entire compiler runs on host CPU. Only the block-Gibbs
sampling loop would ever execute on a TSU, and it has not. Every sampled number in
this repository comes from `thrml` simulating that loop; `provenance.json` records
the JAX backend so an artifact cannot be mistaken for a silicon reading.

## What this is, and what it is not

This is a **host compiler targeting Extropic's published Z1 constraints**, emitting
programs `thrml` can sample, with receipts. It is **Thermalizers-shaped, not
Thermalizers-complete**: Extropic's own `thermalizers` is described in
arXiv:2608.01615 and has no public release, so nothing here is byte-compatible
with it, derived from it, or a substitute for it.

The two address the same class of target through different front ends —
thermalizers compiles *stochastic programs* (Torx kernels); this compiles
*declarative constraints*, or a raw Ising edge list. This exists because that one
is not public, not as a lesser version of it. When it ships, it becomes a
comparison, not a verdict on this.

## Install

    pip install -e .

Requires Python 3.11+, and an interpreter with `thrml` and `extro-torx` available.

## Use

Two questions about any model, before you commit to it:

    tsu preflight --edges model.json --out out/pf   # will it fit on the hardware?
    tsu regime    --edges model.json --out out/rg   # at what coupling should it sample?

`--edges` takes `{"nodes": n, "edges": [[i,j,w],...], "biases": [...], "beta": b}`,
so a model built anywhere goes through the same checks. `--spec` takes this
project's declarative format instead.

The full pipeline:

    tsu inspect   specs/toy.yaml
    tsu compile   specs/toy.yaml --target z1 --out out/toy
    tsu report    out/toy
    tsu replay    out/toy

## The pipeline

    encode -> lower -> analyse -> gate -> place -> mediate -> re-gate -> program -> verify

- **encode** searches representations (domain-wall vs one-hot) and keeps the
  rejected candidates as evidence rather than discarding them.
- **lower** reduces to a pairwise Ising model; spin-power reduction happens
  before the pairwise check, so an unreduced cubic term cannot pass as one.
- **analyse** measures degree, bipartiteness and a chromatic colouring.
- **gate** checks the model against the target's caps, *before* placement,
  because placement raises on a violation rather than returning one.
- **place** embeds on Z1's published offsets, trying the deterministic grid
  embed before the heuristic search; effort-exhausted is reported distinctly
  from geometrically-unreachable, because only the first is something this
  compiler can actually prove.
- **mediate** subdivides edges the bipartite lattice cannot host, preserving the
  exact marginal over the original spins.
- **re-gate** re-checks the mediated model, since mediation raises couplings.
- **verify** compares against an exact reference where one is enumerable, and
  reports `unavailable` with a reason where one is not.

## Guarantees

- **Every compile runs the `ideal` control first.** A hardware verdict is only
  reported when a logically-valid program failed a hardware constraint. A fault in
  this compiler reports as `COMPILER_ERROR`, never as "your model does not fit".
- **Verification never fabricates.** Absent a reference, a field reads
  `unavailable` with the reason. An error bar that could not be estimated prints
  why, never `0.000`.
- **Targets carry provenance, and every report says which is which.** `degree`
  (16) cites `F-14`; `max_abs_coupling` (6.0) is **sourced** to Thermalizers
  Fig. 12's cap-sweep axis; `node_budget` (269,568) cites Fig. 05 of *From One to
  One Billion*. `max_abs_bias` (6.0) is **assumed** — this project's working value,
  not a published Extropic figure — so a verdict resting only on it says so and
  offers `--allow-assumed`.
- **Gates are re-run after mediation.** Mediating a non-bipartite graph raises
  every coupling it touches (`A = acosh(exp(2b|J|))/(2b) > |J|` always), so a model
  can pass the cap and then need a coupling the hardware cannot hold. `preflight`
  predicts that from the closed form before placement runs.
- **No workload-specific code path exists.** Workloads are spec files.

## For Extropic

`EXTROPIC-NOTE.md` is the short note: what this does, what it does not claim,
and a 90-second demo path.

`out/extropic-verify/` compiles Extropic's own published `codon_opt` Ising
models — including the SARS-CoV-2 spike at 3,147 spins and degree 12, the
figures their paper reports — and records every gate result.

`out/connectivity-cost/REPORT.md` measures what Z1's bipartite lattice costs
this compiler. Their paper leaves the connectivity residual explicitly
unsimulated; this compiler has no residual to report, because placement fails
closed rather than dropping a coupling, so the cost appears as mediator spins
instead — 1.60x on the full spike. The report states plainly that this is a
different currency from their residual, not a comparison with it.

## Receipts

`audit/` holds the findings and the oracles. `audit/oracles/exact.py` enumerates
the Boltzmann distribution independently and does **not** import `src/tsu`, so it
cannot inherit a sign convention or an encoding bug from the code it checks — a
test asserts that independence on every run.
