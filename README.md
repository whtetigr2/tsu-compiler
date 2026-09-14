<h1 align='center'>tsu</h1>

<p align='center'>A compiler for thermodynamic sampling units.</p>

`tsu` takes a constrained computational problem, compiles it to an Ising model,
and answers the two questions that decide whether it can run on thermodynamic
hardware: **will it fit**, and **at what coupling should it be sampled**. When it
does not fit, it says which constraint failed, by how much, and which
representation to try instead.

Features include:

- Representation search (domain-wall vs one-hot) with rejected candidates kept as evidence
- Exact mediator insertion for graphs the hardware topology cannot host directly
- Deterministic lattice embedding, with a budgeted search as fallback
- Hardware gates carrying per-limit provenance — sourced figure vs project assumption
- Transition location by finite-size scaling, with τ, effective sample size and Gelman–Rubin
- Replayable receipts that report `unavailable` with a reason rather than a fabricated value

Compilation runs entirely on host CPU. The block-Gibbs sampling loop is executed
by [THRML](https://github.com/extropic-ai/thrml) on JAX; `provenance.json` records
the backend, so an artifact is never mistaken for a hardware measurement.

## Install

    pip install -e .

Requires Python 3.11+, with `thrml` and `extro-torx` available.

## Quick example

Any Ising model, as an edge list:

```json
{"nodes": 4, "edges": [[0,1,0.5],[1,2,0.5],[2,3,0.5]],
 "biases": [0,0,0,0], "beta": 0.4}
```

```bash
tsu preflight --edges model.json --out out/pf   # will it fit?
tsu regime    --edges model.json --out out/rg   # where should it sample?
```

`preflight` reports every gate with its measured value, its limit, and whether
that limit is a documented hardware figure or an assumption:

```
| gate             | value  | limit  | % of limit | status | note                  |
| max_abs_coupling | 0.5    | 6      | 8.3%       | ok     | |J| against the ...  |
| max_abs_bias     | 9      | 6      | 150.0%     | fail   | ... (ASSUMED)         |
```

A verdict resting only on an assumed limit says so, and offers `--allow-assumed`.

## The pipeline

    encode → lower → analyse → gate → place → mediate → re-gate → program → verify

Mediation raises every coupling it touches — `A = acosh(exp(2β|J|)) / (2β) > |J|`
for all `J ≠ 0` — so gates are re-run afterwards, and `preflight` predicts the
result from the closed form before placement runs.

A compilation that hits an unexpected fault reports `COMPILER_ERROR`, never a
hardware verdict about the user's model.

## Working from a spec

```bash
tsu inspect   specs/toy.yaml
tsu compile   specs/toy.yaml --target z1 --out out/toy
tsu report    out/toy
tsu replay    out/toy
```

Workloads are spec files. No workload-specific code path exists in the compiler,
and a test enforces it.

## Receipts

`audit/` holds the findings, the oracles, and the measurements. Every figure names
the script that produced it or is labelled unreproduced at the point of use.

`audit/oracles/exact.py` enumerates the Boltzmann distribution independently and
does not import `src/tsu`, so it cannot inherit a sign convention or an encoding
bug from the code it checks — asserted by a test on every run.

`out/extropic-verify/` compiles published `codon_opt` models, including a 3,147-spin
instance at degree 12. `out/connectivity-cost/` measures what the bipartite lattice
costs: 1.60× in mediator spins on that model, against a bipartite control at 1.00×.

## Target profiles

Hardware targets are declarative — degree, offsets, caps, node budget, schedule —
and each field records whether its value is sourced or assumed. `z1` and an
`ideal` control ship today. The gate and mediation layers are profile-driven; the
lattice placer is specific to a fixed-offset topology.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
