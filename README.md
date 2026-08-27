# tsu — a compiler for thermodynamic sampling units

Describe a constrained computational problem, and this compiles it into a sampling
program that provably fits a documented hardware target — or tells you exactly why
it does not, and which representation to try next.

**No hardware claim.** The entire compiler runs on host CPU. Only the block-Gibbs
sampling loop would ever execute on a TSU, and it has not.

## Install

    pip install -e .

Requires Python 3.11+, and an interpreter with `thrml` and `extro-torx` available.

## Use

    tsu inspect   specs/toy.yaml
    tsu compile   specs/toy.yaml --target z1 --out out/toy
    tsu visualize out/toy --out out/toy.html
    tsu replay    out/toy

## Guarantees

- **Every compile runs the `ideal` control first.** A hardware verdict is only
  reported when a logically-valid program failed a hardware constraint.
- **Verification never fabricates.** Absent a reference, a field reads
  `unavailable` with the reason.
- **Targets carry provenance.** `degree` cites `F-14`; `max_abs_coupling` is marked
  `assumed`, because it is this project's working value and not a published figure.
- **No workload-specific code path exists.** Workloads are spec files.
