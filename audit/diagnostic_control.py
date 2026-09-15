"""Can this project's sampling diagnostics actually report FAILURE?

================================ PROTOCOL ================================
SYSTEM DEFINITION   A 16x16 periodic square-lattice Ising model in ZERO FIELD,
                    swept across beta*J, at both coupling signs. The exact
                    critical point is known: Onsager's Kc = ln(1+sqrt(2))/2 =
                    0.440687 for this lattice, so the ORDERED regime is not a
                    guess -- it is where the answer is known in advance.
STATE VARIABLES     Per (sign, beta*J): R-hat of the folded scalar |m|, R-hat of
                    the signed scalar m, max R-hat over the 256 individual spins,
                    min ESS over those spins, and the count of frozen spins.
TRANSITION RULES    thrml chromatic block Gibbs via `sample_chains`, 16 chains
                    from independent random starts.
ALLOWED OPERATIONS  Sampling, and comparing three diagnostics on the SAME draws.
FORBIDDEN OPERATIONS
                    No tuning any threshold to make a row come out a given way.
                    No hardware. The coupling SIGN is not assumed from the IR
                    convention -- it is confirmed against `audit/oracles/exact.py`,
                    which does not import this package.
ASSUMPTIONS         None beyond Onsager's exact result for this lattice.
INVARIANTS          Deep in the ordered phase, 16 independently started chains
                    CANNOT all have explored both symmetry-related modes. Any
                    diagnostic reporting R-hat ~ 1.000 there is reporting health
                    it did not earn.
MEASUREMENTS        The three diagnostics above, side by side.
NULL HYPOTHESES     "Our sampling diagnostics can detect a stuck chain."
                    This is the hypothesis UNDER TEST, and the scalars FAIL it.
                    A diagnostic that has only ever returned PASS is not evidence
                    of anything; this script is where it is made to return FAIL.
SUCCESS CRITERIA    In the ordered phase the per-spin diagnostic must exceed the
                    R-hat threshold. Whether the scalars do is the finding.
FAILURE CRITERIA    If the per-spin diagnostic ALSO reported health in the
                    ordered phase, this project's sampling verdicts would be
                    worthless and that must be reported, not quietly dropped.
PROVENANCE          Onsager 1944 for Kc. Sign confirmed against the independent
                    oracle. No hardware, no Extropic workload -- this is a test
                    of OUR instruments, not of anyone's model.
SCOPE OF VALIDITY   This shows the diagnostics CAN fail and WHERE the scalars go
                    blind. It does not bound how badly they fail on other graphs.
==========================================================================

WHY THIS EXISTS. Every workload this project has ever measured came back
"sampleable". A verdict function that has never once said no has not been shown
to be capable of saying no, and reporting its passes as evidence is circular.
This script supplies the missing negative.

WHAT IT FOUND -- and it is worse than the defect that prompted it (R20).

Two ways for chains to be stuck, and the scalars miss them differently:

  FERROMAGNETIC (J > 0) below the critical temperature: the two ground states
  are all-up and all-down. Chains split between them. Signed m separates them
  and R-hat fires. The FOLDED |m| maps them onto one value and stays silent.
  This is the R20 defect exactly.

  ANTIFERROMAGNETIC (J < 0) on this bipartite lattice: the two ground states are
  the two checkerboards -- and BOTH have net magnetisation zero. So |m| is blind,
  and signed m is blind TOO. Every chain reports m ~ 0 while sitting in a
  different frozen configuration from its neighbours. Only looking at the
  individual spins reveals it.

So the lesson is stronger than "do not fold a non-symmetric order parameter". It
is that a SCALAR SUMMARY OF ANY KIND can be perfectly healthy while the chains
are catastrophically stuck, because a mean over thousands of spins has room to
hide any structure that averages out. That is why the verdict in
`audit/codon_sampleability.py` rests on every spin individually.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "audit")

import numpy as np

from tsu_compiler.backends.thrml_backend import sample_chains
from tsu_compiler.ess import effective_sample_size
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.program import build_program
from tsu_compiler.preflight.diagnostics import RHAT_THRESHOLD, r_hat

OUT = Path("out/diagnostic-control")
L = 16
N = L * L
KC = float(np.log(1 + np.sqrt(2)) / 2)   # Onsager, exact for this lattice
BETAS = (0.20, 0.44, 0.60, 0.80)
N_CHAINS, N_SAMPLES, N_WARMUP, STEPS = 16, 4000, 2000, 4


def _torus_edges(side: int):
    idx = lambda i, j: (i % side) * side + (j % side)
    return sorted({(min(idx(i, j), idx(i + di, j + dj)),
                    max(idx(i, j), idx(i + di, j + dj)))
                   for i in range(side) for j in range(side)
                   for di, dj in ((1, 0), (0, 1))
                   if idx(i, j) != idx(i + di, j + dj)})


def lattice(beta_j: float, weight: float) -> IsingModel:
    """Periodic LxL square lattice, zero field, uniform coupling."""
    e = _torus_edges(L)
    return IsingModel(nodes=tuple(f"n{i}" for i in range(N)), edges=tuple(e),
                      weights=np.full(len(e), weight), biases=np.zeros(N),
                      beta=beta_j, offset=0.0)


def confirm_sign(weight: float) -> str:
    """Name the coupling sign from the INDEPENDENT oracle, never from our own
    IR convention. A sign error here would invert the entire interpretation of
    this script, and this project has shipped a wrong-signed coupling before."""
    from oracles.exact import exact_boltzmann
    side = 4
    e = _torus_edges(side)
    states, probs = exact_boltzmann({ab: weight for ab in e},
                                    [0.0] * (side * side), 0.6)
    P = np.asarray(probs)
    S = np.array([[2 * v - 1 for v in s] for s in states])
    nn = float(np.mean([P @ (S[:, a] * S[:, b]) for a, b in e]))
    return "ferromagnetic" if nn > 0 else "antiferromagnetic"


def row(weight: float, beta_j: float) -> dict:
    model = lattice(beta_j, weight)
    rep = analyse(model)
    draws = np.asarray(sample_chains(
        build_program(model, rep), n_chains=N_CHAINS, n_samples=N_SAMPLES,
        n_warmup=N_WARMUP, steps_per_sample=STEPS, seed=0))
    mag = (2 * draws.astype(np.int8) - 1).mean(axis=2)
    rh_folded = float(r_hat(np.abs(mag)))
    rh_signed = float(r_hat(mag))

    rhats, esss, frozen, no_ess = [], [], 0, 0
    for j in range(N):
        col = (2.0 * draws[:, :, j] - 1.0).astype(np.float64)
        if col.std() == 0.0:
            frozen += 1
            continue
        rhats.append(float(r_hat(col)))
        est = effective_sample_size(col)
        if est.ess is None:
            no_ess += 1
        else:
            esss.append(est.ess)

    return {
        "beta_j": beta_j,
        "regime": "ordered" if beta_j > KC else "disordered",
        "r_hat_folded_abs_m": round(rh_folded, 4),
        "r_hat_signed_m": round(rh_signed, 4),
        "max_r_hat_per_spin": round(max(rhats), 4) if rhats else None,
        "min_ess_per_spin": round(min(esss)) if esss else None,
        "n_frozen": frozen,
        "n_ess_unavailable": no_ess,
        "per_spin_reports_failure": bool(rhats and max(rhats) > RHAT_THRESHOLD),
        "folded_scalar_reports_failure": bool(rh_folded > RHAT_THRESHOLD),
        "signed_scalar_reports_failure": bool(rh_signed > RHAT_THRESHOLD),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, failures = [], []
    for weight in (1.0, -1.0):
        sign = confirm_sign(weight)
        rows = [row(weight, b) for b in BETAS]
        blocks.append({"ir_weight": weight, "sign_per_exact_oracle": sign,
                       "rows": rows})

        # CONTROL: deep in the ordered phase the per-spin diagnostic MUST fire.
        # If it does not, every sampling verdict this project has published is
        # unsupported, and that is the finding -- not something to smooth over.
        for r in rows:
            if r["beta_j"] >= 0.60 and not r["per_spin_reports_failure"]:
                failures.append(
                    f"{sign} beta*J={r['beta_j']}: the per-spin diagnostic did "
                    f"NOT report failure in the ordered phase (max R-hat "
                    f"{r['max_r_hat_per_spin']}); the verdict cannot detect a "
                    f"stuck chain")

    (OUT / "diagnostic_control.json").write_text(json.dumps(
        {"lattice": f"{L}x{L} periodic, zero field", "n_spins": N,
         "onsager_kc": round(KC, 6), "n_chains": N_CHAINS,
         "draws": N_CHAINS * N_SAMPLES, "rhat_threshold": RHAT_THRESHOLD,
         "blocks": blocks, "control_failures": failures,
         "hardware": "none -- CPU simulation; this tests OUR instruments"},
        indent=2), encoding="utf-8")

    print(f"Onsager Kc = {KC:.6f} (exact for this lattice); {L}x{L} = {N} spins,"
          f" zero field, {N_CHAINS} chains x {N_SAMPLES} samples\n")
    for blk in blocks:
        print(f"  {blk['sign_per_exact_oracle'].upper()}  "
              f"(IR weight {blk['ir_weight']:+.1f}, sign confirmed against the "
              f"independent oracle)")
        print("  beta*J  regime       R-hat |m|   R-hat m   max R-hat/spin  "
              "min ESS/spin  frozen   caught by")
        for r in blk["rows"]:
            caught = []
            if r["folded_scalar_reports_failure"]:
                caught.append("|m|")
            if r["signed_scalar_reports_failure"]:
                caught.append("m")
            if r["per_spin_reports_failure"]:
                caught.append("per-spin")
            print(f"  {r['beta_j']:6.2f}  {r['regime']:11s}  "
                  f"{r['r_hat_folded_abs_m']:9.4f}  {r['r_hat_signed_m']:8.4f}  "
                  f"{(r['max_r_hat_per_spin'] or float('nan')):14.4f}  "
                  f"{(r['min_ess_per_spin'] or -1):12.0f}  {r['n_frozen']:6d}   "
                  + (", ".join(caught) if caught else "NOTHING"))
        print()

    if failures:
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("  In the ordered phase the chains are provably stuck -- 16 chains")
    print("  started independently cannot each have explored both modes. The")
    print("  per-spin diagnostic fires there. Where the scalars do NOT is the")
    print("  point: a mean over hundreds of spins can look perfect while every")
    print("  chain sits in a different frozen configuration.")
    print(f"  -> {OUT / 'diagnostic_control.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
