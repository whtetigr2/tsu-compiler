"""Which of Extropic's published codon models can actually be SAMPLED here?

================================ PROTOCOL ================================
SYSTEM DEFINITION   Extropic's published `codon_opt` Ising models, compiled and
                    mediated to a bipartite program, sampled by thrml's chromatic
                    block Gibbs on CPU.
STATE VARIABLES     Per workload: logical spins, physical spins after mediation,
                    draws taken, tau, ESS, R-hat, wall time, peak draw array size.
TRANSITION RULES    `route.insert_mediators` then `program.build_program`, sampled
                    via `thrml_backend.sample_chains` -- never `sample`, whose own
                    docstring says it flattens the chain boundary on purpose and
                    that autocorrelation is meaningless across it.
ALLOWED OPERATIONS  Compile, mediate, build, sample, and escalate the draw count.
FORBIDDEN OPERATIONS
                    No placement. Placement is a SEPARATE pass with a separate
                    limit, and conflating the two is the confusion this script
                    exists to end. No silicon claim: thrml simulates on CPU.
ASSUMPTIONS         |b| <= 6.0 is this project's working value, not a sourced
                    Extropic figure. Per-edge independent J is an idealisation on
                    the physical die (~9.89 edges per coupling parameter).
INVARIANTS          A mediated model must come back BIPARTITE. Mediation preserves
                    the exact marginal over the original spins (VERDICT R7).
MEASUREMENTS        tau and ESS from `tsu_compiler.ess`, which REFUSES to return a
                    number below its AR(1)-validated reliability floor; R-hat from
                    `tsu_compiler.preflight.diagnostics`.
NULL HYPOTHESES     "The full spike cannot be sampled on this machine." Tested by
                    escalating draws until ESS clears its own floor or the budget
                    is exhausted, and reporting which happened.
SUCCESS CRITERIA    Every workload reports either a trustworthy ESS with the draw
                    count that earned it, or `unavailable` naming the reason.
FAILURE CRITERIA    Any mediated model non-bipartite; any reported ESS that did
                    not clear the reliability floor.
PROVENANCE          Workloads are Extropic's own published models, already in
                    out/extropic-verify/. No hardware anywhere in this script.
SCOPE OF VALIDITY   Results are for THIS machine (CPU, 32 GB) and THIS graph
                    family. Sampleability is not placeability: a model can sample
                    in seconds and still not have a searched geometric embedding.
==========================================================================

WHY THIS EXISTS. The verification pack records that the mediated spike's
"geometric placement after mediation" was not fully searched at 5,025 spins, and
that limitation has since been read as "the full spike is blocked". It is not.
Placement and sampling are different passes with different costs. This script
measures the sampling side directly and states the placement side separately, so
neither limit gets attributed to the other again.
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.backends.thrml_backend import sample_chains
from tsu_compiler.ess import RELIABILITY_MIN_N_OVER_TAU, effective_sample_size
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.program import build_program
from tsu_compiler.passes.route import insert_mediators
from tsu_compiler.preflight.diagnostics import RHAT_THRESHOLD, r_hat
from tsu_compiler.preflight.model import load_model

PACK = Path("out/extropic-verify")
OUT = Path("out/codon-sampleability")

WORKLOADS = ["codon_tiny_10aa", "codon_default_prefix",
             "codon_spike_200aa", "codon_spike_full"]

N_CHAINS = 16
N_WARMUP = 2000
STEPS = 4
START_SAMPLES = 1000
MAX_SAMPLES = 32_000       # stated budget, not a claim about tractability


def measure(stem: str) -> dict:
    path = PACK / f"{stem}.edges.json"
    if not path.exists():
        return {"name": stem, "status": f"unavailable: {path} not present"}

    model = load_model(edges=path)
    rep = analyse(model)
    med, _ = insert_mediators(model, rep)
    mrep = analyse(med)
    prog = build_program(med, mrep)

    ns = START_SAMPLES
    attempts = []
    while True:
        t0 = time.time()
        draws = np.asarray(sample_chains(
            prog, n_chains=N_CHAINS, n_samples=ns, n_warmup=N_WARMUP,
            steps_per_sample=STEPS, seed=0))
        secs = time.time() - t0
        mb = draws.nbytes / 2 ** 20
        # int8 keeps the spin array small; the order parameter is all we need
        mag = (2 * draws.astype(np.int8) - 1).mean(axis=2)
        est = effective_sample_size(np.abs(mag))
        rh = float(r_hat(np.abs(mag)))
        del draws
        gc.collect()

        attempts.append({"n_samples": ns, "draws": N_CHAINS * ns,
                         "seconds": round(secs, 2), "draw_array_mb": round(mb, 1),
                         "ess": est.ess, "tau": est.iat, "r_hat": round(rh, 4),
                         "reliable": bool(est.reliable), "reason": est.reason})
        if est.reliable or ns * 2 > MAX_SAMPLES:
            break
        ns *= 2

    last = attempts[-1]
    return {
        "name": stem,
        "logical_spins": rep.n_nodes, "couplings": rep.n_edges,
        "physical_spins": mrep.n_nodes, "mediators": mrep.n_nodes - rep.n_nodes,
        "bipartite_after_mediation": bool(mrep.bipartite),
        "colour_blocks": mrep.colour_blocks,
        "sampleable": bool(last["reliable"]),
        "draws_needed_for_trustworthy_ess": last["draws"] if last["reliable"] else None,
        "seconds_at_that_size": last["seconds"] if last["reliable"] else None,
        "ess": last["ess"], "tau": last["tau"], "r_hat": last["r_hat"],
        "provisional_r_hat": bool(last["r_hat"] > RHAT_THRESHOLD),
        "ess_note": last["reason"] if not last["reliable"] else "",
        "escalation": attempts,
        "placement_note": (
            "NOT MEASURED HERE. Sampling and placement are different passes; the "
            "verification pack records that the mediated spike's geometric "
            "placement was not fully searched, and that limit says nothing about "
            "whether the model can be sampled."),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [measure(s) for s in WORKLOADS]
    ok = [r for r in rows if "status" not in r]

    failures = [f"{r['name']}: non-bipartite after mediation"
                for r in ok if not r["bipartite_after_mediation"]]
    for r in ok:
        if r["ess"] is not None and r["tau"]:
            if r["draws_needed_for_trustworthy_ess"] / r["tau"] < RELIABILITY_MIN_N_OVER_TAU:
                failures.append(f"{r['name']}: reported an ESS below the floor")

    (OUT / "codon_sampleability.json").write_text(json.dumps(
        {"rows": rows, "control_failures": failures,
         "machine": "CPU only, no hardware", "n_chains": N_CHAINS,
         "reliability_floor_n_over_tau": RELIABILITY_MIN_N_OVER_TAU},
        indent=2), encoding="utf-8")

    print("workload".ljust(24) + "logical".rjust(9) + "physical".rjust(10)
          + "draws".rjust(10) + "tau".rjust(8) + "ESS".rjust(10)
          + "R-hat".rjust(8) + "secs".rjust(8))
    for r in rows:
        if "status" in r:
            print(r["name"].ljust(24) + r["status"])
            continue
        ess = format(r["ess"], ",.0f") if r["ess"] else "unavailable"
        tau = format(r["tau"], ".2f") if r["tau"] else "--"
        print(r["name"].ljust(24)
              + format(r["logical_spins"], ",").rjust(9)
              + format(r["physical_spins"], ",").rjust(10)
              + format(r["draws_needed_for_trustworthy_ess"] or 0, ",").rjust(10)
              + tau.rjust(8) + ess.rjust(10)
              + format(r["r_hat"], ".3f").rjust(8)
              + format(r["seconds_at_that_size"] or 0, ".1f").rjust(8))
    print()
    if failures:
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("  every workload reports a trustworthy ESS with the draw count that")
    print("  earned it, or says why one is unavailable. Placement is a separate")
    print("  pass and is NOT measured here.")
    print("  -> " + str(OUT / "codon_sampleability.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
