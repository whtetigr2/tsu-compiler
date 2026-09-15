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
                    `tsu_compiler.preflight.diagnostics`. Two levels: the SCALAR
                    order parameter (signed mean over logical spins) and the
                    PER-SPIN max R-hat / min ESS over every logical spin.
                    The order parameter is NOT folded by abs(). Folding is right
                    for a Z2-symmetric model; every spin in these carries a
                    nonzero bias, so there is no symmetry to fold, and folding
                    both hid multimodality and inflated ESS.
NULL HYPOTHESES     "The full spike cannot be sampled on this machine." Tested by
                    escalating draws until ESS clears its own floor or the budget
                    is exhausted, and reporting which happened.
                    "A healthy scalar ESS means the model is mixing."
                    CONTROL: the per-spin pass can refute this directly -- one
                    frozen or stuck spin fails the verdict even when the mean
                    looks perfect. The verdict requires both.
SUCCESS CRITERIA    Every workload reports either a trustworthy ESS with the draw
                    count that earned it, or `unavailable` naming the reason --
                    AND a per-spin max R-hat and min ESS over every logical spin.
FAILURE CRITERIA    Any mediated model non-bipartite; any reported ESS that did
                    not clear the reliability floor; any workload called
                    sampleable while a logical spin is frozen or exceeds the
                    R-hat threshold.
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

CORRECTION, found by a claims-vs-evidence audit. The first version of this
script judged sampleability from ONE scalar, and folded it by abs(). Both were
wrong for these models: they carry a bias on every spin, so there is no Z2
symmetry that makes the fold legitimate, and a single mean can be healthy while
an individual spin never moves. The conclusion did not change -- every workload
still samples, and the full spike's worst spin has ESS 5,045 at R-hat 1.0032 --
but it now rests on all 3,147 spins instead of their average. See
audit/findings/R20.md.
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


def _per_spin(draws, n_logical: int) -> dict:
    """Max R-hat and min ESS over every logical spin.

    The scalar order parameter is one number summarising thousands of variables,
    and a chain can mix perfectly in that number while an individual spin never
    moves. Reporting max R-hat and min ESS across parameters is the ordinary
    multi-parameter practice; doing it only for the mean was the weak point this
    harness had. `n_frozen` counts spins that never flipped at all -- those admit
    no R-hat, and hiding them inside a healthy mean is exactly the failure mode.
    """
    rhats, esss, frozen, no_ess = [], [], 0, 0
    for j in range(n_logical):
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
        "n_logical": n_logical,
        "n_frozen": frozen,
        "n_ess_unavailable": no_ess,
        "max_r_hat": round(max(rhats), 4) if rhats else None,
        "median_r_hat": round(float(np.median(rhats)), 4) if rhats else None,
        "min_ess": round(min(esss)) if esss else None,
        "median_ess": round(float(np.median(esss))) if esss else None,
        "all_spins_below_rhat_threshold": bool(rhats and max(rhats) <= RHAT_THRESHOLD),
    }


def measure(stem: str) -> dict:
    path = PACK / f"{stem}.edges.json"
    if not path.exists():
        return {"name": stem, "status": f"unavailable: {path} not present"}

    model = load_model(edges=path)
    rep = analyse(model)
    n_logical = rep.n_nodes
    med, _ = insert_mediators(model, rep)
    mrep = analyse(med)
    if med.mediator_nodes:
        assert min(med.mediator_nodes) == n_logical, (
            "mediators must be appended after the logical spins for the "
            "slice below to select the original model")
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
        # The order parameter is the mean over the LOGICAL spins only. Mediators
        # are auxiliary spins this compiler inserted; including them would dilute
        # the statistic with variables the model never asked for.
        #
        # SIGNED, not |m|. Folding by abs() is the standard trick for a
        # Z2-SYMMETRIC model, where the overall sign is arbitrary and the two
        # modes are the same physics. These models are NOT symmetric -- every
        # spin carries a nonzero bias -- so folding here would map two genuinely
        # different modes at +m and -m onto one value and report an R-hat that
        # was never earned. It also biases tau downward: measured at 4,000
        # samples, abs() understated tau on all four workloads and inflated ESS
        # by 5.9% (spike_full) to 41% (default_prefix). See audit/findings/R20.md.
        mag = (2 * draws[:, :, :n_logical].astype(np.int8) - 1).mean(axis=2)
        est = effective_sample_size(mag)
        rh = float(r_hat(mag))

        # A scalar summary can look healthy while an individual spin is stuck,
        # so the verdict rests on the PER-SPIN diagnostics: max R-hat and min
        # ESS over every logical spin, the standard multi-parameter practice.
        # audit/diagnostic_control.py demonstrates why this is not optional --
        # on an ordered antiferromagnet BOTH scalars report R-hat 1.0000 while
        # the per-spin maximum is 11.4.
        per_spin = _per_spin(draws, n_logical)

        attempts.append({"n_samples": ns, "draws": N_CHAINS * ns,
                         "seconds": round(secs, 2), "draw_array_mb": round(mb, 1),
                         "ess": est.ess, "tau": est.iat, "r_hat": round(rh, 4),
                         "reliable": bool(est.reliable), "reason": est.reason,
                         "per_spin": per_spin})

        # Escalate until EVERY spin is certified, not until the mean is happy.
        # `min_ess` is the minimum over spins that cleared the reliability floor,
        # so a spin that failed to clear it is INVISIBLE to that minimum --
        # reporting a healthy min while dropping the failures is the same defect
        # this harness was corrected for (R20). `n_ess_unavailable` must be 0.
        settled = (est.reliable
                   and per_spin["n_ess_unavailable"] == 0
                   and per_spin["n_frozen"] == 0
                   and per_spin["all_spins_below_rhat_threshold"])
        if settled or ns * 2 > MAX_SAMPLES:
            break
        del draws
        gc.collect()
        ns *= 2

    del draws
    gc.collect()

    last = attempts[-1]
    per_spin = last["per_spin"]
    return {
        "name": stem,
        "logical_spins": rep.n_nodes, "couplings": rep.n_edges,
        "physical_spins": mrep.n_nodes, "mediators": mrep.n_nodes - rep.n_nodes,
        "bipartite_after_mediation": bool(mrep.bipartite),
        "colour_blocks": mrep.colour_blocks,
        # The verdict needs BOTH: a trustworthy scalar ESS and every individual
        # spin mixing. Either alone can look healthy while the other does not.
        "sampleable": bool(last["reliable"]
                           and per_spin["all_spins_below_rhat_threshold"]
                           and per_spin["n_frozen"] == 0
                           and per_spin["n_ess_unavailable"] == 0),
        "sampleable_basis": "scalar ESS cleared its reliability floor AND every "
                            "logical spin mixed: max R-hat within threshold, no "
                            "frozen spin, and NO spin left uncertified -- a spin "
                            "whose ESS fell below the floor does not appear in "
                            "min_ess, so it must be counted separately or it "
                            "vanishes from the verdict",
        "draws_needed_for_trustworthy_ess": last["draws"] if last["reliable"] else None,
        "seconds_at_that_size": last["seconds"] if last["reliable"] else None,
        "ess": last["ess"], "tau": last["tau"], "r_hat": last["r_hat"],
        "provisional_r_hat": bool(last["r_hat"] > RHAT_THRESHOLD),
        "ess_note": last["reason"] if not last["reliable"] else "",
        "per_spin": per_spin,
        "order_parameter": "signed mean over logical spins; NOT |m| -- these "
                           "models carry a bias on every spin and so have no "
                           "Z2 symmetry to fold",
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
        ps = r["per_spin"]
        if r["sampleable"] and ps["n_frozen"]:
            failures.append(f"{r['name']}: called sampleable with "
                            f"{ps['n_frozen']} frozen spin(s)")
        if r["sampleable"] and not ps["all_spins_below_rhat_threshold"]:
            failures.append(f"{r['name']}: called sampleable with max R-hat "
                            f"{ps['max_r_hat']} above {RHAT_THRESHOLD}")

    (OUT / "codon_sampleability.json").write_text(json.dumps(
        {"rows": rows, "control_failures": failures,
         "machine": "CPU only, no hardware", "n_chains": N_CHAINS,
         "reliability_floor_n_over_tau": RELIABILITY_MIN_N_OVER_TAU},
        indent=2), encoding="utf-8")

    print("                          --------- scalar (signed m) ---------"
          "   ------ per logical spin ------")
    print("workload".ljust(24) + "logical".rjust(9) + "physical".rjust(10)
          + "draws".rjust(10) + "tau".rjust(7) + "ESS".rjust(9)
          + "R-hat".rjust(8) + "maxR-hat".rjust(10) + "minESS".rjust(9)
          + "frozen".rjust(8) + "secs".rjust(7))
    for r in rows:
        if "status" in r:
            print(r["name"].ljust(24) + r["status"])
            continue
        ps = r["per_spin"]
        ess = format(r["ess"], ",.0f") if r["ess"] else "unavailable"
        tau = format(r["tau"], ".2f") if r["tau"] else "--"
        print(r["name"].ljust(24)
              + format(r["logical_spins"], ",").rjust(9)
              + format(r["physical_spins"], ",").rjust(10)
              + format(r["draws_needed_for_trustworthy_ess"] or 0, ",").rjust(10)
              + tau.rjust(7) + ess.rjust(9)
              + format(r["r_hat"], ".3f").rjust(8)
              + (format(ps["max_r_hat"], ".4f") if ps["max_r_hat"] else "--").rjust(10)
              + (format(ps["min_ess"], ",") if ps["min_ess"] else "--").rjust(9)
              + str(ps["n_frozen"]).rjust(8)
              + format(r["seconds_at_that_size"] or 0, ".1f").rjust(7))
    print()
    if failures:
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("  Every workload reports a trustworthy ESS with the draw count that")
    print("  earned it, or says why one is unavailable -- and the verdict rests")
    print("  on EVERY logical spin mixing, not on their mean. The order")
    print("  parameter is signed: these models have a bias on every spin, so")
    print("  there is no Z2 symmetry that would justify folding by abs().")
    print("  Placement is a separate pass and is NOT measured here.")
    unavail = sum(r["per_spin"]["n_ess_unavailable"] for r in ok)
    if unavail:
        print(f"  {unavail} spin(s) across all workloads returned no ESS (below "
              f"the reliability floor); they are counted, not hidden.")
    print("  -> " + str(OUT / "codon_sampleability.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
