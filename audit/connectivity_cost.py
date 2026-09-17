"""What does Z1's connectivity actually cost you?

================================ PROTOCOL ================================
SYSTEM DEFINITION   A target Ising model E(s) = -sum J_ij s_i s_j - sum b_i s_i
                    supplied as an edge list, and Z1's published connectivity:
                    degree 16, offsets = rotations of (1,0),(2,1),(2,3),(4,1),
                    every offset dx+dy odd, hence a bipartite (chessboard)
                    lattice.
STATE VARIABLES     Per workload: logical spins, couplings, max degree,
                    bipartiteness, |J|max -- before and after mediation.
TRANSITION RULES    `route.insert_mediators` subdivides every edge lying within
                    one side of a 2-partition through a fresh hidden spin, at
                    coupling A = acosh(exp(2*beta*|J|)) / (2*beta).
ALLOWED OPERATIONS  analyse(), insert_mediators(), gate_checks(), and place()
                    ONLY on models small enough to place in seconds.
FORBIDDEN OPERATIONS
                    No sampling. No claim about Extropic's own residual. No
                    full-lattice placement of the 5,025-spin mediated spike --
                    the existing verification pack states that was never fully
                    searched, and inventing a number here would contradict it.
ASSUMPTIONS         |b| <= 6.0 is this project's working value, NOT a sourced
                    Extropic figure (target.py marks it assumed). The connected
                    -fabric assumption is irrelevant at these sizes: the largest
                    model here is 5,025 spins against one core's 33,696.
INVARIANTS          Mediation preserves the exact marginal over the original
                    spins (audit VERDICT R7: 8.327e-17 one mediator, 1.110e-16
                    K4). A mediated model must come back BIPARTITE or the pass
                    did not do its job.
MEASUREMENTS        spin overhead = physical / logical; coupling inflation =
                    |J|max after / before; mediator count; bipartite before and
                    after.
NULL HYPOTHESES     "The overhead is an artifact of our mediation algorithm
                    rather than a cost imposed by Z1's bipartite lattice."
                    CONTROL: an already-bipartite workload must need ZERO
                    mediators and show overhead exactly 1.00. If a bipartite
                    graph also paid overhead, the number would be measuring our
                    code, not the hardware constraint.
SUCCESS CRITERIA    Every mediated model is bipartite; the bipartite control
                    pays nothing; measured |J|max after mediation matches the
                    closed form to 1e-9.
FAILURE CRITERIA    Any mediated model still non-bipartite; the bipartite
                    control paying overhead; measured inflation departing from
                    the closed form.
PROVENANCE          Workloads are Extropic's own published codon_opt Ising,
                    already in out/extropic-verify/. Z1 caps come from
                    target.py, which records sourced vs assumed per field. No
                    hardware: nothing here samples at all.
SCOPE OF VALIDITY   These are DEGREE-12 biochemistry graphs. The overhead ratio
                    is a property of THIS graph family against a bipartite
                    target -- it is not a universal Z1 constant, and a different
                    interaction structure will pay differently. The comparison
                    to Extropic's residual is a comparison of DESIGN CHOICES,
                    not a benchmark: they fit and accept loss, we mediate and
                    accept spins, and the two numbers are not commensurable.
==========================================================================

WHY THIS IS NOT A RESIDUAL MEASUREMENT. arXiv:2608.01615 App. J states that a
spin which is simultaneously a pairwise neighbour and a three-body partner
"needs both an affine edge to the output and a bilinear edge to a hidden spin,
and on a bipartite graph it cannot keep both", leaving "an irreducible per-site
residual". That residual is real for an approach that FITS a conditional onto
the lattice.

This compiler does not fit. `place.py` raises CompileError with limit=0 the
moment a single edge is unrealized, so it never ships a model with a dropped
coupling -- the structural residual is zero on every successful compile, and
there is no number there to report. The cost lands somewhere else: extra spins.
So this measures what zero residual COSTS, which is the same question in the
currency this design actually spends.

TWO THINGS THAT SENTENCE DOES NOT SAY, because an earlier version of this file
recorded the field as if it did. "Zero on every successful compile" is a
conditional, and this script never runs `place()`, so it establishes no
antecedent for any model here. It also is not true that every model here
compiles: `codon_spike_full` exhausts its placement budget, and the reason is
structural rather than a budget setting (R33). The field is therefore recorded
as unmeasured with that reason attached, which is what it always should have
been -- a hardcoded 0.0 justified by reasoning is a fabricated measurement no
matter how good the reasoning.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.gates import gate_checks
from tsu_compiler.passes.route import insert_mediators, mediator_coupling
from tsu_compiler.target import PROFILES
from tsu_compiler.preflight.model import load_model

PACK = Path("out/extropic-verify")
OUT = Path("out/connectivity-cost")

# The bipartite control comes first on purpose: if it ever pays overhead, every
# number below it is measuring this code rather than Z1's lattice.
WORKLOADS = [
    ("thrml_docs_chain", "CONTROL: already bipartite -- must pay nothing"),
    ("codon_tiny_10aa", "Extropic codon_opt, 10 aa"),
    ("codon_default_prefix", "Extropic codon_opt, ~100 aa"),
    ("codon_spike_200aa", "Extropic codon_opt, SARS-CoV-2 spike 200 aa"),
    ("codon_spike_full", "Extropic codon_opt, full spike 1273 aa"),
]


def measure(stem: str, note: str) -> dict:
    path = PACK / f"{stem}.edges.json"
    if not path.exists():
        return {"name": stem, "status": f"unavailable: {path} not present"}
    model = load_model(edges=path)
    rep = analyse(model)
    jmax_before = float(rep.max_abs_J)

    t0 = time.time()
    med, mrep = insert_mediators(model, rep)
    secs = time.time() - t0
    mrep2 = analyse(med)

    n_log, n_phys = rep.n_nodes, mrep2.n_nodes
    n_med = n_phys - n_log
    jmax_after = float(mrep2.max_abs_J)

    # The closed form this project gates on, checked against what mediation
    # actually produced rather than assumed to agree with it.
    predicted = (mediator_coupling(jmax_before, float(model.beta))
                 if n_med else jmax_before)

    return {
        "name": stem, "note": note,
        "logical_spins": n_log, "couplings": rep.n_edges,
        "max_degree": rep.max_degree,
        "bipartite_before": bool(rep.bipartite),
        "bipartite_after": bool(mrep2.bipartite),
        "mediators": n_med, "physical_spins": n_phys,
        "spin_overhead": round(n_phys / n_log, 4),
        "jmax_before": round(jmax_before, 6),
        "jmax_after": round(jmax_after, 6),
        "jmax_predicted_closed_form": round(float(predicted), 6),
        "closed_form_abs_error": round(abs(float(predicted) - jmax_after), 12),
        # NOT a measurement, and it used to be recorded as one. This field read
        # `0.0` with a note reasoning that place() raises on any unrealized
        # edge, so a successful compile must have realized every coupling. The
        # reasoning is sound and the number was still fabricated: this script
        # never calls place(). Worse, for codon_spike_full place() does NOT
        # succeed -- it exhausts its budget (R33) -- so a reader seeing 0.0
        # concluded every coupling was realized for a model where none were.
        #
        # A control that cannot fail is not a control, which this project has
        # now learned three times (R23, R24, and here). The honest value is
        # that it was not measured, and the honest reason is why.
        "structural_residual": None,
        "structural_residual_note": (
            "unmeasured: this script measures the MEDIATION tax and does not "
            "run place(), so it has no evidence about realized couplings. "
            "Placement for these models is a separate question and a negative "
            "one for the largest of them -- see audit/findings/R33.md"),
        "mediate_seconds": round(secs, 3),
        # D1/D2: the mediated model must still clear every Z1 gate. Recorded
        # per gate with its measured value, its limit, and whether that limit
        # is a sourced Extropic figure or this project's own assumption --
        # a gate that passes against an ASSUMED cap is weaker evidence than
        # one that passes against a documented one, and the receipt says which.
        "gates_after_mediation": [
            {"gate": g.gate, "passed": bool(g.passed),
             "measured": (float(g.measured)
                          if isinstance(g.measured, (int, float)) else g.measured),
             "limit": (float(g.limit)
                       if isinstance(g.limit, (int, float)) else g.limit),
             "limit_is_assumed": bool(g.assumed)}
            for g in gate_checks(med, mrep2, PROFILES["z1"], False)
        ],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [measure(s, n) for s, n in WORKLOADS]
    ok = [r for r in rows if "status" not in r]

    failures = []
    for r in ok:
        if not r["bipartite_after"]:
            failures.append(f"{r['name']}: still non-bipartite after mediation")
        if r["closed_form_abs_error"] > 1e-9:
            failures.append(
                r["name"] + ": |J|max after mediation "
                + str(r["jmax_after"]) + " != closed form "
                + str(r["jmax_predicted_closed_form"]))
        for g in r.get("gates_after_mediation", []):
            if not g["passed"]:
                failures.append(
                    r["name"] + ": gate " + g["gate"] + " FAILED after "
                    + "mediation (" + str(g["measured"]) + " vs "
                    + str(g["limit"]) + ")")
    ctrl = next((r for r in ok if r["name"] == "thrml_docs_chain"), None)
    if ctrl is not None and (ctrl["mediators"] != 0
                             or ctrl["spin_overhead"] != 1.0):
        failures.append(
            "CONTROL FAILED: an already-bipartite graph paid "
            + str(ctrl["mediators"]) + " mediators / "
            + str(ctrl["spin_overhead"])
            + "x -- the overhead below would be measuring this code, not Z1")

    (OUT / "connectivity_cost.json").write_text(
        json.dumps({"rows": rows, "control_failures": failures,
                    "hardware": "none -- no sampling, no silicon"}, indent=2),
        encoding="utf-8")

    hdr = ("workload".ljust(24) + "logical".rjust(9) + "med".rjust(8)
           + "phys".rjust(9) + "overhead".rjust(10) + "|J| before".rjust(12)
           + "|J| after".rjust(11) + "  bip")
    print(hdr)
    for r in rows:
        if "status" in r:
            print(r["name"].ljust(24) + r["status"])
            continue
        print(r["name"].ljust(24)
              + format(r["logical_spins"], ",").rjust(9)
              + format(r["mediators"], ",").rjust(8)
              + format(r["physical_spins"], ",").rjust(9)
              + (format(r["spin_overhead"], ".2f") + "x").rjust(10)
              + format(r["jmax_before"], ".4f").rjust(12)
              + format(r["jmax_after"], ".4f").rjust(11)
              + "  " + str(r["bipartite_before"])[0] + "->"
              + str(r["bipartite_after"])[0])
    print()
    if failures:
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("  controls passed: bipartite control paid nothing; every mediated "
          "model is bipartite; every |J|max matches the closed form to 1e-9")
    print("  -> " + str(OUT / "connectivity_cost.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
