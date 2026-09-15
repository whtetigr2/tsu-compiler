"""Is the 1.19x parity tax on GCA actually required? No -- and 1.00x is reachable.

================================ PROTOCOL ================================
SYSTEM DEFINITION   The time-axis coupling of a Z1T-shaped gated convolutional
                    attention layer, reconstructed as in `z1t_preflight.py`, with
                    the causal stencil's OFFSET SET as the free variable.
STATE VARIABLES     Per offset set: max degree, bipartiteness, mediator count,
                    fabric tax, and whether degree clears Z1's published 16.
TRANSITION RULES    None. Structural, answered before sampling.
ALLOWED OPERATIONS  Building the graph; analyse(); insert_mediators(); checking
                    degree against target.py's sourced Z1 figure.
FORBIDDEN OPERATIONS
                    No weights, so no |J| verdict. No energy claim -- the energy
                    section here CHECKS someone else's arithmetic and does not
                    produce a figure of its own. No claim that Extropic's
                    architecture is wrong: it is a reasonable design, and this
                    asks only whether a cheaper-on-this-fabric variant exists.
ASSUMPTIONS         That colouring by time parity is available, i.e. that a
                    position's spins may be assigned to a colour class by
                    t mod 2. That is what makes an odd offset cross classes and
                    an even one fail to -- the whole mechanism here.
INVARIANTS          Scan-only (offset {1}) is a path in time and MUST come back
                    bipartite. Any offset set containing an even value MUST come
                    back non-bipartite. If either fails, the builder is wrong.
MEASUREMENTS        The table in main().
NULL HYPOTHESES     "The parity tax on attention is intrinsic." Refuted if any
                    offset set with comparable reach reports 1.00x.
SUCCESS CRITERIA    A bipartite variant at 1.00x, within degree 16, whose
                    receptive field is at least the k=4 stencil's.
FAILURE CRITERIA    Reporting 1.00x for a set that still contains an even
                    offset, which would mean the parity check is not working.
PROVENANCE          Rewrite proposed by Grok (via Paul, 2026-09-15). Structural
                    parameters from extropic.ai/writing/z1t. Z1 limits from
                    target.py. This file is the CHECK, not the proposal.
SCOPE OF VALIDITY   TOPOLOGY ONLY. It shows a bipartite stencil EXISTS at no
                    mediator cost. It does NOT show the resulting model trains
                    to the same loss -- dropping the even tap changes the
                    function, and only training settles whether depth recovers
                    it. That is the open question this cannot answer.
==========================================================================

WHAT WAS PROPOSED, AND WHAT SURVIVED THE CHECK.

`z1t_preflight.py` measured a Z1T-shaped attention layer as non-bipartite under
both readings of the published connectivity, costing 1.19x to 1.56x in mediator
spins. The proposal put to it: that tax is not intrinsic. Colour positions by
t mod 2 and a time offset of d crosses colour classes exactly when d is ODD. The
cumulative scan is offset 1 and therefore free. A k=4 causal stencil is offsets
{1,2,3}, and the single EVEN member is the whole problem.

Checked here, and it holds:

    time offsets          deg   <=16   bipartite    tax
    {1} scan only           8   True        True   1.00x
    {1,2,3} as published   10   True       False   1.38x
    {1,3} odd only          8   True        True   1.00x
    {1,3,5} odd extended    8   True        True   1.00x

The odd-only stencil is bipartite at zero mediator cost, uses half the degree
budget, and {1,3,5} REACHES FURTHER than the published {1,2,3} while still
costing nothing. The parity tax is a property of one tap, not of attention.

WHAT THIS DOES NOT SETTLE, and it is the important part: dropping offset 2
changes the function the layer computes. The argument that depth recovers it
(t->t-1->t-2 across layers, as sliding-window attention relies on) is
architectural and plausible, and it is NOT verified here. Nothing in a topology
check can tell you the retrained model reaches the same loss. A bipartite graph
that computes the wrong thing is not a win.

A CORRECTION TO THIS PROJECT'S OWN EARLIER FRAMING. `z1t_preflight.py` reports
that a full T=1024 context needs 38.9x one die. The arithmetic is right and the
framing was misleading: that is the cost of UNROLLING the whole sequence onto
the fabric, which is a thing nobody would do. Streaming decode's working set is
the current token plus the window plus the pool -- about five token slots, or
51,200 pbits, which fits one die with 5.3x headroom. The 38.9x is a prefill
number wearing a decode number's clothes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "audit")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.route import insert_mediators
from tsu_compiler.target import PROFILES

OUT = Path("out/z1t-preflight")
DY4P_BITS = 4
FANIN_PBITS = 4          # 4 of the 16 input slots go to the sparse projection
D_MODEL, N_LAYERS = 512, 4


def layer(n_positions: int, d_model: int, offsets) -> IsingModel:
    """A GCA layer whose time-axis stencil is `offsets`, everything else fixed."""
    n_in = n_positions * d_model
    edges = set()
    for t in range(n_positions):
        for j in range(d_model):
            out_base = (n_in + t * d_model + j) * DY4P_BITS
            for b in range(DY4P_BITS):
                for k in range(FANIN_PBITS):
                    v = (t * d_model + ((j + k) % d_model)) * DY4P_BITS + (k % DY4P_BITS)
                    edges.add((min(out_base + b, v), max(out_base + b, v)))
        for dt in offsets:
            if t - dt < 0:
                continue
            for j in range(d_model):
                a = (n_in + t * d_model + j) * DY4P_BITS
                b = (n_in + (t - dt) * d_model + j) * DY4P_BITS
                for bit in range(DY4P_BITS):
                    edges.add((min(a + bit, b + bit), max(a + bit, b + bit)))
    n = DY4P_BITS * 2 * n_in
    return IsingModel(nodes=tuple(f"p{i}" for i in range(n)),
                      edges=tuple(sorted(edges)),
                      weights=np.full(len(edges), 1.0), biases=np.zeros(n),
                      beta=1.0, offset=0.0)


def measure(offsets, note):
    m = layer(8, 64, offsets)
    rep = analyse(m)
    if rep.bipartite:
        med, tax = 0, 1.0
    else:
        mrep = analyse(insert_mediators(m, rep)[0])
        med, tax = mrep.n_nodes - rep.n_nodes, round(mrep.n_nodes / rep.n_nodes, 3)
    cap = PROFILES["z1"].degree.value
    return {"offsets": list(offsets), "note": note, "max_degree": rep.max_degree,
            "degree_within_z1": rep.max_degree <= cap, "bipartite": bool(rep.bipartite),
            "mediators": med, "fabric_tax": tax,
            "reach": max(offsets), "has_even_offset": any(d % 2 == 0 for d in offsets)}


def check_energy_claim() -> dict:
    """Check the arithmetic offered alongside the rewrite. Reported as a CHECK
    of someone else's number; this file produces no energy figure of its own."""
    tot, fpga, z1, tps = 294.52, 285.78, 8.74, 17_000
    per_token_s = 1 / tps
    claimed_nj, claimed_w, claimed_s = 88.0, 1.5, 58.8e-6
    actual_j = claimed_w * claimed_s
    return {
        "claim": "1.5 W static x 58.8 us ~ 88 nJ, ~30% of the FPGA bill",
        "actual_joules": actual_j,
        "actual_nj": actual_j * 1e9,
        "claimed_nj": claimed_nj,
        "off_by_factor": round(actual_j * 1e9 / claimed_nj),
        "verdict": "WRONG by ~1000x: 1.5 W for 58.8 us is 88.2 uJ, not 88 nJ",
        "and_it_cannot_be_a_unit_slip_alone":
            f"at 1.5 W a {claimed_s*1e6:.1f} us token would cost "
            f"{actual_j*1e9:,.0f} nJ, which is "
            f"{round(actual_j / (tot*1e-9))}x Extropic's entire published "
            f"per-token budget of {tot} nJ",
        "implied_by_published_figures": {
            "per_token_us": round(per_token_s * 1e6, 2),
            "avg_system_power_mw": round(tot * 1e-9 / per_token_s * 1e3, 2),
            "avg_fpga_power_mw": round(fpga * 1e-9 / per_token_s * 1e3, 2),
            "note": "so the FPGA averages about 4.9 mW on these numbers, not 1.5 W"},
        "what_survives": "the CONCLUSION -- that the FPGA cost is orchestration "
                         "and round-trips rather than arithmetic -- may well be "
                         "true. It is simply not supported by this calculation.",
        "unverifiable_here": ["38 us of orchestration", "38 serial hops"],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        measure((1,), "scan only -- a path in time, bipartite by construction"),
        measure((1, 2, 3), "the published k=4 causal stencil"),
        measure((1, 3), "odd taps only -- the proposal"),
        measure((1, 3, 5), "odd taps, extended reach, still free"),
        measure((1, 2, 3, 4), "an earlier reading of the window, for comparison"),
    ]
    failures = []
    for r in rows:
        if r["has_even_offset"] and r["bipartite"]:
            failures.append(f"offsets {r['offsets']} contain an even tap but "
                            f"report bipartite -- the parity check is broken")
        if not r["has_even_offset"] and not r["bipartite"]:
            failures.append(f"offsets {r['offsets']} are all odd but report "
                            f"non-bipartite -- the builder is wrong")

    per_tok = (N_LAYERS + 1) * D_MODEL * DY4P_BITS
    budget = PROFILES["z1"].node_budget.value
    decode = {
        "pbits_per_token_across_L4": per_tok,
        "streaming_slots_needed": 5,
        "streaming_pbits": 5 * per_tok,
        "fits_one_die": 5 * per_tok <= budget,
        "headroom_x": round(budget / (5 * per_tok), 1),
        "corrects": "z1t_preflight.py's 38.9x, which is the cost of UNROLLING "
                    "T=1024 onto the fabric rather than streaming decode. The "
                    "arithmetic was right; the framing implied a cost nobody "
                    "would actually pay.",
    }

    energy = check_energy_claim()
    (OUT / "z1t_bipartite_rewrite.json").write_text(json.dumps(
        {"rows": rows, "control_failures": failures, "decode_working_set": decode,
         "energy_claim_check": energy,
         "proposal_credit": "rewrite proposed by Grok; this file is the check",
         "open_question": "dropping the even tap changes the function. Whether "
                          "depth recovers it is an accuracy question that only "
                          "training settles, and nothing here addresses it.",
         "hardware": "none -- structural analysis only"},
        indent=2), encoding="utf-8")

    cap = PROFILES["z1"].degree.value
    print("IS THE PARITY TAX ON ATTENTION INTRINSIC?  (Z1 degree limit "
          f"{cap})\n")
    print(f"{'time offsets':<24}{'reach':>6}{'deg':>5}{'<=16':>7}"
          f"{'bipartite':>11}{'mediators':>11}{'tax':>8}")
    for r in rows:
        print(f"{str(r['offsets']):<24}{r['reach']:>6}{r['max_degree']:>5}"
              f"{str(r['degree_within_z1']):>7}{str(r['bipartite']):>11}"
              f"{r['mediators']:>11,}{format(r['fabric_tax'], '.2f') + 'x':>8}")
    print()
    print("  No. The tax is a property of ONE TAP, not of attention. Every")
    print("  all-odd stencil is bipartite at 1.00x, and {1,3,5} reaches further")
    print("  than the published {1,2,3} while still costing nothing.")
    print()
    print(f"  Decode working set: {decode['streaming_slots_needed']} token slots "
          f"= {decode['streaming_pbits']:,} pbits, fits one die with "
          f"{decode['headroom_x']}x headroom.")
    print("  That corrects this project's own 38.9x, which priced UNROLLING a")
    print("  full context rather than streaming decode.")
    print()
    print(f"  Energy claim checked: {energy['verdict']}")
    print(f"    {energy['and_it_cannot_be_a_unit_slip_alone']}")
    print(f"    {energy['what_survives']}")
    print()
    print("  NOT SETTLED: dropping the even tap changes what the layer computes.")
    print("  Whether depth recovers it is an accuracy question, and only")
    print("  training answers it. A bipartite graph computing the wrong thing")
    print("  is not a win.")
    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'z1t_bipartite_rewrite.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
