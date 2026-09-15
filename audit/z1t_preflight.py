"""What does Z1's own topology cost a Z1T-shaped layer?

================================ PROTOCOL ================================
SYSTEM DEFINITION   The sparse tanh-linear and gated-convolutional-attention
                    (GCA) layer shapes described in Extropic's Z1T writing
                    (published 2026-09-04), reconstructed from its PUBLISHED
                    STRUCTURAL PARAMETERS ONLY: 4-sparse projections, dy4p
                    encoding at 4 pbits per value, attention windowed to the
                    last N values, L=4, D=512, T=1024.
STATE VARIABLES     Per layer shape: pbits, coupling edges, max degree,
                    bipartiteness, mediator count, fabric tax, and whether the
                    result clears Z1's published node budget.
TRANSITION RULES    None. This is a structural question, answered before any
                    sampling, which is what `preflight` is for.
ALLOWED OPERATIONS  Building the factor graph implied by the published
                    connectivity; running this project's own analyse() and
                    gate_checks() on it; inserting mediators where parity fails.
FORBIDDEN OPERATIONS
                    No weight values. Extropic has not published Z1T's trained
                    parameters, so this CANNOT check the coupling cap and does
                    not try -- the |J| gate is reported as not-applicable rather
                    than passed. No claim about energy, throughput, or loss. No
                    claim that anything Extropic published is wrong.
ASSUMPTIONS         That "each output node is connected to N input nodes" with
                    dy4p at 4 pbits per value means an output pbit couples to
                    4 input values x 4 pbits = 16 input pbits, which is exactly
                    Z1's degree. Stated because the whole measurement rests on
                    it. That a cumulative sum over the token stream couples
                    adjacent positions WITHIN a layer -- that is what a cumsum
                    is -- and that windowed attention couples position t to the
                    preceding N positions.
INVARIANTS          A purely feedforward layered graph must come back BIPARTITE:
                    colour by layer index and every edge crosses. If it does
                    not, the construction is wrong, not the finding.
MEASUREMENTS        Spins, edges, degree, bipartiteness, mediators, fabric tax,
                    node-budget headroom.
NULL HYPOTHESES     "A Z1T-shaped layer needs no mediation." The MLP shape is
                    the control -- it is bipartite by construction and must
                    report a 1.00x tax. If GCA also reports 1.00x, the null
                    stands and attention is free on this fabric.
SUCCESS CRITERIA    Every shape gets a COST, and the bipartite control reports
                    exactly 1.00x.
FAILURE CRITERIA    The feedforward control coming back non-bipartite, which
                    would mean the graph builder is wrong. Reporting a coupling
                    verdict when no weights are known.
PROVENANCE          Structural parameters from extropic.ai/writing/z1t. Z1's
                    degree, node budget and coupler count from target.py, which
                    sources them to Extropic's own published figures. No vendor
                    code. No hardware.
SCOPE OF VALIDITY   TOPOLOGY ONLY, and of OUR reconstruction of a published
                    description -- not of Extropic's implementation, which we
                    have not seen. It says what their stated connectivity costs
                    on their stated fabric. It says nothing about whether their
                    model trains, what it scores, or what it draws in watts.
==========================================================================

WHY THIS IS WORTH ASKING.

Z1T's own writing states the chip's limits -- degree 16, 269,568 pbits,
2,135,904 coupling edges -- and describes a 4-sparse architecture that appears
cut to saturate the first of them exactly: an output pbit taking 4 input values
at 4 pbits each lands on precisely 16. What it does NOT state is whether the
resulting factor graph is bipartite.

That question is not cosmetic. Z1 samples by two-colour block Gibbs, which needs
a 2-colouring. A graph that has one runs natively. A graph that does not has to
buy parity with mediator spins, and those spins come out of the same 269,568.
This project already measures that price for other workloads and calls it the
Fabric Tax; this applies the same meter to the shape Extropic just published.

A feedforward stack is bipartite for free -- colour by layer. Attention is the
part that might not be, because a cumulative sum over the token stream couples
positions to each other INSIDE a layer, and a window on top of that chain closes
short cycles. Whether those cycles are odd is the whole question.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.route import insert_mediators
from tsu_compiler.target import PROFILES

OUT = Path("out/z1t-preflight")

D_MODEL = 512          # published
N_LAYERS = 4           # published
SEQ_LEN = 1024         # published
DY4P_BITS = 4          # published: 4 pbits per value
SPARSITY = 4           # published: 4-sparse projections
WINDOW = 4             # attention windowed to the last N values; N = c = 4


def _pbits(n_values: int) -> int:
    return n_values * DY4P_BITS


def _stencil(j: int, d_model: int):
    """The CONVOLUTIONAL projection: output j reads a contiguous wrapped run of
    SPARSITY inputs.

    An earlier version of this file picked the 4 inputs at RANDOM, which is what
    "4-sparse" says if you read it as a sparsity fraction rather than as a
    stencil. Random selection gives every output fan-in 4 but leaves fan-OUT to
    chance, and the Poisson tail pushed measured degree to 44 -- nearly three
    times Z1's 16. A convolution is regular by definition: every input feeds
    exactly as many outputs as every output takes inputs, and the degree lands
    on 4 values x 4 dy4p bits = 16 exactly, which is Z1's limit and is almost
    certainly why the architecture is cut this way."""
    return [(j + k) % d_model for k in range(SPARSITY)]


def tanh_linear(n_positions: int, d_model: int, seed: int = 0):
    """A 4-sparse tanh-linear layer in dy4p: input values -> output values.

    Each OUTPUT pbit couples to every pbit of the 4 input values that output
    depends on: 4 values x 4 bits = 16, which is Z1's degree exactly.
    """
    n_in = n_positions * d_model
    edges = set()
    for t in range(n_positions):
        for j in range(d_model):
            src = _stencil(j, d_model)
            out_base = (n_in + t * d_model + j) * DY4P_BITS
            for b_out in range(DY4P_BITS):
                u = out_base + b_out
                for i in src:
                    in_base = (t * d_model + i) * DY4P_BITS
                    for b_in in range(DY4P_BITS):
                        v = in_base + b_in
                        edges.add((min(u, v), max(u, v)))
    n = _pbits(2 * n_in)
    return _model(n, edges)


def gca(n_positions: int, d_model: int, seed: int = 0):
    """The same, plus what attention adds: a cumulative sum chaining adjacent
    positions WITHIN the layer, and a window coupling position t to the
    preceding WINDOW positions. Both live on the same side of the layer, which
    is what makes the parity question interesting."""
    n_in = n_positions * d_model
    edges = set()
    for t in range(n_positions):
        for j in range(d_model):
            src = _stencil(j, d_model)
            out_base = (n_in + t * d_model + j) * DY4P_BITS
            for b_out in range(DY4P_BITS):
                u = out_base + b_out
                for i in src:
                    in_base = (t * d_model + i) * DY4P_BITS
                    for b_in in range(DY4P_BITS):
                        edges.add((min(u, in_base + b_in), max(u, in_base + b_in)))
        # within-layer: the cumsum chain, and the attention window on top of it
        for dt in range(1, WINDOW + 1):
            if t - dt < 0:
                continue
            for j in range(d_model):
                a = (n_in + t * d_model + j) * DY4P_BITS
                b = (n_in + (t - dt) * d_model + j) * DY4P_BITS
                for bit in range(DY4P_BITS):
                    edges.add((min(a + bit, b + bit), max(a + bit, b + bit)))
    n = _pbits(2 * n_in)
    return _model(n, edges)


def gca_budgeted(n_positions: int, d_model: int):
    """GCA under the OTHER reading of the same sentence.

    Extropic writes that "each output node of the network is connected to N
    input nodes ... the attention scores are also windowed to the last N
    values." That admits two readings and the published text does not settle
    which, so both are measured rather than one being picked:

      ADDITIVE  (`gca`)          -- the window couples positions ON TOP of the
                                    projection, so degree = projection + window.
      BUDGETED  (this function)  -- the window lives INSIDE the same N-node
                                    budget, so the projection gives up slots to
                                    pay for it and degree stays at N.

    Budgeted is the reading consistent with Z1's degree being a hard silicon
    limit, and it is probably theirs. It is built here so the parity question
    can be answered WITHOUT resting on the degree reading -- because parity does
    not depend on how many edges there are, only on whether any of them couple
    positions within a layer. Both readings share that property, which is why
    the bipartiteness finding survives the ambiguity.
    """
    n_in = n_positions * d_model
    edges = set()
    keep = SPARSITY - 2          # give up 2 projection slots to the window
    for t in range(n_positions):
        for j in range(d_model):
            out_base = (n_in + t * d_model + j) * DY4P_BITS
            for b_out in range(DY4P_BITS):
                u = out_base + b_out
                for i in _stencil(j, d_model)[:keep]:
                    in_base = (t * d_model + i) * DY4P_BITS
                    for b_in in range(DY4P_BITS):
                        v = in_base + b_in
                        edges.add((min(u, v), max(u, v)))
        for dt in (1, 2):
            if t - dt < 0:
                continue
            for j in range(d_model):
                a = (n_in + t * d_model + j) * DY4P_BITS
                b = (n_in + (t - dt) * d_model + j) * DY4P_BITS
                for bit in range(DY4P_BITS):
                    edges.add((min(a + bit, b + bit), max(a + bit, b + bit)))
    return _model(_pbits(2 * n_in), edges)


def _model(n: int, edges) -> IsingModel:
    e = tuple(sorted(edges))
    # Weights are a PLACEHOLDER. Extropic has not published Z1T's trained
    # parameters, so this file makes no coupling-magnitude claim -- see the
    # coupling_cap handling in main(), which reports not-applicable rather than
    # a pass it has not earned.
    return IsingModel(nodes=tuple(f"p{i}" for i in range(n)), edges=e,
                      weights=np.full(len(e), 1.0), biases=np.zeros(n),
                      beta=1.0, offset=0.0)


def measure(name: str, model: IsingModel, note: str) -> dict:
    rep = analyse(model)
    row = {"shape": name, "note": note,
           "pbits": rep.n_nodes, "coupling_edges": rep.n_edges,
           "max_degree": rep.max_degree, "bipartite": bool(rep.bipartite)}
    if rep.bipartite:
        row.update(mediators=0, physical_pbits=rep.n_nodes, fabric_tax=1.0)
    else:
        med, _ = insert_mediators(model, rep)
        mrep = analyse(med)
        row.update(mediators=mrep.n_nodes - rep.n_nodes,
                   physical_pbits=mrep.n_nodes,
                   fabric_tax=round(mrep.n_nodes / rep.n_nodes, 3),
                   bipartite_after_mediation=bool(mrep.bipartite))
    z1 = PROFILES["z1"]
    row["degree_within_z1"] = rep.max_degree <= z1.degree.value
    row["z1_degree"] = z1.degree.value
    row["within_node_budget"] = row["physical_pbits"] <= z1.node_budget.value
    row["z1_node_budget"] = z1.node_budget.value
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    POS = 8          # a tractable slice; the scaling question is answered separately
    rows = [
        measure("tanh_linear (MLP)", tanh_linear(POS, 64),
                "feedforward only -- the bipartite CONTROL, must report 1.00x"),
        measure("GCA, additive", gca(POS, 64),
                "window couples positions ON TOP of the projection"),
        measure("GCA, budgeted", gca_budgeted(POS, 64),
                "window shares the same N-node budget -- degree stays within Z1"),
    ]
    failures = []
    ctrl = rows[0]
    if not ctrl["bipartite"]:
        failures.append("the feedforward control came back NON-bipartite; the "
                        "graph builder is wrong, not the architecture")
    if ctrl["fabric_tax"] != 1.0:
        failures.append(f"bipartite control paid {ctrl['fabric_tax']}x, not 1.00x")

    z1 = PROFILES["z1"]
    v = DY4P_BITS
    per_vec = D_MODEL * v
    per_layer = 2 * per_vec
    per_token_all_layers = (N_LAYERS + 1) * per_vec
    scaling = {
        "pbits_per_activation_vector": per_vec,
        "pbits_per_layer_in_plus_out": per_layer,
        "pbits_per_token_across_L4": per_token_all_layers,
        "token_positions_resident_across_L4": z1.node_budget.value // per_token_all_layers,
        "pbits_for_full_T1024_sequence": SEQ_LEN * per_token_all_layers,
        "dies_for_full_T1024_sequence":
            round(SEQ_LEN * per_token_all_layers / z1.node_budget.value, 1),
    }

    (OUT / "z1t_preflight.json").write_text(json.dumps(
        {"rows": rows, "control_failures": failures, "scaling": scaling,
         "published_config": {"layers": N_LAYERS, "d_model": D_MODEL,
                              "seq_len": SEQ_LEN, "dy4p_bits": DY4P_BITS,
                              "sparsity": SPARSITY, "window": WINDOW},
         "measured_slice": {"positions": POS, "d_model": 64,
                            "why": "degree and parity are local properties and "
                                   "do not change with width; the node-budget "
                                   "question is answered by the scaling block "
                                   "at the published D=512 instead"},
         "coupling_cap": "NOT CHECKED -- Extropic has not published Z1T's "
                         "trained weights, so no |J| verdict is available and "
                         "none is claimed",
         "source": "structural parameters from extropic.ai/writing/z1t "
                   "(2026-09-04); Z1 limits from target.py",
         "not_an_error_report": "this measures what the published connectivity "
                                "costs on the published fabric; nothing here "
                                "says their work is wrong",
         "published_energy_split_nj_per_token": {
             "z1_fabric": 8.74, "fpga": 285.78, "total": 294.52,
             "z1_share_pct": 2.97,
             "source": "extropic.ai/writing/z1t",
             "observation": "Extropic states Z1 is responsible for only sparse "
                            "tanh-linear sampling. That is the shape this "
                            "preflight finds BIPARTITE. The shape it finds "
                            "non-bipartite under both readings -- attention -- "
                            "is the part that stayed on the FPGA. The "
                            "hardware/software boundary coincides with the "
                            "parity boundary.",
             "not_a_criticism": "putting the non-bipartite part on an FPGA is a "
                                "reasonable engineering choice; this is their "
                                "own published figure, and it does not bear on "
                                "their 14x-vs-H100 comparison, which measures a "
                                "different pair of things"},
         "hardware": "none -- structural analysis only"},
        indent=2), encoding="utf-8")

    print("A Z1T-SHAPED LAYER ON Z1'S OWN FABRIC (our reconstruction, topology only)\n")
    print(f"{'shape':<22}{'pbits':>9}{'edges':>10}{'deg':>5}{'bipartite':>11}"
          f"{'mediators':>11}{'tax':>8}")
    for r in rows:
        print(f"{r['shape']:<22}{r['pbits']:>9,}{r['coupling_edges']:>10,}"
              f"{r['max_degree']:>5}{str(r['bipartite']):>11}"
              f"{r['mediators']:>11,}{format(r['fabric_tax'], '.2f') + 'x':>8}")
    print()
    worst = max(r["max_degree"] for r in rows)
    within = all(r["degree_within_z1"] for r in rows)
    print(f"  Z1 degree limit {z1.degree.value}; worst measured {worst} -- "
          f"{'within it' if within else 'OVER IT'}.")
    print("  The coupling CAP is not checked: Z1T's trained weights are not")
    print("  published, so no |J| verdict is available and none is claimed.\n")
    print("  At the published D=512, L=4, dy4p=4:")
    print(f"    {scaling['pbits_per_activation_vector']:,} pbits per activation vector")
    print(f"    {scaling['pbits_per_token_across_L4']:,} pbits per token across 4 layers")
    print(f"    -> {scaling['token_positions_resident_across_L4']} token positions "
          f"resident on one die")
    print(f"    -> a full T=1024 context would need "
          f"{scaling['pbits_for_full_T1024_sequence']:,} pbits, "
          f"{scaling['dies_for_full_T1024_sequence']}x one die")
    print()
    print("  WHAT THIS LINES UP WITH.")
    print("  Extropic's published per-token split is 8.74 nJ on Z1 against")
    print("  285.78 nJ on FPGA -- 2.97% on the fabric -- and their writing says")
    print("  Z1 is \"responsible for only sparse tanh-linear sampling\".")
    print("  That is exactly the piece this preflight finds bipartite. The part")
    print("  it finds NON-bipartite under either reading is the part that stayed")
    print("  off-chip. The hardware/software split falls on the parity line.")
    print()
    print("  That is an observation about where the boundary sits, NOT a claim")
    print("  that anything is wrong: putting the awkward part on an FPGA is a")
    print("  reasonable engineering choice, and it is their own number. It also")
    print("  does not touch their 14x-vs-H100 figure, which compares a different")
    print("  pair of things than this does.")
    print()
    print("  What it does suggest is where the next gain is: while attention")
    print("  stays off-fabric, the Z1 portion is capped at ~3% of per-token")
    print("  energy no matter how efficient the fabric gets. Making attention")
    print("  bipartite -- or cheap to mediate -- is what moves that ceiling.")
    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'z1t_preflight.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
