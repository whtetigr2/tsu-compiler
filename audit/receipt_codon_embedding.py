"""THE RECEIPT: Extropic's published codon models, placed on Z1's topology.

================================ PROTOCOL ================================
SYSTEM DEFINITION   The four `codon_opt` Ising models Extropic publishes, as
                    shipped in `out/extropic-verify/*.edges.json`, embedded into
                    a patch of Z1's own offset lattice (rotations of (1,0),
                    (2,1), (2,3), (4,1); `target.py` records the source).
STATE VARIABLES     One chain of physical p-bits per logical spin.
TRANSITION RULES    None. No sampling happens here. This measures placement.
ALLOWED OPERATIONS  Building the host from `target.offsets`; calling
                    `minorminer.find_embedding`; validating the result with
                    this project's own checker; comparing against the direct
                    placer.
FORBIDDEN OPERATIONS
                    No hardware claim. Nothing here ran on silicon, and an
                    embedding is a statement about a published topology, not
                    about a die whose coupler sharing is unpublished (R2). No
                    reporting of `minorminer`'s own success flag as evidence --
                    every embedding below is re-derived and re-checked by
                    `validate_embedding`, which is asserted to be able to fail
                    in `tests/test_chain_embedding.py`.
ASSUMPTIONS         Z1's published offsets and node budget. That every edge
                    carries an independently programmable coupling remains an
                    ASSUMPTION of this project and is marked so in target.py.
INVARIANTS          An embedding is accepted only if every chain is connected
                    in the host, no cell is used twice, and every logical edge
                    is realized by some adjacent pair of cells. Any failure
                    aborts rather than downgrades.
MEASUREMENTS        Logical spins and edges; physical cells; chain length max
                    and mean; overhead; fraction of Z1's node budget; wall
                    time. Direct placement's verdict on the same model, for
                    contrast.
NULL HYPOTHESES     "This compiler cannot place Extropic's published models on
                    Z1's topology." Refuted for every model below.
SUCCESS CRITERIA    A validated embedding for each model, reproducible from a
                    clean checkout with one command.
FAILURE CRITERIA    Reporting an embedding that does not validate; reporting
                    the library's flag instead of the check; omitting the
                    direct placer's failure, which is the thing that makes
                    these numbers worth reporting.
PROVENANCE          Models: Extropic's `codon_opt`, recorded in
                    `out/extropic-verify/`. Embedder: `minorminer` (D-Wave,
                    Apache-2.0), not written here. Host topology, validation,
                    gates and this harness: this project.
==========================================================================

Run:
    PYTHONIOENCODING=utf-8 python audit/receipt_codon_embedding.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsu_compiler.passes.embed import (  # noqa: E402
    EmbeddingUnavailable,
    build_host,
    find_chain_embedding,
    validate_embedding,
)
from tsu_compiler.preflight.check import preflight  # noqa: E402
from tsu_compiler.preflight.model import load_model  # noqa: E402
from tsu_compiler.target import PROFILES  # noqa: E402

PACK = Path("out/extropic-verify")
OUT = Path("out/codon-embedding")
WORKLOADS = ["codon_tiny_10aa", "codon_default_prefix",
             "codon_spike_200aa", "codon_spike_full"]


def main() -> int:
    target = PROFILES["z1"]
    rows = []
    print("Extropic codon_opt models on Z1's published topology")
    print("chain (minor) embedding, validated independently of the embedder\n")
    hdr = (f"{'workload':<22}{'spins':>7}{'edges':>7}{'cells':>8}{'over':>7}"
           f"{'chain':>7}{'budget':>9}{'valid':>7}{'sec':>7}")
    print(hdr)
    print("-" * len(hdr))

    for name in WORKLOADS:
        path = PACK / f"{name}.edges.json"
        if not path.exists():
            print(f"{name:<22}  unavailable: {path} not present")
            continue
        model = load_model(edges=path)
        try:
            emb = find_chain_embedding(model, target, seed=1)
        except EmbeddingUnavailable as exc:
            print(f"\n{exc}")
            return 2
        except RuntimeError as exc:
            print(f"{name:<22}  REFUSED: {exc}")
            rows.append({"name": name, "embedded": False, "reason": str(exc)})
            continue

        # Re-validate here, separately from the call that produced it. The
        # embedding has to survive a check made by code that did not build it.
        host = build_host(emb.host_side, target.offsets.value)
        edges = [(int(u), int(v)) for u, v in model.edges]
        failures = validate_embedding(edges, host, emb.chains,
                                      nodes=range(len(model.nodes)))
        ok = not failures
        pct = 100.0 * emb.n_physical / target.node_budget.value
        print(f"{name:<22}{emb.n_logical:>7,}{len(model.edges):>7,}"
              f"{emb.n_physical:>8,}{emb.overhead:>6.2f}x"
              f"{emb.max_chain:>7}{pct:>8.1f}%{'yes' if ok else 'NO':>7}"
              f"{emb.seconds:>7.1f}")
        rows.append({
            "name": name,
            "logical_spins": emb.n_logical, "logical_edges": len(model.edges),
            "physical_cells": emb.n_physical, "overhead": round(emb.overhead, 4),
            "chain_max": emb.max_chain, "chain_mean": round(emb.mean_chain, 3),
            "host_side": emb.host_side,
            "node_budget": target.node_budget.value,
            "budget_fraction": round(pct / 100.0, 6),
            "validated": ok, "validation_failures": list(failures[:5]),
            "seconds": round(emb.seconds, 2),
            "embedded": True,
        })
        if failures:
            print(f"   VALIDATION FAILED: {failures[0]}")
            return 1

    # The contrast that makes the numbers mean something. Direct placement is
    # what this repository had before, and R33 measures why it fails.
    print("\nthe same models under DIRECT placement (one spin, one cell):")
    for name in WORKLOADS:
        path = PACK / f"{name}.edges.json"
        if not path.exists():
            continue
        model = load_model(edges=path)
        if len(model.nodes) > 600:
            print(f"  {name:<22} not run here: takes ~10 min to exhaust "
                  f"its budget; measured separately in audit/findings/R33.md")
            for r in rows:
                if r["name"] == name:
                    r["direct_placement"] = "effort (measured in R33)"
            continue
        t0 = time.time()
        rep = preflight(model)
        for r in rows:
            if r["name"] == name:
                r["direct_placement"] = rep.verdict
                r["direct_placed"] = bool(rep.placed)
        print(f"  {name:<22} verdict {rep.verdict:<8} placed={rep.placed}  "
              f"({time.time()-t0:.1f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "what": "Extropic codon_opt models minor-embedded into Z1's published "
                "offset lattice, validated by tsu_compiler.passes.embed",
        "embedder": "minorminer (D-Wave, Apache-2.0) -- not written by this project",
        "validated_by": "tsu_compiler.passes.embed.validate_embedding",
        "target": "z1",
        "offsets_source": target.offsets.source,
        "node_budget": target.node_budget.value,
        "hardware_claim": "none: simulation and topology only, nothing ran on silicon",
        "rows": rows,
    }
    (OUT / "codon_embedding.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT / 'codon_embedding.json'}")
    print("  Topology only. Nothing here ran on Extropic silicon.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
