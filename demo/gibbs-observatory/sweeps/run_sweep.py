#!/usr/bin/env python3
"""Z1 real-world workload compile sweep for Gibbs Observatory."""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/workspace/gibbs-observatory")
TSU = Path("/workspace/tsu-compiler-review")
sys.path.insert(0, str(TSU))
sys.path.insert(0, str(ROOT))

from tsu_compiler.spec import load_spec
from tsu_compiler.target import PROFILES
from tsu_compiler.passes.search import compile_spec
from tsu_compiler.receipt import write_receipt

WORKLOADS_DIR = ROOT / "sweeps" / "workloads"
RECEIPTS = ROOT / "receipts"
OUT_JSON = ROOT / "sweeps" / "SWEEP_RESULTS.json"

# Ordered ladder: credible apps a Z1 owner would try, then stress failures.
LADDER = [
    ("codon_opt_tiny", "codon_opt_tiny.yaml",
     "Tiny codon optimization (usage + adjacent repeat); Extropic/THRML class."),
    ("placement_8x8_exclusion", "placement_8x8_exclusion.yaml",
     "Hard-core placement / keep-out on 8x8 grid."),
    ("ilt_mask_adjacency_2x2", "ilt_mask_adjacency_2x2.yaml",
     "ILT/mask forbidden feature adjacency on 2x2."),
    ("roster_shift_conflicts", "roster_shift_conflicts.yaml",
     "Nurse/machine roster with pairwise shift conflicts."),
    ("maxcut_cycle7", "maxcut_cycle7.yaml",
     "Max-Cut / network partition on odd cycle C7."),
    ("mimo_detect_4", "mimo_detect_4.yaml",
     "Tiny 4-bit MIMO ML-detection QUBO."),
    ("floorplan_4zone", "floorplan_4zone.yaml",
     "Facility floorplan zone adjacency (4 rooms, k=3)."),
    ("mrf_denoise_8x8", "mrf_denoise_8x8.yaml",
     "Image MRF pairwise smoothness prior 8x8."),
    ("terrain_layout_6x6", "terrain_layout_6x6.yaml",
     "Mid-size landcover layout 6x6 k=3 (non-freeze weights)."),
    ("dense_qubo_clique20", "dense_qubo_clique20.yaml",
     "Dense all-pairs QUBO — expect degree > 16 HARDWARE fail."),
    ("highk_biome_infeasible", "highk_biome_infeasible.yaml",
     "k=5 16x16 biome with over-budget partners — expect HARDWARE fail."),
]


def _gate_summary(comp) -> dict:
    rows = []
    fail = []
    for g in comp.gate_checks or ():
        rows.append({
            "gate": g.gate,
            "passed": bool(g.passed),
            "measured": g.measured if not isinstance(g.measured, float) or g.measured == g.measured else None,
            "limit": g.limit if g.limit != float("inf") else None,
            "assumed": bool(g.assumed),
            "downgraded": bool(g.downgraded),
        })
        if not g.passed:
            fail.append(f"{g.gate}: measured={g.measured} limit={g.limit}")
    return {"gates": rows, "fail_summary": "; ".join(fail) if fail else "all_pass"}


def _encoding(comp) -> str | None:
    if not comp.repset:
        return None
    for cand in comp.repset.candidates:
        st = getattr(cand.state, "value", None) or str(cand.state)
        if st == "SELECTED":
            return str(cand.encoding)
    return None


def _errors(comp) -> list[str]:
    errs = []
    if comp.repset:
        for cand in comp.repset.candidates:
            st = getattr(cand.state, "value", None) or str(cand.state)
            if st not in ("SELECTED", "FEASIBLE", "VIABLE_NOT_SELECTED") and getattr(cand, "reason", None):
                errs.append(f"{cand.encoding}/{st}: {cand.reason}")
            elif st == "VIABLE_NOT_SELECTED":
                pass
            elif st not in ("SELECTED", "FEASIBLE", "VIABLE_NOT_SELECTED"):
                reason = getattr(cand, "reason", "") or ""
                errs.append(f"{cand.encoding}/{st}: {reason}")
    # placement failure note
    pl = comp.placement
    if pl is not None and getattr(pl, "mediated_report", None):
        rep = pl.mediated_report
        note = getattr(rep, "note", None) or getattr(rep, "error", None)
        if note:
            errs.append(str(note))
    return errs


def _metrics_from_receipt(receipt_dir: Path) -> dict:
    mpath = receipt_dir / "metrics.json"
    if not mpath.is_file():
        return {}
    return json.loads(mpath.read_text())


def run_one(wid: str, yaml_name: str, discovery_hint: str) -> dict:
    path = WORKLOADS_DIR / yaml_name
    receipt_id = f"sweep_{wid}"
    out_dir = RECEIPTS / receipt_id
    row = {
        "id": wid,
        "name": wid,
        "yaml": str(path),
        "receipt_id": receipt_id,
        "verdict": None,
        "encoding": None,
        "n_nodes": None,
        "n_mediators": None,
        "n_edges": None,
        "max_degree": None,
        "bipartite": None,
        "gates_summary": None,
        "gates": [],
        "elapsed_s": None,
        "errors": [],
        "discovery_note": discovery_hint,
        "ideal_passed": None,
        "receipt_path": None,
    }
    print(f"\n=== {wid} ===", flush=True)
    t0 = time.time()
    try:
        spec = load_spec(str(path))
        row["name"] = spec.name
        comp = compile_spec(spec, PROFILES["z1"], False)
        elapsed = round(time.time() - t0, 4)
        row["elapsed_s"] = elapsed
        row["verdict"] = comp.verdict
        row["ideal_passed"] = bool(comp.ideal_passed)
        row["encoding"] = _encoding(comp)
        gsum = _gate_summary(comp)
        row["gates"] = gsum["gates"]
        row["gates_summary"] = gsum["fail_summary"]
        row["errors"] = _errors(comp)

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_receipt(comp, str(out_dir))
            row["receipt_path"] = str(out_dir)
            metrics = _metrics_from_receipt(out_dir)
            row["n_nodes"] = metrics.get("n_nodes")
            row["n_mediators"] = metrics.get("mediators")
            row["n_edges"] = metrics.get("n_edges")
            row["max_degree"] = metrics.get("max_degree")
            row["bipartite"] = metrics.get("bipartite")
            # enrich discovery
            notes = [discovery_hint]
            if comp.verdict == "COMPILED":
                notes.append(
                    f"COMPILED encoding={row['encoding']} nodes={row['n_nodes']} "
                    f"med={row['n_mediators']} deg={row['max_degree']} bipartite={row['bipartite']}"
                )
            else:
                notes.append(f"verdict={comp.verdict}; gates={row['gates_summary']}")
                if row["errors"]:
                    notes.append("errors=" + " | ".join(row["errors"][:3]))
            row["discovery_note"] = " — ".join(notes)
        except Exception as exc:  # noqa: BLE001
            row["errors"].append(f"write_receipt: {type(exc).__name__}: {exc}")
            row["discovery_note"] = f"{discovery_hint} — verdict={comp.verdict} but receipt write failed: {exc}"

        print(f"  verdict={row['verdict']} enc={row['encoding']} "
              f"nodes={row['n_nodes']} med={row['n_mediators']} "
              f"deg={row['max_degree']} t={elapsed}s", flush=True)
        if row["gates_summary"] != "all_pass":
            print(f"  gates: {row['gates_summary']}", flush=True)
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.time() - t0, 4)
        row["elapsed_s"] = elapsed
        row["verdict"] = "ERROR"
        row["errors"] = [f"{type(exc).__name__}: {exc}"]
        row["discovery_note"] = f"{discovery_hint} — EXCEPTION: {exc}"
        print(f"  ERROR {exc}", flush=True)
        traceback.print_exc()
    return row


def main():
    rows = []
    for wid, yaml_name, hint in LADDER:
        rows.append(run_one(wid, yaml_name, hint))
        # checkpoint after each
        OUT_JSON.write_text(json.dumps({"workloads": rows, "target": "z1",
                                        "honesty": "software compile only — not Extropic silicon",
                                        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                       indent=2))
    compiled = sum(1 for r in rows if r["verdict"] == "COMPILED")
    failed = len(rows) - compiled
    print(f"\nDONE: {compiled} COMPILED, {failed} failed/other of {len(rows)}", flush=True)
    return rows


if __name__ == "__main__":
    main()
