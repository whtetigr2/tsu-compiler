#!/usr/bin/env python3
"""Compile programs/*.yaml → receipts/prog_<id>/ via program_service."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/gibbs-observatory")
TSU = Path("/workspace/tsu-compiler-review")
sys.path.insert(0, str(TSU))
sys.path.insert(0, str(ROOT))

from backend.app.program_service import compile_program, ProgramServiceError

PROGRAMS = ROOT / "programs"
RECEIPTS = ROOT / "receipts"
RESULTS = PROGRAMS / "COMPILE_RESULTS.json"

# Fast → slow-ish; skip docs and this script
ORDER = [
    "number_partition",
    "knapsack_tiny",
    "jobshop_tiny",
    "sat_3tiny",
    "market_binary_factors",
    "maxcut_cycle7",
    "mimo_detect_4",
    "roster_shift_conflicts",
    "ilt_mask_adjacency_2x2",
    "ecology_lotka_lite",
    "floorplan_4zone",
    "mrf_denoise_8x8",
    "placement_8x8_exclusion",
    "codon_opt_tiny",
    "seq_design_longer",
    "terrain_layout_6x6",
]


def main() -> None:
    rows = []
    for name in ORDER:
        path = PROGRAMS / f"{name}.yaml"
        if not path.is_file():
            print(f"MISSING {name}", flush=True)
            rows.append({"id": name, "verdict": "MISSING", "error": "no yaml"})
            continue
        yaml_text = path.read_text(encoding="utf-8")
        receipt_id = f"prog_{name}"
        print(f"\n=== compile {name} → {receipt_id} ===", flush=True)
        t0 = time.time()
        try:
            result = compile_program(
                yaml_text,
                allow_assumed=False,
                target="z1",
                receipt_id=receipt_id,
                root=RECEIPTS,
            )
            elapsed = round(time.time() - t0, 4)
            row = {
                "id": name,
                "receipt_id": receipt_id,
                "yaml": str(path),
                "verdict": result.get("verdict"),
                "ok": result.get("ok"),
                "n_nodes": result.get("n_nodes"),
                "mediator_count": result.get("mediator_count"),
                "bipartite": result.get("bipartite"),
                "elapsed_s": elapsed,
                "errors": result.get("errors") or [],
                "candidates": result.get("candidates") or [],
                "receipt_path": result.get("receipt_path"),
                "label": result.get("label"),
            }
            # enrich from metrics if present
            mpath = RECEIPTS / receipt_id / "metrics.json"
            if mpath.is_file():
                m = json.loads(mpath.read_text())
                row["n_nodes"] = row["n_nodes"] or m.get("n_nodes")
                row["n_edges"] = m.get("n_edges")
                row["max_degree"] = m.get("max_degree")
                row["encoding"] = m.get("encoding") or (
                    next((c["encoding"] for c in row["candidates"] if c.get("state") == "SELECTED"), None)
                )
            print(
                f"  → {row['verdict']} nodes={row.get('n_nodes')} med={row.get('mediator_count')} "
                f"in {elapsed}s",
                flush=True,
            )
            rows.append(row)
        except ProgramServiceError as exc:
            elapsed = round(time.time() - t0, 4)
            print(f"  → ERROR {exc}", flush=True)
            rows.append({
                "id": name,
                "receipt_id": receipt_id,
                "yaml": str(path),
                "verdict": "ERROR",
                "ok": False,
                "elapsed_s": elapsed,
                "errors": [str(exc)],
            })
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - t0, 4)
            print(f"  → EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            rows.append({
                "id": name,
                "receipt_id": receipt_id,
                "yaml": str(path),
                "verdict": "ERROR",
                "ok": False,
                "elapsed_s": elapsed,
                "errors": [f"{type(exc).__name__}: {exc}"],
            })

    summary = {
        "compiled": sum(1 for r in rows if r.get("verdict") == "COMPILED"),
        "hardware": sum(1 for r in rows if r.get("verdict") == "HARDWARE"),
        "error": sum(1 for r in rows if r.get("verdict") not in ("COMPILED", "HARDWARE")),
        "total": len(rows),
    }
    out = {
        "target": "z1",
        "honesty": "software compile + THRML sim only — not Extropic silicon",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "programs": rows,
    }
    RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== SUMMARY {summary} → {RESULTS} ===", flush=True)


if __name__ == "__main__":
    main()
