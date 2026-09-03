"""Helper spawned by r17_reproducibility_check.py's TEST B -- runs exactly
ONE simulate() call in a brand-new interpreter process and writes the result
(raw draws + simulation.json contents) to the path given as argv[5]. Not a
standalone audit deliverable; exists only so TEST B can compare two REAL,
independent process launches rather than two calls sharing one interpreter's
warm JAX/thrml state.

argv: receipt_dir seed clamp_json params_json out_path
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "demo"))


def main():
    receipt_dir, seed_s, clamp_json, params_json, out_path = sys.argv[1:6]
    seed = int(seed_s)
    clamp = json.loads(clamp_json)
    params = json.loads(params_json)

    from tsu.simulate import simulate
    _path, got, _im = simulate(receipt_dir, seed=seed, clamp=clamp, **params)
    doc = json.loads((Path(receipt_dir) / "simulation.json").read_text())
    doc["raw_draws_flat"] = got.tolist()  # independent capture, not re-derived from doc
    Path(out_path).write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
