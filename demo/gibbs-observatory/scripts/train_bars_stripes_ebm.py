#!/usr/bin/env python3
"""Train bars-and-stripes pairwise Ising EBM and export Observatory artifacts.

Usage (from repo root, Observatory venv):
  .venv/bin/python scripts/train_bars_stripes_ebm.py
  .venv/bin/python scripts/train_bars_stripes_ebm.py --lite
  .venv/bin/python scripts/train_bars_stripes_ebm.py --epochs 100 --compile
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/workspace/tsu-compiler-review")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lite", action="store_true", help="Short retrain for API/demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="Compile YAML → receipt after train")
    args = ap.parse_args()

    from backend.app.ebm_bars_stripes import (
        EBM_RECEIPT_ID,
        train_and_export,
        YAML_PATH,
    )

    print("=== bars-and-stripes EBM train ===", flush=True)
    result = train_and_export(n_epochs=args.epochs, lite=args.lite, seed=args.seed)
    print(
        f"final_cd_moment_l1={result['final_cd_moment_l1']:.4f} "
        f"elapsed={result['elapsed_s']}s n_train={result['n_train']}",
        flush=True,
    )
    print("paths:", json.dumps(result["paths"], indent=2), flush=True)

    if args.compile:
        from backend.app.program_service import compile_program

        yaml_text = Path(YAML_PATH).read_text(encoding="utf-8")
        receipts = ROOT / "receipts"
        print(f"=== compile → {EBM_RECEIPT_ID} ===", flush=True)
        out = compile_program(
            yaml_text,
            allow_assumed=False,
            target="z1",
            receipt_id=EBM_RECEIPT_ID,
            root=receipts,
        )
        print(
            f"verdict={out.get('verdict')} nodes={out.get('n_nodes')} "
            f"mediators={out.get('mediator_count')} ok={out.get('ok')}",
            flush=True,
        )
        # Copy edges.json into programs dir for loader
        art = Path(result["paths"]["artifact_dir"])
        prog = ROOT / "programs" / "ebm_bars_stripes"
        if (art / "edges.json").is_file():
            (prog / "edges.json").write_text((art / "edges.json").read_text())


if __name__ == "__main__":
    main()
