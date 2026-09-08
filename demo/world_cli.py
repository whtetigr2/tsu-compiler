"""Generate one world and write it to disk.

    python demo/world_cli.py --size 64 --seed 1234 --out worlds/1234
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "demo")

from world.generate import generate
from world.export import write, to_text
from world.spline import TERRAINS


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a LATTICE world.")
    ap.add_argument("--size", type=int, default=64, help="multiple of 64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-j", type=float, default=0.42,
                    help="measured usable band is [0.38, 0.45]")
    ap.add_argument("--mode", choices=("bilinear", "nearest"),
                    default="bilinear")
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--out", default=None,
                    help="output directory (default worlds/<seed>)")
    ap.add_argument("--show", action="store_true", help="print the glyph grid")
    a = ap.parse_args()

    t0 = time.time()
    w = generate(size=a.size, seed=a.seed, beta_j=a.beta_j, mode=a.mode,
                 warmup=a.warmup)
    paths = write(w, a.out or f"worlds/{a.seed}")

    if a.show:
        print(to_text(w))
        print()
    total = w.terrain.size
    for i, t in enumerate(TERRAINS):
        share = float((w.terrain == i).mean())
        print(f"  {t.glyph}  {t.name:<14} {share * 100:5.1f}%  cost {t.cost}")
    print(f"\n  size {w.size}x{w.size}  seed {w.seed}  beta*J {w.beta_j} "
          f"({w.mode}, warmup {w.warmup})  {total} cells  {time.time() - t0:.1f}s")
    for p in paths:
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
