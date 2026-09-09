"""Does terrain actually SHAPE ROUTING? (spec sections 8.1-8.3)

PRE-REGISTERED. Committed before it is run.

WHY THIS IS THE BAR. Every cell is walkable by construction, so "can you cross
it" is trivially yes and proves nothing. The question that distinguishes a world
from a coloured grid is whether terrain changes HOW you cross it. A world where
every route costs the same is traversable and boring.

MEASURED, over independent worlds:
  detour ratio  optimal path cell count / grid width   (a straight path = 1.0)
  cost saving   (best straight cost - optimal cost) / best straight cost
  variety       share of cells per terrain

PRE-REGISTERED BANDS (spec 8.1):
  cost saving > 20% AND detour > 1.15   -> PASS, terrain shapes routing
  cost saving 10-20%                    -> MARGINAL, tune thresholds, re-measure
  cost saving < 10% AND detour < 1.05   -> FAIL, fields or spline need rework

VARIETY (spec 8.2): all six terrains present, none above 50% of cells. A world
that is 80% grass passes routing trivially and is still a bad world, so variety
is reported beside the verdict rather than folded into it.

PER-WORLD VARIETY (controller addition, not spec, not gated). Spec 8.2 averages
variety over 30 worlds -- but a player experiences ONE world, and a single
ocean-dominated world with zero mountains would pass an averaged check
comfortably while being monotonous to play. So beside the averaged share this
also reports, per mode: how many of the 30 worlds individually contain all six
terrains, and the worst single world's maximum terrain share. Reported only --
the pre-registered bands are on detour and cost saving and are not touched by
this.

UPSAMPLE ARM (spec 8.3): bilinear against nearest on both. IF BILINEAR DOES NOT
BEAT NEAREST, NEAREST WINS -- it is the simpler operation and carries values
without inventing them.

A FAIL HERE REFUTES THE ARCHITECTURE, not the implementation, and gets reported
exactly as loudly as a pass. The parameter-field-plus-spline design is
Minecraft's, adapted; that it produces good terrain HERE has never been measured.
"""
import sys
import json
import heapq

sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np

from world.generate import generate
from world.spline import TERRAINS

SEEDS = list(range(30))
SIZE = 64
BETA_J = 0.42
MODES = ("bilinear", "nearest")


def cheapest_path_cost(cost: np.ndarray) -> tuple[int, int]:
    """Dijkstra from any left-edge cell to any right-edge cell, 4-connected.

    The cost of a route is the sum of the costs of the cells it ENTERS,
    including the one it starts on. Returns (total_cost, path_cell_count).
    """
    n, m = cost.shape
    INF = float("inf")
    dist = np.full((n, m), INF)
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    heap = []
    for y in range(n):
        dist[y, 0] = cost[y, 0]
        prev[(y, 0)] = None
        heapq.heappush(heap, (float(cost[y, 0]), y, 0))
    best_end = None
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        if x == m - 1:
            best_end = (y, x)
            break
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < m:
                nd = d + float(cost[ny, nx])
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(heap, (nd, ny, nx))
    if best_end is None:
        raise RuntimeError("no left-to-right path exists -- impossible when "
                           "every cell is walkable; the cost grid is wrong")
    cells, node = 0, best_end
    while node is not None:
        cells += 1
        node = prev[node]
    return int(dist[best_end]), cells


def best_straight_cost(cost: np.ndarray) -> int:
    """Cheapest purely horizontal traversal -- the best you can do WITHOUT
    deviating vertically. This is the baseline detouring has to beat."""
    return int(cost.sum(axis=1).min())


def main():
    print(f"Routing at {SIZE}x{SIZE}, beta*J = {BETA_J}, {len(SEEDS)} worlds "
          f"per mode\n")
    out = {}
    for mode in MODES:
        detours, savings, shares = [], [], []
        worlds_with_all_six = 0
        worst_world_max_share = 0.0
        for s in SEEDS:
            w = generate(size=SIZE, seed=s, beta_j=BETA_J, mode=mode)
            total, cells = cheapest_path_cost(w.cost)
            straight = best_straight_cost(w.cost)
            detours.append(cells / SIZE)
            savings.append((straight - total) / straight)
            world_share = [float((w.terrain == i).mean())
                           for i in range(len(TERRAINS))]
            shares.append(world_share)
            world_present = sum(1 for x in world_share if x > 0)
            if world_present == len(TERRAINS):
                worlds_with_all_six += 1
            world_max_share = max(world_share)
            if world_max_share > worst_world_max_share:
                worst_world_max_share = world_max_share
        share = np.mean(shares, axis=0)
        detour = float(np.mean(detours))
        saving = float(np.mean(savings))
        if saving > 0.20 and detour > 1.15:
            verdict = "PASS"
        elif saving >= 0.10:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"
        present = int((share > 0).sum())
        out[mode] = dict(detour=detour, saving=saving, verdict=verdict,
                         share=share.tolist(),
                         terrains_present=present,
                         max_share=float(share.max()),
                         worlds_with_all_six_terrains=worlds_with_all_six,
                         worlds_total=len(SEEDS),
                         worst_world_max_share=worst_world_max_share)
        print(f"  {mode:>8}:  detour {detour:.3f}   saving {saving * 100:5.1f}%"
              f"   -> {verdict}")
        for i, t in enumerate(TERRAINS):
            print(f"             {t.glyph} {t.name:<14} {share[i] * 100:5.1f}%")
        variety = ("OK" if present == len(TERRAINS) and share.max() <= 0.50
                   else "POOR")
        print(f"             variety (averaged): {present}/{len(TERRAINS)} "
              f"present, max share {share.max() * 100:.1f}% -> {variety}")
        print(f"             variety (per-world, not gated): "
              f"{worlds_with_all_six}/{len(SEEDS)} worlds contain all six "
              f"terrains; worst single world's max terrain share "
              f"{worst_world_max_share * 100:.1f}%\n")

    b, n = out["bilinear"], out["nearest"]
    winner = "bilinear" if b["saving"] > n["saving"] else "nearest"
    print(f"  UPSAMPLE ARM: {winner} wins on cost saving "
          f"({b['saving'] * 100:.1f}% vs {n['saving'] * 100:.1f}%)")
    if winner == "nearest":
        print("  -> nearest is also the simpler operation and carries values "
              "without inventing them; it should become the default.")
    with open("audit/routing.json", "w") as fh:
        json.dump(dict(size=SIZE, beta_j=BETA_J, seeds=SEEDS, modes=out,
                       upsample_winner=winner), fh, indent=2)
    print("\n-> audit/routing.json")


if __name__ == "__main__":
    main()
