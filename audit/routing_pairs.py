"""Task 7B: endpoint-constrained navigation. PRE-REGISTERED.

Committed BEFORE it is run. The bands are NOT adjusted afterwards.

WHY THIS EXISTS. Task 7 returned FAIL (bilinear 4.1%, nearest 7.4% cost saving)
and THAT RESULT STANDS -- it answered the question it was registered to answer.
But the question was the wrong one. Its baseline was `min(row_costs)`, the
cheapest of 64 straight traversals anywhere on the map, which is a best-of-64
order statistic no player is ever handed. It asked "can the optimal route beat
the luckiest straight line?" when the design question is "does terrain
meaningfully affect how an agent moves between two places?"

Task 7's post-hoc diagnostic found 35.6% saving against a median row. That number
is DIAGNOSTIC EVIDENCE THAT TASK 7'S BASELINE WAS DEFECTIVE, NOTHING MORE. It was
computed with the result already known, so it cannot be retro-fitted as a
pre-registration and is not offered as evidence of a pass anywhere below.

THE PROTOCOL, fixed here before any measurement:

  Worlds        30 per arm, seeds 0..29, arms "nearest" and "bilinear".
                Identical procedure across every world and both arms.

  Endpoints     12 pairs per world. A dedicated RNG seeded `1000 + world_seed`
                draws uniform random cells; a pair is kept when its Euclidean
                separation is >= MIN_SEP (32 cells, half the map width).
                ENDPOINT SELECTION NEVER READS TERRAIN, COST, OR HEIGHT -- it
                cannot be biased toward interesting routes. Attempts are capped
                at MAX_ATTEMPTS and any shortfall is recorded rather than
                silently backfilled. Every sampled pair is written to the JSON,
                including degenerate ones.

  Movement      8-connected. The cost of a move is the cost of the cell ENTERED;
                a path's cost is cost(start) plus the cost of every cell entered.
                Diagonal and orthogonal moves cost the same, which is what makes
                the baseline below a fair comparison rather than a handicap.

  Baseline      The direct line is discretised by INTEGER BRESENHAM between the
                two endpoints, inclusive of both, and its cost is the sum of the
                costs of exactly those cells. Bresenham and an 8-connected
                optimal path need the same number of cells on a uniform map
                (max(|dx|,|dy|) + 1), so on flat terrain the two costs coincide
                and the measured saving is ~0. That is what makes arm C a real
                control rather than a formality.

  A  geometric detour   = geometric path length / Euclidean(start, end), where
                          geometric length counts 1 per orthogonal move and
                          sqrt(2) per diagonal. Flat control should give ~1.0.
  B  cost saving        = (bresenham_cost - optimal_cost) / bresenham_cost
  C  flat control       = A and B recomputed on a uniform cost-1 world over the
                          IDENTICAL endpoints. Quantifies how much of any
                          measured effect is an artifact of the method rather
                          than of terrain.

BANDS -- unchanged from Task 7, deliberately. Choosing new bands after a failure
is how this goes wrong.
    PASS      cost saving > 20%
    MARGINAL  cost saving 10-20%
    FAIL      cost saving < 10%

ONE ADDITION, stricter than Task 7 rather than looser: the verdict requires the
MEAN AND THE MEDIAN to both clear the band. A world where two spectacular routes
carry a 30% average while most routes are untouched is not a world where terrain
shapes movement, and a mean alone cannot tell those apart. Distributions --
median, quartiles, min, max, and every per-pair result -- are reported, not just
aggregate means.

IF THIS FAILS, the next step is to diagnose the terrain COST MODEL and terrain
DISTRIBUTION, not to build presentation or 3-D rendering. IF IT PASSES, Task 8
unlocks.
"""
import sys
import json
import math
import heapq

sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np

from world.generate import generate

SIZE = 64
BETA_J = 0.42
WORLD_SEEDS = list(range(30))
MODES = ("nearest", "bilinear")
PAIRS_PER_WORLD = 12
MIN_SEP = 32.0
MAX_ATTEMPTS = 500
PAIR_RNG_OFFSET = 1000

_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1),
          (-1, -1), (-1, 1), (1, -1), (1, 1))


def choose_pairs(size: int, world_seed: int) -> tuple[list, int, int]:
    """Deterministic, terrain-blind endpoint selection.

    Returns (pairs, attempts, shortfall). Draws uniform random cells and keeps a
    pair when the endpoints are at least MIN_SEP apart. Reads nothing about the
    world -- it is handed only the grid size -- so it cannot prefer interesting
    routes. A shortfall is returned rather than backfilled.
    """
    rng = np.random.default_rng(PAIR_RNG_OFFSET + world_seed)
    pairs, attempts = [], 0
    while len(pairs) < PAIRS_PER_WORLD and attempts < MAX_ATTEMPTS:
        attempts += 1
        y0, x0, y1, x1 = (int(v) for v in rng.integers(0, size, 4))
        if math.hypot(x1 - x0, y1 - y0) >= MIN_SEP:
            pairs.append((y0, x0, y1, x1))
    return pairs, attempts, PAIRS_PER_WORLD - len(pairs)


def bresenham(y0: int, x0: int, y1: int, x1: int) -> list[tuple[int, int]]:
    """Integer Bresenham, inclusive of both endpoints. This is the direct-line
    discretisation named in the protocol; it is fixed here so the baseline
    cannot drift."""
    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        cells.append((y, x))
        if (y, x) == (y1, x1):
            return cells
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def line_cost(cost: np.ndarray, cells) -> int:
    return int(sum(int(cost[y, x]) for y, x in cells))


def optimal_route(cost: np.ndarray, y0: int, x0: int, y1: int, x1: int):
    """8-connected Dijkstra from one point to another.

    Cost of entering a cell is that cell's cost; the total includes the start
    cell, matching `line_cost` so the two are directly comparable. Returns
    (total_cost, orthogonal_moves, diagonal_moves).
    """
    n, m = cost.shape
    dist = np.full((n, m), np.inf)
    prev: dict = {}
    dist[y0, x0] = float(cost[y0, x0])
    heap = [(float(cost[y0, x0]), y0, x0)]
    prev[(y0, x0)] = None
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        if (y, x) == (y1, x1):
            break
        for dy, dx in _MOVES:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < m:
                nd = d + float(cost[ny, nx])
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(heap, (nd, ny, nx))
    if not np.isfinite(dist[y1, x1]):
        raise RuntimeError(
            f"no route from ({y0},{x0}) to ({y1},{x1}) -- impossible when every "
            f"cell is walkable; the cost grid is wrong")
    ortho = diag = 0
    node = (y1, x1)
    while prev[node] is not None:
        py, px = prev[node]
        if py != node[0] and px != node[1]:
            diag += 1
        else:
            ortho += 1
        node = (py, px)
    return float(dist[y1, x1]), ortho, diag


def summarise(vals: list[float]) -> dict:
    a = np.array(vals, dtype=float)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                min=float(a.min()), max=float(a.max()))


def verdict_for(mean: float, median: float) -> str:
    """Both the mean AND the median must clear a band. See this module's
    docstring for why a mean alone is not sufficient."""
    if mean > 0.20 and median > 0.20:
        return "PASS"
    if mean >= 0.10 and median >= 0.10:
        return "MARGINAL"
    return "FAIL"


def run_arm(mode: str, flat: bool) -> tuple[dict, list]:
    savings, detours, rows = [], [], []
    for ws in WORLD_SEEDS:
        w = generate(size=SIZE, seed=ws, beta_j=BETA_J, mode=mode)
        cost = np.ones_like(w.cost) if flat else w.cost
        pairs, attempts, shortfall = choose_pairs(SIZE, ws)
        for (y0, x0, y1, x1) in pairs:
            line = bresenham(y0, x0, y1, x1)
            base = line_cost(cost, line)
            opt, ortho, diag = optimal_route(cost, y0, x0, y1, x1)
            euclid = math.hypot(x1 - x0, y1 - y0)
            glen = ortho + diag * math.sqrt(2.0)
            saving = (base - opt) / base if base else 0.0
            detour = glen / euclid if euclid else float("nan")
            savings.append(saving)
            detours.append(detour)
            rows.append(dict(world_seed=ws, mode=mode, flat=flat,
                             start=[y0, x0], end=[y1, x1],
                             euclid=euclid, line_cells=len(line),
                             baseline_cost=base, optimal_cost=opt,
                             ortho=ortho, diag=diag, geometric_length=glen,
                             saving=saving, detour=detour,
                             attempts=attempts, shortfall=shortfall))
    return dict(saving=summarise(savings), detour=summarise(detours)), rows


def main():
    print(f"Task 7B -- endpoint-constrained navigation, PRE-REGISTERED")
    print(f"{SIZE}x{SIZE}, beta*J={BETA_J}, {len(WORLD_SEEDS)} worlds/arm, "
          f"{PAIRS_PER_WORLD} pairs/world, min separation {MIN_SEP:.0f} cells\n")
    out, all_rows = {}, []
    arms = [(m, False) for m in MODES] + [("nearest", True)]
    for mode, flat in arms:
        label = f"{mode} FLAT-CONTROL" if flat else mode
        stats, rows = run_arm(mode, flat)
        all_rows += rows
        s, d = stats["saving"], stats["detour"]
        v = verdict_for(s["mean"], s["median"])
        out[label] = dict(stats=stats, verdict=v)
        print(f"  {label}")
        print(f"    B cost saving   mean {s['mean']*100:6.1f}%   "
              f"median {s['median']*100:6.1f}%   "
              f"IQR {s['p25']*100:.1f}%..{s['p75']*100:.1f}%   "
              f"range {s['min']*100:.1f}%..{s['max']*100:.1f}%")
        print(f"    A detour        mean {d['mean']:6.3f}    "
              f"median {d['median']:6.3f}    "
              f"IQR {d['p25']:.3f}..{d['p75']:.3f}")
        print(f"    -> {v}   (n={s['n']} pairs)\n")

    print("  C FLAT-WORLD CONTROL reads as the method's own bias: a saving far "
          "from 0%\n    or a detour far from 1.000 would mean the comparison is "
          "unfair before\n    terrain is even considered.")
    with open("audit/routing_pairs.json", "w") as fh:
        json.dump(dict(size=SIZE, beta_j=BETA_J, world_seeds=WORLD_SEEDS,
                       pairs_per_world=PAIRS_PER_WORLD, min_sep=MIN_SEP,
                       arms=out, pairs=all_rows), fh, indent=2)
    print("\n-> audit/routing_pairs.json (every per-pair result)")


if __name__ == "__main__":
    main()
