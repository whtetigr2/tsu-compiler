"""Task 7C: does sampled terrain beat its own shuffle? PRE-REGISTERED.

Committed BEFORE running. The verdict rule below is fixed here.

WHY THIS EXISTS. Task 7B returned PASS and that verdict is VOID -- not FAIL, but
void, because the criterion did not test the claim. Its null was a FLAT world
(no cost variance). The claim "terrain shapes routing" needs the null where the
cost distribution is IDENTICAL but the spatial structure is destroyed. Measured
in the controller session over 10 worlds, 120 pairs:

    null                            saving mean   detour mean
    REAL (TSU-sampled)                27.2%         1.2585
    SHUFFLED (same cost multiset)     37.7%         1.2256
    FLAT (uniform)                     0.0%         1.2400

Shuffling a real world's costs RAISES the saving score. So metric B rewards
fine-grained cost heterogeneity -- which noise maximises and which clumping at
beta*J = 0.42 actively reduces -- and cannot distinguish a sampled field from
white noise. Task 7B's PASS therefore supports "movement cost is not uniform",
not "terrain shapes routing".

The same probe showed metric A (geometric detour) DOES separate real from
shuffled, in the direction structure predicts. Task 7B had withdrawn metric A
because it looked dead against the FLAT null (1.2400), which sits between real
and shuffled -- the wrong comparison made the working metric look broken and the
broken metric look working.

THIS IS A CONFIRMATORY TEST, NOT A BLIND ONE, AND SAYS SO. The direction was
already seen in a 10-world probe. What is genuinely unknown, and what this
measures: whether the separation survives a per-world PAIRED test at 30 worlds
across both upsample arms, against each world's OWN shuffle distribution rather
than a pooled average.

PROTOCOL, fixed before running:

  Worlds      30 per mode (seeds 0..29), modes "nearest" and "bilinear".
  Endpoints   the same terrain-blind deterministic selection Task 7B used --
              `choose_pairs(size, world_seed)` from audit/routing_pairs.py,
              12 pairs per world, minimum separation 32 cells. Reused verbatim
              so endpoint choice cannot differ between the two studies.
  Arms        REAL      the generated cost grid
              SHUFFLED  N_SHUFFLE independent permutations of that same grid,
                        seeded deterministically per (world, rep). Identical
                        multiset, spatial structure destroyed.
              FLAT      uniform cost 1, retained ONLY as a method-bias check.

  Metrics     A geometric detour = geometric path length / Euclidean distance
              B cost saving      = (bresenham - optimal) / bresenham

  PERMUTATION TEST, the primary analysis. For each world, the per-world mean
  detour is computed for REAL and for each of the N_SHUFFLE shuffles. The
  world's p-value is (count of shuffles with detour >= real, plus one) divided
  by (N_SHUFFLE + 1). With N_SHUFFLE = 19 a world reaches p = 0.05 exactly when
  its real detour exceeds ALL nineteen of its own shuffles.

VERDICT RULE, fixed here. Under the null of no spatial structure, 5% of worlds
reach p <= 0.05 by chance -- 1.5 of 30.

    PASS      >= 15 of 30 worlds reach p <= 0.05, in BOTH modes (ten times the
              chance rate)
    MARGINAL  5 to 14 worlds in either mode
    FAIL      < 5 worlds -- indistinguishable from the 1.5 expected by chance

Metric B is reported against the shuffled null but NOT gated: the probe already
showed it runs the wrong way, and this run's job is to establish that formally
rather than to give it another chance to pass.

IF THIS FAILS, the sampled fields do not produce spatial structure that affects
navigation, and the next step is the terrain COST MODEL and DISTRIBUTION -- not
presentation, not 3-D. IF IT PASSES, Task 7B's void verdict is replaced by this
one and the claim is stated in these terms only.
"""
import sys
import json
import math

sys.path.insert(0, "src")
sys.path.insert(0, "demo")
sys.path.insert(0, "audit")
import numpy as np

from world.generate import generate
from routing_pairs import (choose_pairs, bresenham, line_cost, optimal_route,
                           SIZE, BETA_J)

WORLD_SEEDS = list(range(30))
MODES = ("nearest", "bilinear")
N_SHUFFLE = 19
SHUFFLE_SEED_BASE = 90000


def measure(cost: np.ndarray, pairs) -> tuple[float, float]:
    """Mean geometric detour and mean cost saving over one world's pairs."""
    det, sav = [], []
    for (y0, x0, y1, x1) in pairs:
        base = line_cost(cost, bresenham(y0, x0, y1, x1))
        opt, ortho, diag = optimal_route(cost, y0, x0, y1, x1)
        euclid = math.hypot(x1 - x0, y1 - y0)
        det.append((ortho + diag * math.sqrt(2.0)) / euclid)
        sav.append((base - opt) / base if base else 0.0)
    return float(np.mean(det)), float(np.mean(sav))


def shuffled(cost: np.ndarray, world_seed: int, rep: int) -> np.ndarray:
    """Same multiset, no spatial structure. Deterministic per (world, rep)."""
    rng = np.random.default_rng(SHUFFLE_SEED_BASE + world_seed * 100 + rep)
    flat = cost.flatten().copy()
    rng.shuffle(flat)
    return flat.reshape(cost.shape)


def main():
    print("Task 7C -- sampled terrain against its own shuffle. PRE-REGISTERED.")
    print(f"{SIZE}x{SIZE}, beta*J={BETA_J}, {len(WORLD_SEEDS)} worlds/mode, "
          f"{N_SHUFFLE} shuffles/world\n")
    out, rows = {}, []
    for mode in MODES:
        sig = 0
        d_real, d_shuf, s_real, s_shuf, pvals = [], [], [], [], []
        for ws in WORLD_SEEDS:
            w = generate(size=SIZE, seed=ws, beta_j=BETA_J, mode=mode)
            pairs = choose_pairs(SIZE, ws)[0]
            dr, sr = measure(w.cost, pairs)
            ds = [measure(shuffled(w.cost, ws, r), pairs) for r in range(N_SHUFFLE)]
            det_s = [d for d, _ in ds]
            sav_s = [s for _, s in ds]
            p = (sum(1 for d in det_s if d >= dr) + 1) / (N_SHUFFLE + 1)
            sig += (p <= 0.05)
            d_real.append(dr); d_shuf.append(float(np.mean(det_s)))
            s_real.append(sr); s_shuf.append(float(np.mean(sav_s)))
            pvals.append(p)
            rows.append(dict(mode=mode, world_seed=ws, detour_real=dr,
                             detour_shuffled_mean=float(np.mean(det_s)),
                             detour_shuffled_max=float(np.max(det_s)),
                             saving_real=sr, saving_shuffled_mean=float(np.mean(sav_s)),
                             p_value=p, significant=bool(p <= 0.05)))
        flat_d, flat_s = [], []
        for ws in WORLD_SEEDS[:5]:
            w = generate(size=SIZE, seed=ws, beta_j=BETA_J, mode=mode)
            d, s = measure(np.ones_like(w.cost), choose_pairs(SIZE, ws)[0])
            flat_d.append(d); flat_s.append(s)

        verdict = ("PASS" if sig >= 15 else "MARGINAL" if sig >= 5 else "FAIL")
        out[mode] = dict(worlds_significant=sig, n_worlds=len(WORLD_SEEDS),
                         detour_real=float(np.mean(d_real)),
                         detour_shuffled=float(np.mean(d_shuf)),
                         saving_real=float(np.mean(s_real)),
                         saving_shuffled=float(np.mean(s_shuf)),
                         flat_detour=float(np.mean(flat_d)),
                         flat_saving=float(np.mean(flat_s)),
                         median_p=float(np.median(pvals)), verdict=verdict)
        o = out[mode]
        print(f"  {mode}")
        print(f"    A detour   real {o['detour_real']:.4f}   "
              f"shuffled {o['detour_shuffled']:.4f}   flat {o['flat_detour']:.4f}")
        print(f"    B saving   real {o['saving_real']*100:5.1f}%  "
              f"shuffled {o['saving_shuffled']*100:5.1f}%  "
              f"flat {o['flat_saving']*100:5.1f}%   (reported, NOT gated)")
        print(f"    permutation test: {sig}/{len(WORLD_SEEDS)} worlds with "
              f"p <= 0.05   median p {o['median_p']:.3f}")
        print(f"    -> {verdict}\n")

    verdicts = [out[m]["verdict"] for m in MODES]
    overall = ("PASS" if all(v == "PASS" for v in verdicts)
               else "FAIL" if any(v == "FAIL" for v in verdicts) else "MARGINAL")
    print(f"  OVERALL: {overall}   (PASS requires both modes)")
    print(f"  chance expectation under the null: 1.5 of 30 worlds")
    with open("audit/routing_shuffled.json", "w") as fh:
        json.dump(dict(size=SIZE, beta_j=BETA_J, n_shuffle=N_SHUFFLE,
                       world_seeds=WORLD_SEEDS, modes=out, overall=overall,
                       per_world=rows), fh, indent=2)
    print("\n-> audit/routing_shuffled.json (per-world p-values)")


if __name__ == "__main__":
    main()
