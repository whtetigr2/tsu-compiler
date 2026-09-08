"""Is the sampler actually mixing at the operating point? (spec section 8.4)

PRE-REGISTERED. This file is committed BEFORE it is run, so the prediction
cannot be edited to fit whatever comes back. Same discipline that produced two
refutations already in this project.

WHY THIS IS OWED. beta*J = 0.42 is 0.95x Onsager's Kc, and block Gibbs suffers
CRITICAL SLOWING DOWN near a phase transition: the autocorrelation time
diverges, so a warmup that was ample in the disordered phase can be far too
short here. The criticality sweep already contains a demonstrable artifact of
exactly this kind -- at beta*J = 1.1 the magnetization FELL as coupling rose
(0.992 -> 0.924), which is freezing, not physics. No mixing diagnostic has ever
been run on this project, and every downstream number inherits the answer.

PREDICTION: the field's mean level, its standard deviation and its level
histogram are statistically indistinguishable between warmup 4000 and 16000.
FALSIFIER: any of those moves materially (mean level by more than 0.1 of a
level, or any histogram bucket by more than 5 percentage points). If it does,
warmup 4000 is too short, the criticality sweet-spot numbers are under-warmed,
and every measurement built on them is provisional until re-run.

A NULL RESULT HERE IS THE GOOD OUTCOME and gets reported as plainly as a
failure would.
"""
import sys
import json

sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np

from world.fields import sample_field

SIZE = 64
LAYERS = 3
BETA_J = 0.42
SEEDS = [0, 1, 2, 3, 4]
WARMUPS = [4000, 16000]


def summarise(f: np.ndarray) -> dict:
    hist = [float((f == lv).mean()) for lv in range(LAYERS + 1)]
    return dict(mean=float(f.mean()), std=float(f.std()), hist=hist)


def main():
    print(f"Mixing check: {SIZE}x{SIZE}, beta*J = {BETA_J}, "
          f"warmups {WARMUPS}, seeds {SEEDS}\n")
    out = {}
    for w in WARMUPS:
        stats = [summarise(sample_field(SIZE, LAYERS, BETA_J, s, warmup=w))
                 for s in SEEDS]
        out[str(w)] = dict(
            mean=float(np.mean([s["mean"] for s in stats])),
            std=float(np.mean([s["std"] for s in stats])),
            hist=[float(np.mean([s["hist"][i] for s in stats]))
                  for i in range(LAYERS + 1)])
        print(f"  warmup {w:>6}:  mean={out[str(w)]['mean']:.4f}  "
              f"std={out[str(w)]['std']:.4f}  "
              f"hist={[round(h, 4) for h in out[str(w)]['hist']]}")

    a, b = out[str(WARMUPS[0])], out[str(WARMUPS[1])]
    d_mean = abs(a["mean"] - b["mean"])
    d_hist = max(abs(x - y) for x, y in zip(a["hist"], b["hist"]))
    verdict = "MIXED" if (d_mean <= 0.1 and d_hist <= 0.05) else "NOT MIXED"
    print(f"\n  delta mean = {d_mean:.4f} (threshold 0.1)")
    print(f"  max delta hist bucket = {d_hist:.4f} (threshold 0.05)")
    print(f"\n  VERDICT: {verdict}")
    if verdict == "NOT MIXED":
        print("  -> warmup 4000 is too short. Every downstream measurement is "
              "provisional until re-run at the longer warmup.")

    with open("audit/mixing.json", "w") as fh:
        json.dump(dict(size=SIZE, beta_j=BETA_J, seeds=SEEDS, runs=out,
                       delta_mean=d_mean, delta_hist=d_hist,
                       verdict=verdict), fh, indent=2)
    print("\n-> audit/mixing.json")


if __name__ == "__main__":
    main()
