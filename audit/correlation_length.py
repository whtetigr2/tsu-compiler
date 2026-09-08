"""THE DECISIVE MEASUREMENT (cascade plan Task 4, Step 2).

Correlation length of a CASCADED 64x64 world against a FLAT 64x64 sampled with
the same rules and the same sampler settings. The flat control is what makes
this a measurement rather than an anecdote.

Hypothesis under test: local pairwise rules have a correlation length, so a
bigger flat grid gives more noise rather than more structure; deciding structure
at a coarse scale and refining should lengthen it.

If the cascaded correlation length is NOT materially longer than flat, the
hypothesis is REFUTED -- a legitimate outcome, and the search moves to the rules.

Pre-registered in commit bf5456d before any measurement was taken.
"""
import sys, json, math, time
sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np
from cascade_world import run_cascade, _compile_fine_layer, _decode_codewords, SIM_OUTPUT_DIR
from cascade import upsample, cascade_patch


def grid_to_array(decoded: dict, n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=int)
    for k, v in decoded.items():
        x, y = k[1:].split("_")
        a[int(y), int(x)] = v
    return a


def same_value_correlation(a: np.ndarray, max_r: int) -> list[float]:
    """C(r) = P(same value at distance r) - P(same by chance), normalised so
    C(0) = 1. Uses the categorical same-value indicator rather than a numeric
    product, because the state labels are NOT ordinal -- treating 0/1/2 as
    magnitudes would invent an ordering the model never had."""
    n = a.shape[0]
    vals, counts = np.unique(a, return_counts=True)
    p = counts / counts.sum()
    chance = float((p ** 2).sum())          # P(two independent cells match)
    out = []
    for r in range(max_r + 1):
        if r == 0:
            out.append(1.0); continue
        m = []
        if r < n:
            m.append(np.mean(a[:, :-r] == a[:, r:]))   # horizontal
            m.append(np.mean(a[:-r, :] == a[r:, :]))   # vertical
        obs = float(np.mean(m)) if m else chance
        out.append((obs - chance) / (1.0 - chance))    # 1 at r=0, 0 at chance
    return out


def corr_length(c: list[float]) -> float:
    """First r where C(r) drops below 1/e, linearly interpolated. Reported as
    `unavailable` by the caller if it never does."""
    thr = 1.0 / math.e
    for r in range(1, len(c)):
        if c[r] < thr:
            prev = c[r - 1]
            if prev == c[r]:
                return float(r)
            return (r - 1) + (prev - thr) / (prev - c[r])
    return float("nan")


def flat_world(n: int, seed: int) -> dict:
    """A FLAT n x n world: the same bipartite fine layer the cascade uses, with
    NO conditioning patch at all. This is the control -- same rules, same
    sampler, same size, no scale separation."""
    spec, enc, prog, report = _compile_fine_layer(n)
    from tsu.passes.lower import lower
    from tsu.backends.thrml_backend import sample as thrml_sample
    im = lower(enc.model)
    rows = thrml_sample(prog, n_chains=6, n_samples=40, n_warmup=600,
                        steps_per_sample=4, seed=seed)
    dec = _decode_codewords(rows, im, enc)
    if not dec:
        raise RuntimeError(f"flat {n}x{n} seed={seed}: no valid draw")
    return dec[-1]


if __name__ == "__main__":
    N, MAXR, STRENGTH = 64, 24, 1.0
    res = {"n": N, "strength": STRENGTH, "max_r": MAXR}
    t0 = time.perf_counter()
    print(f"cascade 8->16->32->{N} at strength {STRENGTH} ...", flush=True)
    levels = run_cascade([8, 16, 32, N], strength=STRENGTH, seed=0)
    casc = grid_to_array(levels[-1], N)
    print(f"  done {time.perf_counter()-t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    print(f"flat {N}x{N} control (no cascade) ...", flush=True)
    flat = grid_to_array(flat_world(N, seed=0), N)
    print(f"  done {time.perf_counter()-t1:.1f}s", flush=True)

    for label, arr in (("cascaded", casc), ("flat", flat)):
        c = same_value_correlation(arr, MAXR)
        xi = corr_length(c)
        res[label] = {"corr": [round(v, 4) for v in c],
                      "corr_length": (None if math.isnan(xi) else round(xi, 3)),
                      "mix": {int(v): round(float(f), 4) for v, f in
                              zip(*np.unique(arr, return_counts=True))}}
        shown = "unavailable: C(r) never falls below 1/e within max_r" if math.isnan(xi) else f"{xi:.3f} cells"
        print(f"\n{label:>9}: corr_length = {shown}")
        print(f"           C(r) r=0..8: {[round(v,3) for v in c[:9]]}")
    json.dump(res, open("audit/correlation_length.json", "w"), indent=1)
    print("\n-> audit/correlation_length.json")
