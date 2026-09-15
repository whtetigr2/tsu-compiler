"""Does the in-browser sampler on the published page sample the right thing?

================================ PROTOCOL ================================
SYSTEM DEFINITION   The JavaScript Gibbs sampler embedded in the interactive
                    page "The Diagnostic That Lied", ported line-for-line into
                    Python and run against a brute-force reference.
STATE VARIABLES     Mean nearest-neighbour correlation on a 4x4 periodic Ising
                    lattice in zero field, at both coupling signs, across
                    beta*J spanning Onsager's Kc = 0.440687.
TRANSITION RULES    The page's own update, transcribed exactly:
                        h = sign * beta_j * sum(neighbour spins)
                        p(up) = 1 / (1 + exp(-2h))
                    swept in checkerboard parity order, periodic boundaries.
ALLOWED OPERATIONS  Sampling; enumeration; comparing the two.
FORBIDDEN OPERATIONS
                    No tuning the tolerance to fit the result. No changing the
                    page to match this -- if they disagree, the PAGE is wrong
                    and the finding is that it was published unverified.
ASSUMPTIONS         That this transcription matches the shipped JavaScript. That
                    is the one link a test cannot close, so it is stated rather
                    than implied: the rule above is reproduced verbatim in the
                    page and in this docstring so a reader can diff them by eye.
INVARIANTS          sign = +1 must give POSITIVE neighbour correlation
                    (ferromagnetic) and sign = -1 NEGATIVE (antiferromagnetic),
                    matching the labels the page puts on its own toggle.
MEASUREMENTS        Mean neighbour correlation, sampled vs enumerated.
NULL HYPOTHESES     "The page's sampler samples the model it claims to."
                    Refutable: a sign error, a wrong factor of 2, or a broken
                    parity sweep all move the correlation off the exact value.
SUCCESS CRITERIA    Agreement within 0.01 at every configuration.
FAILURE CRITERIA    Any disagreement beyond that, or a sign that contradicts the
                    page's own labelling of the toggle.
PROVENANCE          `audit/oracles/exact.py` enumerates all 2**16 states from
                    E = -sum J s s - sum b s and does NOT import this package,
                    so it cannot inherit a convention from the code it checks.
SCOPE OF VALIDITY   Two of the page's three load-bearing computations. The
                    update rule is checked at 4x4 only, because the reference
                    enumerates 2**n. The R-hat accumulator is checked against
                    tsu_compiler's own across six series. NOT covered: the free
                    energy histogram, and the page's 16x16 rendering.
==========================================================================

WHY THIS EXISTS. The page was written, published, and only then checked -- the
exact ordering this project warns other people about. A fresh sampler with no
oracle behind it is where sign conventions bite, and this project has shipped a
wrong-signed coupling before. Worse, the page's whole argument is that a number
can look right while being structurally wrong; publishing it on top of an
unverified sampler would have been the same mistake one level up.

RESULT (2026-09-15):

     sign   beta*J     JS rule      oracle       diff
        1     0.20      0.2278      0.2281     0.0003
        1     0.44      0.7861      0.7814     0.0047
        1     0.80      0.9925      0.9924     0.0001
       -1     0.20     -0.2284     -0.2281     0.0004
       -1     0.44     -0.7845     -0.7814     0.0031
       -1     0.80     -0.9923     -0.9924     0.0001

The largest gap sits at beta*J = 0.44, essentially on the critical point, where
fluctuations are largest and a finite sample is least precise -- the expected
place for it, not a suspicious one. Signs match the page's labels.
"""
import math
import sys

sys.path.insert(0, "audit")
sys.path.insert(0, "src")

import numpy as np

from oracles.exact import exact_boltzmann

L = 4
N = L * L
TOL = 0.01
NBR = [[((i + 1) % L) * L + j, ((i - 1) % L) * L + j,
        i * L + (j + 1) % L, i * L + (j - 1) % L]
       for i in range(L) for j in range(L)]
EDGES = sorted({(min(k, n), max(k, n)) for k in range(N) for n in NBR[k]})
PARITY = [(i + j) % 2 for i in range(L) for j in range(L)]


def page_sampler(sign: int, beta_j: float, n_sweep=60_000, burn=2_000, seed=7):
    """The page's update rule, transcribed. Deliberately written the way the
    JavaScript is written -- loops over parity then sites -- rather than
    vectorised, so the two can be compared by reading them side by side."""
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], N).astype(np.int64)
    total, count = 0.0, 0
    for t in range(n_sweep):
        for par in (0, 1):
            for k in range(N):
                if PARITY[k] != par:
                    continue
                f = sum(s[n] for n in NBR[k])
                h = sign * beta_j * f
                s[k] = 1 if rng.random() < 1 / (1 + math.exp(-2 * h)) else -1
        if t > burn:
            total += float(np.mean([s[a] * s[b] for a, b in EDGES]))
            count += 1
    return total / count


def exact(sign: int, beta_j: float) -> float:
    states, probs = exact_boltzmann({ab: float(sign) for ab in EDGES},
                                    [0.0] * N, beta_j)
    P = np.asarray(probs)
    S = np.array([[2 * v - 1 for v in x] for x in states])
    return float(np.mean([P @ (S[:, a] * S[:, b]) for a, b in EDGES]))


def page_rhat(series) -> float:
    """The page's OTHER load-bearing computation, transcribed.

    It accumulates Gelman-Rubin from STREAMING Welford moments -- one sample at
    a time, no history retained -- where `tsu_compiler` computes it two-pass
    over a stored array. The two are algebraically identical, which is exactly
    why this is worth checking numerically rather than by reading: streaming
    accumulation is where floating-point drift hides, and the page's two
    headline numbers are both outputs of this function. A correct sampler feeding
    a broken accumulator would still put wrong numbers on the screen."""
    m, n = len(series), len(series[0])
    means, m2, cnt = [0.0] * m, [0.0] * m, [0] * m
    for c in range(m):
        for x in series[c]:
            cnt[c] += 1
            d = x - means[c]
            means[c] += d / cnt[c]
            m2[c] += d * (x - means[c])
    variances = [m2[c] / (n - 1) for c in range(m)]
    W = sum(variances) / m
    gm = sum(means) / m
    if W <= 0:
        return math.inf if any((mm - gm) ** 2 > 0 for mm in means) else 1.0
    B = sum((mm - gm) ** 2 for mm in means) * n / (m - 1)
    return math.sqrt((((n - 1) / n) * W + B / n) / W)


def check_accumulators(failures: list) -> None:
    """Six series chosen so each could break the accumulator differently."""
    from tsu_compiler.preflight.diagnostics import r_hat

    rng = np.random.default_rng(11)
    cases = {}
    cases["iid noise"] = rng.normal(size=(16, 500))
    x = rng.normal(size=(16, 500))
    for t in range(1, 500):
        x[:, t] = 0.9 * x[:, t - 1] + x[:, t]
    cases["correlated AR(1)"] = x
    # identical chains are the published edge case: R-hat is exactly
    # sqrt((n-1)/n), strictly BELOW 1, and an accumulator that quietly clamps
    # to 1 would pass every other row here
    cases["identical chains"] = np.tile(rng.normal(size=(1, 500)), (16, 1))
    split = rng.normal(size=(16, 500)) * 0.01
    split[:8] += 1.0
    split[8:] -= 1.0
    cases["two frozen modes"] = split
    cases["spin-like +-1"] = rng.choice([-1.0, 1.0], size=(16, 500))
    # a large offset breaks a naive sum-of-squares accumulator while leaving
    # Welford unharmed -- the specific failure Welford exists to prevent
    cases["large offset 1e6"] = rng.normal(size=(16, 500)) + 1e6

    print(f"\n{'series':>20} {'tsu_compiler':>15} {'page (Welford)':>16} {'rel diff':>11}")
    for name, v in cases.items():
        ours = float(r_hat(v))
        theirs = page_rhat([list(r) for r in v])
        rel = abs(ours - theirs) / max(1e-12, abs(ours))
        print(f"{name:>20} {ours:>15.9f} {theirs:>16.9f} {rel:>11.2e}")
        if rel >= 1e-9:
            failures.append(f"R-hat accumulator drifts on '{name}': "
                            f"{ours:.9f} vs {theirs:.9f} (relative {rel:.2e})")
    identical = page_rhat([list(r) for r in cases["identical chains"]])
    expected = math.sqrt(499 / 500)
    if abs(identical - expected) > 1e-9:
        failures.append(f"identical chains gave {identical:.9f}, not the "
                        f"published sqrt((n-1)/n) = {expected:.9f}")


def main() -> int:
    failures = []
    print(f"{'sign':>5} {'beta*J':>7} {'page rule':>11} {'oracle':>10} {'diff':>9}")
    for sign in (1, -1):
        for bj in (0.20, 0.44, 0.80):
            got, want = page_sampler(sign, bj), exact(sign, bj)
            d = abs(got - want)
            print(f"{sign:>5} {bj:>7.2f} {got:>11.4f} {want:>10.4f} {d:>9.4f}")
            if d >= TOL:
                failures.append(f"sign={sign} beta*J={bj}: {got:.4f} vs exact "
                                f"{want:.4f}, off by {d:.4f} (tolerance {TOL})")
            # the toggle's own labels must hold
            if sign > 0 and want <= 0:
                failures.append(f"sign=+1 at beta*J={bj} is not ferromagnetic")
            if sign < 0 and want >= 0:
                failures.append(f"sign=-1 at beta*J={bj} is not antiferromagnetic")
    check_accumulators(failures)
    print()
    if failures:
        print("CONTROL FAILURES -- the published page is wrong:")
        for f in failures:
            print("  " + f)
        return 1
    print("  The page's update rule reproduces the exact distribution, its")
    print("  ferro/antiferro labels match the sign of the correlation it")
    print("  actually produces, and its streaming R-hat accumulator agrees with")
    print("  tsu_compiler's two-pass one to floating-point noise -- including")
    print("  the identical-chains edge case, where the published answer is")
    print("  sqrt((n-1)/n), strictly BELOW 1.")
    print("  Still NOT covered: the free energy histogram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
