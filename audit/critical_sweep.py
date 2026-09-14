"""Correlation length vs coupling strength, against Onsager's critical point.

WHY THIS EXPERIMENT EXISTS. Three separate attempts to get spatial structure
out of a sampled world failed: the 8x8 emergence run (diagnosed "canvas too
small"), the flat 64x64 (corr length 0.634 cells), and the coarse-to-fine
cascade (0.679 cells, refuted in 5a21393). A single piece of physics explains
all three at once, and it is checkable.

Onsager's exact result for the uniform 2-D square-lattice Ising ferromagnet in
ZERO FIELD puts the critical coupling at Kc = ln(1+sqrt(2))/2 = 0.4407. Below
it the model is DISORDERED and its correlation length is order one lattice
spacing -- by physics, not by any defect. Every world this project has sampled
sat at beta*|J| = 0.2, which is 45% of Kc. So a correlation length near one
cell is the CORRECT behaviour of the model we actually built, and no amount of
canvas size or scale separation could have changed it.

Two things had to be fixed before the question could even be asked:

  1. ZERO FIELD. The one-sided clumping rule every layer uses
     (a_value=1, b_value=1) is not a pure ferromagnet. Under the binary
     encoding x=(s+1)/2, `w*x_i*x_j` summed over edges expands to a coupling
     PLUS a uniform field of w*deg/4 -- FOUR TIMES the coupling at degree 4.
     Measured, exactly: |b|/|J| = 4.000. That field pins every cell toward one
     value and is the mechanism behind the previously observed P(bit=0) of
     11-15%. Adding the mirrored term (a_value=0, b_value=0) cancels the field
     to exactly 0.0000 and doubles the coupling -- measured, and still
     bipartite, so instant placement survives.

  2. beta*|J| READ FROM THE COMPILED MODEL, never assumed from the spec
     weight. The encoder's weight -> J mapping is its own business; this
     script asks the lowered IsingModel what J actually is.

PRE-REGISTERED PREDICTION (written before the sweep was run):
  - well below Kc: corr length ~1 cell, |m| small (disordered)
  - approaching Kc: corr length grows sharply
  - above Kc: long-range order, |m| -> 1
FALSIFIER: if corr length stays ~1 cell all the way past Kc, the
disordered-phase explanation is WRONG and something else suppresses
correlation. That outcome gets reported exactly as loudly as a confirmation.

CAVEAT, stated up front: Onsager is exact for an INFINITE lattice with UNIFORM
coupling and no field. This is a finite 32x32 without periodic boundaries, so
the transition is rounded and shifted -- Kc is an ORIENTING LINE, not a
predicted crossing point. The claim under test is the qualitative one (a
disordered phase with short correlation, and growth toward Kc), not the exact
location.
"""
import sys, os, json, math, time, tempfile
sys.path.insert(0, "src")
sys.path.insert(0, "demo")
import numpy as np

from tsu_compiler.spec import load_spec
from tsu_compiler.passes.encode import encode
from tsu_compiler.passes.lower import lower
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.place import place
from tsu_compiler.passes.route import route
from tsu_compiler.passes.program import build_program
from tsu_compiler.backends.thrml_backend import sample as thrml_sample
from tsu_compiler.target import PROFILES

N = 32
MAX_R = 12
WEIGHTS = [-0.15, -0.25, -0.35, -0.40, -0.44, -0.50, -0.60, -0.80, -1.10]
SAMPLE = dict(n_chains=6, n_samples=30, n_warmup=3000, steps_per_sample=8)

BODY = """name: crit_sweep
generate:
  kind: grid
  width: {n}
  height: {n}
  variable_domain: {{domain: binary}}
terms:
  - {{kind: product_over_edges, a_value: 1, b_value: 1, weight: {w}}}
  - {{kind: product_over_edges, a_value: 0, b_value: 0, weight: {w}}}
"""


def onsager_kc() -> float:
    """Kc for the uniform 2-D square-lattice Ising model: sinh(2Kc) = 1, so
    Kc = arcsinh(1)/2 = ln(1+sqrt(2))/2. Computed, never a pasted decimal."""
    return math.asinh(1.0) / 2.0


def compile_layer(n: int, w: float):
    f = os.path.join(tempfile.gettempdir(), "crit_%d_%s.yaml" % (n, w))
    with open(f, "w") as fh:
        fh.write(BODY.format(n=n, w=w))
    spec = load_spec(f)
    enc = encode(spec, "domain_wall")
    ising = lower(enc.model)
    report = analyse(ising)
    assert report.bipartite is True, (
        "w=%s: analyse() reports bipartite=False -- refusing place() per "
        "THE SAFETY RULE" % w)
    placement = place(ising, report, PROFILES["z1"], restarts=12, iters=200_000)
    assert placement.mediation is None, "w=%s: unexpected mediation" % w
    j = float(np.abs(ising.weights).max()) if ising.weights.size else 0.0
    b = float(np.abs(ising.biases).max()) if ising.biases.size else 0.0
    prog = build_program(route(ising, report, PROFILES["z1"]), report)
    return spec, enc, ising, prog, j, b


def grid_of(decoded: dict, n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=int)
    for k, v in decoded.items():
        x, y = k[1:].split("_")
        a[int(y), int(x)] = v
    return a


def connected_corr(s: np.ndarray, max_r: int):
    """Connected spin correlation C(r) = (<s_i s_{i+r}> - m^2)/(1 - m^2) on a
    +-1 array, averaged over the two axes and normalised so C(0) = 1. This is
    the standard estimator -- for a BINARY layer the spins are genuinely +-1
    valued, so unlike the categorical case no ordering is being invented."""
    n = s.shape[0]
    m = float(s.mean())
    denom = 1.0 - m * m
    out = [1.0]
    for r in range(1, max_r + 1):
        if r >= n or denom <= 1e-12:
            out.append(float("nan"))
            continue
        acc = [float((s[:, :-r] * s[:, r:]).mean()),
               float((s[:-r, :] * s[r:, :]).mean())]
        out.append((float(np.mean(acc)) - m * m) / denom)
    return out


def corr_length(c):
    """First r where C(r) drops below 1/e, linearly interpolated. Returns None
    when it never does -- correlation exceeding the measured window is a
    RESULT (long-range order), not a number to invent."""
    thr = 1.0 / math.e
    for r in range(1, len(c)):
        if not math.isfinite(c[r]):
            return None
        if c[r] < thr:
            prev = c[r - 1]
            if prev == c[r]:
                return float(r)
            return (r - 1) + (prev - thr) / (prev - c[r])
    return None


def main():
    kc = onsager_kc()
    print("Onsager Kc = %.6f  (grid %dx%d, beta = 1.0)\n" % (kc, N, N))
    rows = []
    for w in WEIGHTS:
        t0 = time.time()
        spec, enc, ising, prog, j, b = compile_layer(N, w)
        draws = thrml_sample(prog, seed=0, **SAMPLE)
        grids = []
        for row in draws:
            bits = dict(zip(ising.nodes, row.tolist()))
            if enc.is_codeword(bits):
                grids.append(grid_of(enc.decode(bits), N))
        if not grids:
            print("w=%s: NO valid codewords drawn -- reporting, not skipping" % w)
            rows.append(dict(weight=w, j=j, b=b, error="no codewords"))
            continue
        spins = [2 * g - 1 for g in grids]
        mags = [abs(float(s.mean())) for s in spins]
        cs = np.array([connected_corr(s, MAX_R) for s in spins], dtype=float)
        cbar = np.nanmean(cs, axis=0).tolist()
        xi = corr_length(cbar)
        beta_j = float(ising.beta) * j
        rows.append(dict(weight=w, j=j, b=b, beta_j=beta_j, kc=kc,
                         ratio=beta_j / kc, abs_m=float(np.mean(mags)),
                         xi=xi, c=cbar, n_grids=len(grids),
                         seconds=round(time.time() - t0, 1)))
        xs = "exceeds window" if xi is None else "%.2f cells" % xi
        print("w=%6s  |J|=%.3f |b|=%.3f  beta*J=%.3f (%.2f x Kc)  "
              "|m|=%.3f  xi=%s   [%d grids, %.0fs]"
              % (w, j, b, beta_j, beta_j / kc, float(np.mean(mags)), xs,
                 len(grids), time.time() - t0))
        with open("audit/critical_sweep.json", "w") as fh:
            json.dump(dict(grid=N, beta=1.0, kc=kc, sample=SAMPLE, rows=rows),
                      fh, indent=2)
    print("\n-> audit/critical_sweep.json")


if __name__ == "__main__":
    main()
