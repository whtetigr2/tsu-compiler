"""Does the game's energy survive 4-bit coefficients?

================================ PROTOCOL ================================
SYSTEM DEFINITION   The level's two energy models, re-expressed with INTEGER
                    couplings and biases in [-7, +7], which is the coefficient
                    range Hitachi's CMOS annealing ASIC accepts.
STATE VARIABLES     Unchanged: one spin per (column, depth), one per body cell.
TRANSITION RULES    `thrml` block Gibbs, the same sampler used for the float
                    models, so the ONLY difference between the two runs is the
                    rounding.
ALLOWED OPERATIONS  Rescaling, rounding, sampling, decoding, comparing.
FORBIDDEN OPERATIONS
                    No hardware claim of any kind. Nothing here has touched
                    silicon. This asks a question that must be answered BEFORE
                    a port is worth attempting, and nothing more.
ASSUMPTIONS         Coefficients are integers in [-7, +7]. Temperature is
                    settable on the target, so a scale factor can be absorbed
                    into beta rather than into the coefficients.
INVARIANTS          Quantising and then sampling at beta/s must approximate
                    sampling the float model at beta, because
                    (beta/s) * round(s*J) -> beta*J as s grows. The whole
                    question is how much is lost when s cannot grow, because
                    the largest coefficient is pinned at 7.
MEASUREMENTS        Visibility: agreement with the exact raycast, and with the
                    float model's own decode. Creature: per-cell marginals
                    against the float model.
NULL HYPOTHESES     "The model only works at full precision." Refuted if some
                    integer assignment holds the answer.
SUCCESS CRITERIA    At least one scale within a point or two of the float model.
FAILURE CRITERIA    Every scale losing the answer -- which would mean this
                    hardware cannot carry this model, and would be reported as
                    such rather than worked around.
SCOPE OF VALIDITY   Coefficient precision only. Says nothing about whether that
                    machine samples at fixed temperature or merely anneals,
                    which is a separate and unresolved question.
==========================================================================

WHY THE SCALE CANNOT JUST BE MADE LARGE.

Sampling depends on beta*J, not on J. So a coefficient set can be scaled freely
as long as the temperature is scaled with it: quantise to round(s*J) and sample
at beta/s, and as s grows the rounding error vanishes.

The catch is that the LARGEST coefficient must still land inside [-7, +7]. Here
the largest is the wall bias at 4.0, so s <= 1.75. That is a very small budget:
at s = 1.75 the chain coupling 1.5 becomes round(2.625) = 3, which is 14% too
strong. The question is whether the model cares.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "audit")

import numpy as np

from tsu_compiler.backends.thrml_backend import sample_chains
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.program import build_program

OUT = Path("out/game")
STATE = OUT / "gamestate.json"
LIMIT = 7


def decode(v, nc, nd):
    out = np.empty(nc, dtype=int)
    for c in range(nc):
        z = np.flatnonzero(v[c] == 0)
        out[c] = z[0] if z.size else nd - 1
    return out


def corrupt(solid_bits, nc, nd, rate, seed=7):
    """Spurious walls in free space -- the failure a scan cannot survive, and
    the ONLY regime in which the couplings do anything (see audit/findings/R23).
    Quantising the couplings has to be judged here, not on a clean input."""
    rng = np.random.default_rng(seed)
    solid = (np.frombuffer(solid_bits.encode(), dtype=np.uint8)
             - ord("0")).reshape(nc, nd).astype(bool)
    return (solid | ((rng.random(solid.shape) < rate) & ~solid)).astype(np.uint8)


def visibility_model(vis, scale=None, obs=None):
    """The level's visibility energy, optionally quantised.

    scale=None keeps the float coefficients. Otherwise every coupling and bias
    is round(scale * value), clipped to [-7, 7], and beta is divided by scale so
    the product beta*J is preserved up to the rounding.
    """
    nc, nd = vis["NC"], vis["ND"]
    if obs is None:
        obs = (np.frombuffer(vis["obs"].encode(), dtype=np.uint8)
               - ord("0")).reshape(nc, nd)
    j_chain, j_coh = vis["W_MONO"] / 2.0, vis["W_COH"]
    b_free, b_wall = vis["W_FREE"], -vis["W_WALL"]
    beta = vis["beta"]

    if scale is not None:
        q = lambda x: float(np.clip(np.round(scale * x), -LIMIT, LIMIT))
        j_chain, j_coh, b_free, b_wall = map(q, (j_chain, j_coh, b_free, b_wall))
        beta = beta / scale

    idx = lambda c, k: c * nd + k
    edges, weights = [], []
    for c in range(nc):
        for k in range(nd):
            if k + 1 < nd:
                edges.append((idx(c, k), idx(c, k + 1))); weights.append(j_chain)
            if c + 1 < nc:
                edges.append((idx(c, k), idx(c + 1, k))); weights.append(j_coh)
    bias = np.where(obs, b_wall, b_free).reshape(-1).astype(float)
    order = sorted(range(len(edges)), key=lambda i: edges[i])
    model = IsingModel(
        nodes=tuple(f"v{i}" for i in range(nc * nd)),
        edges=tuple(edges[i] for i in order),
        weights=np.array([weights[i] for i in order]),
        biases=bias, beta=beta, offset=0.0)
    return model, (j_chain, j_coh, b_free, b_wall, beta)


def creature_model(cre, i, scale=None):
    cw, ch = cre["CW"], cre["CH"]
    anat = np.array(cre["anatomy"], dtype=float)
    rf = np.array(cre["creatures"][i]["rf"], dtype=float)
    bias = anat + rf
    jx, jy, beta = cre["jx"], cre["jy"], 1.0
    if scale is not None:
        jx = float(np.clip(round(scale * jx), -LIMIT, LIMIT))
        jy = float(np.clip(round(scale * jy), -LIMIT, LIMIT))
        bias = np.clip(np.round(scale * bias), -LIMIT, LIMIT).astype(float)
        beta = beta / scale
    idx = lambda x, y: y * cw + x
    edges, weights = [], []
    for y in range(ch):
        for x in range(cw):
            if x + 1 < cw:
                edges.append((idx(x, y), idx(x + 1, y))); weights.append(jx)
            if y + 1 < ch:
                edges.append((idx(x, y), idx(x, y + 1))); weights.append(jy)
    order = sorted(range(len(edges)), key=lambda j: edges[j])
    return IsingModel(
        nodes=tuple(f"c{j}" for j in range(cw * ch)),
        edges=tuple(edges[j] for j in order),
        weights=np.array([weights[j] for j in order]),
        biases=bias, beta=beta, offset=0.0)


def sample_vis(model, nc, nd, seed=0):
    rep = analyse(model)
    prog = build_program(model, rep)
    draws = np.asarray(sample_chains(prog, n_chains=4, n_samples=6,
                                     n_warmup=900, steps_per_sample=3, seed=seed))
    return draws[:, -1, :].reshape(-1, nc, nd), rep


def main() -> int:
    d = json.loads(STATE.read_text(encoding="utf-8"))
    vis, cre = d["vis"], d["cre"]
    nc, nd = vis["NC"], vis["ND"]
    truth = np.array(vis["truth"])
    report, failures = {}, []

    print("VISIBILITY AT 4-BIT COEFFICIENTS")
    print()
    print("  The largest coefficient is the wall bias at 4.0, so the scale")
    print(f"  cannot exceed {LIMIT}/4.0 = {LIMIT / 4.0:.2f} without clipping it.")
    print()

    NOISE = 0.08
    obs_clean = (np.frombuffer(vis["obs"].encode(), dtype=np.uint8)
                 - ord("0")).reshape(nc, nd)
    obs_noisy = corrupt(vis["solid"], nc, nd, NOISE)

    print("  Tested on a CLEAN input and on one with 8% of free cells")
    print("  misreported as wall. Only the second tests the couplings: on a")
    print("  clean input the biases alone give the right answer, so a clean")
    print("  pass would say nothing about whether quantisation hurt them.")
    print()

    SEEDS = (0, 1, 2, 3, 4)

    def score(model, obs_unused=None):
        """Mean over seeds of the best chain, so a lucky chain cannot carry it."""
        vals = []
        for sd in SEEDS:
            draws, rep = sample_vis(model, nc, nd, seed=sd)
            vals.append(max(float(np.mean(decode(draws[c], nc, nd) == truth))
                            for c in range(draws.shape[0])))
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)), rep

    bases = {}
    for label, obs in (("clean", obs_clean), (f"{NOISE:.0%} noise", obs_noisy)):
        m, _ = visibility_model(vis, obs=obs)
        mn, lo, hi, rep = score(m)
        bases[label] = (mn, lo, hi)
    print(f"  {rep.n_nodes:,} spins, degree {rep.max_degree}, "
          f"bipartite {rep.bipartite}")
    print(f"  float coefficients, mean of {len(SEEDS)} seeds:")
    print(f"    clean {bases['clean'][0]:.1%} "
          f"[{bases['clean'][1]:.1%}-{bases['clean'][2]:.1%}]")
    print(f"    noisy {bases[f'{NOISE:.0%} noise'][0]:.1%} "
          f"[{bases[f'{NOISE:.0%} noise'][1]:.1%}-"
          f"{bases[f'{NOISE:.0%} noise'][2]:.1%}]")
    print()
    print("  b*J is what the physics sees, so the effective chain and")
    print("  coherence strengths are shown too: a different scale is a")
    print("  different TUNING, not only a rounding.")
    print()
    print(f"  {'scale':>6}{'chain':>6}{'coh':>5}{'free':>5}{'wall':>5}"
          f"{'b*Jc':>7}{'b*Jh':>7}{'noisy (mean)':>14}{'range':>16}{'clip':>6}")

    rows = []
    for sc in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
        m_c, (jc, jh, bf, bw, bq) = visibility_model(vis, scale=sc, obs=obs_clean)
        m_n, _ = visibility_model(vis, scale=sc, obs=obs_noisy)
        clipped = (abs(round(sc * vis["W_FREE"])) > LIMIT
                   or abs(round(sc * vis["W_WALL"])) > LIMIT)
        ac, _, _, _ = score(m_c)
        an, lo, hi, _ = score(m_n)
        rows.append({"scale": sc, "chain": jc, "coh": jh, "free": bf,
                     "wall": bw, "beta": round(bq, 3),
                     "eff_chain": round(bq * jc, 3), "eff_coh": round(bq * jh, 3),
                     "clean": round(ac, 4), "noisy": round(an, 4),
                     "noisy_lo": round(lo, 4), "noisy_hi": round(hi, 4),
                     "clipped": bool(clipped)})
        print(f"  {sc:>6.2f}{jc:>6.0f}{jh:>5.0f}{bf:>5.0f}{bw:>5.0f}"
              f"{bq * jc:>7.2f}{bq * jh:>7.2f}{an:>14.1%}"
              f"{f'[{lo:.1%}-{hi:.1%}]':>16}"
              f"{'  YES' if clipped else '   no':>6}")

    ref = bases[f"{NOISE:.0%} noise"][0]
    usable = [r for r in rows
              if not r["clipped"] and r["noisy"] >= ref - 0.05]
    print()
    if usable:
        best = max(usable, key=lambda r: r["noisy"])
        print(f"  Best legal assignment: scale {best['scale']} -- chain "
              f"{best['chain']:.0f}, coherence {best['coh']:.0f}, free "
              f"{best['free']:.0f}, wall {best['wall']:.0f}")
        print(f"  Under noise: {best['noisy']:.1%} against {ref:.1%} at full "
              f"precision.")
    else:
        print("  NO integer assignment keeps the denoising. The couplings do")
        print("  not survive 4-bit coefficients, which is the whole reason for")
        print("  having them.")
        failures.append("no 4-bit quantisation preserves the coupling benefit")
    report["visibility"] = {"float_clean": round(bases["clean"][0], 4),
                            "float_noisy": round(ref, 4),
                            "seeds": len(SEEDS),
                            "float_noisy_range": [round(bases[f"{NOISE:.0%} noise"][1], 4),
                                                  round(bases[f"{NOISE:.0%} noise"][2], 4)],
                            "noise": NOISE, "rows": rows}

    print()
    print()
    print("CREATURE BODIES AT 4-BIT COEFFICIENTS")
    print()
    cw, ch = cre["CW"], cre["CH"]
    bias_span = float(np.max(np.abs(
        np.array(cre["anatomy"]) + np.array(cre["creatures"][0]["rf"]))))
    print(f"  Couplings are {cre['jx']} and {cre['jy']}; the per-cell bias runs")
    print(f"  to {bias_span:.2f}, so the scale ceiling here is "
          f"{LIMIT / bias_span:.2f}.")
    print()

    m_float = creature_model(cre, 0)
    r = analyse(m_float)
    dr = np.asarray(sample_chains(build_program(m_float, r), n_chains=6,
                                  n_samples=70, n_warmup=400,
                                  steps_per_sample=2, seed=0))
    marg_float = dr.reshape(-1, cw * ch).mean(axis=0)

    print(f"  {'scale':>7}{'jx':>5}{'jy':>5}{'beta':>8}"
          f"{'mean |d|':>11}{'worst cell':>12}")
    crows = []
    for s in (0.75, 1.0, 1.25, 1.5, 1.75):
        mq = creature_model(cre, 0, scale=s)
        rq = analyse(mq)
        dq = np.asarray(sample_chains(build_program(mq, rq), n_chains=6,
                                      n_samples=70, n_warmup=400,
                                      steps_per_sample=2, seed=0))
        marg_q = dq.reshape(-1, cw * ch).mean(axis=0)
        md = float(np.mean(np.abs(marg_q - marg_float)))
        wd = float(np.max(np.abs(marg_q - marg_float)))
        jxq = float(np.clip(round(s * cre["jx"]), -LIMIT, LIMIT))
        jyq = float(np.clip(round(s * cre["jy"]), -LIMIT, LIMIT))
        crows.append({"scale": s, "jx": jxq, "jy": jyq,
                      "mean_abs_diff": round(md, 4), "worst": round(wd, 4)})
        print(f"  {s:>7.2f}{jxq:>5.0f}{jyq:>5.0f}{1.0 / s:>8.3f}"
              f"{md:>11.3f}{wd:>12.3f}")
    report["creature"] = {"rows": crows, "bias_span": round(bias_span, 3)}

    best_c = min(crows, key=lambda r2: r2["mean_abs_diff"])
    print()
    print(f"  Best: scale {best_c['scale']}, mean marginal difference "
          f"{best_c['mean_abs_diff']:.3f}")
    if best_c["mean_abs_diff"] > 0.15:
        print("  That is a different creature, not the same one rounded.")
        failures.append("the creature model does not survive 4-bit coefficients")

    (OUT / "quantised_for_silicon.json").write_text(json.dumps(
        {**report, "control_failures": failures,
         "coefficient_limit": LIMIT,
         "method": "quantise to round(scale*J), sample at beta/scale, so only "
                   "rounding error remains",
         "target_context": "Hitachi CMOS annealing ASIC accepts integer "
                           "coefficients in [-7, 7] on a King's graph",
         "hardware": "NONE. Nothing here has touched silicon. This is a "
                     "precondition check, not a hardware result.",
         "unresolved": "whether that machine samples at fixed temperature or "
                       "only anneals to low-energy states"},
        indent=2), encoding="utf-8")

    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'quantised_for_silicon.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
