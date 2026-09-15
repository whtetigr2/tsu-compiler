"""The game's own models, compiled and handed to THRML.

================================ PROTOCOL ================================
SYSTEM DEFINITION   The two energy models the playable level actually samples
                    -- first-hit visibility (200 x 56) and creature bodies
                    (15 x 21) -- exported live from the running page and
                    re-sampled through this project's passes and `thrml`.
STATE VARIABLES     Visibility: one spin per (column, depth). Creature: one
                    spin per body cell.
TRANSITION RULES    `thrml.block_sampling.sample_states`, via
                    `tsu_compiler.backends.thrml_backend.sample_chains`.
ALLOWED OPERATIONS  Loading the exported state; building the IR; analysing it;
                    sampling; comparing against what the page produced from the
                    same input.
FORBIDDEN OPERATIONS
                    No re-deriving the level in Python -- the occupancy, the
                    quenched fields and the page's own answers are IMPORTED, so
                    a transcription mistake cannot be mistaken for agreement.
                    No hardware claim: thrml simulates on CPU/JAX.
ASSUMPTIONS         That the page's constants are the model. They are exported
                    rather than retyped.
INVARIANTS          Block Gibbs requires that no two spins updated in the same
                    colour block are coupled to each other. If the compiler
                    cannot 2-colour a model, the page's own two-colour sweep is
                    not a valid sampler for it either.
MEASUREMENTS        Visibility: agreement between thrml's decode and the page's
                    decode on identical occupancy. Creatures: per-cell
                    marginals, because any single draw is stochastic.
NULL HYPOTHESES     "The game only works because of the sampler written into the
                    page." Refuted for any model thrml reproduces.
SUCCESS CRITERIA    thrml matching the page on both models.
FAILURE CRITERIA    Disagreement at the same beta, or a model the compiler
                    cannot colour. The second is not a compiler bug -- it is the
                    compiler reporting that the page's sweep is invalid.
SCOPE OF VALIDITY   The energy models only. The level geometry, the doors, the
                    line-of-sight test and the shot resolution are LOOKUPS and
                    are deliberately not ported: writing a lookup as an energy
                    is how R23 happened.
==========================================================================

WHAT THIS FOUND ON ITS FIRST RUN.

Visibility ported unchanged and thrml reproduced the page exactly.

The creature model did not. As first written it carried a BILATERAL coupling
tying each cell to its mirror, on top of a grid the page sweeps by (x+y)&1.
For an odd width a cell and its mirror always share that parity -- 294 of 315
pairs -- so the page was updating coupled spins in the same block, which is not
a legal Gibbs step, and the centre column was coupled to itself. The graph is
not bipartite and no two-colour hardware blocking exists for it.

The term was also unnecessary. `anatomy()` is a function of |x - centre| and the
quenched field is built by mirroring one half, so the creature was already
bilateral before any coupling was added. Removing it makes the model legal and
costs nothing it was there for.
"""
import json
import sys
import time
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


def load():
    return json.loads(STATE.read_text(encoding="utf-8"))


def build_visibility(vis) -> IsingModel:
    """The model the level samples every frame, as a real IsingModel."""
    nc, nd = vis["NC"], vis["ND"]
    obs = (np.frombuffer(vis["obs"].encode(), dtype=np.uint8)
           - ord("0")).reshape(nc, nd)
    idx = lambda c, k: c * nd + k
    edges, weights = [], []
    for c in range(nc):
        for k in range(nd):
            if k + 1 < nd:
                edges.append((idx(c, k), idx(c, k + 1)))
                weights.append(vis["W_MONO"] / 2.0)
            if c + 1 < nc:
                edges.append((idx(c, k), idx(c + 1, k)))
                weights.append(vis["W_COH"])
    bias = np.where(obs, -vis["W_WALL"], vis["W_FREE"]).reshape(-1).astype(float)
    order = sorted(range(len(edges)), key=lambda i: edges[i])
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(nc * nd)),
        edges=tuple(edges[i] for i in order),
        weights=np.array([weights[i] for i in order]),
        biases=bias, beta=vis["beta"], offset=0.0)


def decode(v, nc, nd):
    out = np.empty(nc, dtype=int)
    for c in range(nc):
        z = np.flatnonzero(v[c] == 0)
        out[c] = z[0] if z.size else nd - 1
    return out


def build_creature(cre, i, with_sym=False, sym=1.35):
    """Anatomy field + this individual's quenched field + grid couplings.

    `with_sym` re-adds the bilateral cell-to-mirror coupling the page used to
    carry, so its illegality can be demonstrated rather than asserted.
    """
    cw, ch = cre["CW"], cre["CH"]
    anat = np.array(cre["anatomy"], dtype=float)
    rf = np.array(cre["creatures"][i]["rf"], dtype=float)
    idx = lambda x, y: y * cw + x
    edges, weights = [], []
    self_loops = 0
    for y in range(ch):
        for x in range(cw):
            if x + 1 < cw:
                edges.append((idx(x, y), idx(x + 1, y)))
                weights.append(cre["jx"])
            if y + 1 < ch:
                edges.append((idx(x, y), idx(x, y + 1)))
                weights.append(cre["jy"])
            if with_sym:
                mx = cw - 1 - x
                if mx == x:
                    self_loops += 1
                elif x < mx:
                    edges.append((idx(x, y), idx(mx, y)))
                    weights.append(sym)
    order = sorted(range(len(edges)), key=lambda j: edges[j])
    model = IsingModel(
        nodes=tuple(f"c{j}" for j in range(cw * ch)),
        edges=tuple(edges[j] for j in order),
        weights=np.array([weights[j] for j in order]),
        biases=(anat + rf), beta=1.0, offset=0.0)
    return model, self_loops


def parity_check(cw, ch):
    """Mirror partners that share the page's (x+y)&1 block."""
    same = 0
    for y in range(ch):
        for x in range(cw):
            mx = cw - 1 - x
            if mx != x and ((x + y) & 1) == ((mx + y) & 1):
                same += 1
    return same


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    vis, cre = d["vis"], d["cre"]
    report, failures = {}, []

    print("VISIBILITY -- the model the level samples every frame")
    print()
    model = build_visibility(vis)
    rep = analyse(model)
    print(f"  {rep.n_nodes:,} spins, {len(model.edges):,} couplings, "
          f"degree {rep.max_degree}, bipartite {rep.bipartite}, "
          f"{rep.colour_blocks} colour blocks")
    if not rep.bipartite:
        failures.append("the visibility lattice is not bipartite")

    t0 = time.time()
    prog = build_program(model, rep)
    draws = np.asarray(sample_chains(prog, n_chains=4, n_samples=6,
                                     n_warmup=900, steps_per_sample=3, seed=0))
    dt = time.time() - t0
    nc, nd = vis["NC"], vis["ND"]
    last = draws[:, -1, :].reshape(-1, nc, nd)
    truth, page = np.array(vis["truth"]), np.array(vis["depth"])
    agree_t = [float(np.mean(decode(last[c], nc, nd) == truth))
               for c in range(last.shape[0])]
    agree_p = [float(np.mean(decode(last[c], nc, nd) == page))
               for c in range(last.shape[0])]
    print()
    print(f"  thrml vs ground truth : {np.max(agree_t):.1%} best, "
          f"{np.mean(agree_t):.1%} mean")
    print(f"  thrml vs the page     : {np.max(agree_p):.1%} best, "
          f"{np.mean(agree_p):.1%} mean")
    print(f"  the page vs truth     : {np.mean(page == truth):.1%}")
    print(f"  ({dt:.1f}s, 4 chains x 900 warmup, CPU)")
    report["visibility"] = {
        "spins": rep.n_nodes, "edges": len(model.edges),
        "degree": rep.max_degree, "bipartite": bool(rep.bipartite),
        "colour_blocks": rep.colour_blocks,
        "thrml_vs_truth_best": round(float(np.max(agree_t)), 4),
        "thrml_vs_page_best": round(float(np.max(agree_p)), 4),
        "page_vs_truth": round(float(np.mean(page == truth)), 4),
        "seconds": round(dt, 1)}
    if np.max(agree_t) < 0.95:
        failures.append(f"thrml reached only {np.max(agree_t):.1%} on visibility")

    cw, ch = cre["CW"], cre["CH"]
    print()
    print()
    print("CREATURE BODIES -- with a bilateral cell-to-mirror coupling")
    print()
    same_block = parity_check(cw, ch)
    m_sym, self_loops = build_creature(cre, 0, with_sym=True)
    rep_sym = analyse(m_sym)
    print(f"  {rep_sym.n_nodes} spins, {len(m_sym.edges):,} couplings, "
          f"degree {rep_sym.max_degree}, bipartite {rep_sym.bipartite}")
    print(f"  mirror pairs sharing the page's (x+y)&1 block : "
          f"{same_block} of {cw * ch}")
    print(f"  spins coupled to themselves (centre column)   : {self_loops}")
    print()
    print("  Width is odd, so a cell and its mirror always share (x+y)&1. A")
    print("  two-colour sweep updates them together while they are coupled,")
    print("  which block Gibbs does not permit, and no two-colour hardware")
    print("  blocking exists for the graph either.")
    report["creature_with_bilateral_coupling"] = {
        "bipartite": bool(rep_sym.bipartite), "degree": rep_sym.max_degree,
        "mirror_pairs_in_same_block": same_block, "self_loops": self_loops}

    print()
    print()
    print("CREATURE BODIES -- symmetry carried by the field, as shipped")
    print()
    print("  anatomy() is a function of |x - centre| and the quenched field is")
    print("  built by mirroring one half, so the body is bilateral before any")
    print("  coupling is added. The coupling was never what made it symmetric.")
    print()
    m0, _ = build_creature(cre, 0)
    rep_c = analyse(m0)
    print(f"  {rep_c.n_nodes} spins, {len(m0.edges):,} couplings, "
          f"degree {rep_c.max_degree}, bipartite {rep_c.bipartite}, "
          f"{rep_c.colour_blocks} colour blocks")
    if not rep_c.bipartite:
        failures.append("the creature lattice is not bipartite")

    gaps, shapes, mean_abs = [], [], None
    for i in range(len(cre["creatures"])):
        mi, _ = build_creature(cre, i)
        ri = analyse(mi)
        pi = build_program(mi, ri)
        dr = np.asarray(sample_chains(pi, n_chains=6, n_samples=70,
                                      n_warmup=400, steps_per_sample=2, seed=i))
        marg_t = dr.reshape(-1, cw * ch).mean(axis=0)
        marg_p = np.array(cre["creatures"][i]["marg"])
        gaps.append(float(np.max(np.abs(marg_t - marg_p))))
        if i == 0:
            mean_abs = float(np.mean(np.abs(marg_t - marg_p)))
        if i < 4:
            shapes.append((marg_t > 0.5).reshape(ch, cw))

    print()
    print(f"  per-cell marginals, thrml vs the page, {len(gaps)} individuals:")
    print(f"    mean |difference|, individual 0 : {mean_abs:.3f}")
    print(f"    worst single cell, any of them  : {max(gaps):.3f}")
    print(f"    median per-individual worst     : {float(np.median(gaps)):.3f}")
    print()
    print("  thrml's draw of four individuals, same species field, four seeds:")
    print()
    for y in range(ch):
        print("   " + "   ".join(
            "".join("#" if sh[y, x] else "." for x in range(cw))
            for sh in shapes))
    diffs = [int(np.sum(shapes[i] != shapes[j]))
             for i in range(len(shapes)) for j in range(i + 1, len(shapes))]
    print()
    print(f"  mean pairwise difference: {np.mean(diffs):.1f} of {cw * ch} cells")

    report["creature"] = {
        "spins": rep_c.n_nodes, "edges": len(m0.edges),
        "degree": rep_c.max_degree, "bipartite": bool(rep_c.bipartite),
        "colour_blocks": rep_c.colour_blocks,
        "marginal_mean_abs_diff": round(mean_abs, 4),
        "marginal_worst_cell": round(max(gaps), 4),
        "mean_pairwise_diff": round(float(np.mean(diffs)), 1)}

    if max(gaps) > 0.25:
        failures.append(f"thrml and the page differ by up to {max(gaps):.2f} on "
                        f"a per-cell marginal; not the same distribution")
    if np.mean(diffs) < 5:
        failures.append("thrml's creatures are near-identical")

    (OUT / "game_on_thrml.json").write_text(json.dumps(
        {**report, "control_failures": failures,
         "sampler": "thrml via tsu_compiler.backends.thrml_backend",
         "input": "exported live from level.html, not re-derived",
         "not_ported": ["level geometry", "doors", "line of sight",
                        "shot resolution"],
         "not_ported_reason": "these are lookups; writing a lookup as an energy "
                              "is how R23 happened",
         "hardware": "none -- thrml on CPU/JAX"},
        indent=2), encoding="utf-8")

    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'game_on_thrml.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
