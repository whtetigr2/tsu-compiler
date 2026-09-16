"""Surfaces with no pixels in them: texture as a sampled Ising field.

================================ PROTOCOL ================================
SYSTEM DEFINITION   Greyscale surface texture on a grid, produced by sampling a
                    pairwise Ising model whose couplings and biases are written
                    by hand. No image data of any kind is stored, loaded or fit.
STATE VARIABLES     b[bit][y][x] in {0,1}: bit `bit` of the value at cell (y,x),
                    with the value read as a binary code. One Ising layer per
                    bit, each with its own coupling strength.
TRANSITION RULES    Chromatic block Gibbs on a checkerboard, per bit layer.
ALLOWED OPERATIONS  Choosing couplings and periodic biases by hand; sampling;
                    decoding to a greyscale value; counting the parameters used.
FORBIDDEN OPERATIONS
                    No texture, photograph or asset is read, embedded, or fit
                    against -- not even to tune a constant. Every number in a
                    recipe below was chosen by reasoning about what the energy
                    does, and the parameter count is reported so the claim can
                    be checked by counting rather than believed.
ASSUMPTIONS         None beyond the Ising model itself.
INVARIANTS          Every layer is a plain grid with orthogonal couplings, so a
                    checkerboard colouring must 2-colour it. Non-bipartite means
                    the builder is wrong.
MEASUREMENTS        Per recipe: parameter count, bits, lattice size, degree,
                    bipartiteness, Z1 fit, and simple image statistics that
                    distinguish the families from each other and from noise.
NULL HYPOTHESES     "Hand-written couplings cannot produce distinguishable
                    surface families -- it will all look like noise."
                    Refuted if the recipes separate on measured statistics, and
                    the CONTROL that keeps this honest is an isotropic
                    zero-structure recipe: it must look like noise, and must
                    measure like noise.
SUCCESS CRITERIA    Recipes that differ measurably from each other and from the
                    noise control, at a parameter count small enough to print.
FAILURE CRITERIA    Recipes indistinguishable from the noise control; any layer
                    non-bipartite; any stored image data.
PROVENANCE          Anisotropic couplings and periodic fields are standard
                    statistical mechanics. Z1 limits from target.py.
SCOPE OF VALIDITY   These are TEXTURE FAMILIES, not reproductions. Nothing here
                    reproduces any particular surface from any particular game,
                    and it cannot: there is no pixel data to reproduce from.
                    Each draw is a different member of the same family.
==========================================================================

WHY THIS EXISTS.

The DOOM demo's surfaces are `idx[]` arrays -- the game's actual pixel data,
baked into the page. That is what makes "this is not the DOOM assets" false
today, and no amount of sampling around a stored image changes it.

So: can a surface come out of an energy instead of a lookup?

Yes, and the mechanism is ordinary. A ferromagnetic Ising grid already produces
blobs whose size is set by the coupling. Make the coupling ANISOTROPIC and the
blobs stretch into bands. Add a PERIODIC BIAS and regular structure appears --
mortar lines, panel seams, grating. None of that requires an image; it requires
a few constants.

The part that makes it look like a surface rather than a blob field is giving
each BIT of the value its own coupling strength. The high bit, strongly coupled,
lays down large regions. The low bits, weakly coupled, add grain on top. One
lattice, several scales, and the scales are just numbers.

WHAT MAKES THE CLAIM CHECKABLE. Each recipe below prints its parameter count.
A stored 64x64 greyscale tile is 4,096 numbers. These recipes are around a
dozen. You do not have to take "there are no assets in here" on faith -- you can
count, and the numbers are all visible in RECIPES.

THE CONTROL. `flat_noise` is deliberately structureless: no anisotropy, no
periodic field, all bits coupled the same. It must come out looking like noise
and measuring like noise. If the interesting recipes did not separate from it on
the statistics below, this file would be showing pareidolia rather than texture.
"""
# NO-PREFLIGHT: surface appearance only. The model is sampled for how it looks, and is never proposed for a fabric, so the gates have nothing to decide.

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.target import PROFILES

OUT = Path("out/texture")
Z1 = PROFILES["z1"]
SIDE, BITS = 64, 4
SWEEPS, BURN = 400, 200

# Every recipe is this handful of numbers and nothing else.
#   jx, jy      coupling along x and along y, per bit (high bit first)
#   period_y    rows between horizontal seams   (0 = none)
#   period_x    columns between vertical seams  (0 = none)
#   stagger     offset alternate bands, the thing that makes brick brick
#   seam        how hard a seam pushes its cells dark. Must exceed the
#               coupling pull from its neighbours (2*jx + 2*jy) or the
#               mortar is simply absorbed into the face -- measured, not guessed
#   grain       an unstructured field on the lowest bit
#   base        the surface's own lightness, before any seam darkens it
RECIPES = {
    "flat_noise": dict(jx=[.35, .35, .35, .35], jy=[.35, .35, .35, .35],
                       period_y=0, period_x=0, stagger=0, seam=0.0, grain=0.0, base=0.0,
                       note="CONTROL -- no anisotropy, no period, equal bits"),
    "brick":      dict(jx=[1.5, .9, .5, .25], jy=[1.5, .9, .5, .25],
                       period_y=8, period_x=16, stagger=8, seam=9.0, grain=.35, base=0.9,
                       note="staggered courses with mortar seams"),
    "grating":    dict(jx=[.15, .12, .1, .08], jy=[2.2, 1.6, .9, .4],
                       period_x=4, period_y=0, stagger=0, seam=7.0, grain=.25, base=0.5,
                       note="strong vertical coupling, weak horizontal"),
    "panel":      dict(jx=[1.7, 1.1, .6, .3], jy=[1.7, 1.1, .6, .3],
                       period_y=16, period_x=16, stagger=0, seam=8.0, grain=.15, base=0.8,
                       note="large plates divided by seams on a square grid"),
    "rough_stone": dict(jx=[.9, .55, .35, .2], jy=[.85, .5, .3, .18],
                        period_y=0, period_x=0, stagger=0, seam=0.0, grain=.55, base=0.0,
                       note="near-critical, slightly anisotropic, heavy grain"),
}


def n_params(r) -> int:
    return len(r["jx"]) + len(r["jy"]) + 6        # the six scalars below them


def seam_field(r, side=SIDE):
    """Periodic bias: where a seam runs, push the cell dark."""
    f = np.zeros((side, side))
    py, px, st, s = r["period_y"], r["period_x"], r["stagger"], r["seam"]
    if s == 0.0:
        return f
    if py:
        for y in range(0, side, py):
            f[y, :] -= s
    if px:
        for y in range(side):
            band = (y // py) if py else 0
            off = (band * st) % px if st else 0
            f[y, (np.arange(side) + off) % px == 0] -= s
    return f


def sample_texture(r, side=SIDE, seed=0):
    rng = np.random.default_rng(seed)
    bits = np.array([rng.integers(0, 2, size=(side, side)) for _ in range(BITS)],
                    dtype=np.int8)
    seam = seam_field(r, side)
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    parity = (yy + xx) % 2
    acc = np.zeros((side, side))
    kept = 0
    for s in range(SWEEPS):
        for colour in (0, 1):
            for b in range(BITS):
                jx, jy = r["jx"][b], r["jy"][b]
                v = bits[b]
                f = np.zeros((side, side))
                f += jx * (2 * np.roll(v, 1, axis=1) - 1)
                f += jx * (2 * np.roll(v, -1, axis=1) - 1)
                f += jy * (2 * np.roll(v, 1, axis=0) - 1)
                f += jy * (2 * np.roll(v, -1, axis=0) - 1)
                # the seam field acts on every bit; grain only on the lowest
                f += r["base"] * (1.0 if b == 0 else 0.5)
                f += seam * (1.0 if b == 0 else 0.45)
                if b == BITS - 1 and r["grain"]:
                    f += r["grain"] * rng.standard_normal((side, side))
                p = 1.0 / (1.0 + np.exp(-2.0 * f))
                draw = (rng.random((side, side)) < p).astype(np.int8)
                bits[b] = np.where(parity == colour, draw, v)
        if s >= BURN:
            val = np.zeros((side, side))
            for b in range(BITS):
                val += bits[b] * (1 << (BITS - 1 - b))
            acc += val / (2 ** BITS - 1)
            kept += 1
    return acc / max(kept, 1)


def stats(img) -> dict:
    """Numbers that separate the families. Chosen before looking at output:
    banding is what anisotropy should produce, seam contrast is what a periodic
    field should produce, and neighbour correlation is what coupling does."""
    gy = float(np.mean(np.abs(np.diff(img, axis=0))))
    gx = float(np.mean(np.abs(np.diff(img, axis=1))))
    corr = float(np.mean(img[:, :-1] * img[:, 1:]) - np.mean(img) ** 2)
    row = img.mean(axis=1)
    return {"contrast": round(float(img.std()), 4),
            "grad_y": round(gy, 4), "grad_x": round(gx, 4),
            "anisotropy": round(gy / gx if gx > 1e-9 else float("inf"), 3),
            "neighbour_corr": round(corr, 4),
            "row_structure": round(float(row.std()), 4)}


def lattice_check(side=SIDE) -> dict:
    """One bit layer, as an IsingModel, weighed against Z1."""
    idx = lambda y, x: y * side + x
    e = set()
    for y in range(side):
        for x in range(side):
            e.add((min(idx(y, x), idx(y, (x + 1) % side)),
                   max(idx(y, x), idx(y, (x + 1) % side))))
            e.add((min(idx(y, x), idx((y + 1) % side, x)),
                   max(idx(y, x), idx((y + 1) % side, x))))
    n = side * side
    ed = tuple(sorted(e))
    m = IsingModel(nodes=tuple(f"p{i}" for i in range(n)), edges=ed,
                   weights=np.full(len(ed), 1.0), biases=np.zeros(n),
                   beta=1.0, offset=0.0)
    rep = analyse(m)
    total = n * BITS
    return {"pbits_per_bit_layer": rep.n_nodes, "bit_layers": BITS,
            "pbits_total": total, "max_degree": rep.max_degree,
            "bipartite": bool(rep.bipartite),
            "degree_within_z1": rep.max_degree <= Z1.degree.value,
            "fits_node_budget": total <= Z1.node_budget.value}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lat = lattice_check()
    imgs, rows = {}, []
    for name, r in RECIPES.items():
        img = sample_texture(r)
        imgs[name] = img
        rows.append({"recipe": name, "note": r["note"],
                     "parameters": n_params(r), **stats(img)})

    ctrl = next(x for x in rows if x["recipe"] == "flat_noise")
    failures = []
    if not lat["bipartite"]:
        failures.append("a bit layer came back non-bipartite; it is a plain grid, "
                        "so the builder is wrong")
    for r in rows:
        if r["recipe"] == "flat_noise":
            continue
        same_aniso = abs(r["anisotropy"] - ctrl["anisotropy"]) < 0.08
        same_struct = abs(r["row_structure"] - ctrl["row_structure"]) < 0.01
        if same_aniso and same_struct:
            failures.append(f"{r['recipe']} is not distinguishable from the noise "
                            f"control on either anisotropy or row structure")
    if abs(ctrl["anisotropy"] - 1.0) > 0.15:
        failures.append(f"the isotropic control measured anisotropy "
                        f"{ctrl['anisotropy']}, which should be ~1.0 -- the "
                        f"statistic is not measuring what it claims")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(imgs), figsize=(3.0 * len(imgs), 3.6),
                                 facecolor="#0d1117")
        for ax, (name, img) in zip(axes, imgs.items()):
            ax.imshow(img, cmap="copper", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{name}\n{n_params(RECIPES[name])} parameters",
                         color="#e6e9f2", fontsize=10)
            ax.axis("off")
        fig.suptitle("Surfaces sampled from hand-written couplings -- no image data anywhere",
                     color="#e6e9f2", fontsize=12)
        fig.savefig(OUT / "textures.png", dpi=120, facecolor="#0d1117",
                    bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:                       # rendering is not the result
        print(f"  (figure skipped: {exc})")

    (OUT / "texture_from_energy.json").write_text(json.dumps(
        {"lattice": lat, "recipes": rows, "control_failures": failures,
         "stored_image_data": 0,
         "comparison": f"a stored {SIDE}x{SIDE} greyscale tile is "
                       f"{SIDE * SIDE:,} numbers; these recipes are "
                       f"{n_params(RECIPES['brick'])} each",
         "scope": "texture FAMILIES, not reproductions. Nothing here reproduces "
                  "any particular surface from any particular game, and it "
                  "cannot -- there is no pixel data to reproduce from.",
         "hardware": "none -- simulation"},
        indent=2), encoding="utf-8")

    print("SURFACES FROM COUPLINGS, NOT PIXELS\n")
    print(f"  one bit layer: {lat['pbits_per_bit_layer']:,} pbits, degree "
          f"{lat['max_degree']}, bipartite {lat['bipartite']}")
    print(f"  {BITS} bit layers -> {lat['pbits_total']:,} pbits, "
          f"{'fits' if lat['fits_node_budget'] else 'OVER'} Z1's "
          f"{Z1.node_budget.value:,}\n")
    print(f"  {'recipe':<13}{'params':>7}{'contrast':>10}{'anisotropy':>12}"
          f"{'row struct':>12}{'nbr corr':>10}")
    for r in rows:
        print(f"  {r['recipe']:<13}{r['parameters']:>7}{r['contrast']:>10.4f}"
              f"{r['anisotropy']:>12.3f}{r['row_structure']:>12.4f}"
              f"{r['neighbour_corr']:>10.4f}")
    print()
    print(f"  A stored {SIDE}x{SIDE} greyscale tile is {SIDE * SIDE:,} numbers.")
    print(f"  Each of these is {n_params(RECIPES['brick'])}. You can count them; "
          f"they are all in RECIPES.")
    print()
    print("  flat_noise is the control. It has no anisotropy and no periodic")
    print("  field, so it must measure like noise -- and the others must not,")
    print("  or this would be pareidolia rather than texture.")
    if failures:
        print()
        print("CONTROL FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print()
    print(f"  -> {OUT / 'texture_from_energy.json'}")
    print(f"  -> {OUT / 'textures.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
