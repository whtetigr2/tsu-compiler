"""Render a GENUINELY SAMPLED world from a compiled receipt.

Nothing here is hand-authored. The 8x8 terrain grid is decoded from a chromatic
block Gibbs sample of the compiled Ising model in demo/receipts/small, and is
validated against the spec's own contract before being drawn. Shading and
sub-cell decoration are DETERMINISTIC functions of the decoded cell (spec S1):
same cell -> same pixels, always. They invent no world facts.
"""
import sys, numpy as np
sys.path.insert(0, "src")
from scipy import ndimage
from PIL import Image
from tsu_compiler.simulate import simulate

W = H = 8
WATER, ROCK, GRASS = 0, 1, 2
NAMES = {WATER: "water", ROCK: "rock", GRASS: "grass"}
PAL = np.array([[46, 92, 132], [124, 116, 106], [126, 158, 84]], float)

# Reuse the compiler's OWN decode/validate path -- is_codeword, decode, and
# contract.validate all come from tsu, not reimplemented here.
from pathlib import Path
from tsu_compiler.spec import load_spec
from tsu_compiler.passes.encode import encode
from tsu_compiler.simulate import reconstruct_program, _selected_encoding
from tsu_compiler.backends.thrml_backend import sample as thrml_sample

R = "demo/receipts/small"
spec = load_spec(str(Path(R) / "spec.yaml"))
enc = encode(spec, _selected_encoding(Path(R)))
prog = reconstruct_program(R)
im = prog.ising

got = thrml_sample(prog, n_chains=8, n_samples=60, n_warmup=1200,
                   steps_per_sample=4, seed=7)
worlds, noncodeword, invalid = [], 0, 0
for row in got:
    bits = dict(zip(im.nodes, row.tolist()))
    if not enc.is_codeword(bits):
        noncodeword += 1
        continue
    dec = enc.decode(bits)
    if spec.contract.validate(dec).ok:
        worlds.append(dec)
    else:
        invalid += 1
print(f"drew {len(got)} samples: {noncodeword} non-codewords, {invalid} "
      f"codewords failing the contract, {len(worlds)} VALID WORLDS "
      f"({100*len(worlds)/len(got):.1f}%)")
print(f"distinct valid worlds: {len({tuple(sorted(w.items())) for w in worlds})}")
if not worlds:
    raise SystemExit("no valid sample -- a non-codeword is NOT a world")

d = worlds[0]
grid = np.array([[int(d[f"g{x}_{y}"]) for x in range(W)] for y in range(H)])
print("cell counts:", {NAMES[k]: int((grid == k).sum()) for k in (0, 1, 2)})

# contract re-check on the thing we are about to draw
bad = []
for y in range(H):
    for x in range(W):
        for (y2, x2) in ([(y, x+1)] if x+1 < W else []) + ([(y+1, x)] if y+1 < H else []):
            if {int(grid[y, x]), int(grid[y2, x2])} == {WATER, ROCK}:
                bad.append(((y, x), (y2, x2)))
print(f"water-adjacent-to-rock violations in the rendered world: {len(bad)}")

UP = 64
base = PAL[np.kron(grid, np.ones((UP, UP), int))]
# pseudo-relief DERIVED from terrain class (water low, grass mid, rock high) --
# derived, not sampled; there is no elevation layer in this spec.
hgt = np.choose(grid, [0.0, 2.0, 1.0])
ef = ndimage.gaussian_filter(ndimage.zoom(hgt, UP, order=3), UP * 0.5)
gy, gx = [g * UP * 1.4 for g in np.gradient(ef)]
slope, aspect = np.arctan(np.hypot(gy, gx)), np.arctan2(-gx, gy)
az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
shade = np.clip(np.sin(alt)*np.cos(slope) + np.cos(alt)*np.sin(slope)*np.cos(az-aspect), 0, None)**0.85
shade = 0.58 + 0.92*shade

deco = np.zeros(base.shape[:2]); yy, xx = np.mgrid[0:UP, 0:UP] / float(UP)
for r in range(H):
    for c in range(W):
        t, rg = grid[r, c], np.random.default_rng(r*1000 + c)   # cell-derived seed
        tile = np.zeros((UP, UP))
        if t == GRASS:
            for _ in range(14):
                cy, cx, s = rg.uniform(.1,.9), rg.uniform(.1,.9), rg.uniform(.03,.055)
                tile -= 0.5*np.exp(-(((xx-cx)**2 + (yy-cy)**2)/(2*s*s)))
        elif t == ROCK:
            tile += 0.22*np.sin(15*(xx*0.7 - yy*0.8) + rg.uniform(0, 6.28)) + 0.10*rg.normal(0,1,(UP,UP))
        else:
            tile += 0.05*np.sin(9*yy + rg.uniform(0, 6.28))
        deco[r*UP:(r+1)*UP, c*UP:(c+1)*UP] = tile
deco = ndimage.gaussian_filter(deco, 1.1)

img = base * shade[..., None] * (1 + 0.40*deco[..., None])
wet = (np.kron(grid, np.ones((UP, UP), int)) == WATER)
shore = np.clip(ndimage.gaussian_filter(wet.astype(float), UP*0.22) - wet, 0, 1)
img = img*(1 - 0.45*shore[..., None]) + np.array([222, 214, 180])*0.45*shore[..., None]
g = np.zeros(base.shape[:2]); g[::UP, :] = g[:, ::UP] = 1
img *= (1 - 0.05*g[..., None])
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save("demo/world.png")
print("wrote demo/world.png", img.shape)
