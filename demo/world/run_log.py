"""Saving a World Studio run so someone else can check the work.

SCOPE. This is spec sections 5.4 (the world) and 5.5 (provenance). The sampler
diagnostics and plot panels (5.1-5.3) need instrumented sampling and belong to
Plan B. They are ABSENT here rather than stubbed: an empty trace plot would be
worse than no trace plot, because a reader would take its presence as evidence
the diagnostic ran.

The README written into every run says which panels are missing, for the same
reason.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import replace
from pathlib import Path

from world.export import write as export_write
from world.generate import World
from world.studio import COST_TABLES, KC, Readout, Sampled, Studio


def _current_world(studio: Studio) -> World:
    """A World carrying the CURRENT view's arrays.

    `studio.world` is the world as SAMPLED, before any readout slider moved.
    Exporting that would mean a user who reshaped their world and saved got a
    picture of the world they started with -- and the round-trip test could not
    notice, because it compares views rather than the written file."""
    v = studio.view()
    return replace(studio.world, height=v.height, terrain=v.terrain, cost=v.cost)


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("thrml", "jax", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "unavailable"
    try:
        import jax
        out["jax_backend"] = jax.default_backend()
    except Exception:
        out["jax_backend"] = "unknown"
    try:
        out["commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        out["commit"] = "unknown"
    return out


README = """# World Studio run

Sampled with thrml on the **{backend}** backend. thrml simulates the block
Gibbs sampling a Z1 would perform; no Z1 hardware was involved.

  seed {seed} | beta*J {beta_j} | size {size} | mode {mode} | warmup {warmup}
  sea level {sea_level} | mountain line {mountain_line} | relief {relief}
  cost table {cost_table}

Sampling took {total:.2f}s in total. Per-field timings in `studio.json` are
APPORTIONED BY SPIN COUNT, not measured individually.

beta*J = {beta_j} is {ratio:.3f} x Onsager's Kc = {kc:.6f}, i.e. {side}.

## Which parts of this map are the lattice

`decodes.json` holds the SAME sampled draws decoded under both upsample modes,
which differ here in {differing} of {cells} cells ({pct:.1f}%). `mode` is an
interpolation choice on the measure-to-decode path, so both come from identical
spins: **a feature present in both was sampled; a feature present only under
bilinear was introduced by the interpolator.**

`world.json` also carries `layers` -- the per-field binary draws whose sum is
each field's levels, so the sampling can be re-verified rather than only the
decode replayed. The corresponding spins are `2 * layers - 1`.

## What this run does NOT show

**Structure-stability is not mixing, and only the first was measured.** Warmup
was fixed at {warmup} sweeps because 4,000 and 100,000 were measured to give the
same STRUCTURE. Whether the chain has CONVERGED is a different question that has
never been measured here.

Sampler diagnostics were **not run**: no magnetization or energy trace, no
integrated autocorrelation time, no Gelman-Rubin R-hat. This run therefore makes
**no claim about mixing or convergence**. Those require instrumented sampling and
are produced by a separate reporting path.
"""


def save_run(studio: Studio, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    view = studio.view()
    paths = list(export_write(_current_world(studio), out))

    s_, r = studio.sampled, studio.readout
    studio_json = out / "studio.json"
    studio_json.write_text(json.dumps(dict(
        sampled=dict(seed=s_.seed, beta_j=s_.beta_j, size=s_.size, mode=s_.mode,
                     warmup=s_.warmup),
        readout=dict(sea_level=r.sea_level, mountain_line=r.mountain_line,
                     relief=r.relief, cost_table=r.cost_table),
        resolved_bounds=[float(b) for b in view.bounds],
        resolved_costs=list(COST_TABLES[r.cost_table]),
        kc=KC,
        beta_j_over_kc=s_.beta_j / KC,
        timings_apportioned_by_spin_count=studio.timings,
    ), indent=2), encoding="utf-8")
    paths.append(studio_json)

    near = studio.terrain_for_mode("nearest")
    bil = studio.terrain_for_mode("bilinear")
    dec = out / "decodes.json"
    dec.write_text(json.dumps(dict(
        note=("The same sampled draws decoded under both upsample modes. `mode` "
              "is an interpolation choice on the measure-to-decode path, not a "
              "sampling one, so these two terrains come from IDENTICAL spins. "
              "A feature present in BOTH was sampled; a feature present only "
              "under bilinear was introduced by the interpolator. Without this "
              "diff, no single world can distinguish the two."),
        mode_used=studio.world.mode,
        terrain_nearest=near.tolist(),
        terrain_bilinear=bil.tolist(),
        cells_differing=int((near != bil).sum()),
        fraction_differing=float((near != bil).mean()),
    ), indent=2), encoding="utf-8")
    paths.append(dec)

    prov = out / "provenance.json"
    prov.write_text(json.dumps(_versions(), indent=2), encoding="utf-8")
    paths.append(prov)

    readme = out / "README.md"
    readme.write_text(README.format(
        backend=_versions()["jax_backend"], seed=s_.seed, beta_j=s_.beta_j,
        size=s_.size, mode=s_.mode, warmup=s_.warmup, sea_level=r.sea_level,
        mountain_line=r.mountain_line, relief=r.relief,
        cost_table=r.cost_table, total=studio.timings.get("total", 0.0),
        ratio=s_.beta_j / KC, kc=KC,
        side=("subcritical" if s_.beta_j < KC else "supercritical"),
        differing=int((near != bil).sum()), cells=int(near.size),
        pct=float((near != bil).mean()) * 100.0),
        encoding="utf-8")
    paths.append(readme)
    return paths


def load_run(out_dir) -> tuple[Sampled, Readout]:
    d = json.loads((Path(out_dir) / "studio.json").read_text(encoding="utf-8"))
    return Sampled(**d["sampled"]), Readout(**d["readout"])
