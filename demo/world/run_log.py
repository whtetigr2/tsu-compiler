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
from world.studio import COST_TABLES, Readout, Sampled, Studio


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

## What this run does NOT show

Sampler diagnostics were **not run**. There is no magnetization or energy
trace, no integrated autocorrelation time, and no Gelman-Rubin R-hat in this
directory, so this run makes **no claim about mixing or convergence**. Those
require instrumented sampling and are produced by a separate reporting path.
"""


def save_run(studio: Studio, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    view = studio.view()
    paths = list(export_write(_current_world(studio), out))

    s, r = studio.sampled, studio.readout
    studio_json = out / "studio.json"
    studio_json.write_text(json.dumps(dict(
        sampled=dict(seed=s.seed, beta_j=s.beta_j, size=s.size, mode=s.mode,
                     warmup=s.warmup),
        readout=dict(sea_level=r.sea_level, mountain_line=r.mountain_line,
                     relief=r.relief, cost_table=r.cost_table),
        resolved_bounds=[float(b) for b in view.bounds],
        resolved_costs=list(COST_TABLES[r.cost_table]),
        timings_apportioned_by_spin_count=studio.timings,
    ), indent=2), encoding="utf-8")
    paths.append(studio_json)

    prov = out / "provenance.json"
    prov.write_text(json.dumps(_versions(), indent=2), encoding="utf-8")
    paths.append(prov)

    readme = out / "README.md"
    readme.write_text(README.format(
        backend=_versions()["jax_backend"], seed=s.seed, beta_j=s.beta_j,
        size=s.size, mode=s.mode, warmup=s.warmup, sea_level=r.sea_level,
        mountain_line=r.mountain_line, relief=r.relief,
        cost_table=r.cost_table, total=studio.timings.get("total", 0.0)),
        encoding="utf-8")
    paths.append(readme)
    return paths


def load_run(out_dir) -> tuple[Sampled, Readout]:
    d = json.loads((Path(out_dir) / "studio.json").read_text(encoding="utf-8"))
    return Sampled(**d["sampled"]), Readout(**d["readout"])
