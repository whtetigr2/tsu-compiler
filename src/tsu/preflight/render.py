"""The artifact a reader checks the work from.

Two rules shape every function here. A number the reader cannot act on is
noise, so every gate ships its limit and every estimate ships its
uncertainty. And a number the tool does not trust must SAY so on the page
rather than in the JSON, because a reader who has to open the JSON to find
out will not.

This module follows `tsu.report`'s discipline (see that module's docstring):
it never recomputes a number the compiler or the sweep did not already
record, and it prints `unavailable: <reason>` -- never a blank, a zero, or an
invented value -- wherever one is missing. `_fmt_opt` below is this module's
equivalent of `tsu.report._as_unavailable`.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsu.preflight.diagnostics import RHAT_THRESHOLD


def provenance() -> dict:
    """Everything a skeptical reader needs to know THIS RUN never touched
    real Z1 hardware and to reproduce it: interpreter, library versions, the
    JAX backend actually in effect (jax.default_backend(), not assumed --
    this project runs on CPU, but a reader must be told that rather than
    guess it), and the compiler commit. Never raises: any piece that cannot
    be determined is reported as "unavailable"/"unknown" rather than aborting
    the whole report over one missing fact.
    """
    out: dict[str, Any] = {"python": platform.python_version()}
    for mod in ("thrml", "jax", "numpy", "matplotlib"):
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


_HEADER = ("# Pre-flight report\n\n"
           "Produced by the `tsu` compiler passes. **No hardware was involved**"
           " — where sampling appears below, thrml *simulates* on the "
           "`{backend}` backend the block Gibbs a TSU would perform.\n\n")


def _as_unavailable(text: str) -> str:
    """Mirrors `tsu.report._as_unavailable`: every stand-in for a missing
    number in this codebase reads `unavailable: <reason>`."""
    return text if text.lower().startswith("unavailable") else f"unavailable: {text}"


def _format_band(band) -> str:
    """Render `usable_band`'s return value for a human reader.

    `band` is a `(lo, hi)` pair, either fully closed or OPEN ABOVE (`hi is
    None`, meaning no row at or above the crossing reached SATURATION
    within the sweep -- see `tsu.preflight.sweep.usable_band`'s own
    docstring). The open case is spelled out as an explicit half-open
    interval rather than left to fall through to Python's own tuple repr:
    `f"{band}"` on `(0.2, None)` prints the literal substring "None" next
    to a beta value, which reads as a formatting bug, not as "unbounded" --
    exactly the "where I stopped looking, printed as a measurement"
    failure this project already fixed once for the band's edges
    themselves. `None` is never printed bare anywhere else in this
    module's report (see `_fmt_opt`/`_as_unavailable` above); this keeps
    that rule.
    """
    lo, hi = band
    if hi is None:
        return (f"[{lo:g}, +inf) -- open above (no row in this sweep "
                "reached saturation at or above the crossing)")
    return f"({lo:g}, {hi:g})"


def _fmt_opt(value, reason: str, fmt: str) -> str:
    """Format `value` with `fmt`, or `unavailable: <reason>` when it is None.

    `RegimeRow.abs_m_err`/`tau`/`n_eff` are None exactly when tsu.ess judged
    the run too short to support a trustworthy estimate (see RegimeRow's own
    docstring in sweep.py). Formatting None directly (`f"{None:.3f}"`) raises
    TypeError; silently substituting 0.0 would print a number that reads as
    an exact measurement of zero -- the opposite of what happened. This is
    the one place both failure modes are avoided.
    """
    if value is None:
        return _as_unavailable(reason)
    return format(value, fmt)


def write_preflight(report, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance()

    j = out / "preflight.json"
    j.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    p = out / "provenance.json"
    p.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    lines = [_HEADER.format(backend=prov["jax_backend"]),
             f"**Verdict: {report.verdict.upper()}**\n",
             f"- {report.n_spins:,} spins, {report.n_couplings:,} couplings, "
             f"max degree {report.max_degree}",
             f"- bipartite: **{report.bipartite}** — embedding path "
             f"`{report.embedding}`, {report.mediators} mediators",
             f"- placement took {report.place_seconds:.3f}s\n",
             "| gate | value | limit | % of limit | status |",
             "|---|---|---|---|---|"]
    for g in report.gates:
        pct = 100.0 * g.value / g.limit if g.limit else float("nan")
        lines.append(f"| {g.name} | {g.value:g} | {g.limit:g} | "
                     f"{pct:.1f}% | **{g.status}** |")
    if not report.bipartite:
        lines.append(
            "\n> The interaction graph is **not bipartite**, so it cannot be "
            "a direct subgraph of a chessboard lattice. Placement falls back "
            "to an annealing search, which on this project has taken "
            "6–32 minutes *to fail* at sizes above 8×8.")
    if report.place_error is not None:
        # check.py's own docstring: a pre-flight report that says only
        # "degree exceeded" throws away the actionable half of the answer
        # place() had already worked out. Surface both the error and the
        # compiler's own remediations rather than only the failed verdict.
        lines.append(f"\n> **Placement failed:** {report.place_error}")
        if report.remediations:
            lines.append("\n**Remediations:**")
            lines += [f"- {r}" for r in report.remediations]
    md = out / "report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [j, md, p]


def write_regime(rows, band, cross, out_dir, *, onsager,
                 sweep_config: dict | None = None) -> list[Path]:
    """Render a regime sweep's rows, usable band and Binder crossing.

    `sweep_config` is NOT part of this task's stated interface
    (`write_regime(rows, band, cross, out_dir, *, onsager)`) -- it is added
    here, keyword-only and defaulted to None, because `RegimeRow` carries no
    seed or sweep-parameter fields at all, so nothing about the run that
    produced `rows` (seed, n_chains, n_samples, n_warmup, steps, max_samples,
    the sweep's own sizes/couplings) is otherwise reachable from this
    function's arguments. Rule 3 requires provenance.json to record "seeds,
    and the full sweep configuration"; with the stated signature alone that
    is structurally impossible to satisfy without inventing values. This
    keyword lets a caller that HAS that information (e.g. a script driving
    `sweep()` directly) pass it through; when it is not supplied,
    provenance.json says so explicitly (`unavailable: ...`) rather than
    omitting the key or fabricating a config. Every call in this task's own
    brief omits this argument and continues to work unchanged.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    prov["sweep_config"] = (sweep_config if sweep_config is not None else
                            "unavailable: not supplied to write_regime "
                            "(RegimeRow carries no seed/sweep-parameter "
                            "fields; pass sweep_config= from the caller "
                            "that ran sweep() to record it)")
    prov_path = out / "provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    j = out / "regime.json"
    j.write_text(json.dumps(dict(
        rows=[asdict(r) for r in rows],
        usable_band=list(band) if band else None,
        binder_crossing=cross,
        onsager_kc=onsager,
        onsager_note=("printed only because the graph is a uniform square "
                      "lattice in zero field" if onsager is not None else
                      "not applicable to this graph"),
    ), indent=2), encoding="utf-8")

    sizes = sorted({r.size for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for s in sizes:
        rs = sorted((r for r in rows if r.size == s), key=lambda r: r.beta_j)
        x = [r.beta_j for r in rs]
        # yerr entries may be None (ess-unavailable rows); matplotlib's
        # errorbar cannot take a mixed None/float yerr array, so a missing
        # bar is drawn as 0 height here ONLY on the plot (never in the
        # table or the JSON, where it stays None/"unavailable: ..."), which
        # is the least misleading option a picture allows -- there is no way
        # to draw "no bar at all" for one point on a shared errorbar call
        # without a second plotting pass per row.
        yerr = [0.0 if r.abs_m_err is None else r.abs_m_err for r in rs]
        ax[0].errorbar(x, [r.abs_m for r in rs], yerr=yerr, marker="o",
                       capsize=3, label=f"L={s}")
        ax[1].plot(x, [r.chi for r in rs], marker="o", label=f"L={s}")
        ax[2].plot(x, [r.binder for r in rs], marker="o", label=f"L={s}")
    for a, t in zip(ax, ("<|m|>  (error bars from N_eff; 0-height = no "
                         "reliable ESS, see report.md)",
                         "susceptibility  (peak drifts with size)",
                         "Binder cumulant U  (crossing locates Kc)")):
        a.set_xlabel("beta*J"); a.set_title(t, fontsize=9); a.legend(fontsize=8)
        if cross is not None:
            a.axvline(cross, color="crimson", ls="--", lw=1)
    if onsager is not None:
        ax[2].axvline(onsager, color="black", ls=":", lw=1,
                      label="Onsager Kc")
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    p1 = out / "regime.png"
    fig.savefig(p1, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for s in sizes:
        rs = sorted((r for r in rows if r.size == s), key=lambda r: r.beta_j)
        x = [r.beta_j for r in rs]
        # tau is None for the same ess-unavailable rows; plot only the
        # points that have one rather than substitute a fabricated value.
        tx = [b for b, r in zip(x, rs) if r.tau is not None]
        ty = [r.tau for r in rs if r.tau is not None]
        ax[0].plot(tx, ty, marker="o", label=f"L={s}")
        ax[1].plot(x, [r.r_hat for r in rs], marker="o", label=f"L={s}")
    ax[0].set_title("integrated autocorrelation time tau (unavailable rows "
                    "omitted -- see report.md)", fontsize=9)
    ax[1].set_title("Gelman-Rubin R-hat")
    ax[1].axhline(RHAT_THRESHOLD, color="crimson", ls="--", lw=1)
    for a in ax:
        a.set_xlabel("beta*J"); a.legend(fontsize=8)
    fig.tight_layout()
    p2 = out / "diagnostics.png"
    fig.savefig(p2, dpi=120); plt.close(fig)

    lines = [_HEADER.format(backend=prov["jax_backend"]),
             "## Regime\n"]
    if cross is not None:
        lines.append(f"Binder crossing at **beta*J = {cross:.4f}**")
    else:
        lines.append("**No Binder crossing was observed in this sweep** — "
                     "the range may not bracket the transition. No "
                     "transition is reported rather than one being "
                     "inferred.")
    if onsager is not None:
        lines.append(f"\nOnsager's exact `Kc = {onsager:.6f}` is shown for "
                     f"comparison **because this graph is a uniform square "
                     f"lattice in zero field**, the only condition under "
                     f"which it applies.")
    lines.append(f"\nUsable band: **{_format_band(band)}**\n" if band else
                 "\nNo usable band was found in this sweep.\n")
    lines += ["| beta*J | L | <\\|m\\|> | error | chi | U | tau | N_eff | "
              "R-hat | N_samples | flags |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r.size, r.beta_j)):
        flags = []
        if r.provisional:
            flags.append("**provisional**")
        if r.ess_unavailable:
            flags.append("no error bar")
        lines.append(
            f"| {r.beta_j:g} | {r.size} | {r.abs_m:.3f} | "
            f"{_fmt_opt(r.abs_m_err, r.ess_reason, '.3f')} | {r.chi:.3g} | "
            f"{r.binder:.3f} | {_fmt_opt(r.tau, r.ess_reason, '.2f')} | "
            f"{_fmt_opt(r.n_eff, r.ess_reason, '.0f')} | {r.r_hat:.3f} | "
            f"{r.n_samples_used:,} | {', '.join(flags)} |")
    lines.append("\nRows marked **provisional** have R-hat above "
                 f"{RHAT_THRESHOLD}: their chains disagree, so those numbers "
                 "are not trusted and are excluded from the band. Rows "
                 "marked **no error bar** have a usable mean (abs_m/binder/"
                 "chi are still the best estimate the draws support) but "
                 "tsu.ess judged tau/ESS itself too noisy to quantify an "
                 "uncertainty on it -- see that row's `error`/`tau`/`N_eff` "
                 "cells for why. `N_samples` is the draw count sweep()'s own "
                 "escalation loop actually settled on for that row: rows "
                 "near the transition need more draws because tau diverges "
                 "there (critical slowing down), so a larger N_samples is "
                 "itself a physics readout, not bookkeeping.")
    md = out / "report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [j, p1, p2, md, prov_path]
