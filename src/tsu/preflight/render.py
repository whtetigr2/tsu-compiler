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


# F8 (branch review): write_preflight and write_regime used to SHARE one
# hardcoded `_HEADER` literal -- "# Pre-flight report" as the title of
# EVERY regime run too, and "where sampling appears below, thrml
# simulates..." on the preflight report, where (by design --
# test_preflight_imports_no_sampler_at_all) sampling never appears at
# all. Two distinct headers now, built by one shared helper so the common
# "no hardware was involved" disclaimer (task-5-brief.md's own required
# sentence) cannot drift between them independently.
def _header(title: str, backend: str, *, sampling_appears_below: bool) -> str:
    text = (f"# {title}\n\n"
           f"Produced by the `tsu` compiler passes. **No hardware was "
           f"involved**")
    if sampling_appears_below:
        text += (f" — where sampling appears below, thrml *simulates* on "
                f"the `{backend}` backend the block Gibbs a TSU would "
                f"perform.")
    else:
        text += " — this pass runs compiler passes only; no sampling occurs."
    return text + "\n\n"


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


def _pct_cell(value: float, limit: float) -> str:
    """Residue 1 (coordinator review, post-F1-F10): the colouring gate's
    limit is 0 -- a violation COUNT, not a headroom threshold this project
    scales a percentage toward -- so `100*value/limit` is literally 0/0.
    Formatting that NaN with `.1f}%` prints the literal "nan%" on the
    page, which no test caught (grep -rn "nan" tests/ found nothing) and
    which reads as carelessness in a report whose whole pitch is rigor.
    A zero limit is reported as "n/a" with the reason a reader needs
    (this column measures headroom, and a count-against-zero gate has
    none to measure) rather than a raw NaN."""
    if not limit:
        return "n/a (zero-limit gate: any violation fails)"
    return f"{100.0 * value / limit:.1f}%"


def _verdict_qualifier(report) -> str:
    """Residue 2 (coordinator review, substantive): when EVERY gate
    driving the current verdict (fail, or warn/downgraded for a warn
    verdict) traces to an ASSUMED limit -- target.py's `Sourced(...,
    "assumed", ...)`, not a sourced Extropic fact -- the verdict line
    must say so. Marking the individual gate row (F2's `note` column) is
    NECESSARY but not SUFFICIENT: a reader who reads only
    "**Verdict: FAIL**" and stops has been told their model does not fit
    real hardware, when what actually happened is that it exceeds a
    project WORKING ASSUMPTION.

    Returns "" (no qualifier) whenever at least one gate driving the
    verdict is a SOURCED FACT -- a genuine hardware failure/warning must
    never be softened by wording that only applies to a DIFFERENT gate.
    This is what stops the qualifier being pasted onto every failure
    regardless of cause (see
    test_verdict_is_not_qualified_when_a_sourced_fact_gate_fails and its
    mixed-set sibling in tests/test_preflight_render.py).

    Residue 3 (coordinator review): "pass --allow-assumed to downgrade
    it" is only offered for gates that are STILL failing -- i.e. NOT
    already `downgraded`. A gate whose own status IS "downgraded" means
    --allow-assumed was already passed and already worked (verdict WARN,
    exit code 0); re-offering the same flag tells the reader to run
    something they just ran, which either loops them or reads as if the
    flag did nothing. For an already-downgraded gate this instead states
    what actually happened -- the assumed-limit failure was overridden to
    a warning AT THE USER'S REQUEST -- so a later reader of the artifact
    knows this WARN exists only because someone overrode an assumption,
    not because the model cleanly passed. `--allow-assumed` is still
    NAMED in that case, but as provenance for what happened, never as an
    imperative to run it again."""
    if report.verdict == "fail":
        driving = [g for g in report.gates if g.status == "fail"]
    elif report.verdict == "warn":
        driving = [g for g in report.gates if g.status in ("warn", "downgraded")]
    else:
        return ""
    if not driving or not all(g.assumed for g in driving):
        return ""

    still_failing = [g for g in driving if not g.downgraded]
    already_downgraded = [g for g in driving if g.downgraded]

    clauses = []
    if still_failing:
        names = ", ".join(g.name for g in still_failing)
        clauses.append(
            f"rests entirely on ASSUMED limit(s) ({names}), not a sourced "
            f"hardware fact; pass --allow-assumed to downgrade "
            f"{'it' if len(still_failing) == 1 else 'them'}")
    if already_downgraded:
        names = ", ".join(g.name for g in already_downgraded)
        clauses.append(
            f"ASSUMED limit(s) ({names}) "
            f"{'was' if len(already_downgraded) == 1 else 'were'} "
            f"downgraded to a warning via --allow-assumed at your request, "
            f"not a clean pass")
    return " — " + "; ".join(clauses)


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

    lines = [_header("Pre-flight report", prov["jax_backend"],
                     sampling_appears_below=False),
             f"**Verdict: {report.verdict.upper()}**{_verdict_qualifier(report)}\n",
             f"- {report.n_spins:,} spins, {report.n_couplings:,} couplings, "
             f"max degree {report.max_degree}",
             f"- bipartite: **{report.bipartite}** — embedding path "
             f"`{report.embedding}`, {report.mediators} mediators",
             # F5 (branch review): `.3g`, not `.3f` -- a real, measured
             # place_seconds below 5e-4 (this project's own live value:
             # 7.128715515136719e-05) rounds to the literal "0.000" under
             # `.3f`, indistinguishable from an instantaneous placement.
             f"- placement took {report.place_seconds:.3g}s\n",
             "| gate | value | limit | % of limit | status | note |",
             "|---|---|---|---|---|---|"]
    for g in report.gates:
        # F2/F9 (branch review): the note column -- previously written to
        # preflight.json only, never rendered here -- is what lets a
        # reader see an ASSUMED (not Extropic-sourced) limit marked
        # visibly next to the gate it applies to, and what distinguishes
        # node_budget's pre-placement mediator ESTIMATE from the
        # `mediators` line above (placement's ACTUAL count), rather than
        # leaving two numbers that can disagree unexplained on the page.
        lines.append(f"| {g.name} | {g.value:g} | {g.limit:g} | "
                     f"{_pct_cell(g.value, g.limit)} | **{g.status}** | "
                     f"{g.note} |")
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
                 onsager_note: str | None = None,
                 sweep_config: dict | None = None) -> list[Path]:
    """Render a regime sweep's rows, usable band and Binder crossing.

    `onsager_note` (F1, branch review) makes explicit a caller's THREE
    distinct possible claims about Onsager's Kc, never collapsed to a
    binary applies/does-not: "applies" (`onsager` is numeric), "checked
    and does not apply" (a caller POSITIVELY VERIFIED a precondition
    fails), or "not determined" (never checked, or the check could not
    decide either way). Keyword-only and defaulted to None, matching the
    `sweep_config` precedent immediately below, so every call in this
    task's own brief keeps working unchanged. When a caller does not
    supply one, the fallback below is honest about NOT knowing rather
    than F1's shipped "not applicable to this graph" for every run that
    never checked at all -- `tsu.preflight.sweep.detect_uniform_square_
    lattice` is what a caller should use to get a real three-state answer;
    `tsu.cli.sweep_single_model` is the one caller in this codebase that
    does.

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

    if onsager_note is not None:
        note = onsager_note
    elif onsager is not None:
        note = ("printed only because the graph is a uniform square "
               "lattice in zero field")
    else:
        # F1 (branch review): NOT "not applicable to this graph" -- this
        # branch is reached only when a caller passed onsager=None without
        # ever saying WHY, so the honest claim is that applicability was
        # never determined here, not that it was checked and failed.
        note = ("not determined: this call did not report whether "
               "Onsager's solution applies to this graph (no onsager_note "
               "was supplied)")

    j = out / "regime.json"
    j.write_text(json.dumps(dict(
        rows=[asdict(r) for r in rows],
        usable_band=list(band) if band else None,
        binder_crossing=cross,
        onsager_kc=onsager,
        onsager_note=note,
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

    lines = [_header("Regime report", prov["jax_backend"],
                     sampling_appears_below=True),
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
        # F5 (branch review): `.3g`, not `.3f`, for `error` and `tau` -- a
        # real, measured abs_m_err below 5e-4 (this project's own live
        # value: 0.0004600829406846147) rounded to the literal "0.000"
        # under `.3f`, indistinguishable from `_fmt_opt`'s None case even
        # though `_fmt_opt` was built specifically so a None never renders
        # that way (see its own docstring). `n_eff` stays `.0f` -- an
        # effective sample COUNT rounds to a whole number sensibly and was
        # never the field that collided with zero.
        lines.append(
            f"| {r.beta_j:g} | {r.size} | {r.abs_m:.3f} | "
            f"{_fmt_opt(r.abs_m_err, r.ess_reason, '.3g')} | {r.chi:.3g} | "
            f"{r.binder:.3f} | {_fmt_opt(r.tau, r.ess_reason, '.3g')} | "
            f"{_fmt_opt(r.n_eff, r.ess_reason, '.0f')} | {r.r_hat:.3f} | "
            f"{r.n_samples_used:,} | {', '.join(flags)} |")
    lines.append("\n`tau` here is tau_A = 1 + 2 * sum_k rho_k (Sokal/emcee "
                 "automatic windowing) and `N_eff` = N_total / tau_A, "
                 "floored at tau_A = 1.0 so N_eff never prints larger than "
                 "the draws actually taken (tau_A = 1.0 exactly for i.i.d. "
                 "draws; an estimate below that is a windowing artifact, "
                 "not a real property -- see `tsu.preflight.sweep.sweep`).")
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
