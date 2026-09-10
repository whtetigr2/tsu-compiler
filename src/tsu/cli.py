"""Operations, separated so each is independently useful (spec section 11)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from .passes.analyse import analyse
from .passes.encode import encode
from .passes.lower import lower
from .passes.route import BetaMismatchError, assert_beta_consistent
from .passes.search import compile_spec
from .preflight.check import preflight as run_preflight
from .preflight.model import load_model
from .preflight.render import write_preflight, write_regime
from .preflight.sweep import detect_uniform_square_lattice, sweep, usable_band
from .receipt import replay, write_receipt
from .report import render_explain, render_report
from .simulate import simulate
from .spec import load_spec
from .target import PROFILES
from .viz import render


def susceptibility_note(rows) -> str:
    """Describe the swept susceptibility for a reader, WITHOUT claiming a
    peak that was not actually observed.

    A maximum sitting at either EDGE of the swept couplings is not a peak
    -- it means the true maximum (if one exists at all) lies outside the
    range that was actually measured. Reporting "chi peaks at beta*J = X"
    when X is simply the last (or first) point the sweep happened to try
    is the identical mistake `usable_band`'s old upper edge made ("where I
    stopped looking" printed as a measurement of the model), one layer
    down: a coordinator review caught this live on a 16-node 1D Ising
    ring, whose chi rose monotonically to its largest value at the LAST
    swept beta_j (1.11 at beta*J=0.6, the top of the range) -- correctly
    NOT a peak, since the ring has no finite-temperature transition at any
    coupling (Ising 1925) and chi was simply still climbing when the sweep
    stopped.

    Only an INTERIOR maximum (strictly between the smallest and largest
    swept beta_j among the non-provisional rows) is reported as a peak.
    At either edge, this reports the direction chi is still moving and
    which CLI flag would extend the sweep to look further -- genuinely
    actionable, unlike a location that was never actually observed.
    """
    good = sorted((r for r in rows if not r.provisional), key=lambda r: r.beta_j)
    if not good:
        return ("Every row in this sweep was provisional (R-hat above "
                "threshold, chains disagree), so even the susceptibility "
                "peak's location cannot be reported here.")
    peak = max(good, key=lambda r: r.chi)
    lo_beta, hi_beta = good[0].beta_j, good[-1].beta_j
    if lo_beta < peak.beta_j < hi_beta:
        return (
            f"The susceptibility peaks at beta*J = {peak.beta_j:g} (chi = "
            f"{peak.chi:.3g}) within this sweep's own range. That is "
            "reported here as an INDICATIVE signal only, not a located "
            "transition: the susceptibility peak drifts with system size "
            "(see `tsu.preflight.sweep`'s module docstring), and this "
            "sweep has only one size to measure it at, so there is no way "
            "to tell how much it would move on a different-size graph. It "
            "plays no part in the (absent) band above.")
    if len(good) < 2:
        return (
            "This sweep has only one non-provisional row, so nothing can "
            "be said about where chi peaks (interior or otherwise) from a "
            "single point.")
    if peak.beta_j >= hi_beta:
        return (
            f"chi is still rising at the TOP of this sweep's range "
            f"(largest measured chi = {peak.chi:.3g}, at beta*J = "
            f"{peak.beta_j:g}, the last point swept, not an interior "
            "point) -- no peak was observed within this range, so the "
            "range may not bracket one and it may lie above --beta-max. "
            "Raising --beta-max would show whether chi turns over.")
    return (
        f"chi is still rising toward the BOTTOM of this sweep's range "
        f"(largest measured chi = {peak.chi:.3g}, at beta*J = "
        f"{peak.beta_j:g}, the first point swept, not an interior point) "
        "-- no peak was observed within this range, so the range may not "
        "bracket one and it may lie below --beta-min. Lowering --beta-min "
        "would show whether chi turns over.")


def sweep_single_model(model, *, beta_min, beta_max, beta_steps, out_dir):
    """Sweep ONE fixed-size model across beta -- what `regime` actually does.

    `--spec`/`--edges` (via `load_model`) give exactly one fixed-size
    IsingModel, not the model_fn(size, beta_j) family `sweep()` was written
    to generate. But beta is a scalar multiplier on the whole energy, so a
    sweep over it on a FIXED graph is well posed on its own -- a temperature
    sweep -- and yields <|m|>, chi, the Binder cumulant, tau, N_eff and
    R-hat against beta, plus the susceptibility peak. What it cannot give is
    the Binder CROSSING, which genuinely needs two or more sizes; `crossing`
    is always None here, and the report explains why (see the appended
    paragraph below -- write_regime's own crossing=None message is generic
    and would otherwise mislead a reader into thinking the range just
    didn't happen to bracket a transition).

    Every candidate beta_j is checked against `model`'s OWN (unreplaced)
    beta via `assert_beta_consistent` before it is ever sampled: a model
    carrying mediator spins (`model.mediator_nodes`) has temperature-
    dependent couplings baked in (spec 5.3.5, `route.py`), so resampling it
    at a beta other than the one those couplings were computed at would
    silently reproduce the WRONG physical couplings. An unmediated model
    (`mediator_nodes` empty) has nothing to protect and sweeps freely --
    `assert_beta_consistent` no-ops for it. Raises `BetaMismatchError`
    (from the FIRST refused beta_j, before any sampling) when the model is
    mediated and the swept range does not sit at its own beta throughout.
    """
    couplings = np.linspace(beta_min, beta_max, beta_steps).tolist()
    n_spins = len(model.nodes)

    def model_fn(size, beta_j):
        # Checked against the ORIGINAL `model`, not the beta-replaced copy
        # returned below -- checking the copy would always trivially agree
        # with itself and never catch anything.
        assert_beta_consistent(model, beta_j)
        return dataclasses.replace(model, beta=beta_j)

    rows = sweep(model_fn, sizes=[n_spins], couplings=couplings)
    # crossing is always None here (no size family -- see below), and
    # usable_band's lower edge IS the crossing (tsu.preflight.sweep's own
    # docstring), so band is always None too. That is not a second,
    # independent limitation; it is a direct consequence of the first, and
    # the appended report text below says so rather than leaving the two
    # facts looking unrelated.
    band = usable_band(rows, None)
    # F1 (branch review): actually CHECK whether Onsager's Kc applies to
    # this model instead of hardcoding None. The old comment here claimed
    # the fact "cannot be determined" -- that was itself wrong: nothing
    # about `load_model`'s output prevents checking uniform |J|, zero
    # field, and grid structure directly from nodes/edges/weights/biases,
    # only the absence of a check did. `detect_uniform_square_lattice` is
    # deliberately conservative (three states: applies / checked and does
    # not apply / not determined -- see its own docstring), so a false
    # positive here remains as unlikely as under the old hardcoded-None
    # behaviour, while a genuine uniform square lattice -- the one graph
    # Onsager solved, and the graph on which the old hardcoded None
    # produced a flatly false "not applicable to this graph" claim -- now
    # gets its Kc printed and cross-checked against the measured crossing.
    onsager_kc, onsager_note = detect_uniform_square_lattice(model)
    paths = write_regime(rows, band, None, out_dir, onsager=onsager_kc,
                         onsager_note=onsager_note)

    # write_regime's own crossing=None message ("the range may not bracket
    # the transition") is correct for a genuine two-size sweep that simply
    # never crossed, but MISLEADING here: this sweep has no size family at
    # all, so the crossing is not unobserved, it is structurally
    # unavailable from a single model -- and since the band's lower edge IS
    # the crossing, the same is true of "No usable band was found" a few
    # lines later in write_regime's own output. Append the real reason for
    # both, tied together, rather than let them stand as two apparently
    # independent, contradictory-looking limitations.
    peak_text = susceptibility_note(rows)
    report_path = Path(out_dir) / "report.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8") +
        "\n## Why no Binder crossing (and no usable band)\n\n"
        f"This sweep covers a single size ({n_spins} spins) because "
        "`--spec`/`--edges` load exactly one fixed-size model -- there is "
        "no size family to sweep. Locating the transition by finite-size "
        "scaling requires two or more sizes' Binder curves to cross; a "
        "single `--spec`/`--edges` model cannot provide that, so no "
        "crossing is computed or inferred here.\n\n"
        "The usable band's lower edge IS the Binder crossing, so with no "
        "crossing there is no usable band either -- this is the same "
        "limitation stated twice, not two independent ones. " + peak_text +
        "\n\nEverything else above (<|m|>, chi, U, tau, N_eff, R-hat) is "
        "measured directly from this model's own sweep and is unaffected "
        "by the missing size family.\n",
        encoding="utf-8")

    return rows, band, paths


def _force_utf8_stdout() -> None:
    """Print box-drawing and check marks on a console that defaults to cp1252.

    Windows terminals default to a legacy code page, so writing the report's
    U+2500 rule or its check marks raises UnicodeEncodeError before a single
    line reaches the user. Reconfiguring is preferred over degrading the output;
    `errors="replace"` is the floor so a report is never lost to an encoding.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv=None) -> int:
    _force_utf8_stdout()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="tsu")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile"); c.add_argument("spec")
    c.add_argument("--target", default="z1"); c.add_argument("--out", required=True)
    c.add_argument("--allow-assumed", action="store_true")
    c.add_argument("--placement-restarts", type=int, default=None)
    c.add_argument("--placement-iters", type=int, default=None)
    c.add_argument("--coefficient-scale", type=float, default=1.0,
                   help="uniform multiplier on every energy coefficient "
                        "(including the representation penalty); beta is "
                        "compensated automatically so the induced "
                        "distribution is unchanged (spec section 4.9)")

    i = sub.add_parser("inspect"); i.add_argument("spec")
    v = sub.add_parser("visualize"); v.add_argument("receipt")
    v.add_argument("--out", required=True)
    r = sub.add_parser("replay"); r.add_argument("receipt")
    rp = sub.add_parser("report"); rp.add_argument("receipt")
    ex = sub.add_parser("explain"); ex.add_argument("receipt")

    sm = sub.add_parser("simulate"); sm.add_argument("receipt")
    sm.add_argument("--samples", type=int, default=200)
    sm.add_argument("--chains", type=int, default=32)
    sm.add_argument("--warmup", type=int, default=400)
    sm.add_argument("--beta", type=float, default=None)
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--clamp", action="append", default=None,
                    help="VAR=VAL; may be given multiple times")

    # preflight/regime: two questions about an Ising model -- will it fit
    # (preflight) and at what coupling should it be sampled (regime). Both
    # parsers accept the same --spec/--edges/--out shape as load_model().
    for name in ("preflight", "regime"):
        q = sub.add_parser(name)
        q.add_argument("--spec"); q.add_argument("--edges")
        q.add_argument("--out", default=f"{name}_out")
    rg = sub.choices["regime"]
    rg.add_argument("--beta-min", type=float, default=0.05)
    rg.add_argument("--beta-max", type=float, default=0.60)
    rg.add_argument("--beta-steps", type=int, default=10)
    # "--beta-steps", NOT "--steps": sweep()'s own `steps` parameter means
    # Gibbs steps per sample, a different axis entirely -- one flag meaning
    # two things in this CLI is how a user ends up sweeping the wrong one.
    # No "--sizes": `--spec`/`--edges` (via load_model()) give exactly ONE
    # fixed-size model, so there is no family of sizes for this flag to
    # mean anything over -- `regime` always sweeps beta at that one size
    # (see sweep_single_model). Dropped rather than accepted-and-ignored or
    # accepted-and-rejected-by-hand: argparse's own "unrecognized
    # arguments" error for `--sizes` on this subcommand is already an
    # honest, correct message (the flag genuinely does not exist here), so
    # a second hand-written check would just duplicate it.

    a = p.parse_args(argv)

    if a.cmd == "inspect":
        spec = load_spec(a.spec)
        rep = analyse(lower(encode(spec).model))
        # rep.mediators == -1 means "not computed" (exact max-cut is exponential
        # and this graph exceeded MAXCUT_EXACT_LIMIT), not zero mediators (I5).
        mediators = rep.mediators if rep.mediators >= 0 else \
            "not computed: graph exceeds the exact max-cut limit"
        print(json.dumps({"nodes": rep.n_nodes, "edges": rep.n_edges,
                          "max_degree": rep.max_degree, "bipartite": rep.bipartite,
                          "mediators": mediators,
                          "colour_blocks": rep.colour_blocks,
                          "max_abs_J": rep.max_abs_J}, indent=2))
        return 0

    if a.cmd == "compile":
        _eff = {k: v for k, v in (("restarts", a.placement_restarts),
                                  ("iters", a.placement_iters)) if v is not None}
        comp = compile_spec(load_spec(a.spec), PROFILES[a.target], a.allow_assumed,
                            coefficient_scale=a.coefficient_scale,
                            placement_effort=_eff or None)
        d = write_receipt(comp, a.out)
        print(f"verdict: {comp.verdict}")
        if comp.verdict != "COMPILED":
            for cand in comp.repset.candidates:
                print(f"  {cand.state.value}: {cand.reason}")
        print(f"receipt: {d}")
        return 0 if comp.verdict == "COMPILED" else 2

    if a.cmd == "visualize":
        print(render(a.receipt, a.out))
        return 0

    if a.cmd == "replay":
        res = replay(a.receipt)
        print("MATCHES" if res.matches else f"DIVERGED: {res.diffs}")
        return 0 if res.matches else 3

    if a.cmd == "report":
        print(render_report(a.receipt))
        return 0

    if a.cmd == "explain":
        print(render_explain(a.receipt))
        return 0

    if a.cmd == "simulate":
        clamp = None
        if a.clamp:
            clamp = {}
            for item in a.clamp:
                name, sep, val = item.partition("=")
                if not sep:
                    raise ValueError(f"--clamp expects VAR=VAL, got {item!r}")
                clamp[name] = int(val)
        path, _got, _im = simulate(
            a.receipt, n_chains=a.chains, n_samples=a.samples,
            n_warmup=a.warmup, beta=a.beta, seed=a.seed, clamp=clamp)
        print(f"simulation: {path}")
        return 0

    if a.cmd == "preflight":
        model = load_model(spec=a.spec, edges=a.edges)
        rep = run_preflight(model)
        for out_path in write_preflight(rep, a.out):
            print(f"  -> {out_path}")
        print(f"  verdict: {rep.verdict.upper()}  "
              f"(bipartite={rep.bipartite}, path={rep.embedding})")
        return 0 if rep.verdict != "fail" else 1

    if a.cmd == "regime":
        # --spec/--edges (via load_model) give exactly ONE fixed-size
        # IsingModel -- sweep_single_model sweeps beta at that one size
        # (see its own docstring for why that is well posed, and what it
        # cannot give: the Binder crossing, which needs a size family).
        model = load_model(spec=a.spec, edges=a.edges)
        try:
            rows, band, paths = sweep_single_model(
                model, beta_min=a.beta_min, beta_max=a.beta_max,
                beta_steps=a.beta_steps, out_dir=a.out)
        except BetaMismatchError as exc:
            # Refuse cleanly (spec 5.3.5): a mediated model's couplings are
            # only correct at the beta they were computed at, so this is a
            # real refusal to report, not a crash to hide.
            print(f"  regime: refused -- {exc}", file=sys.stderr)
            return 2
        for out_path in paths:
            print(f"  -> {out_path}")
        print(f"  usable band: {band}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
