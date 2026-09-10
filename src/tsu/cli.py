"""Operations, separated so each is independently useful (spec section 11)."""
from __future__ import annotations

import argparse
import json
import sys

from .passes.analyse import analyse
from .passes.encode import encode
from .passes.lower import lower
from .passes.search import compile_spec
from .preflight.check import preflight as run_preflight
from .preflight.model import load_model
from .preflight.render import write_preflight
from .receipt import replay, write_receipt
from .report import render_explain, render_report
from .simulate import simulate
from .spec import load_spec
from .target import PROFILES
from .viz import render


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
    rg.add_argument("--sizes", type=int, nargs="+", default=[8, 16])
    # "--beta-steps", NOT "--steps": sweep()'s own `steps` parameter means
    # Gibbs steps per sample, a different axis entirely -- one flag meaning
    # two things in this CLI is how a user ends up sweeping the wrong one.

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
        # IsingModel; sweep() needs model_fn(size, beta_j) -> IsingModel, a
        # GENERATOR across the --sizes this parser accepts. Nothing in this
        # branch's inputs bridges that gap (task 1-4's load_model has no
        # size parameter), so this command reports that honestly rather than
        # silently sweeping only the one size --spec/--edges happened to
        # describe. It does not call load_model at all: failing that first,
        # on a user's otherwise-valid spec, would blame the wrong thing.
        print("  regime: --spec/--edges give one model; a sweep needs a "
              "family of sizes.\n  Provide a spec whose generator takes a "
              "size, or import tsu.preflight.sweep.sweep directly with your "
              "own model_fn.", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
