"""Operations, separated so each is independently useful (spec section 11)."""
from __future__ import annotations

import argparse
import json
import sys

from .passes.analyse import analyse
from .passes.encode import encode
from .passes.lower import lower
from .passes.search import compile_spec
from .receipt import replay, write_receipt
from .report import render_explain, render_report
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

    i = sub.add_parser("inspect"); i.add_argument("spec")
    v = sub.add_parser("visualize"); v.add_argument("receipt")
    v.add_argument("--out", required=True)
    r = sub.add_parser("replay"); r.add_argument("receipt")
    rp = sub.add_parser("report"); rp.add_argument("receipt")
    ex = sub.add_parser("explain"); ex.add_argument("receipt")

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
        comp = compile_spec(load_spec(a.spec), PROFILES[a.target], a.allow_assumed)
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

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
