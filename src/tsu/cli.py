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
from .spec import load_spec
from .target import PROFILES
from .viz import render


def main(argv=None) -> int:
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

    a = p.parse_args(argv)

    if a.cmd == "inspect":
        spec = load_spec(a.spec)
        rep = analyse(lower(encode(spec).model))
        print(json.dumps({"nodes": rep.n_nodes, "edges": rep.n_edges,
                          "max_degree": rep.max_degree, "bipartite": rep.bipartite,
                          "mediators": rep.mediators,
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

    return 1
