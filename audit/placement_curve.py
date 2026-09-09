"""Measure how placement scales with grid size. Writes incrementally so a
killed run still leaves every row it completed -- placement is the only
pass whose cost grows with grid size (all gates are size-invariant), so
this curve is what decides how big a world can be."""
import sys, time, tempfile, os, json
sys.path.insert(0, "src")
from tsu.spec import load_spec
from tsu.passes.encode import encode
from tsu.passes.lower import lower
from tsu.passes.analyse import analyse
from tsu.passes.place import place
from tsu.target import PROFILES
from tsu.failures import CompileError

BASE = """name: p
generate: {{kind: grid, width: {n}, height: {n}, variable_domain: {{domain: categorical, k: 3}}}}
terms:
  - {{kind: product_over_edges, a_value: 0, b_value: 1, weight: 4.0}}
  - {{kind: product_over_edges, a_value: 2, b_value: 2, weight: -1.5}}
"""
OVL = """name: o
generate: {{kind: grid, width: {n}, height: {n}, variable_domain: {{domain: binary}}}}
terms:
  - {{kind: product_over_edges, a_value: 1, b_value: 1, weight: -0.4}}
"""
out = "audit/placement_curve.json"
rows = []
for n in (8, 10, 12, 14, 16, 20):
    for label, body in (("base_k3", BASE), ("overlay", OVL)):
        f = os.path.join(tempfile.gettempdir(), "pc.yaml")
        open(f, "w").write(body.format(n=n))
        m = lower(encode(load_spec(f), "domain_wall").model)
        rep = analyse(m)
        t0 = time.perf_counter()
        try:
            p = place(m, rep, PROFILES["z1"], restarts=12, iters=200000)
            el = time.perf_counter() - t0
            r = dict(grid=n, kind=label, spins=rep.n_nodes, edges=rep.n_edges,
                     degree=rep.max_degree, bipartite=rep.bipartite,
                     place_s=round(el, 1), result="PLACED",
                     unrealized=len(p.unrealized),
                     mediators=(p.mediation.mediator_count if p.mediation else 0))
        except CompileError as e:
            el = time.perf_counter() - t0
            r = dict(grid=n, kind=label, spins=rep.n_nodes, edges=rep.n_edges,
                     degree=rep.max_degree, bipartite=rep.bipartite,
                     place_s=round(el, 1), result=e.failures[0].failure_class,
                     unrealized=None, mediators=None)
        rows.append(r)
        print(f"{n}x{n} {label:>8} spins={r['spins']:>5} {r['place_s']:>8.1f}s "
              f"{r['result']} med={r['mediators']}", flush=True)
        json.dump(rows, open(out, "w"), indent=1)
print("done ->", out)
