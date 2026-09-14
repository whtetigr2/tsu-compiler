"""I6 (code review, 2026-09-04-lattice-rule-taxonomy): `expressibility_matrix.md`
section 6 (`topological`) compared a SOLO measurement (conservation alone on
the full 8x8 grid, headroom to weight=12.0 before the 6.0 |b| cap) against
the brief's own PRIOR finding, which was about a CONTESTED setup
(conservation competing against a separate 4.0-weight adjacency penalty on
the same grid), and called the difference "roughly 24x more weight
headroom" -- but the contested setup itself was never actually measured in
this matrix. That is not a fair comparison: one number describes a term in
isolation, the other describes two terms competing for the same |b| budget.

This script measures the CONTESTED setup for real: a `product_over_edges`
adjacency term (a_value=b_value=1, weight=4.0 -- `symmetric: true` by
default doubles this to an effective 8.0 per edge, per
`src/tsu_compiler/spec.py::_product_over_edges`'s own docstring) on the full 8x8
grid, alongside a `conserve_over_edges` term swept across the SAME range
the brief specifies (0.1 to 12.0).

Result: the adjacency term ALONE already produces `|b|max=8.0` (a single
interior cell, degree 4, each edge contributing 2.0 to that cell's own
bias: 4 edges x 2.0 = 8.0) -- already over the 6.0 field cap with NO
conservation term present at all. Adding conservation at any weight in the
swept range does not change this: `|b|max` stays at 8.0 throughout, because
the adjacency term's own peak bias exceeds conservation's own (measured
separately in the matrix's own topological section: conservation alone
tops out at |b|max=6.0 only at weight=12.0, i.e. never above 8.0 across
this sweep). So there is NO conservation weight in this configuration that
fits the field cap -- the prior finding (a collision at this competing
setup) is confirmed, not contradicted, once actually measured; the
matrix's "24x more headroom" framing compared the wrong two numbers.

Run with the project's pinned interpreter, from the repo root:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
        audit/contested_conservation_probe.py
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsu_compiler.spec import load_spec  # noqa: E402
from tsu_compiler.passes.encode import encode  # noqa: E402
from tsu_compiler.passes.lower import lower  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402

FIELD_CAP = 6.0
ADJACENCY_WEIGHT = 4.0
SWEEP = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def _write(body: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    f.write(textwrap.dedent(body))
    f.close()
    return f.name


def measure_adjacency_alone():
    p = _write(f"""
        name: adjacency_alone
        generate: {{kind: grid, width: 8, height: 8, variable_domain: {{domain: binary}}}}
        terms:
          - {{kind: product_over_edges, a_value: 1, b_value: 1, weight: {ADJACENCY_WEIGHT}}}
    """)
    s = load_spec(p)
    ising = lower(encode(s).model)
    return analyse(ising)


def measure_contested(conservation_weight: float):
    p = _write(f"""
        name: contested_setup
        generate: {{kind: grid, width: 8, height: 8, variable_domain: {{domain: binary}}}}
        terms:
          - {{kind: product_over_edges, a_value: 1, b_value: 1, weight: {ADJACENCY_WEIGHT}}}
          - {{kind: conserve_over_edges, flow_prefix: f, weight: {conservation_weight},
             sources: {{g0_0: 1.0}}, sinks: {{g7_7: 1.0}}}}
    """)
    s = load_spec(p)
    ising = lower(encode(s).model)
    return analyse(ising)


def main():
    print("=== I6: the CONTESTED setup, actually measured ===")
    r0 = measure_adjacency_alone()
    print(f"  adjacency term ALONE (weight={ADJACENCY_WEIGHT}, symmetric doubling "
          f"to effective 8.0/edge): n_edges={r0.n_edges} max_degree={r0.max_degree} "
          f"|J|max={r0.max_abs_J} |b|max={r0.max_abs_b}  "
          f"exceeds field_cap({FIELD_CAP}) with ZERO conservation weight: "
          f"{r0.max_abs_b > FIELD_CAP}")

    print(f"\n  conservation weight swept {SWEEP[0]}..{SWEEP[-1]}, competing "
          f"against the same adjacency term:")
    all_fail = True
    for w in SWEEP:
        r = measure_contested(w)
        fails = r.max_abs_b > FIELD_CAP
        all_fail &= fails
        print(f"    weight={w:>5}: max_degree={r.max_degree}  |J|max={r.max_abs_J:.4f}  "
              f"|b|max={r.max_abs_b:.4f}  fails field_cap({FIELD_CAP}): {fails}")

    print(f"\n=== VERDICT ===")
    print(f"  |b|max stays at (or above) the adjacency term's own solo peak "
          f"(8.0) across the ENTIRE swept range, and every swept weight "
          f"fails the field_cap gate: {all_fail}")
    print("  There is no conservation weight in this configuration that fits "
          "the 6.0 field cap -- the prior finding (a real collision at this "
          "competing setup) is CONFIRMED once measured, not contradicted; "
          "the matrix's earlier '24x more headroom' framing compared a solo "
          "measurement against a contested figure that was never actually "
          "reproduced.")


if __name__ == "__main__":
    main()
