"""Task 5: `neighbourhood_count` -- a generic term template for "a cell's own
neighbourhood count, squared-deviation from a target". Named for the
MATHEMATICS, not a domain: it is the honest EXACT form only for a rule whose
OWN intent is symmetric ("as close to target as possible"), the identical
argument `audit/expressibility_matrix.md`'s `neighbourhood` row (#2) makes.
It is documented as a SHAPE, never a domain rule -- the same distinction
`product_over_edges`/`conserve_over_edges` already draw in `src/tsu_compiler/spec.py`.

Consumes ONLY what Task 3's matrix marked EXACT. `morphology`/`ecological`
(the one-sided "at least N" classes) are DISTORTED under this exact shape --
this module does not attempt to detect or special-case that; the distortion
is pinned by `test_neighbourhood_count_energy_is_symmetric_about_the_target`
below so nobody later mistakes this template for a one-sided rule.
"""
import os
import tempfile
import textwrap

import pytest

from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.encode import encode
from tsu_compiler.passes.lower import lower
from tsu_compiler.spec import load_spec


def _write(body: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        f.write(textwrap.dedent(body))
        return f.name


def test_neighbourhood_count_lowers_to_pairwise():
    """A squared deviation over a neighbourhood is Product(L, L, w) and must
    survive `lower` without raising ThreeBodyError."""
    body = (
        "name: nb\n"
        "generate: {kind: grid, width: 3, height: 3, "
        "variable_domain: {domain: binary}}\n"
        "terms:\n"
        "  - {kind: neighbourhood_count, value: 1, target: 4, weight: 0.5}\n"
    )
    p = os.path.join(tempfile.gettempdir(), "nb.yaml")
    open(p, "w").write(body)
    rep = analyse(lower(encode(load_spec(p), "domain_wall").model))
    assert rep.n_nodes == 9
    assert rep.max_degree <= 16


def test_neighbourhood_count_energy_is_symmetric_about_the_target():
    """Pins the known distortion: `neighbourhood_count`'s emitted energy is a
    squared linear form, hence symmetric about the target (N=target+k costs
    exactly what N=target-k costs, for every k). This is the honest form for
    a rule whose OWN intent is symmetric (`neighbourhood`, matrix row #2) --
    and it is precisely what makes this shape the WRONG one for a one-sided
    rule (`morphology`/`ecological`, matrix rows #3/#9), whose intent is
    "at least N" and is DISTORTED by squaring. Pinning the symmetry here
    means nobody later mistakes `neighbourhood_count` for a one-sided
    template: any change that made it asymmetric would be a bug, and any
    caller reaching for a one-sided rule needs a different mechanism
    entirely (Task 4's own probe: no auxiliary-variable escape is known).

    `neighbourhood_count` emits ONE term per node of the edge set (the same
    "one per node" shape `conserve_over_edges` uses), so a 3-node PATH
    (g0_0 - g1_0 - g2_0, a 3x1 grid) is used rather than a denser grid: it is
    the smallest topology where exactly one node (the middle) has more than
    one neighbour, so its term is the only one whose energy varies with the
    swept assignment -- g0_0's and g2_0's own terms each reference only
    g1_0 (held fixed at 0 throughout), so they contribute a fixed constant,
    and the comparison below isolates the middle node's term cleanly rather
    than mixing in cross-term contributions from other cells' own
    neighbourhood_count terms (which a denser grid, e.g. 3x3, would not
    isolate: a corner cell's term there also references two of the swept
    cells).

    Verified via the IR's own `model.energy` over the full swept assignment,
    not by inspecting the emitted term's shape alone; hand-computed
    reference values are asserted alongside the symmetry itself so a
    silently-wrong-but-still-symmetric implementation cannot pass by
    accident.
    """
    body = (
        "name: nb_sym\n"
        "generate: {kind: grid, width: 3, height: 1, "
        "variable_domain: {domain: binary}}\n"
        "terms:\n"
        "  - {kind: neighbourhood_count, value: 1, target: 1, weight: 1.0}\n"
    )
    p = os.path.join(tempfile.gettempdir(), "nb_sym.yaml")
    open(p, "w").write(body)
    s = load_spec(p)
    enc = encode(s, "domain_wall")

    def energy_at(g0_0: int, g2_0: int) -> float:
        return enc.model.energy({"g0_0": g0_0, "g1_0": 0, "g2_0": g2_0})

    e_deficit = energy_at(0, 0)   # g1_0's own neighbour-count N=0, target=1
    e_target = energy_at(1, 0)    # N=1 == target (or energy_at(0, 1), by symmetry)
    e_surplus = energy_at(1, 1)   # N=2, target=1

    # Hand-computed reference: g0_0's and g2_0's own terms are each fixed at
    # weight*(0-1)**2 = 1.0 (they reference only g1_0 == 0) regardless of the
    # sweep, so total energy = 2.0 + weight*(N-1)**2 for the middle node's
    # own term, N = g0_0 + g2_0.
    assert e_deficit == pytest.approx(3.0)   # 2.0 + (0-1)**2
    assert e_target == pytest.approx(2.0)    # 2.0 + (1-1)**2
    assert e_surplus == pytest.approx(3.0)   # 2.0 + (2-1)**2

    assert e_deficit == e_surplus, (
        f"N=0 ({e_deficit}) must equal N=2 ({e_surplus}): symmetric about "
        f"target=1 -- a squared linear form penalises a deficit and an "
        f"equal surplus identically")
    assert e_target < e_deficit and e_target < e_surplus, \
        "the target itself must be the unique minimum"
