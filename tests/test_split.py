"""Task 2 (plan 2026-09-04-degree-space-trade): Route A -- node splitting.

`split_high_degree` is a compiler pass over an arbitrary graph: a logical
spin of degree > max_degree becomes a CHAIN of copies, each carrying a
fraction of the original edges, bound together by a ferromagnetic coupling
strong enough that every copy takes the same value. The whole risk is a
BROKEN chain -- copies disagreeing, which makes the logical spin's value
meaningless while the model still looks "successfully split" (degree drops,
nothing else visibly wrong). A degree assertion alone cannot see this; only
a marginal-preservation check against an independent oracle can.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

# audit/ (the parent of oracles/) -- `audit/oracles/exact.py` is written from
# the physics and does NOT import src/tsu (plan Task A4's own rule, restated
# in split.py's own module docstring): "an oracle sharing code with the thing
# under test verifies nothing." Same sys.path pattern `audit/tests/test_oracles.py`
# already uses to reach the same module from outside its own directory.
AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit"
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))

from oracles.exact import exact_boltzmann  # noqa: E402

# How close the split model's marginal over the ORIGINAL spins must be to the
# unsplit model's own exact joint (both computed by full brute-force
# enumeration -- no sampling noise on either side, so this is not the same
# kind of tolerance as this project's sampling-execution TV checks, which
# accept up to `max(5*noise_floor, 0.05)` precisely because THOSE compare an
# exact reference against a FINITE sample). 1e-3 is chosen to sit two orders
# of magnitude below that finite-sample floor while remaining comfortably
# above the TVs a genuinely broken chain produces on this instance (0.01-0.3
# at chain_strength <= 3.0, measured below) -- tight enough that passing it
# means the chain actually held, loose enough not to chase floating-point
# noise.
AGREEMENT_TV = 1e-3


def _marginal_tv_after_split(im, leaves, max_degree, chain_strength):
    """Split `im`'s hub, marginalise the hub's chain copies back out (using
    chain-copy-0 as the logical spin's own representative value -- NOT a
    majority vote, which could paper over exactly the disagreement this
    check exists to catch), and return the total-variation distance between
    that marginal and the ORIGINAL model's own exact joint over the same
    spins. Both distributions come from full brute-force enumeration via
    `audit/oracles/exact.py`, independent of `src/tsu`."""
    from tsu.passes.split import split_high_degree

    edges = im.edges
    J_before = {edges[i]: float(im.weights[i]) for i in range(len(edges))}
    states_before, probs_before = exact_boltzmann(J_before, list(im.biases), im.beta)
    p_before = dict(zip(states_before, probs_before))

    out, rep = split_high_degree(im, max_degree=max_degree, chain_strength=chain_strength)
    idx = {n: i for i, n in enumerate(out.nodes)}
    J_after = {out.edges[i]: float(out.weights[i]) for i in range(len(out.edges))}
    states_after, probs_after = exact_boltzmann(J_after, list(out.biases), out.beta)

    hub_copy0 = idx["h__chain0"]
    leaf_idx = [idx[l] for l in leaves]
    p_after: dict[tuple[int, ...], float] = {}
    for s, p in zip(states_after, probs_after):
        key = (s[hub_copy0],) + tuple(s[i] for i in leaf_idx)
        p_after[key] = p_after.get(key, 0.0) + p

    n_original = 1 + len(leaves)
    keys = list(itertools.product((0, 1), repeat=n_original))
    tv = 0.5 * sum(abs(p_before.get(k, 0.0) - p_after.get(k, 0.0)) for k in keys)
    return tv, rep


def _hub_and_leaves_model():
    """A hub of degree 5 (5 leaves) with EVERY coupling and EVERY bias a
    distinct, nonzero value -- chosen non-degenerate on purpose: this
    project's single most repeated trap is a degenerate small instance
    (`aux_variable_probe.py`'s original probe, `boundary`'s 2-edge case)
    producing a measurement that is an artifact of accidental symmetry
    rather than a real result. A symmetric-coupling or zero-bias instance
    here could let a BROKEN chain's marginal coincidentally match the
    correct one (e.g. if every leaf pulled the hub identically, a chain that
    settles on the "wrong" copy value could still reproduce the right
    AGGREGATE statistics by symmetry) -- distinct, nonzero, mixed-sign
    values on every edge and every bias remove that possibility, so an
    agreement measured here is a genuine agreement, not a coincidence.

    max_degree=4 forces hub (degree 5) to split into k=ceil(5/(4-2))=3
    copies -- not a trivial k=1 (no split at all) or a degenerate 2-node
    chain; the same chain length (k=3) the real assignment_8x8 benchmark's
    own worst-case (degree 32) needs against Z1's own degree-16 cap
    (`ceil(32/(16-2))=3`), which is why this small instance's own
    chain_strength measurement (see the sweep test below) is not an
    arbitrary toy number. 64 states before splitting, 256 after -- both
    exhaustively enumerable by `audit/oracles/exact.py`, nowhere near
    `analyse.MAXCUT_EXACT_LIMIT`/`thrml_backend.EXACT_LIMIT`."""
    from tsu.passes.lower import IsingModel

    leaves = [f"l{i}" for i in range(5)]
    nodes = ("h",) + tuple(leaves)
    edges = tuple((0, i) for i in range(1, 6))
    J = [1.3, -0.7, 0.9, -1.1, 0.4]
    b = np.array([0.6, 0.2, -0.3, 0.5, -0.1, 0.15])
    beta = 0.7
    im = IsingModel(nodes=nodes, edges=edges, weights=np.array(J),
                    biases=b, beta=beta, offset=0.0)
    return im, leaves


def test_split_reduces_degree_below_the_cap():
    """A star graph: one hub of degree 6 and six leaves. With max_degree=4 the
    hub must be split, and every node in the result must be within the cap."""
    from tsu.passes.split import split_high_degree
    from tsu.passes.lower import IsingModel
    from tsu.passes.analyse import analyse
    n = 7
    edges = tuple((0, i) for i in range(1, 7))     # hub 0, leaves 1..6
    im = IsingModel(nodes=tuple(f"v{i}" for i in range(n)), edges=edges,
                    weights=np.full(len(edges), -1.0), biases=np.zeros(n),
                    beta=1.0, offset=0.0)
    assert analyse(im).max_degree == 6
    out, rep = split_high_degree(im, max_degree=4, chain_strength=4.0)
    assert analyse(out).max_degree <= 4
    assert rep.splits == 1
    assert rep.added_nodes >= 1


def test_split_preserves_the_marginal_over_original_spins():
    """A split model must sample the SAME distribution over the original
    spins. Verified against audit/oracles/exact.py, which is written from
    the physics and does not import src/tsu -- an oracle sharing code with
    the thing under test verifies nothing. Chain strength must be high
    enough that no chain breaks; if the marginal disagrees, the chain broke
    and the model is silently wrong rather than approximately right.

    First run with chain_strength=0.1 (deliberately too weak) confirmed this
    test can actually DETECT a broken chain -- a degree assertion alone
    cannot: `split_high_degree` happily reports max_degree_after<=4
    regardless of chain_strength. That run's exact watched failure:
    `AssertionError: marginal disagreement TV=0.279932 at chain_strength=0.1
    -- the chain broke ... assert 0.27993176005225545 < 0.001`. The
    load-bearing assertion is the marginal comparison, not the degree one.

    chain_strength=6.0 here is the documented Z1 `|J| <= 6.0` cap itself
    (Extropic-sourced) -- the most any single edge, chain or otherwise, is
    allowed to cost. The sweep test below measures the LOWEST value that
    still agrees (~4.6 on this instance); this test fixes the cap itself so
    a future regression that pushed the required strength above the cap
    would show up here even if nobody re-reads the sweep's own printed
    numbers.
    """
    im, leaves = _hub_and_leaves_model()
    tv, rep = _marginal_tv_after_split(im, leaves, max_degree=4, chain_strength=6.0)
    assert rep.splits == 1
    assert tv < AGREEMENT_TV, (
        f"marginal disagreement TV={tv:.6f} at chain_strength=6.0 (the |J| "
        f"cap) -- the chain broke: the split model no longer samples the "
        f"same distribution over the original spins")


def test_split_chain_strength_sweep_finds_the_lowest_agreeing_value():
    """Task 2 Step 7 -- the real cost of Route A. Sweep chain_strength on the
    SAME fixed instance `test_split_preserves_the_marginal_over_original_spins`
    uses, and find the lowest value (grid resolution 0.1) at which the
    marginal still agrees (TV < AGREEMENT_TV). Measured, not estimated: every
    point on the grid is a fresh call to `_marginal_tv_after_split`, i.e. a
    fresh exact enumeration on both sides.

    Reported against the documented `|J| <= 6.0` Z1 cap (Extropic-sourced,
    the Thermalizers cap sweep's own "6 (Z1)" axis annotation) -- the chain
    coupling IS a |J| edge weight like any other, so whatever this sweep
    finds is real |J| budget spent, not a separate currency."""
    im, leaves = _hub_and_leaves_model()
    J_CAP = 6.0

    grid = [round(0.1 * i, 1) for i in range(0, 101)]   # 0.0 .. 10.0
    results = []
    lowest_passing = None
    for cs in grid:
        tv, _rep = _marginal_tv_after_split(im, leaves, max_degree=4, chain_strength=cs)
        results.append((cs, tv))
        if lowest_passing is None and tv < AGREEMENT_TV:
            lowest_passing = cs

    print("\nchain_strength sweep (hub degree=5, max_degree=4, k=3 copies):")
    for cs, tv in results:
        if cs <= 5.0 or (lowest_passing is not None and abs(cs - lowest_passing) < 0.25):
            print(f"  chain_strength={cs:5.2f}  TV={tv:.8f}")

    assert lowest_passing is not None, (
        "no chain_strength up to 10.0 preserved the marginal on this "
        "instance -- Route A would be unusable here even far past the |J| cap")
    print(f"  LOWEST chain_strength with TV < {AGREEMENT_TV}: {lowest_passing}")
    print(f"  |J| cap (Extropic-sourced, Z1): {J_CAP}")
    print(f"  lowest agreeing chain_strength / cap = {lowest_passing / J_CAP:.3f}")

    # Never weaken this assertion (Global Constraint) -- it is the actual
    # finding, not a guard rail: if this instance needed chain_strength >=
    # 6.0, that would be exactly the "Route A may be unusable" architectural
    # finding the plan asks to report plainly rather than work around.
    assert lowest_passing <= J_CAP, (
        f"chain_strength={lowest_passing} exceeds the documented |J| <= "
        f"{J_CAP} cap on this instance -- Route A's node-splitting pass "
        f"would need more coupling budget than Z1 hardware provides")
