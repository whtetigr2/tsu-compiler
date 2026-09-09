"""Task 4 (plan 2026-09-04-lattice-rule-taxonomy): does an auxiliary binary
z per neighbourhood rescue a one-sided threshold (morphology/ecological's
max(0, target-N), Task 3's DISTORTED classes)?

The probe, as specified by the plan: z*(target-N) IS pairwise (Product(a, b,
weight) with a=VarRef(z), b=the deficit LinearForm) -- no ThreeBodyError
concern there at all. The open question is whether z can be CONSTRAINED, by
more pairwise terms, to actually indicate [N < target] without requiring
another non-pairwise rule to build that constraint -- the recursion the
plan's brief names directly.

C2 (code review, 2026-09-04-lattice-rule-taxonomy): an EARLIER version of
this script used a 2-neighbour instance (N=x0+x1, target=1) as its "smallest
exact-enumeration instance". That instance is DEGENERATE: at target=1, n=2,
max(0, 1-N) is ALREADY exactly (1-x0)*(1-x1) -- a plain Product(L, L, 1.0)
with ZERO auxiliaries needed at all (confirmed below by
`_multilinear_reference_polynomial`: its exact multilinear expansion has
total degree 2, matching what a bare Product(L,L,w) can already express
directly, no z required). Two of the probe's three evidence lines came from
an instance where the very mechanism under test is unnecessary -- this is
the EXACT MIRROR IMAGE of the near-miss trap `audit/expressibility_matrix.md`
documents so carefully for `boundary` (a 2-edge instance that happens to
collapse to a pairwise form by coincidence, mistaken for a general proof of
EXPRESSIBILITY): here a degenerate small instance was mistaken for the
general case in the OTHER direction, producing a false negative (the
mechanism looks like it fails on this instance, but the instance never
exercised it) rather than boundary's false positive (the mechanism looks
like it succeeds, but only by coincidence). Same lesson, both directions:
always check whether "the smallest instance" is smallest-and-representative
or smallest-and-accidentally-special before trusting what it shows.

Fixed here by moving to n=3, target=1 -- confirmed non-degenerate below: the
exact multilinear polynomial for max(0, 1-N) at n=3 has a genuine degree-3
term (-x0*x1*x2, coefficient -1), which NO Product(L, L, w) (max degree 2)
can ever produce, auxiliary or not, without an aux actually doing work. This
is the smallest instance of this rule shape that a direct zero-auxiliary
pairwise form cannot already match by coincidence.

This script:
  0. Confirms, by computing the EXACT multilinear polynomial of
     max(0, target-N) independently for both the old (n=2) and this
     script's (n=3) instance, that this one is not the degenerate case --
     before running any probe on it (C2).
  1. Builds the auxiliary formulation on the smallest NON-DEGENERATE
     instance checkable by exact enumeration (3-neighbour neighbourhood,
     N = x0+x1+x2, target=1).
  2. Searches a grid of (weight, z-bias) pairs -- the most general PAIRWISE
     extension of the probe's own stated shape (Product(z, deficit, w) plus
     an optional Linear(z, mu) bias, the only way to touch z without
     re-introducing a non-pairwise term) -- and reports whether ANY
     reproduces max(0, target-N) exactly, by comparing GROUND STATES
     (min-over-z energy at every fixed neighbourhood configuration) against
     the intended function, computed independently of src/tsu.
  3. Gives the GENERAL structural reason no such (w, mu) can exist: z=0 is
     always an available choice and every z-containing term evaluates to 0
     when z=0, so min_z(...) <= 0 for EVERY neighbourhood configuration --
     but max(0, target-N) needs to reach values up to `target` (strictly
     positive) at maximum deficit. This generalises beyond the 2-parameter
     search: ANY finite set of additional terms that are purely
     z-multiplicative (contain z, or any auxiliary sharing this property,
     as a factor in every term) has the same all-zero escape at
     z=0(=z1=z2=...=0), so no finite tower of "one more auxiliary" rescues
     it without a term that is NOT purely auxiliary-multiplicative -- which
     is exactly the recursion the plan's brief names.
  4. Cross-checks the failure at the FULL DISTRIBUTION level (not just
     ground states) using `audit/oracles/exact.py`'s `exact_boltzmann`/
     `exact_energy` -- independent of src/tsu by construction -- comparing
     the auxiliary formulation's marginal distribution over the
     neighbourhood against the distribution induced by the honest
     DISTORTED alternative ((target-N)^2, already measured EXACT-pairwise
     in Task 3) at the same beta, via total variation distance over P(N),
     to quantify how the aux probe's failure looks in practice, not
     just at zero temperature. (Both models' (J, b) are BUILT via
     `tsu.passes.lower.lower` -- already independently validated exact in
     Task 3 by brute force -- rather than hand-derived here; an earlier
     version of this script hand-derived them and silently mis-collected a
     same-spin square as a bias term, exactly the class of bug
     `encode.py`'s own docstrings warn a second, independent computation
     can introduce. `lower()`'s OUTPUT is then cross-checked against
     `model.energy()` before being handed to the independent oracle, so
     nothing downstream of that check depends on trusting `lower()` blindly.)

Run with the project's pinned interpreter, from the repo root:
    PYTHONIOENCODING=utf-8 "C:/Users/whtet/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
        audit/aux_variable_probe.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "audit"))

from tsu.ir import Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef  # noqa: E402
from tsu.passes.lower import lower  # noqa: E402 -- to BUILD (J,b); see step4's own note
from oracles.exact import exact_boltzmann, exact_energy  # noqa: E402 -- independent oracle


TARGET = 1.0
NAMES = ["x0", "x1", "x2"]


def intended(N: float, target: float) -> float:
    """max(0, target - N) -- the rule Task 3's morphology/ecological rows
    found DISTORTED by the nearest pairwise form. Computed independently,
    not by calling anything under src/tsu."""
    return max(0.0, target - N)


def _multilinear_reference_polynomial(n: int, target: float):
    """The EXACT multilinear (Mobius/Lagrange) polynomial for
    max(0, target-N) over n binary neighbours, independent of src/tsu and of
    `build_aux_model` -- one indicator monomial per state, weighted by that
    state's own `intended` value. Used only to VERIFY, before probing
    anything, that a given (n, target) instance is not degenerate: a bare
    Product(L, L, w) (or any zero-auxiliary pairwise form) can express this
    function directly iff its multilinear polynomial has total degree <= 2.
    C2 (code review): the ORIGINAL probe's (n=2, target=1) instance has
    degree exactly 2 here -- already expressible with zero auxiliaries, so
    the probe never exercised the mechanism under test. (n=3, target=1)
    has a genuine degree-3 term, ruling that out."""
    import sympy as sp
    xs = sp.symbols(f"x0:{n}")
    expr = sp.Integer(0)
    for combo in itertools.product((0, 1), repeat=n):
        val = intended(sum(combo), target)
        if val == 0:
            continue
        term = sp.Integer(1)
        for xi, ci in zip(xs, combo):
            term *= xi if ci == 1 else (1 - xi)
        expr += val * term
    expr = sp.expand(expr)
    poly = sp.Poly(expr, *xs) if expr != 0 else None
    max_degree = max((sum(m) for m in poly.monoms()), default=0) if poly else 0
    return expr, max_degree


def build_aux_model(weight: float, mu: float) -> EnergyModel:
    """The probe exactly as specified: Product(z, target-N, weight), plus
    the only OTHER way to touch z without introducing a non-pairwise term --
    a plain Linear bias on z itself (Linear(z, mu)). Any additional term
    that constrains z against N MORE tightly than this would have to
    reference [N < target] directly, which is the one-sided predicate under
    test -- using it here would be circular, not a rescue."""
    z = Var("z", Binary())
    xs = [Var(n, Binary()) for n in NAMES]
    deficit_form = LinearForm({VarRef(n): -1.0 for n in NAMES}, const=TARGET)  # target - N
    z_form = LinearForm({VarRef("z"): 1.0})
    terms = [Product(z_form, deficit_form, weight)]
    if mu != 0.0:
        terms.append(Linear(z_form, mu))
    return EnergyModel(tuple(xs) + (z,), tuple(terms), 1.0)


def min_over_z(model: EnergyModel, xvals: tuple[int, ...]) -> tuple[float, int]:
    """The T->0 ground-state value at fixed neighbourhood values: the energy
    the sampler settles into once z has also been optimised, which is
    exactly what Step 1 asks to compare against `intended`."""
    candidates = []
    for z in (0, 1):
        asg = {n: v for n, v in zip(NAMES, xvals)}
        asg["z"] = z
        candidates.append((model.energy(asg), z))
    return min(candidates)


def step0_confirm_instance_is_not_degenerate():
    print("=== Step 0: confirm this instance is NOT the degenerate one (C2) ===")
    for n, target, label in ((2, 1.0, "the OLD (degenerate) instance"),
                              (len(NAMES), TARGET, "THIS script's instance")):
        expr, deg = _multilinear_reference_polynomial(n, target)
        zero_aux_expressible = deg <= 2
        print(f"    n={n}, target={target} ({label}): exact multilinear poly = {expr}")
        print(f"      total degree = {deg}  ->  expressible with ZERO auxiliaries "
              f"as a bare pairwise form: {zero_aux_expressible}")
    print("    This script uses n=3, target=1: degree 3, genuinely NOT "
          "zero-aux-expressible -- the auxiliary mechanism under test is "
          "actually needed here, unlike the old n=2 instance.")


def step1_ground_state_comparison():
    n = len(NAMES)
    print("\n=== Step 1: ground-state comparison, smallest NON-DEGENERATE "
          "exact-enumeration instance ===")
    print(f"    N = {' + '.join(NAMES)}, target = {TARGET}  (neighbourhood of {n})")
    print("    intended max(0, target-N):",
          {k: intended(k, TARGET) for k in range(n + 1)})

    search_values = [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4]
    exact_match = None
    for w in search_values:
        for mu in search_values + [0.0]:
            model = build_aux_model(w, mu)
            ok = True
            for xvals in itertools.product((0, 1), repeat=n):
                N = sum(xvals)
                want = intended(N, TARGET)
                got_e, _ = min_over_z(model, xvals)
                if abs(got_e - want) > 1e-9:
                    ok = False
                    break
            if ok:
                exact_match = (w, mu)
                break
        if exact_match:
            break

    print(f"    grid searched: weight in {search_values}, bias(mu) in "
          f"{search_values + [0.0]} ({len(search_values) * (len(search_values)+1)} pairs)")
    if exact_match:
        print(f"    EXACT MATCH FOUND at weight={exact_match[0]}, mu={exact_match[1]} "
              f"(unexpected -- see Step 3 for why this should be impossible)")
    else:
        print("    NO (weight, mu) pair reproduces max(0, target-N) exactly.")
        model = build_aux_model(1.0, 0.0)
        print("    representative case, weight=1.0, mu=0.0:")
        for xvals in itertools.product((0, 1), repeat=n):
            N = sum(xvals)
            want = intended(N, TARGET)
            got_e, got_z = min_over_z(model, xvals)
            xdesc = " ".join(f"{name}={v}" for name, v in zip(NAMES, xvals))
            print(f"      {xdesc} N={N}  want={want}  got={got_e}  (z*={got_z})"
                  f"{'  MATCH' if abs(got_e-want)<1e-9 else '  DISAGREE'}")
    return exact_match


def step3_structural_argument():
    print("\n=== Step 3: why this fails, structurally (not just on this grid) ===")
    print("""    Every term in build_aux_model contains z as a factor:
        Product(z, target-N, weight).evaluate(z=0, ...) == 0
        Linear(z, mu).evaluate(z=0, ...)                == 0
    so setting z=0 makes EVERY auxiliary term vanish, for ANY (x0, x1) and
    ANY (weight, mu). z=0 is always a legal choice, so:

        min_over_z(...) <= energy(z=0) == 0        for every configuration.

    But intended = max(0, target-N) needs to reach values up to `target`
    (here, 1.0) -- strictly POSITIVE -- at maximum deficit (N=0). A
    quantity that is always <= 0 cannot equal a quantity that is sometimes
    > 0. No choice of weight or mu escapes this: it is not a search-
    resolution artefact, it is a structural property of "every term the
    probe is allowed to add contains z as a factor."

    This generalises past a single z: replace z with any FINITE set of
    auxiliaries z_1..z_k, each entering ONLY as a factor in a product term
    (the probe's own stated shape, extended to k auxiliaries) -- the
    all-zero point z_1=...=z_k=0 still makes every such term vanish, so the
    same min_over_z <= 0 argument applies verbatim. Escaping it requires a
    term that is NOT purely auxiliary-multiplicative -- i.e. a term that
    fires based on [N < target] WITHOUT going through an auxiliary that can
    be switched off for free. Building that term correctly is exactly the
    one-sided-threshold problem this probe set out to solve: the recursion
    the plan's brief names is real, not a manner of speaking.""")


def _to_oracle_JB(ising):
    """IsingModel -> (J, b) in the oracle's own indexing, plus its offset.
    `lower()` is the module ALREADY independently validated exact in Task 3
    (measure_expressibility.py's hydrological/climate/gameplay rows, each
    confirmed against a from-scratch brute-force truth table) -- reusing it
    here to BUILD the two models' (J, b) is not a re-verification of
    `lower()` itself, only a convenient, already-trusted way to get correct
    coefficients (hand-deriving them, as an earlier version of this script
    did, is exactly the "second, weaker computation" class of bug this
    project's own encode.py docstrings warn against -- it silently mis-
    collected a same-spin square as a bias term; see git history). The
    INDEPENDENCE requirement (spec Task A4: "do NOT import src/tsu") binds
    `audit/oracles/exact.py` itself, which this function feeds but never
    modifies or reimplements."""
    J = {ising.edges[i]: float(ising.weights[i]) for i in range(len(ising.edges))}
    b = [float(x) for x in ising.biases]
    return J, b, float(ising.offset)


def step4_oracle_distribution_check():
    n = len(NAMES)
    print("\n=== Step 4: distribution-level check against the independent oracle ===")
    print("    (audit/oracles/exact.py's exact_boltzmann/exact_energy -- does not import src/tsu)")
    weight, mu, beta = 1.0, 0.0, 1.0
    model = build_aux_model(weight, mu)
    ising = lower(model)
    J, b, offset = _to_oracle_JB(ising)

    # sanity: (J, b, offset) must reproduce model.energy() exactly before we
    # trust anything the independent oracle computes from them.
    ok = True
    for combo in itertools.product((0, 1), repeat=n + 1):
        asg = {name: v for name, v in zip(NAMES, combo)}
        asg["z"] = combo[n]
        e_model = model.energy(asg)
        e_oracle = exact_energy(list(combo), J, b) + offset
        ok &= abs(e_model - e_oracle) < 1e-9
    print(f"    lower()-derived (J,b,offset) matches model.energy() over all "
          f"{2 ** (n + 1)} states: {ok}")
    assert ok, "cross-check failed -- do not trust the distribution below"

    states, probs = exact_boltzmann(J, b, beta)
    # marginal over the neighbourhood: sum out z. `ising.nodes` fixes the
    # column order.
    xi = {name: i for i, name in enumerate(ising.nodes)}
    marg = {}
    for s, p in zip(states, probs):
        key = tuple(s[xi[name]] for name in NAMES)
        marg[key] = marg.get(key, 0.0) + p
    print(f"    aux-model marginal P({','.join(NAMES)}) at beta={beta}: "
          f"{ {k: round(v,4) for k,v in sorted(marg.items())} }")

    # reference: the honestly-DISTORTED squared form (target-N)^2 -- Task 3's
    # measured EXACT-pairwise alternative, built and lowered the same way.
    xs2 = [Var(name, Binary()) for name in NAMES]
    L2 = LinearForm({VarRef(name): -1.0 for name in NAMES}, const=TARGET)
    model2 = EnergyModel(tuple(xs2), (Product(L2, L2, 1.0),), 1.0)
    ising2 = lower(model2)
    J2, b2, offset2 = _to_oracle_JB(ising2)
    ok2 = True
    for combo in itertools.product((0, 1), repeat=n):
        asg = {name: v for name, v in zip(NAMES, combo)}
        ok2 &= abs(model2.energy(asg) - (exact_energy(list(combo), J2, b2) + offset2)) < 1e-9
    print(f"    lower()-derived reference (J,b,offset) matches model2.energy(): {ok2}")
    assert ok2

    states2, probs2 = exact_boltzmann(J2, b2, beta)
    xi2 = {name: i for i, name in enumerate(ising2.nodes)}
    marg2 = {tuple(s[xi2[name]] for name in NAMES): p for s, p in zip(states2, probs2)}
    print(f"    (target-N)^2 reference P({','.join(NAMES)}) at beta={beta}: "
          f"{ {k: round(v,4) for k,v in sorted(marg2.items())} }")

    # Total variation distance between the two, against N (the only thing
    # either model can possibly be tracking): a cleaner, size-independent
    # summary than eyeballing a growing joint table.
    def marg_by_N(m):
        out = {}
        for xvals, p in m.items():
            out[sum(xvals)] = out.get(sum(xvals), 0.0) + p
        return out

    aux_by_N, ref_by_N = marg_by_N(marg), marg_by_N(marg2)
    tv = 0.5 * sum(abs(aux_by_N.get(k, 0.0) - ref_by_N.get(k, 0.0)) for k in range(n + 1))
    print(f"    P(N) from aux-model:        {[round(aux_by_N.get(k,0.0),4) for k in range(n+1)]}")
    print(f"    P(N) from (target-N)^2 ref: {[round(ref_by_N.get(k,0.0),4) for k in range(n+1)]}")
    print(f"    total variation distance between the two P(N) distributions "
          f"at beta={beta}: {tv:.4f}")
    print("    Both are WRONG relative to the intended one-sided rule (neither "
          "is claimed correct here) -- this quantifies how differently they "
          "are wrong, not which one is right. Reported as a plain fact "
          "about this instance, not a ranking claim; see "
          "expressibility_matrix.md for how this relates to the SEPARATE "
          "assignment-gadget construction (C1) that DOES reproduce the "
          "intended rule (up to the additive constant every energy model "
          "carries), unlike either distribution measured here.")


def main():
    step0_confirm_instance_is_not_degenerate()
    step1_ground_state_comparison()
    step3_structural_argument()
    step4_oracle_distribution_check()
    print("\n=== VERDICT ===")
    print("    The single/finite-auxiliary z*(target-N) probe FAILS: it")
    print("    cannot reproduce max(0, target-N), at any weight/bias, ground")
    print("    states or full distribution. Confirmed by exact enumeration")
    print("    against an independent oracle, and by a general structural")
    print("    argument that holds for any finite tower of purely")
    print("    auxiliary-multiplicative terms. This is the negative result")
    print("    the plan's brief calls the most valuable possible finding.")


if __name__ == "__main__":
    main()
