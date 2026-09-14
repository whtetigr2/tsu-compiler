"""R7 -- representation correctness (plan Wave 2).

Three independent checks, none of them trusting the existing test suite's
own verdict (plan: "verify it yourself against the oracle rather than
trusting the existing test"):

  (1) Domain-wall encode/decode round trip over the FULL codeword space
      (every value 0..k-1, for several k) -- not a sample.
  (2) `is_codeword` over the FULL bit-pattern space 2**(k-1) for several k
      -- checked against a hand-written, independent definition of
      "codeword" (monotone non-increasing chain), not against the
      compiler's own internal logic restated.
  (3) Mediator insertion preserves the marginal: a frustrated (odd-cycle)
      3-spin triangle, mediated via `tsu_compiler.passes.route.insert_mediators`,
      its EXACT distribution computed via `audit.oracles.exact.
      exact_boltzmann` (independent of src/tsu_compiler), the mediator spin summed
      out, and the result compared numerically against the oracle's own
      exact distribution of the UNMEDIATED 3-spin model. Two frustration
      signs (J<0 valley used by route.py's own docstring worked example,
      and J>0) and two beta values are tried.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "audit"))

import numpy as np  # noqa: E402

from tsu_compiler.passes.encode import encode, _chain_names_domain_wall  # noqa: E402
from tsu_compiler.spec import WorkloadSpec, TaskContract, Var as SpecVar, Categorical  # noqa: E402
from tsu_compiler.passes.lower import IsingModel  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.route import insert_mediators, assert_beta_consistent, BetaMismatchError  # noqa: E402

from oracles.exact import exact_boltzmann  # noqa: E402

print("=" * 78)
print("(1)+(2) DOMAIN-WALL FULL CODEWORD SPACE -- encode/decode round trip")
print("        and is_codeword, k = 2..6, EVERY bit pattern (not a sample)")
print("=" * 78)


def independent_is_codeword_domain_wall(bits: tuple[int, ...]) -> bool:
    """Hand-written, independent of encode.py's own `is_codeword`: a
    domain-wall chain is a codeword iff it is monotone NON-INCREASING
    (once a 0 appears, every subsequent bit is 0 too) -- the textbook
    Chancellor (2019) domain-wall definition, not a restatement of the
    compiler's own `n_{j-1} >= n_j` scan."""
    return all(bits[i] >= bits[i + 1] for i in range(len(bits) - 1))


all_ok = True
for k in range(2, 7):
    spec = WorkloadSpec(
        name="r7_probe", variables=(SpecVar("c", Categorical(k)),), terms=(),
        contract=TaskContract(rules=()))
    enc = encode(spec, encoding="domain_wall")
    chain = enc.categorical["c"]
    assert len(chain) == k - 1, (chain, k)

    n_patterns = 2 ** (k - 1)
    codeword_count = 0
    roundtrip_failures = []
    codeword_misclassifications = []

    for pattern in itertools.product((0, 1), repeat=k - 1):
        bits = dict(zip(chain, pattern))
        expected_codeword = independent_is_codeword_domain_wall(pattern)
        actual_codeword = enc.is_codeword(bits)
        if expected_codeword != actual_codeword:
            codeword_misclassifications.append((pattern, expected_codeword, actual_codeword))
        if expected_codeword:
            codeword_count += 1
            decoded = enc.decode(bits)["c"]
            # independent expected value: number of leading 1s
            expected_v = 0
            for b in pattern:
                if b == 1:
                    expected_v += 1
                else:
                    break
            if decoded != expected_v:
                roundtrip_failures.append((pattern, expected_v, decoded))

    # encode_assignment(v) -> bits -> decode -> v, for EVERY value 0..k-1
    encode_roundtrip_failures = []
    for v in range(k):
        bits = enc.encode_assignment({"c": v})
        if not enc.is_codeword(bits):
            encode_roundtrip_failures.append((v, bits, "encode_assignment produced a non-codeword"))
            continue
        decoded_v = enc.decode(bits)["c"]
        if decoded_v != v:
            encode_roundtrip_failures.append((v, bits, decoded_v))

    ok = (not codeword_misclassifications and not roundtrip_failures
          and not encode_roundtrip_failures)
    all_ok = all_ok and ok
    print(f"k={k}: chain_len={k-1} patterns={n_patterns} codewords_found={codeword_count} "
          f"(expected {k}) is_codeword_misclassifications={len(codeword_misclassifications)} "
          f"decode_roundtrip_failures={len(roundtrip_failures)} "
          f"encode_assignment_roundtrip_failures={len(encode_roundtrip_failures)} "
          f"-> {'OK' if ok else 'FAIL'}")
    if codeword_misclassifications:
        print(f"   MISCLASSIFIED: {codeword_misclassifications[:5]}")
    if roundtrip_failures:
        print(f"   ROUNDTRIP FAILURES: {roundtrip_failures[:5]}")
    if encode_roundtrip_failures:
        print(f"   ENCODE ROUNDTRIP FAILURES: {encode_roundtrip_failures[:5]}")

print(f"\nFull codeword-space check across k=2..6: {'ALL OK' if all_ok else 'FAILURES FOUND'}")


print("\n" + "=" * 78)
print("(3) MEDIATOR INSERTION PRESERVES THE MARGINAL -- verified against the")
print("    independent oracle, not the existing test suite")
print("=" * 78)


def build_triangle(J: float, beta: float, biases=(0.0, 0.0, 0.0)) -> IsingModel:
    return IsingModel(
        nodes=("s0", "s1", "s2"),
        edges=((0, 1), (1, 2), (0, 2)),
        weights=np.array([J, J, J], dtype=float),
        biases=np.array(biases, dtype=float),
        beta=beta,
        offset=0.0,
    )


def marginalize_mediator_out(states, probs, n_original: int):
    """Sum probability mass over every mediator-spin value for each
    distinct pattern of the first `n_original` columns. Returns
    (original_states, marginal_probs) in the SAME enumeration order as a
    fresh itertools.product(range(2), repeat=n_original) so the caller can
    zip it directly against the unmediated oracle's own state order."""
    agg: dict[tuple[int, ...], float] = {}
    for s, p in zip(states, probs):
        key = tuple(int(x) for x in s[:n_original])
        agg[key] = agg.get(key, 0.0) + p
    ordered_states = list(itertools.product((0, 1), repeat=n_original))
    ordered_probs = [agg.get(s, 0.0) for s in ordered_states]
    return ordered_states, ordered_probs


cases = [
    ("J=-1.25 (route.py's own worked example, antiferromagnetic)", -1.25, 4.0),
    ("J=-1.25, beta=1.0", -1.25, 1.0),
    ("J=+0.8 (ferromagnetic)", 0.8, 2.0),
    ("J=+2.4 (large coupling)", 2.4, 0.5),
]

worst_overall = 0.0
for label, J, beta in cases:
    ising = build_triangle(J, beta)
    report = analyse(ising)
    med, mreport = insert_mediators(ising, report)
    n_mediators = mreport.mediator_count

    # Oracle: exact distribution of the ORIGINAL (unmediated) 3-spin model.
    J_dict = {(0, 1): J, (1, 2): J, (0, 2): J}
    b_list = [0.0, 0.0, 0.0]
    orig_states, orig_probs = exact_boltzmann(J_dict, b_list, beta)

    # Oracle: exact distribution of the MEDIATED model (n=3+n_mediators spins).
    med_J = {(u, v): float(w) for (u, v), w in zip(med.edges, med.weights)}
    med_b = [float(x) for x in med.biases]
    med_states, med_probs = exact_boltzmann(med_J, med_b, beta)

    marg_states, marg_probs = marginalize_mediator_out(med_states, med_probs, 3)

    # orig_states from exact_boltzmann is itertools.product((0,1), repeat=3)
    # in the SAME order as marg_states by construction -- verify that
    # assumption explicitly rather than silently relying on it.
    assert list(orig_states) == marg_states, "state ordering mismatch -- oracle enumeration order changed"

    max_dev = max(abs(a - b) for a, b in zip(orig_probs, marg_probs))
    worst_overall = max(worst_overall, max_dev)
    print(f"\n{label}: beta={beta}  bipartite_after={mreport.bipartite_after}  "
          f"mediators_inserted={n_mediators}  partition_method={mreport.partition_method}")
    print(f"  {'state':>10} | {'P(unmediated, oracle)':>22} | {'P(mediated, marginalised, oracle)':>34} | diff")
    for s, p_orig, p_marg in zip(orig_states, orig_probs, marg_probs):
        print(f"  {str(s):>10} | {p_orig:22.14f} | {p_marg:34.14f} | {abs(p_orig-p_marg):.3e}")
    print(f"  max|P_unmediated - P_mediated_marginalised| = {max_dev:.3e}")

    # beta-mismatch refusal, cross-checked here since it's directly implied
    # by this same mediated model (R14 territory too, checked once here
    # since the model is already built).
    try:
        assert_beta_consistent(med, beta + 0.37)
        print("  BetaMismatchError: NOT RAISED at a different beta -- FINDING")
    except BetaMismatchError as exc:
        print(f"  BetaMismatchError correctly raised at a different beta: {exc}"[:140] + "...")
    try:
        assert_beta_consistent(med, beta)
        print("  Sampling at the SAME beta the mediators were computed at: correctly allowed (no raise)")
    except BetaMismatchError:
        print("  BetaMismatchError incorrectly raised at the SAME beta -- FINDING")

print(f"\nWorst marginal-preservation deviation across all {len(cases)} cases: {worst_overall:.3e}")

print("\n" + "=" * 78)
print("(3b) MULTI-MEDIATOR CASE: K4 (4-spin complete graph), which needs")
print("     MORE THAN ONE mediated edge -- closes the single-mediator gap")
print("     the triangle cases above cannot exercise")
print("=" * 78)


def build_k4(J: float, beta: float) -> IsingModel:
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return IsingModel(
        nodes=("s0", "s1", "s2", "s3"),
        edges=tuple(edges),
        weights=np.array([J] * len(edges), dtype=float),
        biases=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        beta=beta, offset=0.0)


for label, J, beta in [("K4, J=-0.9", -0.9, 1.5), ("K4, J=+1.1", 1.1, 1.0)]:
    ising = build_k4(J, beta)
    report = analyse(ising)
    med, mreport = insert_mediators(ising, report)
    print(f"\n{label}: mediators_inserted={mreport.mediator_count} "
          f"bipartite_after={mreport.bipartite_after}")

    J_dict = {e: J for e in ising.edges}
    b_list = [0.0, 0.0, 0.0, 0.0]
    orig_states, orig_probs = exact_boltzmann(J_dict, b_list, beta)

    med_J = {(u, v): float(w) for (u, v), w in zip(med.edges, med.weights)}
    med_b = [float(x) for x in med.biases]
    med_states, med_probs = exact_boltzmann(med_J, med_b, beta)

    marg_states, marg_probs = marginalize_mediator_out(med_states, med_probs, 4)
    assert list(orig_states) == marg_states
    max_dev = max(abs(a - b) for a, b in zip(orig_probs, marg_probs))
    worst_overall = max(worst_overall, max_dev)
    print(f"  max|P(unmediated) - P(mediated, {mreport.mediator_count} mediators marginalised)| = {max_dev:.3e}")

print(f"\nWorst marginal-preservation deviation INCLUDING multi-mediator K4 cases: {worst_overall:.3e}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  Full codeword-space (encode/decode + is_codeword), k=2..6: {'ALL OK' if all_ok else 'FAILURES FOUND'}")
print(f"  Mediator marginal preservation, worst case over {len(cases)} configurations: {worst_overall:.3e}")
