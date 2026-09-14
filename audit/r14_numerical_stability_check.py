"""R14 -- numerical stability, adversarially (plan Wave 2).

Every check below hits a genuinely PUBLIC entry point (importable and
callable from outside its own module -- not a leading-underscore private
helper) directly, bypassing the full spec/encode/compile pipeline on
purpose: `validate_coefficient_scale` (called by `encode()`) already
guards `coefficient_scale` (hence, transitively, the ONLY beta value the
real `compile_spec()` pipeline can ever produce: `beta = 1.0 /
coefficient_scale`, always finite and > 0 -- see search.py:161,391) --
but `lower()`, `tsu_compiler.passes.route.insert_mediators`,
`tsu_compiler.backends.thrml_backend.*`, `tsu_compiler.ess.*`, `demo.lattice_app.
energy_of_draw` and `audit.oracles.exact.*` are ALL independently
importable and callable with values the front door's own guard never
reaches (a hand-edited receipt JSON, a future caller, a test, a REPL) --
so each is tested here in isolation, adversarially, exactly as the plan's
own wording asks ("at every public entry point").

Every result below is classified into exactly one of three buckets the
plan's own wording distinguishes: CRASH (an exception propagates out),
SILENT WRONG ANSWER (returns without complaint, but the number is
NaN/inf/nonsensical), or HONEST UNAVAILABLE (returns a labelled
`unavailable: ...`/raises a clearly-named guard exception BEFORE producing
a number). These are not treated as equally acceptable -- see this
script's own summary at the end.
"""
from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "audit"))
sys.path.insert(0, str(REPO_ROOT / "demo"))

import numpy as np  # noqa: E402

from tsu_compiler.ir import Binary, EnergyModel, Linear, LinearForm, Product, Var, VarRef  # noqa: E402
from tsu_compiler.passes.lower import lower, IsingModel  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.program import build_program  # noqa: E402
from tsu_compiler.passes.route import insert_mediators, assert_beta_consistent, BetaMismatchError  # noqa: E402
from tsu_compiler.backends.thrml_backend import (  # noqa: E402
    exact_distribution, exact_conditional_distribution, sample_chains, sample)
from tsu_compiler import ess  # noqa: E402

from oracles.exact import exact_boltzmann, exact_energy  # noqa: E402

try:
    import lattice_app
    energy_of_draw = lattice_app.energy_of_draw
except Exception as exc:  # pragma: no cover
    energy_of_draw = None
    print(f"NOTE: demo.lattice_app import failed ({exc!r}); energy_of_draw checks skipped.")

RESULTS = []  # (label, bucket, detail)
CRASH, SILENT_WRONG, HONEST_UNAVAILABLE, OK = "CRASH", "SILENT-WRONG-ANSWER", "HONEST-UNAVAILABLE", "OK-DEFENSIBLE"


def record(label, bucket, detail=""):
    RESULTS.append((label, bucket, detail))
    print(f"[{bucket:22}] {label}: {detail}")


def simple_ising(J, b, beta, offset=0.0):
    n = len(b)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)] if isinstance(J, (int, float)) else list(J.keys())
    if isinstance(J, (int, float)):
        weights = [float(J)] * len(edges)
    else:
        weights = [float(v) for v in J.values()]
    return IsingModel(
        nodes=tuple(f"s{i}" for i in range(n)), edges=tuple(edges),
        weights=np.array(weights, dtype=float), biases=np.array(b, dtype=float),
        beta=beta, offset=offset)


print("=" * 78)
print("1. BETA NEAR ZERO AND VERY LARGE")
print("=" * 78)

# 1a. beta near zero (but not exactly zero) -- should approach the uniform
#     distribution smoothly. thrml_backend.exact_distribution and
#     oracle.exact_boltzmann are tried SEPARATELY (not in one combined
#     try/except) specifically so a crash in one is never misattributed to
#     the other -- src/tsu_compiler's own path (jax.nn.softmax, internally
#     max-shifted) and the audit's own oracle (bare math.exp, NOT
#     max-shifted -- see oracles/exact.py) have materially different
#     numerical behaviour at extreme beta, and conflating them would be
#     exactly the kind of misattributed finding this audit exists to avoid.
for beta in (1e-12, 1e-6, 1e6, 1e12, 1e300):
    im = simple_ising(J={(0, 1): 1.3}, b=[0.4, -0.7], beta=beta)
    thrml_label = f"src/tsu_compiler exact_distribution (thrml/jax.nn.softmax path), beta={beta:g}"
    oracle_label = f"AUDIT ORACLE exact_boltzmann (bare math.exp, no shift), beta={beta:g}"
    thrml_probs = None
    try:
        report = analyse(im)
        prog = build_program(im, report)
        thrml_states, thrml_probs = exact_distribution(prog)
        finite = all(math.isfinite(p) for p in thrml_probs) and math.isclose(sum(thrml_probs), 1.0, abs_tol=1e-6)
        record(thrml_label, OK if finite else SILENT_WRONG,
               f"sum(probs)={sum(thrml_probs):.6f}" if finite else f"probs not finite/normalised: {thrml_probs}")
    except OverflowError as exc:
        record(thrml_label, CRASH, f"OverflowError: {exc}")
    except Exception as exc:
        record(thrml_label, CRASH, f"{type(exc).__name__}: {exc}")

    try:
        o_states, o_probs = exact_boltzmann({(0, 1): 1.3}, [0.4, -0.7], beta)
        if thrml_probs is not None:
            idx = {tuple(int(x) for x in s): i for i, s in enumerate(thrml_states)}
            max_dev = max(abs(thrml_probs[idx[s]] - p) for s, p in zip(o_states, o_probs))
            record(oracle_label, OK, f"sum(probs)={sum(o_probs):.6f}, max_dev vs src/tsu_compiler = {max_dev:.3e}")
        else:
            record(oracle_label, OK, f"sum(probs)={sum(o_probs):.6f} (src/tsu_compiler path crashed above, no cross-check possible)")
    except OverflowError as exc:
        record(oracle_label, CRASH, f"OverflowError IN THE AUDIT'S OWN ORACLE (not src/tsu_compiler): {exc}")
    except Exception as exc:
        record(oracle_label, CRASH, f"{type(exc).__name__}: {exc}")

# 1c. beta = 0 exactly: uniform distribution over ALL 2**n states,
#     independent of J/b entirely (p ~ exp(0) = 1 for every state).
im0 = simple_ising(J={(0, 1): 3.0}, b=[5.0, -5.0], beta=0.0)
try:
    report = analyse(im0)
    prog = build_program(im0, report)
    states, probs = exact_distribution(prog)
    uniform = 1.0 / len(states)
    max_dev = max(abs(p - uniform) for p in probs)
    if max_dev < 1e-9:
        record("exact_distribution, beta=0.0 EXACTLY", OK,
               f"uniform to {max_dev:.2e} as required by p~exp(0)=1 for all states")
    else:
        record("exact_distribution, beta=0.0 EXACTLY", SILENT_WRONG,
               f"NOT uniform (max_dev={max_dev:.3e}) -- beta=0 should erase all J/b influence")
except Exception as exc:
    record("exact_distribution, beta=0.0 EXACTLY", CRASH, f"{type(exc).__name__}: {exc}")

# beta_to_temperature's own explicit guard (scope.py) -- confirm it refuses
# beta<=0 rather than returning inf/nan silently.
import scope as demo_scope  # noqa: E402
for beta in (0.0, -1.0, -1e300):
    try:
        t = demo_scope.beta_to_temperature(beta)
        record(f"beta_to_temperature(beta={beta})", SILENT_WRONG, f"returned {t} instead of refusing")
    except ValueError as exc:
        record(f"beta_to_temperature(beta={beta})", HONEST_UNAVAILABLE, f"correctly refused: {exc}")

print("\n" + "=" * 78)
print("2. J = 0 (both a zero-WEIGHT edge, and the fully edgeless model)")
print("=" * 78)

# 2a. edge present but weight exactly 0.0 -- goes through the GENERAL
#     (edged) thrml path, not the _edgeless_* bypass.
im_zero_weight = simple_ising(J={(0, 1): 0.0}, b=[0.5, -0.3], beta=1.0)
try:
    report = analyse(im_zero_weight)
    prog = build_program(im_zero_weight, report)
    states, probs = exact_distribution(prog)
    o_states, o_probs = exact_boltzmann({(0, 1): 0.0}, [0.5, -0.3], 1.0)
    idx = {tuple(int(x) for x in s): i for i, s in enumerate(states)}
    max_dev = max(abs(probs[idx[s]] - p) for s, p in zip(o_states, o_probs))
    record("edge present, J=0.0 exactly (general/edged thrml path)",
           OK if max_dev < 1e-9 else SILENT_WRONG, f"max_dev vs oracle = {max_dev:.3e}")
except Exception as exc:
    record("edge present, J=0.0 exactly (general/edged thrml path)", CRASH, f"{type(exc).__name__}: {exc}")

# 2b. truly edgeless model (no edges at all) -- the documented bypass of
#     thrml's own IndexError on an empty edge block (thrml_backend.py's own
#     module docstring, C3).
im_edgeless = IsingModel(nodes=("s0", "s1", "s2"), edges=(), weights=np.array([]),
                         biases=np.array([0.3, -0.9, 1.5]), beta=1.0, offset=0.0)
try:
    report = analyse(im_edgeless)
    prog = build_program(im_edgeless, report)
    states, probs = exact_distribution(prog)
    o_states, o_probs = exact_boltzmann({}, [0.3, -0.9, 1.5], 1.0)
    idx = {tuple(int(x) for x in s): i for i, s in enumerate(states)}
    max_dev = max(abs(probs[idx[s]] - p) for s, p in zip(o_states, o_probs))
    record("edgeless model (0 edges, the documented _edgeless_* bypass)",
           OK if max_dev < 1e-9 else SILENT_WRONG, f"max_dev vs oracle = {max_dev:.3e}")
    # exercise sample_chains on the same edgeless model too (a SEPARATE
    # code path per the module's own docstring, C1).
    chains = sample_chains(prog, n_chains=4, n_samples=50, n_warmup=1, steps_per_sample=1, seed=0)
    record("edgeless model sample_chains (separate code path)", OK,
           f"shape={chains.shape}, dtype={chains.dtype}, no crash")
except Exception as exc:
    record("edgeless model (0 edges)", CRASH, f"{type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print("3. DEGENERATE ENERGIES (every state has the SAME energy)")
print("=" * 78)

im_degenerate = IsingModel(nodes=("s0", "s1", "s2"), edges=(), weights=np.array([]),
                           biases=np.array([0.0, 0.0, 0.0]), beta=1.0, offset=0.0)
try:
    report = analyse(im_degenerate)
    prog = build_program(im_degenerate, report)
    states, probs = exact_distribution(prog)
    uniform = 1.0 / len(states)
    max_dev = max(abs(p - uniform) for p in probs)
    record("all-zero b, no edges -> every state E=0 (fully degenerate)",
           OK if max_dev < 1e-9 else SILENT_WRONG, f"uniform to {max_dev:.2e}")
except Exception as exc:
    record("degenerate energies", CRASH, f"{type(exc).__name__}: {exc}")

# ess.effective_sample_size on a genuinely CONSTANT chain (zero variance --
# the module's own documented guard).
const_chain = np.full((4, 100), 3.7)
result = ess.effective_sample_size(const_chain)
record("ess.effective_sample_size on a CONSTANT series (zero variance)",
       HONEST_UNAVAILABLE if result.ess is None else SILENT_WRONG, result.reason or f"ess={result.ess}")

print("\n" + "=" * 78)
print("4. SINGLE-STATE DISTRIBUTIONS (every node clamped)")
print("=" * 78)

im3 = simple_ising(J={(0, 1): 1.3, (1, 2): -0.7, (0, 2): 0.2}, b=[0.1, -0.2, 0.3], beta=1.0)
report3 = analyse(im3)
prog_full_clamp = build_program(im3, report3, clamp={0: 1, 1: 0, 2: 1})
try:
    states, probs = exact_conditional_distribution(prog_full_clamp)
    ok = len(states) == 1 and math.isclose(probs[0], 1.0, abs_tol=1e-12)
    record("exact_conditional_distribution, EVERY node clamped (single remaining state)",
           OK if ok else SILENT_WRONG, f"states={states.tolist()}, probs={probs}")
except Exception as exc:
    record("exact_conditional_distribution, every node clamped", CRASH, f"{type(exc).__name__}: {exc}")

# oracle.exact_boltzmann with n=1 (smallest nonzero instance -- 2 states).
try:
    states, probs = exact_boltzmann({}, [2.5], 1.0)
    record("oracle.exact_boltzmann, n=1 spin (2-state distribution)", OK, f"probs={probs}")
except Exception as exc:
    record("oracle.exact_boltzmann, n=1 spin", CRASH, f"{type(exc).__name__}: {exc}")

# oracle.exact_boltzmann with n=0 (empty valid/spin set) -- documented to raise.
try:
    exact_boltzmann({}, [], 1.0)
    record("oracle.exact_boltzmann, n=0 (b=[]) -- EMPTY set", SILENT_WRONG, "did not raise")
except ValueError as exc:
    record("oracle.exact_boltzmann, n=0 (b=[]) -- EMPTY set", HONEST_UNAVAILABLE, str(exc))

print("\n" + "=" * 78)
print("5. NaN / Inf INGRESS AT PUBLIC ENTRY POINTS")
print("=" * 78)

# 5a. lower(): a term with NaN weight -- does the algebra layer guard this?
model_nan_weight = EnergyModel(
    variables=(Var("x0", Binary()), Var("x1", Binary())),
    terms=(Linear(LinearForm({VarRef("x0"): 1.0}), float("nan")),
          Product(LinearForm({VarRef("x0"): 1.0}), LinearForm({VarRef("x1"): 1.0}), 1.0)),
    beta=1.0)
try:
    im_nan = lower(model_nan_weight)
    has_nan = bool(np.isnan(im_nan.biases).any() or np.isnan(im_nan.weights).any() or math.isnan(im_nan.offset))
    record("lower(): a Linear term with weight=NaN",
           SILENT_WRONG if has_nan else OK,
           f"biases={im_nan.biases.tolist()} weights={im_nan.weights.tolist()} offset={im_nan.offset!r} "
           f"-- {'NaN propagated into the IsingModel with NO guard/error' if has_nan else 'no NaN present'}")
except Exception as exc:
    record("lower(): a Linear term with weight=NaN", HONEST_UNAVAILABLE if "nan" in str(exc).lower() else CRASH,
           f"{type(exc).__name__}: {exc}")

# 5b. lower(): EnergyModel.beta = NaN / inf -- lower() itself doesn't use
# beta at all (only carries it through onto IsingModel.beta), so check
# whether it is validated ANYWHERE before reaching a sampler.
for bad_beta in (float("nan"), float("inf"), -float("inf")):
    model_bad_beta = EnergyModel(
        variables=(Var("x0", Binary()),), terms=(Linear(LinearForm({VarRef("x0"): 1.0}), 1.0),),
        beta=bad_beta)
    im_bad_beta = lower(model_bad_beta)
    label = f"lower(): EnergyModel.beta={bad_beta!r} passed straight through, unvalidated"
    record(label, SILENT_WRONG if not math.isfinite(im_bad_beta.beta) else OK,
           f"IsingModel.beta={im_bad_beta.beta!r} (lower() carries beta through with no check at all)")
    # does exact_distribution/thrml choke on it, or silently produce garbage?
    try:
        report_b = analyse(im_bad_beta)
        prog_b = build_program(im_bad_beta, report_b)
        states_b, probs_b = exact_distribution(prog_b)
        finite = all(math.isfinite(p) for p in probs_b)
        record(f"  -> exact_distribution() fed that beta={bad_beta!r}",
               SILENT_WRONG if not finite else OK,
               f"probs={probs_b}")
    except Exception as exc:
        record(f"  -> exact_distribution() fed that beta={bad_beta!r}", CRASH, f"{type(exc).__name__}: {exc}")

# 5c. oracle.exact_energy: an out-of-domain state entry (already documented
# to raise -- confirm it actually does, and that NaN specifically is caught
# by the same guard).
for bad_state in ((1, 2), (float("nan"), 0), (0, float("inf"))):
    try:
        exact_energy(bad_state, {(0, 1): 1.0}, [0.0, 0.0])
        record(f"oracle.exact_energy(state={bad_state!r})", SILENT_WRONG, "did not raise on an invalid state entry")
    except (ValueError, TypeError) as exc:
        record(f"oracle.exact_energy(state={bad_state!r})", HONEST_UNAVAILABLE, f"{type(exc).__name__}: {exc}")

# 5d. energy_of_draw fed a NaN-containing row (demo's own public function).
if energy_of_draw is not None:
    im_small = simple_ising(J={(0, 1): 1.0}, b=[0.5, -0.5], beta=1.0)
    try:
        e = energy_of_draw(im_small, [float("nan"), 1])
        record("demo.lattice_app.energy_of_draw([NaN, 1])",
               SILENT_WRONG if math.isnan(e) else OK, f"returned {e!r} (no guard, no exception)")
    except Exception as exc:
        record("demo.lattice_app.energy_of_draw([NaN, 1])", HONEST_UNAVAILABLE, f"{type(exc).__name__}: {exc}")

# 5e. oracle.exact_boltzmann: NaN inside b.
try:
    exact_boltzmann({}, [float("nan")], 1.0)
    record("oracle.exact_boltzmann(b=[NaN])", SILENT_WRONG, "did not raise")
except ValueError as exc:
    record("oracle.exact_boltzmann(b=[NaN])", HONEST_UNAVAILABLE, f"Z-usability guard caught it: {exc}")
except Exception as exc:
    record("oracle.exact_boltzmann(b=[NaN])", CRASH, f"{type(exc).__name__}: {exc}")

# 5f. ess module: NaN inside a chain.
chain_with_nan = np.full((4, 50), 1.0)
chain_with_nan[0, 10] = float("nan")
try:
    result = ess.effective_sample_size(chain_with_nan)
    record("ess.effective_sample_size(chain containing one NaN)",
           HONEST_UNAVAILABLE if result.ess is None else SILENT_WRONG,
           result.reason if result.ess is None else f"ess={result.ess} (NaN silently absorbed into a real-looking number!)")
except Exception as exc:
    record("ess.effective_sample_size(chain containing one NaN)", CRASH, f"{type(exc).__name__}: {exc}")

# 5g. insert_mediators: J containing NaN/inf on a non-bipartite (needs
# mediation) graph -- math.acosh(math.exp(2*beta*|J|)) with |J|=inf or nan.
# IMPORTANT: for edges=((0,1),(1,2),(0,2)) on a triangle, this module's own
# BFS-depth-parity colouring (verified by hand: colour=[0,1,1], no flips
# survive the greedy local search) mediates ONLY the (1,2) edge -- (0,1)
# and (0,2) are cross-colour and are KEPT UNTOUCHED, never reaching the
# A=acosh(exp(2*beta*|J|)) formula at all. So the bad value is placed on
# edge INDEX 1 (=(1,2), the mediated one) here, deliberately, to actually
# exercise that formula -- putting it on index 0 or 2 (as a first, less
# careful draft of this script did) would silently test nothing, because
# the bad value would just pass through a KEPT edge untouched and prove
# nothing about the mediation arithmetic itself.
for bad_J in (float("nan"), float("inf")):
    im_triangle_bad = IsingModel(
        nodes=("s0", "s1", "s2"), edges=((0, 1), (1, 2), (0, 2)),
        weights=np.array([1.0, bad_J, 1.0]), biases=np.array([0.0, 0.0, 0.0]),
        beta=1.0, offset=0.0)
    label = f"insert_mediators() on a triangle, bad J ON THE MEDIATED EDGE (1,2)={bad_J!r}"
    try:
        report_t = analyse(im_triangle_bad)
        med, mreport = insert_mediators(im_triangle_bad, report_t)
        bad = (np.isnan(med.weights).any() or np.isinf(med.weights).any()
              or not math.isfinite(med.offset))
        record(label, SILENT_WRONG if bad else OK,
               f"mediator_count={mreport.mediator_count} mediated weights={med.weights.tolist()} offset={med.offset!r}")
    except (ValueError, OverflowError) as exc:
        record(label, HONEST_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        record(label, CRASH, f"{type(exc).__name__}: {exc}")

# 5h. same bad value on a KEPT (non-mediated) edge -- (0,2), index 2 -- to
# show explicitly, side by side, that a KEPT edge's bad value is passed
# through with NO arithmetic touching it at all (neither validated nor
# transformed), which is a DIFFERENT mechanism from 5g's silent
# propagation THROUGH the mediation formula.
for bad_J in (float("nan"), float("inf")):
    im_triangle_kept = IsingModel(
        nodes=("s0", "s1", "s2"), edges=((0, 1), (1, 2), (0, 2)),
        weights=np.array([1.0, 1.0, bad_J]), biases=np.array([0.0, 0.0, 0.0]),
        beta=1.0, offset=0.0)
    label = f"insert_mediators() on a triangle, bad J ON A KEPT EDGE (0,2)={bad_J!r}"
    try:
        report_t = analyse(im_triangle_kept)
        med, mreport = insert_mediators(im_triangle_kept, report_t)
        bad = (np.isnan(med.weights).any() or np.isinf(med.weights).any())
        record(label, SILENT_WRONG if bad else OK,
               f"mediator_count={mreport.mediator_count} mediated weights={med.weights.tolist()} "
               f"(bad value passed through UNTOUCHED, no formula applied to a kept edge)")
    except Exception as exc:
        record(label, CRASH, f"{type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
from collections import Counter
counts = Counter(bucket for _, bucket, _ in RESULTS)
for bucket in (CRASH, SILENT_WRONG, HONEST_UNAVAILABLE, OK):
    print(f"  {bucket:22}: {counts.get(bucket, 0)}")
print(f"\n  Total checks: {len(RESULTS)}")
print("\n  SILENT-WRONG-ANSWER and CRASH entries (the ones a hostile audit cares about):")
for label, bucket, detail in RESULTS:
    if bucket in (SILENT_WRONG, CRASH):
        print(f"    [{bucket}] {label}: {detail}")
