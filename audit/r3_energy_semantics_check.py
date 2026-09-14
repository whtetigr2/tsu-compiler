"""R3 -- energy semantics: three-way energy agreement + sign-flip-of-J
falsification test (plan Wave 2, mandatory tests (a) and (b)).

Builds a small, DELIBERATELY ASYMMETRIC 3-spin instance directly from the
IR (`tsu_compiler.ir.EnergyModel`) and pushes it through the real compiler passes
`tsu_compiler.passes.lower.lower` and `tsu_compiler.passes.analyse.analyse` /
`tsu_compiler.passes.program.build_program` -- the fast, pure passes only. This
script never calls `tsuc compile`, `place`, `route`, or `search` (all
forbidden/expensive per the audit's global constraints); `lower` and
`analyse`/`build_program` are the passes actually under test for R3's own
question (where does E(x) come from, and where does beta enter) and are
each O(n) on an instance this small.

Asymmetry is deliberate (plan's own instruction): "flipping J's sign" and
"double sign flip cancels on a symmetric instance" are both explicitly
named risks, so this instance has THREE spins with DIFFERENT biases and
THREE edges with DIFFERENT weights -- nothing here is invariant under a
global sign flip, a spin relabelling, or an edge permutation. A bug that
double-flips (cancels) would reproduce this exact same asymmetric energy
landscape; a bug that flips zero or one time would not.

Three independent computations of the SAME E(x), per plan Wave 2 (a):
  1. `EnergyModel.energy(assignment)`      -- the IR's OWN definition,
     src/tsu_compiler/ir.py:117-118, "E(x) = sum over terms of weight * term(x)".
     This is "the compiler's own energy" -- the pre-lowering ground truth
     of what the compiler claims a workload's energy IS.
  2. `demo.lattice_app.energy_of_draw(im, row)` -- the live app's function,
     reimplemented independently of `_from_ising` per its own docstring.
  3. `audit.oracles.exact.exact_energy(state, J, b)` -- the audit's oracle,
     written from the physics, never importing `src/tsu_compiler` (see that module's
     own docstring). The oracle carries NO notion of `offset` (it was
     written from the bare two-spin Ising Hamiltonian, which has none by
     construction) -- so the three-way check compares
     `energy_of_draw(im, row) - im.offset` against the oracle's raw
     `exact_energy`, and separately reports `im.offset` on its own so the
     reader can see the constant the IR's own energy carries that the
     oracle's convention does not.

Sign-flip-of-J test (plan Wave 2 (b)): the SAME instance, every J negated
(edge weights *= -1, biases untouched), re-lowered, and the SAME three
energies recomputed on every one of the 2**3 = 8 states, plus the induced
Boltzmann distribution (via `energy_of_draw`) at beta=1. A test that
"passes" under both signs (i.e. produces the SAME distribution) is
declared void by the plan; this script computes and prints the actual
per-state divergence so that cannot happen silently.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "demo"))
sys.path.insert(0, str(REPO_ROOT / "audit"))

from tsu_compiler.ir import Binary, EnergyModel, Linear, LinearForm, Product, Var  # noqa: E402
from tsu_compiler.passes.lower import lower  # noqa: E402

from oracles.exact import exact_energy, exact_boltzmann  # noqa: E402

# demo.lattice_app has a heavy Tk import chain; import only the one pure
# function this script needs, the same way the rest of this module does
# throughout `energy_of_draw`'s own docstring (pure Python/numpy, no Tk).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_lattice_app_energy_only", REPO_ROOT / "demo" / "lattice_app.py")
# NOTE: lattice_app.py imports tkinter/PIL at module level, so a full
# exec_module would require a display. Import guard: try the real module
# first (works headlessly on this box per prior audit scripts' own
# experience -- Tk imports without a display on Windows); if that ever
# stops being true, this script's own local reimplementation (identical
# formula, `_energy_of_draw_reference` below, transcribed from
# demo/lattice_app.py:1071-1085 at audit time) is used as a fallback and
# clearly labelled as such in the report.
try:
    import lattice_app  # noqa: E402
    energy_of_draw = lattice_app.energy_of_draw
    ENERGY_OF_DRAW_SOURCE = "demo.lattice_app.energy_of_draw (live import)"
except Exception as exc:  # pragma: no cover -- environment-dependent
    import numpy as np

    def energy_of_draw(im, row) -> float:
        """FALLBACK ONLY -- transcribed verbatim from
        demo/lattice_app.py:1071-1085 because the live import failed
        ({!r}). Used only if flagged in the report below.""".format(exc)
        s = 2.0 * np.asarray(row, dtype=float) - 1.0
        total = im.offset
        for i in range(len(im.nodes)):
            total -= im.biases[i] * s[i]
        for k, (u, v) in enumerate(im.edges):
            total -= im.weights[k] * s[u] * s[v]
        return float(total)
    ENERGY_OF_DRAW_SOURCE = f"LOCAL FALLBACK TRANSCRIPTION (live import failed: {exc!r})"


def make_asymmetric_model(j_sign: float = 1.0) -> EnergyModel:
    """3 binary spins x0,x1,x2. Terms deliberately asymmetric:
      bias terms:  +0.7*x0, -0.3*x1, +1.1*x2   (three different weights)
      pair terms:  +0.9*x0*x1, -0.5*x1*x2, +0.2*x0*x2   (three different
                   weights, scaled by j_sign for the sign-flip test)
    None of these are invariant under any relabelling of the three spins,
    any global sign flip, or swapping which pair carries which weight --
    exactly the "asymmetric instance" the plan requires to catch a
    double-sign-flip bug a symmetric instance would hide."""
    x0, x1, x2 = Var("x0", Binary()), Var("x1", Binary()), Var("x2", Binary())
    terms = (
        Linear(LinearForm({VarRefLike("x0"): 1.0}), 0.7),
        Linear(LinearForm({VarRefLike("x1"): 1.0}), -0.3),
        Linear(LinearForm({VarRefLike("x2"): 1.0}), 1.1),
        Product(LinearForm({VarRefLike("x0"): 1.0}), LinearForm({VarRefLike("x1"): 1.0}), 0.9 * j_sign),
        Product(LinearForm({VarRefLike("x1"): 1.0}), LinearForm({VarRefLike("x2"): 1.0}), -0.5 * j_sign),
        Product(LinearForm({VarRefLike("x0"): 1.0}), LinearForm({VarRefLike("x2"): 1.0}), 0.2 * j_sign),
    )
    return EnergyModel(variables=(x0, x1, x2), terms=terms, beta=1.0)


def VarRefLike(name: str):
    from tsu_compiler.ir import VarRef
    return VarRef(name)


def ising_to_oracle_J(im) -> dict[tuple[int, int], float]:
    return {(u, v): float(w) for (u, v), w in zip(im.edges, im.weights)}


def report_three_way(im, beta: float) -> None:
    n = len(im.nodes)
    J = ising_to_oracle_J(im)
    b = [float(x) for x in im.biases]
    print(f"  nodes={im.nodes}  offset={im.offset!r}  beta={im.beta!r}")
    print(f"  J (oracle form) = {J}")
    print(f"  b (oracle form) = {b}")
    print(f"  {'state':>10} | {'EnergyModel.energy (IR)':>24} | "
          f"{'energy_of_draw (demo)':>22} | {'oracle.exact_energy + offset':>29} | agree?")
    worst_ir_vs_demo = 0.0
    worst_demo_vs_oracle = 0.0
    for state in itertools.product((0, 1), repeat=n):
        assignment = dict(zip(im.nodes, state))
        e_ir = MODEL_FOR_REPORT.energy(assignment)
        e_demo = energy_of_draw(im, list(state))
        e_oracle_raw = exact_energy(state, J, b)
        e_oracle_plus_offset = e_oracle_raw + im.offset
        worst_ir_vs_demo = max(worst_ir_vs_demo, abs(e_ir - e_demo))
        worst_demo_vs_oracle = max(worst_demo_vs_oracle, abs(e_demo - e_oracle_plus_offset))
        agree = math.isclose(e_ir, e_demo, abs_tol=1e-12) and \
                math.isclose(e_demo, e_oracle_plus_offset, abs_tol=1e-12)
        print(f"  {str(state):>10} | {e_ir:24.12f} | {e_demo:22.12f} | "
              f"{e_oracle_plus_offset:29.12f} | {'YES' if agree else 'NO'}")
    print(f"  max|IR - demo|          = {worst_ir_vs_demo:.3e}")
    print(f"  max|demo - (oracle+off)| = {worst_demo_vs_oracle:.3e}")
    return worst_ir_vs_demo, worst_demo_vs_oracle


print("=" * 78)
print(f"energy_of_draw source: {ENERGY_OF_DRAW_SOURCE}")
print("=" * 78)

print("\n--- (a) THREE-WAY ENERGY AGREEMENT, J POSITIVE (as declared) ---")
model_pos = make_asymmetric_model(j_sign=1.0)
MODEL_FOR_REPORT = model_pos
im_pos = lower(model_pos)
w1, w2 = report_three_way(im_pos, beta=1.0)

print("\n--- (b) SIGN-FLIP-OF-J: SAME instance, every J negated ---")
model_neg = make_asymmetric_model(j_sign=-1.0)
MODEL_FOR_REPORT = model_neg
im_neg = lower(model_neg)
w1n, w2n = report_three_way(im_neg, beta=1.0)

print("\n--- Boltzmann distributions at beta=1, J positive vs J negated ---")
n = len(im_pos.nodes)
states = list(itertools.product((0, 1), repeat=n))


def boltzmann_from_energy_of_draw(im, beta):
    es = [energy_of_draw(im, list(s)) for s in states]
    m = min(es)
    ws = [math.exp(-beta * (e - m)) for e in es]
    Z = sum(ws)
    return [w / Z for w in ws]


p_pos = boltzmann_from_energy_of_draw(im_pos, 1.0)
p_neg = boltzmann_from_energy_of_draw(im_neg, 1.0)

# Independent oracle cross-check of the SAME two distributions.
J_pos = ising_to_oracle_J(im_pos)
b_pos = [float(x) for x in im_pos.biases]
_, p_pos_oracle = exact_boltzmann(J_pos, b_pos, 1.0)  # oracle has no offset;
# offset is a per-state-constant additive shift to E(x), which cancels
# exactly in the softmax/normalisation -- so the ORACLE's un-offset
# distribution must still match energy_of_draw's offset-carrying one
# state-for-state. This is itself part of what "offset correctly absorbs
# the constant" (R3's own question) means: offset changes every state's
# raw energy number but must never change the induced PROBABILITY.

J_neg = ising_to_oracle_J(im_neg)
b_neg = [float(x) for x in im_neg.biases]
_, p_neg_oracle = exact_boltzmann(J_neg, b_neg, 1.0)

print(f"  {'state':>10} | {'p(J>0) demo':>14} | {'p(J>0) oracle':>14} | "
      f"{'p(J<0) demo':>14} | {'p(J<0) oracle':>14}")
max_dev_pos = 0.0
max_dev_neg = 0.0
for i, s in enumerate(states):
    max_dev_pos = max(max_dev_pos, abs(p_pos[i] - p_pos_oracle[i]))
    max_dev_neg = max(max_dev_neg, abs(p_neg[i] - p_neg_oracle[i]))
    print(f"  {str(s):>10} | {p_pos[i]:14.10f} | {p_pos_oracle[i]:14.10f} | "
          f"{p_neg[i]:14.10f} | {p_neg_oracle[i]:14.10f}")
print(f"  max|p_demo - p_oracle| (J>0) = {max_dev_pos:.3e}")
print(f"  max|p_demo - p_oracle| (J<0) = {max_dev_neg:.3e}")

tv = 0.5 * sum(abs(a - b) for a, b in zip(p_pos, p_neg))
print(f"\n  Total variation distance between p(J>0) and p(J<0) distributions: {tv:.6f}")
print(f"  (0.0 would mean the sign flip had NO effect -- a test with no teeth.)")

# thrml's OWN exact_distribution as a fourth, independent cross-check
# (beyond R3's mandatory 3, but directly answers "where does beta enter"
# and "does thrml's own softmax(-e) route agree with the IR's p(x) definition").
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.program import build_program  # noqa: E402
from tsu_compiler.backends.thrml_backend import exact_distribution  # noqa: E402

report_pos = analyse(im_pos)
prog_pos = build_program(im_pos, report_pos)
states_thrml, probs_thrml = exact_distribution(prog_pos)
state_index = {tuple(int(x) for x in row): i for i, row in enumerate(states_thrml)}
max_dev_thrml = 0.0
for i, s in enumerate(states):
    j = state_index[tuple(s)]
    max_dev_thrml = max(max_dev_thrml, abs(p_pos[i] - probs_thrml[j]))
print(f"\n  max|p_demo(J>0) - p_thrml(J>0)| (thrml's own IsingEBM/softmax route) "
      f"= {max_dev_thrml:.3e}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"  max|IR.energy - energy_of_draw|  (J>0) = {w1:.3e}   (J<0) = {w1n:.3e}")
print(f"  max|energy_of_draw - (oracle+offset)|  (J>0) = {w2:.3e}   (J<0) = {w2n:.3e}")
print(f"  max|p_demo - p_oracle|                  (J>0) = {max_dev_pos:.3e}   (J<0) = {max_dev_neg:.3e}")
print(f"  max|p_demo - p_thrml| (J>0)             = {max_dev_thrml:.3e}")
print(f"  Total variation p(J>0) vs p(J<0)        = {tv:.6f}")
print(f"  offset(J>0) = {im_pos.offset!r}   offset(J<0) = {im_neg.offset!r}")
