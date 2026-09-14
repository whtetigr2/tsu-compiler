"""Does thrml sample the distribution our compiler intended?

This is the check that backs every sampled number in this project, and until now
it was run by hand in a session and its result was never written down. It runs
every suite instead.

WHY IT IS THE ORACLE AND NOT JUST ANOTHER TEST. `audit/oracles/exact.py`
enumerates all 2**n configurations from `E(s) = -sum J s s - sum b s` and does
NOT import `src/tsu_compiler`. So it cannot inherit a sign convention, an encoding
mistake, or a lowering bug from the code under test -- which is exactly what a
cross-check has to avoid. A test that verified our sampler against our own
energy function would prove only that we are consistent.

THE SIGN IS THE PART THAT MATTERS MOST. Our IR means `E(x) = sum weight*term`
with `p ~ exp(-beta E)`, thrml means `E = -beta(sum b s + sum J s s)`, and this
project has already been bitten once by a chain coupling written with the wrong
sign -- it was antiferromagnetic and silently broke the thing it was meant to
bind. An inverted coupling here would produce a checkerboard that still looks
like plausible terrain at a glance and would very likely pass the shuffled-null
structure test, because a checkerboard is highly structured. Only comparison
against an independent energy function catches it.

Kept small deliberately: 4x4 is 16 spins, 65,536 states, enumerable in about a
second. The bridge to the sizes we actually sample is Onsager, not this test --
the 32x32 sweep put the transition within one step of the exact infinite-lattice
Kc, which is independent evidence the large models behave.
"""
import sys

import numpy as np

sys.path.insert(0, "demo")
sys.path.insert(0, "src")
sys.path.insert(0, "audit")

from world.fields import compile_layer
from tsu_compiler.backends.thrml_backend import sample as thrml_sample
from oracles.exact import exact_boltzmann

N = 4
BETA_J = 0.42
CHAINS, SAMPLES, WARMUP, STEPS = 32, 60, 2000, 8


def _exact_moments(ising):
    """First and second moments by brute-force enumeration, from the oracle."""
    J = {(int(a), int(b)): float(w)
         for (a, b), w in zip(ising.edges, ising.weights)}
    b = [float(v) for v in ising.biases]
    states, probs = exact_boltzmann(J, b, float(ising.beta))
    P = np.asarray(probs)
    S = np.array([[2 * v - 1 for v in s] for s in states])
    mag = P @ S
    corr = {e: float(P @ (S[:, e[0]] * S[:, e[1]])) for e in J}
    return mag, corr, J


def _sampled_moments(prog, ising, J):
    rows = np.asarray(thrml_sample(prog, seed=0, n_chains=CHAINS,
                                   n_samples=SAMPLES, n_warmup=WARMUP,
                                   steps_per_sample=STEPS))
    S = 2 * rows.astype(int) - 1
    mag = S.mean(axis=0)
    corr = {e: float((S[:, e[0]] * S[:, e[1]]).mean()) for e in J}
    return mag, corr, S.shape[0]


def test_thrml_reproduces_the_exact_distribution():
    """Sampled moments must match brute-force enumeration to within Monte Carlo
    error. The tolerance is 5 standard errors of the sample mean, which is a
    real bound rather than a number chosen to pass: a systematically wrong
    distribution fails it no matter how many samples are drawn, while a correct
    one fails it about once in three million per moment."""
    _spec, _enc, ising, prog = compile_layer(N, BETA_J)
    ex_mag, ex_corr, J = _exact_moments(ising)
    th_mag, th_corr, n = _sampled_moments(prog, ising, J)

    tol = 5.0 / np.sqrt(n)
    d_mag = float(np.abs(th_mag - ex_mag).max())
    d_corr = max(abs(th_corr[e] - ex_corr[e]) for e in J)
    assert d_mag < tol, f"magnetisation off by {d_mag:.4f} (tol {tol:.4f})"
    assert d_corr < tol, f"correlation off by {d_corr:.4f} (tol {tol:.4f})"


def test_the_coupling_is_ferromagnetic_not_antiferromagnetic():
    """A same-value clumping rule must make neighbours ALIGN.

    Asserted against the oracle's own enumeration rather than against our
    lowering, and stated as a positive threshold rather than a sign test: an
    inverted coupling gives a strongly NEGATIVE mean neighbour correlation, so
    this cannot pass by accident. This project has shipped a wrong-signed
    coupling before, and a checkerboard is structured enough to survive tests
    that only look for structure."""
    _spec, _enc, ising, _prog = compile_layer(N, BETA_J)
    _mag, corr, _J = _exact_moments(ising)
    mean_nn = float(np.mean(list(corr.values())))
    assert mean_nn > 0.25, (
        f"mean neighbour correlation {mean_nn:.4f} is not ferromagnetic; "
        f"a negative value means the clumping rule is anti-aligning")


def test_the_oracle_does_not_import_the_code_it_checks():
    """The property that makes this a cross-check rather than a tautology. If
    the oracle ever imports src/tsu_compiler it inherits our conventions and stops being
    independent evidence."""
    from pathlib import Path
    src = Path("audit/oracles/exact.py").read_text(encoding="utf-8")
    for banned in ("from tsu", "import tsu_compiler", "from src.tsu"):
        assert banned not in src, f"oracle imports the code under test: {banned}"
