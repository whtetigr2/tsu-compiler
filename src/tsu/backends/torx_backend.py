"""The ONLY file that imports torx.

torx and thrml are two DISJOINT stacks with no lowering path between them (manual
section I.1, verified 2026-08-25). This backend is an INDEPENDENT exact reference,
never chained into the thrml path.

IT MUST ACTUALLY USE TORX. An earlier draft of this file imported torx and then
computed a numpy Boltzmann loop -- which would have agreed with thrml trivially,
because it computed the same thing the same way. That is the hand-rolling failure
WL4-WL7 paid for, and a control that cannot disagree is not a control.

The real route is PISING driven to stationarity under StateVectorSimulator. PISING
is an energy-based generator gate whose repeated application thermalizes a bond
toward its Boltzmann distribution; StateVectorSimulator.density then gives that
distribution EXACTLY, with no sampling error. BranchingSimulator cannot be used --
it requires branch lookup-table gates and rejects generator gates (EXP-TX1-d).

SCOPE, stated rather than fudged: this route is exact for a SINGLE-BOND model.
Composing PISING gates across several edges is a Trotter splitting and introduces
its own error (EXP-TX4), so for multi-edge models this returns None with a reason
and the caller reports "unavailable" rather than a number it cannot defend.

API note: the installed extro-torx 0.0.1 `PISING` takes its two site indices as a
SINGLE `sites: list[int]` argument (`PISING([0, 1])`), not two positional ints as
early documentation implied. `StateVectorSimulator.density(circuit, x)` also does
not take a scalar basis-state index for `x` -- `x` is the INITIAL DISTRIBUTION, a
probability vector of length prod(circuit.dims) (here 4, for two binary pbits).
Both were confirmed against the installed package's source
(torx/psc/gates/_generator.py, torx/psc/simulation/statevector.py) before use.
"""
from __future__ import annotations

import numpy as np

from ..passes.lower import IsingModel

CROSS_CHECK_REPS = (200, 400, 800)
CONVERGENCE_TOL = 1e-9


def torx_cross_check(ising: IsingModel):
    """Independent exact distribution for a single-bond model, via torx.

    Returns (probs, note) where probs is None when the route does not apply, so
    callers can report "unavailable: <note>" instead of fabricating a reference.
    """
    n = len(ising.nodes)
    if n != 2 or len(ising.edges) != 1:
        return None, (f"torx PISING route is exact only for a single-bond model; "
                      f"this has {n} nodes and {len(ising.edges)} edges, and "
                      f"composing PISING across edges is a Trotter splitting "
                      f"with its own error (EXP-TX4)")
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from torx.psc import DiscretePCircuit, PISING, StateVectorSimulator
    except Exception as e:
        return None, f"torx unavailable: {type(e).__name__}: {e}"

    J = float(ising.weights[0])
    h1, h2 = float(ising.biases[0]), float(ising.biases[1])
    beta = float(ising.beta)

    sim = StateVectorSimulator()
    prev = None
    for reps in CROSS_CHECK_REPS:
        # sites is a single list argument, not two positional ints (see module
        # docstring's API note).
        circuit = DiscretePCircuit([PISING([0, 1])], reps=reps)
        theta = jnp.asarray([J, h1, h2, beta, 0.5])
        compiled = sim.build_circuit(circuit, [theta])
        dim = int(np.prod(circuit.dims))
        # x is the INITIAL DISTRIBUTION (a probability vector), not a basis-state
        # index -- start uniform so convergence to stationarity is not an artifact
        # of an already-converged or otherwise privileged starting point.
        start = jnp.full((dim,), 1.0 / dim, dtype=jnp.float64)
        dens = np.asarray(sim.density(compiled, start)).reshape(-1)
        dens = dens / dens.sum()
        if prev is not None and np.abs(dens - prev).max() < CONVERGENCE_TOL:
            return dens, ""
        prev = dens
    return None, (f"torx PISING did not reach stationarity within "
                  f"{CROSS_CHECK_REPS[-1]} reps (last delta "
                  f"{np.abs(dens - prev).max():.2e})" if prev is not None else
                  "torx PISING produced no density")
