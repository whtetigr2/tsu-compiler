"""The ONLY file that imports thrml.

Exact references come from IsingEBM.energy, never a hand-written Boltzmann loop --
WL4-WL7 were computed by hand while that method sat unused, and retrofitting them
reproduced every number to 1e-9. The arithmetic was right; it should not have been
written.

C3 of the final review: a model with zero edges (the simplest legal workload --
one binary variable, one linear term, no products) makes thrml's own
IsingEBM.factors build a SpinEBMFactor over an EMPTY edge block, which raises a raw
`IndexError: tuple index out of range` deep in vendor code (thrml/models/
discrete_ebm.py, `is_spin[type(group.nodes[0])] = True` indexing an empty
`group.nodes`). That happens inside `_verify`, AFTER the verdict is already
COMPILED, and `IndexError` is not `CompileError`, so nothing in this compiler
catches it. `_edgeless_*` below handles the zero-edge case with real maths instead
of routing it through the vendor factor machinery at all: with no couplings the
joint distribution factorises into independent per-site marginals and both the
exact distribution and exact sampling are closed-form.
"""
from __future__ import annotations

import itertools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from ..passes.lower import IsingModel
from ..passes.program import SamplingProgram

EXACT_LIMIT = 18


def _model(prog: SamplingProgram):
    im = prog.ising
    nodes = [SpinNode() for _ in im.nodes]
    ebm = IsingEBM(nodes, [(nodes[u], nodes[v]) for u, v in im.edges],
                   jnp.asarray(im.biases), jnp.asarray(im.weights),
                   jnp.asarray(im.beta))
    return nodes, ebm


def _edgeless_distribution(im: IsingModel):
    """Zero couplings: the joint over spins s in {-1,+1}^n factorises as
    p(s) proportional to exp(beta * sum_i b_i s_i). Enumerated in the SAME
    MSB-first order as the general path (`itertools.product`), so callers cannot
    tell the two branches apart by state ordering."""
    n = len(im.nodes)
    beta = float(im.beta)
    biases = np.asarray(im.biases, dtype=float)
    states = np.asarray(list(itertools.product([False, True], repeat=n)), dtype=bool)
    spins = 2.0 * states.astype(float) - 1.0
    x = beta * (spins @ biases if n else np.zeros(states.shape[0]))
    x = x - x.max()
    ex = np.exp(x)
    probs = ex / ex.sum()
    return states.astype(int), probs


def _edgeless_sample(im: IsingModel, n_chains: int, n_samples: int, seed: int) -> np.ndarray:
    """Zero couplings: each site is an independent Bernoulli, exact by
    construction -- no chain, no warmup, no mixing to worry about."""
    n = len(im.nodes)
    beta = float(im.beta)
    biases = np.asarray(im.biases, dtype=float)
    p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * biases))     # shape (n,)
    rng = np.random.default_rng(seed)
    total = n_chains * n_samples
    draws = rng.random((total, n)) if n else np.zeros((total, 0))
    return (draws < p_plus).astype(int)


def exact_distribution(prog: SamplingProgram):
    n = len(prog.ising.nodes)
    if n > EXACT_LIMIT:
        raise ValueError(f"{n} nodes exceeds the exact-enumeration limit {EXACT_LIMIT}")
    if not prog.ising.edges:
        return _edgeless_distribution(prog.ising)
    nodes, ebm = _model(prog)
    blk = Block(nodes)
    states = jnp.asarray(list(itertools.product([False, True], repeat=n)), dtype=bool)
    e = jax.jit(jax.vmap(lambda s: ebm.energy([s], [blk])))(states)
    probs = np.asarray(jax.nn.softmax(-e))
    return np.asarray(states).astype(int), probs


def sample(prog: SamplingProgram, n_chains: int, n_samples: int, n_warmup: int,
           steps_per_sample: int, seed: int) -> np.ndarray:
    assert n_warmup > 0, \
        "n_warmup=0 makes sample_states record BEFORE stepping; see manual 4.4"
    if not prog.ising.edges:
        return _edgeless_sample(prog.ising, n_chains, n_samples, seed)
    nodes, ebm = _model(prog)
    blocks = [Block([nodes[i] for i in b]) for b in prog.blocks]
    program = IsingSamplingProgram(ebm, blocks, [])
    sched = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples,
                             steps_per_sample=steps_per_sample)
    key = jax.random.key(seed)
    k_i, k_r = jax.random.split(key)
    # uniform-random init, NOT hinton_init, which starts at the mode
    init = [jax.random.bernoulli(k, 0.5, (n_chains, len(b)))
            for k, b in zip(jax.random.split(k_i, len(blocks)), prog.blocks)]
    fn = jax.jit(jax.vmap(lambda i, k: sample_states(
        k, program, sched, i, [], [Block(nodes)])))
    out = np.asarray(fn(init, jax.random.split(k_r, n_chains))[0])
    return out.reshape(-1, len(prog.ising.nodes)).astype(int)
