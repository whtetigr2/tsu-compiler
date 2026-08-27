"""The ONLY file that imports thrml.

Exact references come from IsingEBM.energy, never a hand-written Boltzmann loop --
WL4-WL7 were computed by hand while that method sat unused, and retrofitting them
reproduced every number to 1e-9. The arithmetic was right; it should not have been
written.
"""
from __future__ import annotations

import itertools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from ..passes.program import SamplingProgram

EXACT_LIMIT = 18


def _model(prog: SamplingProgram):
    im = prog.ising
    nodes = [SpinNode() for _ in im.nodes]
    ebm = IsingEBM(nodes, [(nodes[u], nodes[v]) for u, v in im.edges],
                   jnp.asarray(im.biases), jnp.asarray(im.weights),
                   jnp.asarray(im.beta))
    return nodes, ebm


def exact_distribution(prog: SamplingProgram):
    n = len(prog.ising.nodes)
    if n > EXACT_LIMIT:
        raise ValueError(f"{n} nodes exceeds the exact-enumeration limit {EXACT_LIMIT}")
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
