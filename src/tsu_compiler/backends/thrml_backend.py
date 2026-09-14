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


def _edgeless_sample(im: IsingModel, n_chains: int, n_samples: int, seed: int,
                     clamp_values: dict[int, int] | None = None) -> np.ndarray:
    """Zero couplings: each site is an independent Bernoulli, exact by
    construction -- no chain, no warmup, no mixing to worry about.

    C1: a clamped site (`clamp_values`) is never drawn -- its column is fixed
    to its clamped value on every draw, the same guarantee the general
    (edged) path gets from thrml's own `state_clamp`. This is a SEPARATE code
    path from the general one (see this module's own docstring on why
    zero-edge models bypass thrml's factor machinery entirely), so it needs
    its own clamp handling."""
    n = len(im.nodes)
    beta = float(im.beta)
    biases = np.asarray(im.biases, dtype=float)
    p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * biases))     # shape (n,)
    rng = np.random.default_rng(seed)
    total = n_chains * n_samples
    draws = rng.random((total, n)) if n else np.zeros((total, 0))
    out = (draws < p_plus).astype(int)
    for i, v in (clamp_values or {}).items():
        out[:, i] = int(v)
    return out


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


def _clamp_consistent_rows(n: int, clamped: tuple[int, ...],
                           clamp_values) -> list[list[bool]]:
    """Every FULL n-length row consistent with the clamp: clamped columns
    fixed to their clamped value, free columns ranging over every
    combination. Shared by both the edged and edgeless conditional paths so
    the two can never enumerate a different row order or disagree on which
    rows are clamp-consistent."""
    clamped_set = set(clamped)
    free = [i for i in range(n) if i not in clamped_set]
    combos = list(itertools.product([False, True], repeat=len(free)))
    rows = []
    for combo in combos:
        row = [False] * n
        for i in clamped:
            row[i] = bool(clamp_values[i])
        for fi, val in zip(free, combo):
            row[fi] = val
        rows.append(row)
    return rows


def _edgeless_conditional_distribution(im: IsingModel, clamped: tuple[int, ...],
                                       clamp_values):
    """Clamp-aware sibling of `_edgeless_distribution`: same closed-form
    independent-Bernoulli factorisation, restricted to the clamp-consistent
    rows and renormalised over THOSE alone -- clamping a factorised
    distribution just fixes some of its independent coordinates, so the
    conditional is still closed-form, no enumeration-and-discard needed for
    correctness (though `_clamp_consistent_rows` is reused anyway so both
    conditional paths share exactly one definition of "clamp-consistent
    row")."""
    n = len(im.nodes)
    beta = float(im.beta)
    biases = np.asarray(im.biases, dtype=float)
    rows = _clamp_consistent_rows(n, clamped, clamp_values)
    states = np.array(rows, dtype=bool).astype(int)
    spins = 2.0 * states.astype(float) - 1.0
    x = beta * (spins @ biases if n else np.zeros(states.shape[0]))
    x = x - x.max()
    ex = np.exp(x)
    probs = ex / ex.sum()
    return states, probs


def exact_conditional_distribution(prog: SamplingProgram):
    """Clamp-aware exact reference: `exact_distribution` restricted to the
    states consistent with `prog`'s clamp and renormalised over just those --
    the distribution a clamped sampler is actually drawing from, so a caller
    comparing sampled draws against it (`_verify`'s execution_tv) is checking
    the sampler against the reference it is actually running under, not an
    unconditional one the clamp has already made irrelevant.

    Equal to `exact_distribution(prog)` when the program carries no clamp --
    conditioning on nothing is the unconditional distribution -- so callers
    can call this unconditionally without a clamped/unclamped branch of
    their own.

    Same `IsingEBM.energy` route as `exact_distribution` (never a hand-
    rolled Boltzmann loop): enumerates only the clamp-consistent full-node
    rows (via `_clamp_consistent_rows`) and asks thrml for each one's energy,
    exactly as the unconditional path does over every row.
    """
    n = len(prog.ising.nodes)
    if n > EXACT_LIMIT:
        raise ValueError(f"{n} nodes exceeds the exact-enumeration limit {EXACT_LIMIT}")
    if not prog.clamped:
        return exact_distribution(prog)
    if not prog.ising.edges:
        return _edgeless_conditional_distribution(
            prog.ising, prog.clamped, prog.clamp_values)
    nodes, ebm = _model(prog)
    blk = Block(nodes)
    rows = _clamp_consistent_rows(n, prog.clamped, prog.clamp_values)
    states = jnp.asarray(rows, dtype=bool)
    e = jax.jit(jax.vmap(lambda s: ebm.energy([s], [blk])))(states)
    probs = np.asarray(jax.nn.softmax(-e))
    return np.asarray(states).astype(int), probs


def sample_chains(prog: SamplingProgram, n_chains: int, n_samples: int,
                  n_warmup: int, steps_per_sample: int, seed: int) -> np.ndarray:
    """Same draws as `sample`, but UNFLATTENED: shape (n_chains, n_samples,
    n_spins). `sample`'s own (n_chains*n_samples, n_spins) contract flattens
    the chain boundary away on purpose -- fine for a caller that only wants
    i.i.d.-looking draws to compare against a stationary reference (energy_tv,
    execution_tv, task_validity), but autocorrelation/effective-sample-size
    (tsu_compiler.ess) is meaningless across that boundary. Added as a SEPARATE
    function, rather than a flag on `sample`, so `sample`'s existing shape
    contract -- several callers already depend on it -- never changes based on
    a caller's intent.

    C1: `prog.clamped`/`prog.clamp_values` (set by `build_program`) are
    passed straight through to thrml -- `IsingSamplingProgram`'s own
    `clamped_blocks` and `sample_states`' own `state_clamp`, exactly the
    mechanism thrml provides for this, never a hand-rolled "hold it fixed"
    loop. `nodes_to_sample` stays every node (free AND clamped), so a
    returned sample's clamped columns show the clamped value, never an
    ambiguous absence.
    """
    assert n_warmup > 0, \
        "n_warmup=0 makes sample_states record BEFORE stepping; see manual 4.4"
    if not prog.ising.edges:
        n = len(prog.ising.nodes)
        flat = _edgeless_sample(prog.ising, n_chains, n_samples, seed,
                                clamp_values=prog.clamp_values)
        return flat.reshape(n_chains, n_samples, n)
    nodes, ebm = _model(prog)
    free_blocks = [Block([nodes[i] for i in b]) for b in prog.blocks]
    clamped = tuple(prog.clamped)
    if clamped:
        clamped_blocks = [Block([nodes[i] for i in clamped])]
        state_clamp = [jnp.asarray(
            [bool(prog.clamp_values[i]) for i in clamped], dtype=bool)]
    else:
        clamped_blocks = []
        state_clamp = []
    program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks)
    sched = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples,
                             steps_per_sample=steps_per_sample)
    key = jax.random.key(seed)
    k_i, k_r = jax.random.split(key)
    # uniform-random init, NOT hinton_init, which starts at the mode
    init = [jax.random.bernoulli(k, 0.5, (n_chains, len(b)))
            for k, b in zip(jax.random.split(k_i, len(free_blocks)), prog.blocks)]
    fn = jax.jit(jax.vmap(lambda i, k: sample_states(
        k, program, sched, i, state_clamp, [Block(nodes)])))
    out = np.asarray(fn(init, jax.random.split(k_r, n_chains))[0])
    return out.astype(int)


def sample(prog: SamplingProgram, n_chains: int, n_samples: int, n_warmup: int,
           steps_per_sample: int, seed: int) -> np.ndarray:
    chains = sample_chains(prog, n_chains, n_samples, n_warmup,
                           steps_per_sample, seed)
    return chains.reshape(-1, chains.shape[-1]).astype(int)
