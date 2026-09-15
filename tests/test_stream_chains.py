"""Does streaming give the same distribution, from genuinely continuing chains?

`stream_chains` exists so a live view can watch ONE set of chains evolve rather
than a new set every frame. That makes two things load-bearing, and both are
easy to get wrong in ways that still produce plausible-looking pictures:

  1. The draws must still come from the model's distribution. A resumed chain
     that quietly restarts, or that carries state back into the wrong block,
     would still yield well-shaped arrays full of wrong numbers.
  2. The chains must actually continue. If each batch secretly re-initialised,
     every frame of a live view would show fresh chains while the caption said
     "the same chains" -- the animation equivalent of the R20 defect.

The distribution test is against `audit/oracles/exact.py`, which enumerates the
Boltzmann distribution and does NOT import `src/tsu_compiler`, so it cannot
inherit a convention from the code under test.
"""
import itertools
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "audit")

from oracles.exact import exact_boltzmann
from tsu_compiler.backends.thrml_backend import sample_chains, stream_chains
from tsu_compiler.passes.analyse import analyse
from tsu_compiler.passes.lower import IsingModel
from tsu_compiler.passes.program import build_program


def ring(n=6, weight=0.6, beta=1.0):
    """An even ring: bipartite, so block Gibbs applies, and small enough to
    enumerate exactly."""
    edges = tuple((i, (i + 1) % n) for i in range(n))
    return IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=edges,
                      weights=np.full(len(edges), weight), biases=np.zeros(n),
                      beta=beta, offset=0.0)


def test_streamed_draws_match_brute_force_enumeration():
    """WHAT THIS PINS: streaming samples the model's actual distribution. This
    is the property that makes a live view evidence rather than decoration.

    HOW IT FAILS: carry the resumed state back in the wrong block order and the
    chains sample a permuted model. The arrays still have the right shape and
    the picture still looks like a landscape.

    PROVENANCE: `audit/oracles/exact.py` enumerates all 2**n states from
    E = -sum J s s - sum b s and does not import this package. The tolerance is
    5 standard errors of the sample mean -- a real bound, not a number chosen
    to pass."""
    model = ring()
    prog = build_program(model, analyse(model))
    J = {(int(a), int(b)): float(w)
         for (a, b), w in zip(model.edges, model.weights)}
    states, probs = exact_boltzmann(J, [0.0] * len(model.nodes),
                                    float(model.beta))
    P = np.asarray(probs)
    S = np.array([[2 * v - 1 for v in s] for s in states])
    exact_corr = {e: float(P @ (S[:, e[0]] * S[:, e[1]])) for e in J}

    gen = stream_chains(prog, n_chains=16, batch_samples=250, n_warmup=2000,
                        steps_per_sample=4, seed=0)
    draws = np.concatenate([next(gen) for _ in range(8)], axis=1)
    flat = 2 * draws.reshape(-1, draws.shape[-1]).astype(int) - 1

    n = flat.shape[0]
    tol = 5.0 / np.sqrt(n)
    worst = max(abs(float((flat[:, a] * flat[:, b]).mean()) - exact_corr[(a, b)])
                for a, b in J)
    assert worst < tol, (
        f"streamed neighbour correlations are off the exact answer by {worst:.4f}, "
        f"beyond {tol:.4f} (5 s.e. on {n:,} draws)")


def test_the_chains_actually_continue_across_batches():
    """WHAT THIS PINS: the state really carries. A deeply ordered model barely
    moves between consecutive samples, so the last configuration of one batch
    and the first of the next must be nearly identical -- far closer than two
    independently initialised chains would be.

    HOW IT FAILS: drop the resume and re-initialise each batch. The two
    configurations then agree only at chance level, and the gap between this
    test's two numbers collapses.

    PROVENANCE: at beta*J = 1.2 a 16-spin ring is frozen; agreement across the
    seam is the signature of continuation, and the independent-restart baseline
    is measured in the same test rather than assumed."""
    model = ring(n=16, weight=1.2)
    prog = build_program(model, analyse(model))
    gen = stream_chains(prog, n_chains=24, batch_samples=30, n_warmup=1500,
                        steps_per_sample=2, seed=3)
    a, b = next(gen), next(gen)
    seam = float((a[:, -1, :] == b[:, 0, :]).mean())

    # baseline: two runs that share no state at all
    x = sample_chains(prog, n_chains=24, n_samples=2, n_warmup=1500,
                      steps_per_sample=2, seed=11)
    y = sample_chains(prog, n_chains=24, n_samples=2, n_warmup=1500,
                      steps_per_sample=2, seed=12)
    restart = float((x[:, -1, :] == y[:, 0, :]).mean())

    assert seam > 0.9, (
        f"only {seam:.3f} of spins agree across the batch seam; the chains are "
        f"not continuing")
    assert seam > restart + 0.2, (
        f"agreement across the seam ({seam:.3f}) is not meaningfully above two "
        f"independent restarts ({restart:.3f}), so continuation is unproven")


def test_shape_contract_matches_sample_chains():
    """WHAT THIS PINS: a caller can swap one for the other. Both return
    (n_chains, n_samples, n_spins).

    HOW IT FAILS: yield the flattened shape and every collective coordinate
    computed per chain silently averages across chains instead.

    PROVENANCE: `sample_chains`' own documented contract."""
    model = ring()
    prog = build_program(model, analyse(model))
    gen = stream_chains(prog, n_chains=5, batch_samples=7, n_warmup=50,
                        steps_per_sample=2, seed=0)
    batch = next(gen)
    assert batch.shape == (5, 7, len(model.nodes))
    assert next(gen).shape == batch.shape
    assert set(np.unique(batch)) <= {0, 1}


def test_an_edgeless_model_is_refused_rather_than_faked():
    """WHAT THIS PINS: the edgeless path in `sample_chains` draws independent
    samples and has no chain state, so there is nothing to continue. Streaming
    it would produce a live view of a chain that does not exist.

    HOW IT FAILS: silently fall through to the independent-draw path and the
    viewer shows "continuing chains" that restart every frame.

    PROVENANCE: `sample_chains`' `_edgeless_sample` branch."""
    n = 4
    model = IsingModel(nodes=tuple(f"n{i}" for i in range(n)), edges=(),
                       weights=np.zeros(0), biases=np.zeros(n),
                       beta=1.0, offset=0.0)
    prog = build_program(model, analyse(model))
    with pytest.raises(ValueError, match="no chain state"):
        next(stream_chains(prog, n_chains=2, batch_samples=2, n_warmup=10,
                           steps_per_sample=1, seed=0))


def test_zero_warmup_is_refused():
    """WHAT THIS PINS: the same guard `sample_chains` carries. At n_warmup=0
    `sample_states` records before stepping, so the first sample of a resumed
    batch would duplicate the previous batch's last configuration -- inflating
    apparent agreement exactly where this module claims continuity.

    PROVENANCE: thrml manual 4.4, and the identical assert in `sample_chains`."""
    model = ring()
    prog = build_program(model, analyse(model))
    with pytest.raises(AssertionError, match="n_warmup=0"):
        next(stream_chains(prog, n_chains=2, batch_samples=2, n_warmup=0,
                           steps_per_sample=1, seed=0))
