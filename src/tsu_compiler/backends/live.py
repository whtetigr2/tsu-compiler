"""A sampler you can drive frame by frame without recompiling it.

`sample_chains` is written for one independent run: it builds a fresh
`IsingSamplingProgram` and calls `jax.jit` on a new closure every call. The
closure is a new Python object each time, so JAX cannot reuse its compilation
cache and re-traces on every call. Measured on the shipped visibility model,
1536 spins on a 48x32 lattice:

    sample_chains, per call                 183 ms      5.4 fps
    the same work with the trace reused    0.23 ms     4315 fps

The physics is not the cost. Sixty-four Gibbs sweeps per frame on that model
cost 1.10 ms; the other 180 ms is compilation, paid again every frame. Anything
driven interactively -- the game, a dragged residue, a swept parameter -- is
paying for a compiler, not for sampling.

`LiveSampler` traces once. Two things make that possible:

**Biases are a traced argument, not a closure.** `eqx.filter_jit` treats array
leaves as traced and everything else -- the `SpinNode` objects, the blocks, the
schedule -- as static, so a model whose input changes every frame reuses one
compilation. Closing over the biases instead would make every frame a new
program and defeat the whole thing.

**The chain carries forward.** A game resumes from the previous frame rather
than re-thermalising from noise, which is both cheaper and what makes the
picture stable instead of boiling.

**The trace count is exposed.** A silent retrace gives back exactly the cost
this class exists to avoid, and it is invisible: the draws would still be
correct, just slow. So `traces` is a counted fact, and the tests assert on it.

WHAT IS STRUCTURE AND WHAT IS DATA. Which spins exist, which are coupled to
which, and which are clamped are STRUCTURE: changing any of them is a different
program and needs a new sampler. The bias on a spin, the value a clamped spin
is held at, and the inverse temperature are DATA: they can change every frame
for free. A game moves the player by rewriting biases, which is data, which is
why this works at all.

No hardware claim. THRML simulates on CPU through JAX.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["LiveSampler"]


class LiveSampler:
    """Advance one set of chains, frame by frame, from a single compilation.

    Parameters
    ----------
    prog:
        A `SamplingProgram` from `passes.program.build_program`, which carries
        the colour blocks the sampler updates in parallel.
    n_chains:
        Independent chains advanced together. One is the usual choice for a
        game, where the point is a single evolving picture; more is useful when
        the caller wants a spread.
    sweeps:
        Block-Gibbs sweeps per `step`. More sweeps per frame means the field
        settles faster in response to changed input, at linear cost.
    seed:
        Starting key. Every `step` splits a fresh subkey, so frames are not
        correlated through the RNG.
    """

    def __init__(self, prog: Any, *, n_chains: int = 1, sweeps: int = 8,
                 seed: int = 0) -> None:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from thrml import Block, SamplingSchedule, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram

        from .thrml_backend import _model

        if not prog.ising.edges:
            # `_model` has nothing to build a factor from, and a model with no
            # edges has no couplings to sample against anyway. `sample_chains`
            # special-cases it; here it is simply not what this class is for.
            raise ValueError(
                "LiveSampler needs a model with at least one edge; this one "
                "has none, so there is nothing for block Gibbs to do. Use "
                "sample_chains, which handles the edgeless case directly.")

        if sweeps < 1:
            # `sample_states` records BEFORE stepping when warmup is 0 (manual
            # 4.4), which would hand back the previous frame unchanged and look
            # like a frozen picture rather than an error.
            raise ValueError(
                f"sweeps must be at least 1, got {sweeps}; at 0 the sampler "
                f"records before stepping and every frame repeats the last")

        self._eqx = eqx
        self._jax = jax
        self._jnp = jnp

        nodes, ebm = _model(prog)
        self._nodes = nodes
        self._blocks = list(prog.blocks)
        self._free_blocks = [Block([nodes[i] for i in b]) for b in prog.blocks]
        self._all = Block(nodes)
        self._edges = ebm.edges
        self._weights = jnp.asarray(ebm.weights)

        self._clamped = tuple(prog.clamped)
        if self._clamped:
            self._clamped_blocks = [Block([nodes[i] for i in self._clamped])]
            self._clamp_values = jnp.asarray(
                [bool(prog.clamp_values[i]) for i in self._clamped], dtype=bool)
        else:
            self._clamped_blocks = []
            self._clamp_values = None

        self.n_spins = len(nodes)
        self.n_chains = int(n_chains)
        self.sweeps = int(sweeps)
        self.traces = 0

        self._biases = jnp.asarray(np.asarray(prog.ising.biases, dtype=float))
        self._beta = jnp.asarray(float(prog.ising.beta))
        self._schedule = SamplingSchedule(n_warmup=self.sweeps, n_samples=1,
                                          steps_per_sample=1)

        counter = self  # the trace counter increments at TRACE time only

        @eqx.filter_jit
        def _advance(biases, beta, state, clamp_values, keys):
            counter.traces += 1
            model = IsingEBM(self._nodes, self._edges, biases,
                             self._weights, beta)
            program = IsingSamplingProgram(model, self._free_blocks,
                                           self._clamped_blocks)
            state_clamp = [] if clamp_values is None else [clamp_values]

            # `sample_states` takes ONE chain: per-block state is
            # (block_len,), and the chain axis is what vmap adds. Handing it
            # (n_chains, block_len) directly raises "Spin states must be
            # scalar" from thrml's own shape check. The clamp values are
            # closed over rather than mapped, because they are shared across
            # chains rather than one per chain.
            def one(init, key):
                return sample_states(key, program, self._schedule, init,
                                     state_clamp, [self._all])

            return jax.vmap(one)(state, keys)

        self._advance = _advance

        self._key = jax.random.key(seed)
        self._key, k = jax.random.split(self._key)
        # Uniform init, not hinton_init, which starts at the mode. The same
        # choice sample_chains makes and for the same reason.
        self._state = [
            jax.random.bernoulli(sub, 0.5, (self.n_chains, len(b)))
            for sub, b in zip(jax.random.split(k, len(self._free_blocks)),
                              self._blocks)]
        self._last: np.ndarray | None = None

    # -- driving ----------------------------------------------------------

    def step(self, biases: Any = None, *, beta: float | None = None,
             clamp_values: Any = None) -> np.ndarray:
        """Advance every chain by `sweeps` and return the current spins.

        `biases`, `beta` and `clamp_values` are data: pass new ones and they
        take effect this frame without recompiling. Omit them and the last
        values stay in force, which is what a game wants between input events.

        Returns an array of shape (n_chains, n_spins) of 0/1.
        """
        jnp = self._jnp
        if biases is not None:
            arr = np.asarray(biases, dtype=float).reshape(-1)
            if arr.size != self.n_spins:
                raise ValueError(
                    f"biases has {arr.size} entries and this model has "
                    f"{self.n_spins} spins")
            if not np.all(np.isfinite(arr)):
                # A NaN propagates into every draw and the picture simply goes
                # wrong, with nothing on screen saying why.
                bad = int(np.count_nonzero(~np.isfinite(arr)))
                raise ValueError(
                    f"biases contains {bad} non-finite value(s); a NaN or inf "
                    f"spreads silently through every draw")
            self._biases = jnp.asarray(arr)

        if beta is not None:
            if not np.isfinite(beta):
                raise ValueError(f"beta must be finite, got {beta!r}")
            self._beta = jnp.asarray(float(beta))

        if clamp_values is not None:
            if not self._clamped:
                raise ValueError(
                    "this model clamps no spins, so there are no clamp values "
                    "to set; which spins are clamped is structure and is fixed "
                    "when the sampler is built")
            arr = np.asarray(clamp_values).reshape(-1)
            if arr.size != len(self._clamped):
                raise ValueError(
                    f"clamp_values has {arr.size} entries and this model "
                    f"clamps {len(self._clamped)} spins")
            self._clamp_values = jnp.asarray(arr.astype(bool), dtype=bool)

        self._key, k = self._jax.random.split(self._key)
        keys = self._jax.random.split(k, self.n_chains)
        out = self._advance(self._biases, self._beta, self._state,
                            self._clamp_values, keys)

        # One block was requested, so out[0]. vmap prepends the chain axis and
        # the schedule contributes the n_samples axis, giving
        # (n_chains, 1, n_spins); one sample per frame, so drop that axis.
        drawn = np.asarray(out[0])
        self._last = drawn.reshape(self.n_chains, self.n_spins).astype(int)

        # Carry the chain forward: this frame's configuration is the next
        # frame's starting point, per block.
        self._state = [
            self._jnp.asarray(self._last[:, list(b)].astype(bool))
            for b in self._blocks]
        return self._last

    # -- reading ----------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """The most recent draw, (n_chains, n_spins). None before the first
        step, because there is nothing to report and a zero array would be a
        configuration the sampler never produced."""
        if self._last is None:
            raise RuntimeError("no frame has been sampled yet; call step()")
        return self._last

    @property
    def biases(self) -> np.ndarray:
        return np.asarray(self._biases)
