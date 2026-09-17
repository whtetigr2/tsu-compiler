"""A sampler you can drive frame by frame without recompiling it.

`sample_chains` builds a fresh `IsingSamplingProgram` and calls `jax.jit` on a
new closure every call. The closure is a new Python object each time, so JAX
cannot reuse its compilation cache and re-traces on every single call. Measured
on the shipped visibility model, 1536 spins:

    sample_chains, per call                 183 ms      5.4 fps
    the same work with the trace reused    0.23 ms     4315 fps

The physics is not the cost. Even 64 Gibbs sweeps per frame costs 1.10 ms; the
180 ms is compilation, paid again every frame.

That is fine for what `sample_chains` was written for -- one independent run --
and it is why a game could not be built on it. `LiveSampler` traces once and
then takes new biases as a TRACED argument, so a model whose input changes
every frame costs one compilation, not one per frame.

The trace count is exposed rather than assumed. A silent retrace would give
back exactly the 180 ms this class exists to avoid, and it would be invisible:
the numbers would still be right. So `traces` is a counted fact and the tests
below assert on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("thrml", reason="thrml not installed")
pytest.importorskip("equinox", reason="equinox not installed")

from tsu_compiler.backends.live import LiveSampler  # noqa: E402
from tsu_compiler.passes.analyse import analyse  # noqa: E402
from tsu_compiler.passes.program import build_program  # noqa: E402
from tsu_compiler.preflight.model import IsingModel  # noqa: E402


def _grid(cols: int = 6, rows: int = 6, weight: float = 0.4) -> IsingModel:
    """A bipartite grid, so the compiler can colour it in two blocks."""
    def idx(c, r):
        return c * rows + r

    edges, weights = [], []
    for c in range(cols):
        for r in range(rows):
            if r + 1 < rows:
                edges.append((idx(c, r), idx(c, r + 1)))
                weights.append(weight)
            if c + 1 < cols:
                edges.append((idx(c, r), idx(c + 1, r)))
                weights.append(weight)
    order = sorted(range(len(edges)), key=lambda i: edges[i])
    return IsingModel(
        nodes=tuple(f"v{i}" for i in range(cols * rows)),
        edges=tuple(edges[i] for i in order),
        weights=np.array([weights[i] for i in order]),
        biases=np.zeros(cols * rows),
        beta=1.0,
        offset=0.0,
    )


@pytest.fixture
def sampler():
    model = _grid()
    return LiveSampler(build_program(model, analyse(model)), sweeps=4, seed=0)


class TestItTracesOnce:
    """The property the class exists for, asserted rather than hoped for."""

    def test_one_trace_after_many_frames(self, sampler):
        for _ in range(25):
            sampler.step()
        assert sampler.traces == 1

    def test_new_biases_every_frame_do_not_retrace(self, sampler):
        rng = np.random.default_rng(0)
        for _ in range(25):
            sampler.step(biases=rng.normal(size=sampler.n_spins))
        assert sampler.traces == 1, (
            "changing the biases retraced, which costs the compilation this "
            "class exists to pay once")

    def test_the_first_step_does_trace(self, sampler):
        assert sampler.traces == 0
        sampler.step()
        assert sampler.traces == 1


class TestItSamples:
    def test_a_step_returns_one_spin_per_node(self, sampler):
        out = sampler.step()
        assert out.shape == (sampler.n_chains, sampler.n_spins)

    def test_draws_are_binary(self, sampler):
        out = sampler.step()
        assert set(np.unique(out)) <= {0, 1}

    def test_the_chain_carries_forward(self, sampler):
        """A game resumes from the previous frame rather than re-thermalising.
        If state were discarded, every frame would start from noise and the
        picture would boil."""
        sampler.step()
        before = sampler.state.copy()
        sampler.step()
        assert not np.array_equal(before, sampler.state), (
            "the state did not advance")

    def test_a_strong_bias_moves_the_answer(self, sampler):
        """The control. If the biases could be changed with no effect on the
        draws, the port would be decorative -- which is exactly R23."""
        up = sampler.step(biases=np.full(sampler.n_spins, 6.0))
        for _ in range(8):
            up = sampler.step(biases=np.full(sampler.n_spins, 6.0))
        down = sampler.step(biases=np.full(sampler.n_spins, -6.0))
        for _ in range(8):
            down = sampler.step(biases=np.full(sampler.n_spins, -6.0))
        assert up.mean() > 0.9, f"strong positive bias gave {up.mean():.3f}"
        assert down.mean() < 0.1, f"strong negative bias gave {down.mean():.3f}"

    def test_biases_persist_until_changed(self, sampler):
        sampler.step(biases=np.full(sampler.n_spins, 6.0))
        for _ in range(8):
            out = sampler.step()          # no biases passed
        assert out.mean() > 0.9, "the last biases were forgotten"


class TestItRefusesClearly:
    def test_wrong_length_biases_are_refused_with_both_numbers(self, sampler):
        with pytest.raises(ValueError) as e:
            sampler.step(biases=np.zeros(3))
        msg = str(e.value)
        assert "3" in msg and str(sampler.n_spins) in msg

    def test_a_non_finite_bias_is_refused(self, sampler):
        """A NaN propagates silently into every draw and the picture just goes
        wrong, with nothing saying why."""
        bad = np.zeros(sampler.n_spins)
        bad[0] = np.nan
        with pytest.raises(ValueError) as e:
            sampler.step(biases=bad)
        assert "finite" in str(e.value).lower()

    def test_an_edgeless_model_is_refused_by_name(self):
        model = IsingModel(nodes=("a", "b"), edges=(), weights=np.array([]),
                           biases=np.zeros(2), beta=1.0, offset=0.0)
        with pytest.raises(ValueError) as e:
            LiveSampler(build_program(model, analyse(model)))
        assert "edge" in str(e.value).lower()


class TestItIsActuallyFast:
    def test_a_frame_costs_far_less_than_a_fresh_sample_chains_call(self):
        """The whole claim, measured. Not a fixed millisecond budget, which
        would make this a flaky test on a loaded machine, but a ratio against
        the thing it replaces, measured in the same process."""
        import time
        from tsu_compiler.backends.thrml_backend import sample_chains

        model = _grid(10, 10)
        prog = build_program(model, analyse(model))

        sampler = LiveSampler(prog, sweeps=4, seed=0)
        sampler.step()                      # pay the one compilation
        t0 = time.time()
        for _ in range(20):
            sampler.step()
        live = (time.time() - t0) / 20

        sample_chains(prog, 1, 1, 4, 1, 0)  # warm whatever it can warm
        t0 = time.time()
        for i in range(3):
            sample_chains(prog, 1, 1, 4, 1, i)
        fresh = (time.time() - t0) / 3

        assert live * 20 < fresh, (
            f"live frame {live * 1000:.2f}ms vs sample_chains "
            f"{fresh * 1000:.1f}ms; without a large gap this class is not "
            f"worth having")
        assert sampler.traces == 1
