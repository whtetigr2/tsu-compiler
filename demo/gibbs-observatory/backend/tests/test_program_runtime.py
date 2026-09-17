"""Driving a loaded program frame by frame.

The vertical slice: build a model from a program, compile a sampler once, then
step it with new input and decode what comes out.

Two properties carry the whole design and both are asserted rather than
assumed. The sampler must compile ONCE, because a recompile costs ~300ms and is
invisible in the output (R32). And input must not be able to change the graph
behind the sampler's back, because the compiled sampler would then describe a
different model while still producing confident numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "audit"))

pytest.importorskip("tsu_compiler", reason="compiler not importable")

from backend.app.program_runtime import (  # noqa: E402
    RuntimeError_,
    Session,
    open_session,
)

PROGRAM = ROOT / "programs" / "visibility_game.py"
N_COLS, N_DEPTH = 48, 32


@pytest.fixture(scope="module")
def occupancy():
    from visibility_as_inference import corrupt, level_occupancy
    return corrupt(level_occupancy(N_COLS, N_DEPTH), 0.08, seed=7)


@pytest.fixture
def session(occupancy):
    return open_session(PROGRAM, {"occupancy": occupancy}, consent=True, seed=1)


class TestItRuns:
    def test_a_frame_has_one_spin_per_node(self, session):
        frame = session.step()
        assert frame.spins.shape == (1, N_COLS * N_DEPTH)

    def test_frames_advance(self, session):
        assert session.step().frames == 1
        assert session.step().frames == 2

    def test_the_picture_changes_between_frames(self, session):
        a = session.step().spins.copy()
        b = session.step().spins
        assert not np.array_equal(a, b)

    def test_consent_is_required_here_too(self, occupancy):
        from backend.app.program_contract import ConsentRequired
        with pytest.raises(ConsentRequired):
            open_session(PROGRAM, {"occupancy": occupancy})


class TestItCompilesOnce:
    """R32's property, at the level the viewport actually uses."""

    def test_many_frames_one_trace(self, session):
        for _ in range(30):
            frame = session.step()
        assert frame.traces == 1

    def test_new_input_every_frame_still_one_trace(self, session, occupancy):
        from visibility_as_inference import corrupt, level_occupancy
        solid = level_occupancy(N_COLS, N_DEPTH)
        for i in range(20):
            frame = session.step({"occupancy": corrupt(solid, 0.08, seed=i)})
        assert frame.traces == 1, (
            "feeding new input recompiled the sampler, which costs ~300ms a "
            "frame and is invisible in the output")


class TestInputIsDataNotStructure:
    def test_input_that_changes_the_graph_is_refused(self, occupancy):
        """A program whose graph depends on its input would leave the compiled
        sampler describing a different model. Sampling it anyway would produce
        confident numbers for the wrong graph."""
        s = open_session(PROGRAM, {"occupancy": occupancy}, consent=True)
        s.step()
        smaller = np.zeros((N_COLS // 2, N_DEPTH), dtype=bool)
        with pytest.raises(RuntimeError_) as e:
            s.step({"occupancy": smaller})
        msg = str(e.value)
        assert "structure" in msg or "graph" in msg

    def test_the_refusal_reports_both_sizes(self, occupancy):
        s = open_session(PROGRAM, {"occupancy": occupancy}, consent=True)
        s.step()
        with pytest.raises(RuntimeError_) as e:
            s.step({"occupancy": np.zeros((4, 4), dtype=bool)})
        assert str(N_COLS * N_DEPTH) in str(e.value)

    def test_changing_the_biases_is_allowed(self, session, occupancy):
        session.step()
        flipped = ~np.asarray(occupancy)
        frame = session.step({"occupancy": flipped})
        assert frame.frames == 2


class TestDecoding:
    def test_it_decodes_through_the_programs_own_decoder(self, session):
        frame = session.step()
        out = session.decode(frame)
        assert out is not None
        assert out.shape == (N_COLS,)

    def test_the_decoded_answer_is_in_range(self, session):
        out = session.decode(session.step())
        assert out.min() >= 0
        assert out.max() <= N_DEPTH - 1

    def test_the_decoder_shape_comes_from_the_declared_port(self, session):
        assert session._decoder_shape() == (N_COLS, N_DEPTH)


class TestItGetsTheAnswerRight:
    def test_it_agrees_with_the_exact_raycast_on_a_clean_input(self):
        """Speed is worth nothing if the answer moved. The raycast is computed
        independently of anything the sampler does."""
        from visibility_as_inference import (corrupt, exact_raycast,
                                             level_occupancy)
        solid = level_occupancy(N_COLS, N_DEPTH)
        truth = exact_raycast(solid)
        s = open_session(PROGRAM, {"occupancy": corrupt(solid, 0.0, seed=7)},
                         consent=True, seed=1)
        for _ in range(60):
            frame = s.step()
        got = s.decode(frame)
        assert (got == truth).mean() > 0.9

    def test_turning_the_couplings_off_makes_it_worse(self):
        """R23's control, at the level a player would experience. If the
        couplings could be deleted with no effect, the model would be
        decoration."""
        from visibility_as_inference import (corrupt, exact_raycast,
                                             level_occupancy)
        solid = level_occupancy(N_COLS, N_DEPTH)
        truth = exact_raycast(solid)
        observed = corrupt(solid, 0.08, seed=7)

        scores = {}
        for coupled in (True, False):
            s = open_session(PROGRAM,
                             {"occupancy": observed, "couplings": coupled},
                             consent=True, seed=1)
            for _ in range(60):
                frame = s.step()
            scores[coupled] = float((s.decode(frame) == truth).mean())

        assert scores[True] > scores[False] * 2, (
            f"couplings on {scores[True]:.1%}, off {scores[False]:.1%}; "
            f"without a large gap the couplings are not doing the work")
