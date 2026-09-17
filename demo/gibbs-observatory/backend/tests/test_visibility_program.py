"""The shipped game program must be the model that was already verified.

`audit/visibility_on_thrml.py` builds an energy that was checked against an
exact raycast and cross-checked against a numpy reference at the same beta.
`programs/visibility_game.py` is that same model wearing the Workbench's
program contract.

"Same" has to be checked, not asserted. A program contract that quietly
produced a DIFFERENT energy would carry none of the verification across, and
the divergence would be invisible: both files would compile, both would sample,
both would produce plausible pictures. So these tests compare the two models
term by term.

The ablation is here for the same reason it is in the audit script. R23 was a
lattice whose couplings could be deleted with zero effect while the prose
credited them, and the only thing that catches that is turning the term off.
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

from backend.app.program_contract import load_program  # noqa: E402

PROGRAM = ROOT / "programs" / "visibility_game.py"
N_COLS, N_DEPTH = 48, 32


@pytest.fixture(scope="module")
def program():
    return load_program(PROGRAM, consent=True)


@pytest.fixture(scope="module")
def observed():
    """The same occupancy the audit script uses, corrupted the same way, so
    the two models are compared on identical input."""
    from visibility_as_inference import corrupt, level_occupancy
    solid = level_occupancy(N_COLS, N_DEPTH)
    return corrupt(solid, 0.08, seed=7)


@pytest.fixture(scope="module")
def reference(observed):
    """The verified model, built by the audit script's own `build`."""
    import visibility_on_thrml
    return visibility_on_thrml.build(observed, beta=1.0, couplings=True)


class TestItIsTheVerifiedModel:
    def test_the_graph_is_identical(self, program, observed, reference):
        mine = program.build({"occupancy": observed, "beta": 1.0})
        assert len(mine.nodes) == len(reference.nodes)
        assert mine.edges == reference.edges

    def test_every_coupling_is_identical(self, program, observed, reference):
        mine = program.build({"occupancy": observed, "beta": 1.0})
        assert np.array_equal(mine.weights, reference.weights), (
            "the program's couplings differ from the verified model's, so "
            "none of that verification carries across")

    def test_every_bias_is_identical(self, program, observed, reference):
        mine = program.build({"occupancy": observed, "beta": 1.0})
        assert np.array_equal(mine.biases, reference.biases)

    def test_beta_matches(self, program, observed, reference):
        mine = program.build({"occupancy": observed, "beta": 1.0})
        assert mine.beta == reference.beta

    def test_the_lattice_is_bipartite(self, program, observed):
        """Not decoration: `build_program` cannot colour a non-bipartite
        lattice, so this is the difference between a program that runs and one
        that does not."""
        from tsu_compiler.passes.analyse import analyse
        mine = program.build({"occupancy": observed, "beta": 1.0})
        assert analyse(mine).bipartite


class TestTheCouplingsAreNotInert:
    """R23's control. A term that can be deleted with no effect is not doing
    the work the prose credits it with."""

    def test_turning_the_couplings_off_changes_the_model(self, program, observed):
        on = program.build({"occupancy": observed, "couplings": True})
        off = program.build({"occupancy": observed, "couplings": False})
        assert not np.array_equal(on.weights, off.weights)
        assert np.abs(off.weights).max() == 0.0
        assert np.abs(on.weights).max() > 0.0

    def test_the_biases_are_untouched_by_the_ablation(self, program, observed):
        """The ablation must move ONE thing. If it also changed the biases the
        comparison would not isolate the couplings."""
        on = program.build({"occupancy": observed, "couplings": True})
        off = program.build({"occupancy": observed, "couplings": False})
        assert np.array_equal(on.biases, off.biases)


class TestTheContractItDeclares:
    def test_the_occupancy_port_enters_as_a_bias(self, program):
        """The whole point, and the opposite of what the design document says.
        A clamped wrong cell stays wrong; a biased one can be outvoted."""
        port = {p.name: p for p in program.ports}["occupancy"]
        assert port.mode == "bias"

    def test_nothing_is_clamped(self, program):
        assert not [p for p in program.ports if p.mode == "clamp"]

    def test_it_names_its_decoder(self, program):
        assert program.decoder == "visibility"

    def test_the_port_shape_matches_what_build_accepts(self, program, observed):
        port = {p.name: p for p in program.ports}["occupancy"]
        assert tuple(port.shape) == observed.shape


class TestTheDecoder:
    def test_it_returns_one_depth_per_column(self, program, observed):
        model = program.build({"occupancy": observed})
        spins = np.zeros(len(model.nodes), dtype=int)
        out = program.module.decode(spins, (N_COLS, N_DEPTH))
        assert out.shape == (N_COLS,)

    def test_a_column_that_never_stops_reads_as_maximum_depth(self, program):
        """All ones is "still going" the whole way down."""
        spins = np.ones(N_COLS * N_DEPTH, dtype=int)
        out = program.module.decode(spins, (N_COLS, N_DEPTH))
        assert (out == N_DEPTH - 1).all()

    def test_a_column_that_stops_immediately_reads_as_zero(self, program):
        spins = np.zeros(N_COLS * N_DEPTH, dtype=int)
        out = program.module.decode(spins, (N_COLS, N_DEPTH))
        assert (out == 0).all()

    def test_it_matches_the_reference_decoder(self, program):
        from visibility_as_inference import decode as reference_decode
        rng = np.random.default_rng(3)
        v = (rng.random((N_COLS, N_DEPTH)) < 0.5).astype(int)
        assert np.array_equal(
            program.module.decode(v.reshape(-1), (N_COLS, N_DEPTH)),
            reference_decode(v))


class TestItCompiles:
    def test_the_model_passes_the_gates_at_this_size(self, program, observed):
        """A program the Workbench ships must fit the target it ships against,
        or the first thing a reader does is watch it refuse."""
        from tsu_compiler.gates import gate_checks
        from tsu_compiler.passes.analyse import analyse
        from tsu_compiler.target import PROFILES
        model = program.build({"occupancy": observed})
        report = analyse(model)
        failed = [g for g in gate_checks(model, report, PROFILES["z1"])
                  if not g.passed]
        assert not failed, f"gates refused: {[g.gate for g in failed]}"
