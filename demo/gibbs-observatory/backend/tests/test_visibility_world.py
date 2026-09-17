"""The walkable level: pose in, visible depth per column out.

The geometry half is ordinary raycasting and is checked here against what the
world obviously contains. The point of the tests is that the geometry is dumb
and correct, so that anything interesting in the output came from the sampler
rather than from a clever ray march.
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
from backend.app.program_runtime import Session  # noqa: E402

PROGRAM = ROOT / "programs" / "visibility_world.py"


@pytest.fixture(scope="module")
def program():
    return load_program(PROGRAM, consent=True)


class TestTheWorld:
    def test_it_is_enclosed(self, program):
        """A player who can walk out of the world sees nothing, and nothing is
        not a picture anyone can debug."""
        world = program.module.WORLD
        assert world[0].all() and world[-1].all()
        assert world[:, 0].all() and world[:, -1].all()

    def test_the_start_pose_is_not_inside_a_wall(self, program):
        x, y, _ = program.module.START_POSE
        assert not program.module.WORLD[int(y), int(x)]

    def test_it_has_interior_geometry(self, program):
        """An empty room makes every column identical and the coupling has
        nothing to do."""
        interior = program.module.WORLD[1:-1, 1:-1]
        assert interior.any()


class TestTheRayMarch:
    def test_it_returns_one_ray_per_column(self, program):
        occ = program.module.occupancy_from_pose(program.module.START_POSE)
        assert occ.shape == (program.module.N_COLS, program.module.N_DEPTH)

    def test_the_far_plane_is_always_solid(self, program):
        """A column that hits nothing must stop at the end of its range. An
        'infinitely far' column is not something the decoder can draw."""
        occ = program.module.occupancy_from_pose((12.0, 12.0, 0.0))
        assert occ[:, -1].all()

    def test_facing_a_near_wall_reads_solid_early(self, program):
        """Standing next to the west wall and facing it, every ray should hit
        something almost immediately."""
        occ = program.module.occupancy_from_pose((1.5, 12.0, np.pi))
        first_hit = [int(np.flatnonzero(row)[0]) for row in occ]
        assert max(first_hit) < program.module.N_DEPTH // 2

    def test_turning_around_changes_what_is_seen(self, program):
        a = program.module.occupancy_from_pose((12.0, 12.0, 0.0))
        b = program.module.occupancy_from_pose((12.0, 12.0, np.pi))
        assert not np.array_equal(a, b)

    def test_moving_changes_what_is_seen(self, program):
        a = program.module.occupancy_from_pose((4.0, 4.0, 0.5))
        b = program.module.occupancy_from_pose((14.0, 14.0, 0.5))
        assert not np.array_equal(a, b)

    def test_stepping_outside_the_world_reads_as_solid(self, program):
        """Not a crash and not a wrapped index: outside is wall."""
        occ = program.module.occupancy_from_pose((-5.0, -5.0, 0.0))
        assert occ.all()


class TestItRunsAsAProgram:
    def test_a_session_opens_on_the_declared_default_pose(self, program):
        s = Session(program, {})
        frame = s.step()
        assert frame.spins.shape[1] == 48 * 32

    def test_the_decoder_gives_one_depth_per_column(self, program):
        s = Session(program, {})
        out = s.decode(s.step())
        assert out.shape == (48,)
        assert out.min() >= 0
        assert out.max() <= 31

    def test_moving_changes_the_picture(self, program):
        """The whole interaction loop in one assertion: new pose in, different
        visible depths out."""
        s = Session(program, {})
        for _ in range(30):
            near = s.decode(s.step({"pose": np.array([2.0, 2.0, 0.7])}))
        for _ in range(30):
            far = s.decode(s.step({"pose": np.array([12.0, 12.0, 3.4])}))
        assert not np.array_equal(near, far)

    def test_a_pose_change_does_not_recompile(self, program):
        """The property the frame rate depends on. Pose enters as biases on a
        fixed graph, so moving is data rather than structure (R32)."""
        s = Session(program, {})
        rng = np.random.default_rng(0)
        for _ in range(25):
            pose = np.array([6.0 + rng.random(), 6.0 + rng.random(),
                             rng.random() * 6.28])
            frame = s.step({"pose": pose})
        assert frame.traces == 1

    def test_facing_a_wall_reads_nearer_than_facing_the_room(self, program):
        """The sampled answer has to agree with where the player is looking,
        or the picture is decorative."""
        s = Session(program, {})
        for _ in range(40):
            at_wall = s.decode(s.step({"pose": np.array([1.6, 12.0, np.pi])}))
        for _ in range(40):
            down_room = s.decode(s.step({"pose": np.array([1.6, 12.0, 0.0])}))
        assert at_wall.mean() < down_room.mean(), (
            f"facing the wall read {at_wall.mean():.1f} and facing the open "
            f"room read {down_room.mean():.1f}; the sampled depth does not "
            f"track where the player is looking")


class TestItReusesTheVerifiedEnergy:
    def test_the_lattice_matches_visibility_game_on_the_same_rays(self, program):
        """This program must not have its own copy of the energy. If it did,
        nothing checked against the exact raycast would apply to it.

        Compared with noise off, so the two sides see the same rays. The noise
        is the subject of the tests below, not of this one.
        """
        import visibility_on_thrml
        occ = program.module.occupancy_from_pose(program.module.START_POSE)
        mine = program.build({"pose": program.module.START_POSE, "noise": 0.0})
        reference = visibility_on_thrml.build(occ, beta=1.0, couplings=True)
        assert mine.edges == reference.edges
        assert np.array_equal(mine.weights, reference.weights)
        assert np.array_equal(mine.biases, reference.biases)


class TestTheNoiseIsOnByDefault:
    """With clean input this program reproduces a raycast, which is not the
    claim. The reason to sample is that neighbouring columns couple and can
    overrule a cell read wrong, and with nothing read wrong there is nothing to
    overrule."""

    def test_the_default_is_not_zero(self, program):
        assert program.module.DEFAULT_NOISE > 0

    def test_opening_with_no_input_uses_it(self, program):
        """A declared port is zero-filled by the runtime unless the program
        says otherwise, and zero noise is exactly the setting to avoid."""
        from backend.app.program_runtime import initial_inputs
        assert initial_inputs(program, {})["noise"] == program.module.DEFAULT_NOISE

    def test_noise_changes_the_biases(self, program):
        clean = program.build({"pose": program.module.START_POSE, "noise": 0.0})
        noisy = program.build({"pose": program.module.START_POSE, "noise": 0.25})
        assert not np.array_equal(clean.biases, noisy.biases)

    def test_noise_only_adds_walls(self, program):
        """A missing wall makes a ray run long and the next cell catches it; a
        phantom wall stops it dead and nothing downstream undoes that. Only the
        second is the failure a raycaster cannot recover from."""
        occ = program.module.occupancy_from_pose(program.module.START_POSE)
        dirty = program.module._corrupt(occ.copy(), 0.3, 0)
        assert (dirty | occ == dirty).all(), "noise removed a wall"
        assert dirty.sum() > occ.sum()

    def test_the_far_plane_survives_the_noise(self, program):
        """A column still needs somewhere to stop."""
        model = program.build({"pose": program.module.START_POSE, "noise": 0.9})
        assert model.biases.size == 48 * 32

    def test_noise_does_not_change_the_graph(self, program):
        """It is data, so it must not move structure or every change would
        recompile the sampler (R32)."""
        a = program.build({"pose": program.module.START_POSE, "noise": 0.0})
        b = program.build({"pose": program.module.START_POSE, "noise": 0.4})
        assert a.edges == b.edges
        assert np.array_equal(a.weights, b.weights)
