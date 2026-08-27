import numpy as np
from tsu.passes.lower import IsingModel
from tsu.passes.analyse import analyse
from tsu.passes.program import build_program
from tsu.passes.route import route
from tsu.target import Z1


def ising(n, edges):
    return IsingModel(tuple(f"x{i}" for i in range(n)), tuple(edges),
                      np.ones(len(edges)), np.zeros(n), 1.0, 0.0)


def test_every_block_is_an_independent_set():
    """Two adjacent nodes in one block is silently wrong, not an error, in thrml."""
    im = ising(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    prog = build_program(im, analyse(im))
    for block in prog.blocks:
        for u, v in im.edges:
            assert not (u in block and v in block)


def test_blocks_partition_every_node_exactly_once():
    im = ising(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    prog = build_program(im, analyse(im))
    seen = [n for b in prog.blocks for n in b]
    assert sorted(seen) == list(range(5))


def test_schedule_is_the_target_documented_one():
    im = ising(2, [(0, 1)])
    prog = build_program(im, analyse(im))
    assert prog.schedule == "chromatic_block_gibbs"


def test_program_declares_which_FORM_it_is():
    """A monolithic Boltzmann energy is one program form, not the only one. A
    composition of conditional kernels is another (C-033/U-008). The form must be
    stated so a receipt is never ambiguous about what produced it."""
    im = ising(2, [(0, 1)])
    assert build_program(im, analyse(im)).form == "monolithic_energy"


def test_route_is_identity_when_nothing_is_frustrated():
    im = ising(3, [(0, 1), (1, 2)])
    out = route(im, analyse(im), Z1)
    assert out.edges == im.edges
    assert np.array_equal(out.weights, im.weights)
