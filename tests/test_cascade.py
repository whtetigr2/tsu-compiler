"""Task 1 of the coarse-to-fine cascade plan
(SPR/docs/superpowers/plans/2026-09-07-coarse-to-fine-cascade.md):
upsampling a coarse decode and building the conditioning patch for the next
level. These two functions are the whole cascade -- everything else is
composition (demo/cascade_world.py).

Values are carried, never interpolated: these are categorical states
(water/rock/grass, or whatever a coarser level decoded), and averaging them
would invent a state nobody asked for. `upsample` is a pure block-replication
function -- no sampling, no randomness -- so it is tested here as ordinary
Python logic, the same way tests/test_binary_stack.py tests `compose_state`
without touching a compiled program.
"""
import sys

sys.path.insert(0, "demo")


def test_upsample_expands_each_cell_into_a_block():
    """A 2x2 coarse grid at factor 2 becomes 4x4, each coarse cell filling its
    own 2x2 block. Values are carried, never interpolated -- these are
    categorical states, and averaging them would invent a state nobody asked
    for.

    FIXTURE FIX (flagged by code review): the original fixture used 0 for
    BOTH off-diagonal coarse cells (g1_0 and g0_1). An x/y-transpose bug in
    upsample -- e.g. swapping cx/cy on the way in, or swapping
    base_x/base_y when building the fine cell name on the way out -- would
    swap the (1,0) block and the (0,1) block, but since both carried the
    identical value 0, the swap was invisible to every assertion below: the
    test passed whether or not upsample transposed x and y. All four coarse
    cells now carry DISTINCT values (1, 0, 3, 2) specifically so that swap
    changes what lands at g2_0 vs g0_2 (and the rest of their blocks), and
    the assertions check exactly those cells."""
    from cascade import upsample
    coarse = {"g0_0": 1, "g1_0": 0, "g0_1": 3, "g1_1": 2}
    fine = upsample(coarse, src_w=2, factor=2)
    assert len(fine) == 16
    # coarse (0,0) = 1 -> fine block {(0,0),(1,0),(0,1),(1,1)}
    assert fine["g0_0"] == 1 and fine["g1_0"] == 1
    assert fine["g0_1"] == 1 and fine["g1_1"] == 1
    # coarse (1,0) = 0 -> fine block {(2,0),(3,0),(2,1),(3,1)}
    assert fine["g2_0"] == 0 and fine["g3_0"] == 0
    assert fine["g2_1"] == 0 and fine["g3_1"] == 0
    # coarse (0,1) = 3 -> fine block {(0,2),(1,2),(0,3),(1,3)}
    assert fine["g0_2"] == 3 and fine["g1_2"] == 3
    assert fine["g0_3"] == 3 and fine["g1_3"] == 3
    # coarse (1,1) = 2 -> fine block {(2,2),(3,2),(2,3),(3,3)}
    assert fine["g2_2"] == 2 and fine["g3_2"] == 2
    assert fine["g2_3"] == 2 and fine["g3_3"] == 2


def test_cascade_patch_favours_the_inherited_value():
    """The patch must reward a fine cell for keeping what it inherited, and
    only that. A patch that rewards every value equally conditions on nothing."""
    from cascade import cascade_patch
    patch = cascade_patch({"g0_0": 1, "g1_0": 0}, strength=0.5)
    assert patch[("g0_0", 1)] > 0
    assert patch[("g1_0", 0)] > 0
    assert patch.get(("g0_0", 0), 0.0) <= 0.0
