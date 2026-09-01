"""A3 Step 1: chunk-grid bookkeeping -- pure logic, no sampling, no receipt
access. Given a grid of chunks and which neighbours already exist, which of
chunk (i,j)'s cells are clamped and to what.

Four cases per the brief: top-left corner (no clamps), top edge (west
only), left edge (north only), interior (both, with a corner cell that must
resolve to exactly one consistent value). A fifth case demonstrates the
corner-disagreement guard raising -- proof `neighbour_clamp` does not
silently paper over a caller that (mistakenly) hands it a diagonal-neighbour
situation raster order is supposed to make impossible.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import multichunk as mc  # noqa: E402

W = H = 8


def _uniform(v):
    return {f"g{x}_{y}": v for x in range(W) for y in range(H)}


# ---------------------------------------------------------------------------
# raster_order
# ---------------------------------------------------------------------------

def test_raster_order_2x2_is_left_to_right_top_to_bottom():
    assert list(mc.raster_order(2, 2)) == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_raster_order_3x1_row():
    assert list(mc.raster_order(3, 1)) == [(0, 0), (1, 0), (2, 0)]


# ---------------------------------------------------------------------------
# neighbour_clamp: the four bookkeeping cases
# ---------------------------------------------------------------------------

def test_top_left_corner_has_no_clamps():
    clamp = mc.neighbour_clamp(0, 0, {}, W, H)
    assert clamp == {}


def test_top_edge_clamps_west_column_only():
    # chunk (1,0): west neighbour (0,0) exists, north neighbour (1,-1) never
    # can (row -1 doesn't exist).
    west_nb = _uniform(1)
    clamp = mc.neighbour_clamp(1, 0, {(0, 0): west_nb}, W, H)
    assert set(clamp) == {f"g0_{y}" for y in range(H)}
    assert all(v == 1 for v in clamp.values())


def test_left_edge_clamps_north_row_only():
    # chunk (0,1): north neighbour (0,0) exists, west neighbour (-1,1) never
    # can (column -1 doesn't exist).
    north_nb = _uniform(2)
    clamp = mc.neighbour_clamp(0, 1, {(0, 0): north_nb}, W, H)
    assert set(clamp) == {f"g{x}_0" for x in range(W)}
    assert all(v == 2 for v in clamp.values())


def test_interior_clamps_both_edges_and_corner_is_one_consistent_value():
    # Build a raster-consistent trio exactly as real generation would
    # produce it: origin chunk (0,0); its EAST neighbour (1,0), whose west
    # column duplicates origin's east column (as (1,0) would have been
    # clamped against origin when IT was generated); its SOUTH neighbour
    # (0,1), whose north row duplicates origin's south row (same reasoning).
    # Then chunk (1,1)'s north neighbour is (1,0) and west neighbour is
    # (0,1) -- both trace back to origin's own SE corner cell.
    origin = {f"g{x}_{y}": (x + y) % 3 for x in range(W) for y in range(H)}
    se_corner_value = origin[f"g{W-1}_{H-1}"]

    north_nb = {f"g{x}_{y}": 99 for x in range(W) for y in range(H)}  # chunk (1,0)
    for y in range(H):
        north_nb[f"g0_{y}"] = origin[f"g{W-1}_{y}"]

    west_nb = {f"g{x}_{y}": 77 for x in range(W) for y in range(H)}  # chunk (0,1)
    for x in range(W):
        west_nb[f"g{x}_0"] = origin[f"g{x}_{H-1}"]

    chunks = {(0, 0): origin, (1, 0): north_nb, (0, 1): west_nb}
    clamp = mc.neighbour_clamp(1, 1, chunks, W, H)

    assert set(clamp) == ({f"g0_{y}" for y in range(H)} | {f"g{x}_0" for x in range(W)})
    # the corner key appears exactly once in the returned dict (it is a
    # single dict; a Python dict cannot hold two values for one key), and
    # that one value is the consistent one both donors independently named
    assert clamp["g0_0"] == se_corner_value
    # sanity on the test's own construction: both donor cells really do
    # trace back to the same origin cell
    assert north_nb["g0_7"] == se_corner_value
    assert west_nb["g7_0"] == se_corner_value


def test_interior_corner_disagreement_raises():
    # A caller handing (1,1) two neighbours whose corner cells disagree --
    # exactly the situation raster order is supposed to make structurally
    # impossible. The guard must catch it rather than silently pick one.
    north_nb = _uniform(0)   # chunk (1,0): g0_7 = 0
    west_nb = _uniform(1)    # chunk (0,1): g7_0 = 1
    chunks = {(1, 0): north_nb, (0, 1): west_nb}
    with pytest.raises(ValueError, match="corner disagreement"):
        mc.neighbour_clamp(1, 1, chunks, W, H)


def test_neighbour_clamp_ignores_diagonal_and_far_chunks():
    # A chunk present at a coordinate that is neither north nor west of
    # (i,j) must never leak into the clamp.
    diagonal = _uniform(2)
    far = _uniform(3)
    clamp = mc.neighbour_clamp(1, 1, {(0, 0): diagonal, (2, 2): far}, W, H)
    assert clamp == {}
