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


# ---------------------------------------------------------------------------
# This task: edge_span -- which along-edge indices a clamp mode covers.
# ---------------------------------------------------------------------------

def test_edge_span_full_edge_includes_both_ends():
    assert mc.edge_span("full_edge", 8) == tuple(range(8))
    assert 0 in mc.edge_span("full_edge", 8)
    assert 7 in mc.edge_span("full_edge", 8)


def test_edge_span_partial_edge_excludes_both_ends():
    span = mc.edge_span("partial_edge", 8)
    assert span == (1, 2, 3, 4, 5, 6)
    assert 0 not in span
    assert 7 not in span
    assert len(span) == 6


def test_edge_span_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown clamp mode"):
        mc.edge_span("diagonal_edge", 8)


# ---------------------------------------------------------------------------
# This task: neighbour_clamp_free -- the general 4-direction clamp,
# tested per direction and per mode. This is the bookkeeping the free-
# wandering demo below runs on.
# ---------------------------------------------------------------------------

def test_neighbour_clamp_free_no_neighbours_is_empty():
    assert mc.neighbour_clamp_free(1, 1, {}, W, H) == {}


def test_neighbour_clamp_free_north_only_partial_edge_excludes_row_ends():
    north_nb = _uniform(1)
    clamp = mc.neighbour_clamp_free(1, 1, {(1, 0): north_nb}, W, H, mode="partial_edge")
    assert set(clamp) == {f"g{x}_0" for x in range(1, 7)}
    assert "g0_0" not in clamp
    assert "g7_0" not in clamp


def test_neighbour_clamp_free_south_only_partial_edge_excludes_row_ends():
    south_nb = _uniform(2)
    clamp = mc.neighbour_clamp_free(1, 1, {(1, 2): south_nb}, W, H, mode="partial_edge")
    assert set(clamp) == {f"g{x}_{H-1}" for x in range(1, 7)}
    assert f"g0_{H-1}" not in clamp
    assert f"g7_{H-1}" not in clamp


def test_neighbour_clamp_free_west_only_partial_edge_excludes_column_ends():
    west_nb = _uniform(0)
    clamp = mc.neighbour_clamp_free(1, 1, {(0, 1): west_nb}, W, H, mode="partial_edge")
    assert set(clamp) == {f"g0_{y}" for y in range(1, 7)}
    assert "g0_0" not in clamp
    assert "g0_7" not in clamp


def test_neighbour_clamp_free_east_only_partial_edge_excludes_column_ends():
    east_nb = _uniform(1)
    clamp = mc.neighbour_clamp_free(1, 1, {(2, 1): east_nb}, W, H, mode="partial_edge")
    assert set(clamp) == {f"g{W-1}_{y}" for y in range(1, 7)}
    assert f"g{W-1}_0" not in clamp
    assert f"g{W-1}_7" not in clamp


def test_neighbour_clamp_free_north_and_west_partial_edge_share_no_cell():
    # The exact bookkeeping property this task asks for: a chunk with
    # north AND west neighbours, under partial_edge, has NO cell named by
    # both clamps -- deliberately built with DISAGREEING corner-adjacent
    # data (uniform 0 vs uniform 1) to prove it's a structural absence of
    # overlap, not an accidental agreement.
    north_nb = _uniform(0)
    west_nb = _uniform(1)
    clamp = mc.neighbour_clamp_free(1, 1, {(1, 0): north_nb, (0, 1): west_nb},
                                     W, H, mode="partial_edge")
    north_keys = {f"g{x}_0" for x in range(1, 7)}
    west_keys = {f"g0_{y}" for y in range(1, 7)}
    assert north_keys & west_keys == set()
    assert set(clamp) == north_keys | west_keys
    assert "g0_0" not in clamp  # the corner cell itself: named by neither


def test_neighbour_clamp_free_full_edge_north_west_disagreement_raises():
    # Same disagreeing neighbours as above, but under full_edge -- where
    # the corner IS shared, so the disagreement must be caught (this is
    # A1's 52%-disagreement hazard, reproduced deliberately).
    north_nb = _uniform(0)
    west_nb = _uniform(1)
    with pytest.raises(ValueError, match="disagreement"):
        mc.neighbour_clamp_free(1, 1, {(1, 0): north_nb, (0, 1): west_nb},
                                 W, H, mode="full_edge")


def test_neighbour_clamp_free_full_edge_north_west_agreement_is_fine():
    # Consistent corner -> no error, corner IS in the clamp (unlike
    # partial_edge), and its value is the one both neighbours name.
    grid = {f"g{x}_{y}": (x + y) % 3 for x in range(W) for y in range(H)}
    corner_value = grid[f"g{W-1}_{H-1}"]
    north_nb = dict(grid)
    west_nb = dict(grid)
    clamp = mc.neighbour_clamp_free(1, 1, {(1, 0): north_nb, (0, 1): west_nb},
                                     W, H, mode="full_edge")
    assert clamp["g0_0"] == north_nb[f"g0_{H-1}"] == west_nb[f"g{W-1}_0"]


def test_neighbour_clamp_free_all_four_neighbours_partial_edge_no_collision():
    # All four directions present at once (only possible under free
    # generation order, never under raster) -- opposite pairs write
    # different rows/columns so can never collide either way; this just
    # confirms the combined call doesn't raise and covers every expected
    # cell.
    n = _uniform(0)
    s = _uniform(1)
    w = _uniform(2)
    e = _uniform(0)  # deliberately == north's value; must not matter,
    # n and e never touch the same key regardless
    clamp = mc.neighbour_clamp_free(
        1, 1, {(1, 0): n, (1, 2): s, (0, 1): w, (2, 1): e}, W, H, mode="partial_edge")
    expected = ({f"g{x}_0" for x in range(1, 7)} | {f"g{x}_{H-1}" for x in range(1, 7)}
                | {f"g0_{y}" for y in range(1, 7)} | {f"g{W-1}_{y}" for y in range(1, 7)})
    assert set(clamp) == expected


# ---------------------------------------------------------------------------
# This task: center_out_order -- a concrete non-raster traversal.
# ---------------------------------------------------------------------------

def test_center_out_order_3x3_starts_at_centre():
    order = mc.center_out_order(3, 3)
    assert order[0] == (1, 1)


def test_center_out_order_3x3_visits_every_cell_exactly_once():
    order = mc.center_out_order(3, 3)
    assert len(order) == 9
    assert set(order) == {(i, j) for i in range(3) for j in range(3)}


def test_center_out_order_is_not_raster_order():
    order = mc.center_out_order(3, 3)
    assert order != list(mc.raster_order(3, 3))
    # raster order's first coordinate is always the top-left corner (0,0);
    # center_out's is the centre -- the concrete, checkable difference
    # that makes this order actually "non-raster", not just differently
    # shuffled.
    assert list(mc.raster_order(3, 3))[0] == (0, 0)
    assert order[0] == (1, 1)


def test_center_out_order_non_square_grid():
    order = mc.center_out_order(4, 2)
    assert len(order) == 8
    assert set(order) == {(i, j) for i in range(4) for j in range(2)}
