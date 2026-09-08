"""Task 7: the routing measurement's path finder, tested on hand-built grids
where the right answer is known by inspection. The verdict itself is a
measurement, not a test -- but the arithmetic under it has to be right, or the
whole finding is a confident wrong number."""
import sys

import numpy as np

sys.path.insert(0, "demo")
sys.path.insert(0, "src")
sys.path.insert(0, "audit")

from routing import cheapest_path_cost, best_straight_cost


def test_uniform_grid_costs_one_row_and_takes_the_straight_path():
    cost = np.ones((5, 5), dtype=int)
    total, cells = cheapest_path_cost(cost)
    assert total == 5
    assert cells == 5


def test_a_cheap_lane_is_preferred_over_a_straight_expensive_one():
    """Row 0 is expensive, row 4 is cheap. The optimal route must detour to
    row 4 -- if it does not, the search is not exploring vertically."""
    cost = np.full((5, 5), 9, dtype=int)
    cost[4, :] = 1
    total, _cells = cheapest_path_cost(cost)
    assert total < 9 * 5


def test_detour_makes_the_path_longer_than_the_width():
    """TWO staggered walls, whose gaps sit on opposite rows, so no straight
    horizontal route is cheap. Any route must pass column 3 at row 6 and
    column 5 at row 0, forcing vertical travel.

    A single wall with one gap is NOT enough and was rejected during plan
    review: Dijkstra may start on ANY left-edge cell, so it would simply start
    on the gap's row and walk straight across -- passing the test while proving
    nothing about vertical search, which is the only thing this test exists to
    check."""
    cost = np.ones((7, 7), dtype=int)
    cost[:, 3] = 50
    cost[6, 3] = 1
    cost[:, 5] = 50
    cost[0, 5] = 1
    _total, cells = cheapest_path_cost(cost)
    assert cells > 7


def test_best_straight_cost_picks_the_cheapest_row():
    cost = np.full((3, 4), 5, dtype=int)
    cost[1, :] = 2
    assert best_straight_cost(cost) == 8


def test_cheapest_path_never_beats_an_impossible_lower_bound():
    """Sanity: the optimal path cannot cost less than the width times the
    global minimum cell cost."""
    rng = np.random.default_rng(0)
    cost = rng.integers(1, 5, size=(8, 8))
    total, _cells = cheapest_path_cost(cost)
    assert total >= 8 * int(cost.min())
