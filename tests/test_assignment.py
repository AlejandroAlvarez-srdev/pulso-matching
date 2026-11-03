import itertools
import math
import random
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pulso_matching.assignment import (
    BudgetExceededError,
    CostMatrix,
    Pairs,
    greedy_assignment,
    hungarian_assignment,
    hungarian_assignment_np,
    to_float_matrix,
)


def total(cost: CostMatrix, pairs: Pairs) -> float:
    values = [cost[i][j] for i, j in pairs]
    assert all(v is not None for v in values), "solver returned an infeasible pair"
    return math.fsum(v for v in values if v is not None)


def assert_valid(cost: CostMatrix, pairs: Pairs) -> None:
    rows = [i for i, _ in pairs]
    cols = [j for _, j in pairs]
    assert len(set(rows)) == len(rows)
    assert len(set(cols)) == len(cols)
    assert all(cost[i][j] is not None for i, j in pairs)


def brute_force(cost: CostMatrix) -> tuple[int, float]:
    """Reference optimum: (max #feasible matches, min total cost among those). O(m!/(m-n)!)."""
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0:
        return 0, 0.0
    best: tuple[int, float] = (0, 0.0)
    if n <= m:
        candidates = (
            list(zip(range(n), perm, strict=True)) for perm in itertools.permutations(range(m), n)
        )
    else:
        candidates = (
            list(zip(perm, range(m), strict=True)) for perm in itertools.permutations(range(n), m)
        )
    first = True
    for pairs in candidates:
        feasible = [cost[i][j] for i, j in pairs if cost[i][j] is not None]
        key = (-len(feasible), math.fsum(c for c in feasible if c is not None))
        if first or key < (-best[0], best[1]):
            best = (-key[0], key[1])
            first = False
    return best


cell = st.one_of(st.none(), st.floats(min_value=0, max_value=100, allow_nan=False))
dense_cell = st.floats(min_value=-50, max_value=100, allow_nan=False)


@st.composite
def matrices(
    draw: st.DrawFn, cells: st.SearchStrategy[float | None] = cell
) -> list[list[float | None]]:
    n = draw(st.integers(min_value=0, max_value=6))
    m = draw(st.integers(min_value=1, max_value=6))
    return [[draw(cells) for _ in range(m)] for _ in range(n)]


def test_textbook_example() -> None:
    cost: CostMatrix = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    pairs = hungarian_assignment(cost)
    assert pairs == [(0, 1), (1, 0), (2, 2)]
    assert total(cost, pairs) == 5


def test_greedy_is_suboptimal_on_crafted_instance() -> None:
    # Greedy grabs (0,0)=1 and is then forced into (1,1)=100; optimum is 2 + 2 = 4.
    cost: CostMatrix = [[1, 2], [2, 100]]
    assert total(cost, greedy_assignment(cost)) == 101
    assert total(cost, hungarian_assignment(cost)) == 4


def test_greedy_can_lose_cardinality_hungarian_does_not() -> None:
    # Row 1 can only use column 0; greedy gives it away to row 0.
    cost: CostMatrix = [[1, 5], [3, None]]
    assert greedy_assignment(cost) == [(0, 0)]
    assert hungarian_assignment(cost) == [(0, 1), (1, 0)]


@pytest.mark.parametrize(
    "solver", [hungarian_assignment, hungarian_assignment_np, greedy_assignment]
)
def test_rectangular_and_degenerate(solver: Callable[[CostMatrix], Pairs]) -> None:
    assert solver([]) == []
    assert solver([[None, None], [None, None]]) == []
    wide: CostMatrix = [[5, 1, 9, 4]]
    assert solver(wide) == [(0, 1)]
    tall: CostMatrix = [[5], [1], [9]]
    assert solver(tall) == [(1, 0)]


@pytest.mark.parametrize("solver", [hungarian_assignment, hungarian_assignment_np])
def test_rejects_ragged_matrix(solver: Callable[[CostMatrix], Pairs]) -> None:
    with pytest.raises(ValueError, match="rectangular"):
        solver([[1, 2], [3]])


def test_numpy_accepts_ndarray_with_inf_and_nan_as_infeasible() -> None:
    cost = np.array([[1.0, 5.0], [3.0, np.inf], [np.nan, 0.5]])
    assert hungarian_assignment_np(cost) == hungarian_assignment([[1, 5], [3, None], [None, 0.5]])


def test_numpy_deadline_in_the_past_raises() -> None:
    with pytest.raises(BudgetExceededError):
        hungarian_assignment_np([[1.0, 2.0], [3.0, 4.0]], deadline=0.0)


def test_to_float_matrix_marks_none_as_inf() -> None:
    out = to_float_matrix([[1.0, None]])
    assert out[0, 0] == 1.0
    assert np.isinf(out[0, 1])


@settings(max_examples=300, deadline=None)
@given(matrices())
def test_hungarian_matches_brute_force_with_infeasible_pairs(
    cost: list[list[float | None]],
) -> None:
    pairs = hungarian_assignment(cost)
    assert_valid(cost, pairs)
    best_count, best_cost = brute_force(cost)
    assert len(pairs) == best_count
    assert total(cost, pairs) == pytest.approx(best_cost, abs=1e-6)


@settings(max_examples=300, deadline=None)
@given(matrices(dense_cell))
def test_hungarian_matches_brute_force_dense_with_negative_costs(
    cost: list[list[float | None]],
) -> None:
    pairs = hungarian_assignment(cost)
    best_count, best_cost = brute_force(cost)
    assert len(pairs) == best_count == min(len(cost), len(cost[0]) if cost else 0)
    assert total(cost, pairs) == pytest.approx(best_cost, abs=1e-6)


@settings(max_examples=300, deadline=None)
@given(matrices(dense_cell))
def test_greedy_is_valid_and_never_beats_optimum_on_dense(cost: list[list[float | None]]) -> None:
    pairs = greedy_assignment(cost)
    assert_valid(cost, pairs)
    assert len(pairs) == min(len(cost), len(cost[0]) if cost else 0)
    assert total(cost, pairs) >= total(cost, hungarian_assignment(cost)) - 1e-6


@settings(max_examples=300, deadline=None)
@given(matrices())
def test_numpy_variant_matches_brute_force(cost: list[list[float | None]]) -> None:
    pairs = hungarian_assignment_np(cost)
    assert_valid(cost, pairs)
    best_count, best_cost = brute_force(cost)
    assert len(pairs) == best_count
    assert total(cost, pairs) == pytest.approx(best_cost, abs=1e-6)


@settings(max_examples=50, deadline=None)
@given(
    st.integers(min_value=1, max_value=40),
    st.integers(min_value=1, max_value=40),
    st.randoms(use_true_random=False),
)
def test_numpy_variant_agrees_with_reference_on_larger_instances(
    n: int, m: int, rnd: random.Random
) -> None:
    cost: list[list[float | None]] = [
        [None if rnd.random() < 0.3 else rnd.uniform(0, 1) for _ in range(m)] for _ in range(n)
    ]
    ref, fast = hungarian_assignment(cost), hungarian_assignment_np(cost)
    assert len(ref) == len(fast)
    assert total(cost, fast) == pytest.approx(total(cost, ref), abs=1e-9)
