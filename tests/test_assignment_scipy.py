"""Differential tests against SciPy, far beyond the sizes brute force can reach.

Brute force certifies the solvers up to 6x6, and the NumPy variant is compared with the
pure-Python reference up to 40x40 — but both of those are *this* repo's code. Here the
optimum comes from two independent implementations:

- ``scipy.sparse.csgraph.maximum_bipartite_matching`` (Hopcroft-Karp) gives the maximum number of
  feasible matches, with no costs involved — this checks the coverage half of the objective.
- ``scipy.optimize.linear_sum_assignment`` (LAPJV-style shortest augmenting path, C++) gives the
  minimum total cost among maximum matchings. Infeasible pairs are priced with a penalty larger
  than any difference in feasible cost, so SciPy optimises the same lexicographic objective.

SciPy is a dev-only dependency: the service never imports it.
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array
from scipy.sparse.csgraph import maximum_bipartite_matching

from pulso_matching.assignment import (
    greedy_assignment,
    hungarian_assignment,
    hungarian_assignment_np,
)

FloatArray = NDArray[np.float64]

SHAPES = [(1, 1), (7, 3), (3, 7), (50, 80), (80, 50), (120, 120), (200, 300), (300, 200)]
DENSITIES = [1.0, 0.3, 0.05]
SEEDS = [0, 1, 2]


def instance(n: int, m: int, density: float, seed: int) -> FloatArray:
    """Random costs in [-50, 100) with ``inf`` for infeasible pairs; few distinct values on
    purpose (rounded to 0.5), so ties between optimal assignments are common."""
    rng = np.random.default_rng(seed)
    cost = np.round(rng.uniform(-50, 100, size=(n, m)) * 2) / 2
    cost[rng.random((n, m)) >= density] = np.inf
    return cost


def scipy_optimum(cost: FloatArray) -> tuple[int, float]:
    feasible = np.isfinite(cost)
    if not feasible.any():
        return 0, 0.0
    max_matches = int(
        (maximum_bipartite_matching(csr_array(feasible), perm_type="column") >= 0).sum()
    )

    spread = float(cost[feasible].max() - cost[feasible].min())
    penalty = (spread + 1) * (min(cost.shape) + 1) + float(np.abs(cost[feasible]).max())
    rows, cols = linear_sum_assignment(np.where(feasible, cost, penalty))
    chosen = [(i, j) for i, j in zip(rows, cols, strict=True) if feasible[i, j]]
    assert len(chosen) == max_matches, "penalty too small: SciPy traded coverage for cost"
    return max_matches, math.fsum(float(cost[i, j]) for i, j in chosen)


def summarise(cost: FloatArray, pairs: list[tuple[int, int]]) -> tuple[int, float]:
    rows = [i for i, _ in pairs]
    cols = [j for _, j in pairs]
    assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
    assert all(np.isfinite(cost[i, j]) for i, j in pairs), "solver returned an infeasible pair"
    return len(pairs), math.fsum(float(cost[i, j]) for i, j in pairs)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize(("n", "m"), SHAPES)
def test_numpy_hungarian_matches_scipy(n: int, m: int, density: float, seed: int) -> None:
    cost = instance(n, m, density, seed)
    expected_matches, expected_cost = scipy_optimum(cost)

    matches, total = summarise(cost, hungarian_assignment_np(cost))

    assert matches == expected_matches
    assert total == pytest.approx(expected_cost, abs=1e-6)


@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize(("n", "m"), [(50, 80), (80, 50)])
def test_reference_hungarian_matches_scipy(n: int, m: int, density: float) -> None:
    cost = instance(n, m, density, seed=7)
    sparse = [[None if math.isinf(c) else c for c in row] for row in cost.tolist()]

    assert summarise(cost, hungarian_assignment(sparse)) == pytest.approx(scipy_optimum(cost))


@pytest.mark.parametrize("density", DENSITIES)
def test_greedy_never_beats_scipy(density: float) -> None:
    cost = instance(200, 300, density, seed=3)
    expected_matches, expected_cost = scipy_optimum(cost)
    sparse = [[None if math.isinf(c) else c for c in row] for row in cost.tolist()]

    matches, total = summarise(cost, greedy_assignment(sparse))

    assert matches <= expected_matches
    if matches == expected_matches:
        assert total >= expected_cost - 1e-6
