"""NumPy-vectorised Hungarian algorithm — same algorithm as :mod:`.hungarian`, faster constants.

The pure-Python module stays as the readable reference; this one is what the service runs.

What is vectorised
------------------
The potentials algorithm spends all its time in the inner sweep over columns: update ``minv``
with the reduced costs of the newest tree row, pick the free column with the smallest ``minv``,
then shift potentials by ``delta``. Each of those is an O(m) array operation here, so the
Python-level loop runs O(n · m) times (once per tree growth step) instead of O(n · m²)
scalar iterations.

Complexity is unchanged: **O(n² · m)** arithmetic for ``n <= m``, **O(n · m)** memory.

Input
-----
Either a :data:`CostMatrix` (``None`` = infeasible) or a float ndarray where ``inf``/``nan`` mark
infeasible pairs. Infeasible pairs get the same lexicographic ``BIG`` penalty as the reference
implementation: maximise the number of matches first, then minimise cost.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pulso_matching.assignment.types import CostMatrix, FloatMatrix, Pairs


class BudgetExceededError(Exception):
    """Raised when a ``deadline`` passes before the assignment is complete."""


def to_float_matrix(cost: CostMatrix | FloatMatrix) -> FloatMatrix:
    """Convert a ``None``-sparse matrix to float64 with ``inf`` for infeasible pairs."""
    if isinstance(cost, np.ndarray):
        return cost.astype(np.float64, copy=False)
    n = len(cost)
    m = len(cost[0]) if n else 0
    if any(len(row) != m for row in cost):
        raise ValueError("cost matrix must be rectangular")
    out = np.full((n, m), np.inf)
    for i, row in enumerate(cost):
        for j, c in enumerate(row):
            if c is not None:
                out[i, j] = c
    return out


def hungarian_assignment_np(cost: CostMatrix | FloatMatrix, deadline: float | None = None) -> Pairs:
    """Optimal min-cost assignment.

    ``deadline`` is a :func:`time.perf_counter` timestamp; it is checked once per inserted row
    (every O(n · m) work) and raises :class:`BudgetExceededError` when passed.
    """
    a = to_float_matrix(cost)
    if a.ndim != 2 or a.size == 0:
        return []
    feasible = np.isfinite(a)
    if not feasible.any():
        return []
    n, m = a.shape
    big = (2 * float(np.abs(a[feasible]).max()) + 1) * (min(n, m) + 1)
    dense = np.where(feasible, a, big)

    transposed = n > m
    if transposed:
        dense = dense.T
    match_of_col = _solve(np.ascontiguousarray(dense), deadline)

    pairs: Pairs = []
    for col, row in enumerate(match_of_col.tolist()):
        if row < 0:
            continue
        i, j = (col, row) if transposed else (row, col)
        if feasible[i, j]:
            pairs.append((i, j))
    pairs.sort()
    return pairs


def _solve(a: NDArray[np.float64], deadline: float | None) -> NDArray[np.intp]:
    n, m = a.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.intp)  # p[j] = row (1-based) matched to column j, 0 = free
    way = np.zeros(m + 1, dtype=np.intp)

    for i in range(1, n + 1):
        if deadline is not None and time.perf_counter() > deadline:
            raise BudgetExceededError(f"hungarian stopped after {i - 1}/{n} rows")
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used
            free[0] = False
            cur = a[i0 - 1] - u[i0] - v[1:]
            better = free[1:] & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            candidates = np.where(free[1:], minv[1:], np.inf)
            j1 = int(candidates.argmin()) + 1
            delta = candidates[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    return p[1:] - 1
