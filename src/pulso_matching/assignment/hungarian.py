"""Hungarian algorithm (Kuhn-Munkres) with dual potentials, written from scratch.

Complexity
----------
For an ``n x m`` matrix with ``n <= m`` (the input is transposed otherwise) this runs in
**O(n² · m)** time — O(n³) for square inputs — and **O(n · m)** memory for the dense copy.
Each of the ``n`` rows is inserted with one Dijkstra-like sweep over the columns that keeps the
reduced costs ``c[i][j] - u[i] - v[j]`` non-negative; every inner iteration either finds a free
column or adds one column to the alternating tree, so a row costs at most ``m`` iterations of
O(m) work.

Infeasible pairs
----------------
``None`` entries are replaced by a penalty ``BIG`` strictly larger than any possible difference
in total finite cost. Minimising ``cost + BIG · (#infeasible pairs)`` is therefore lexicographic:
the solver first **maximises the number of feasible matches**, then minimises their total cost.
Penalised pairs are dropped from the result, which leaves those rows unassigned.
"""

from __future__ import annotations

import math

from pulso_matching.assignment.types import CostMatrix, Pairs


def hungarian_assignment(cost: CostMatrix) -> Pairs:
    """Optimal min-cost assignment; see the module docstring for guarantees and Big-O."""
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0:
        return []
    if any(len(row) != m for row in cost):
        raise ValueError("cost matrix must be rectangular")

    finite = [abs(c) for row in cost for c in row if c is not None]
    if not finite:
        return []
    big = (2 * max(finite) + 1) * (min(n, m) + 1)

    transposed = n > m
    if transposed:
        dense = [[_fill(cost[i][j], big) for i in range(n)] for j in range(m)]
        n, m = m, n
    else:
        dense = [[_fill(c, big) for c in row] for row in cost]

    match_of_col = _solve(dense, n, m)

    pairs: Pairs = []
    for col, row in enumerate(match_of_col):
        if row < 0:
            continue
        i, j = (col, row) if transposed else (row, col)
        if cost[i][j] is not None:
            pairs.append((i, j))
    pairs.sort()
    return pairs


def _fill(value: float | None, big: float) -> float:
    return big if value is None else value


def _solve(a: list[list[float]], n: int, m: int) -> list[int]:
    """Core O(n²·m) routine for ``n <= m``. Returns, per column, its row or -1.

    Indices are 1-based internally; column 0 is a virtual root of the alternating tree.
    """
    inf = math.inf
    u = [0.0] * (n + 1)  # row potentials
    v = [0.0] * (m + 1)  # column potentials
    p = [0] * (m + 1)  # p[j] = row matched to column j (0 = free)
    way = [0] * (m + 1)  # previous column on the shortest augmenting path

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            row = a[i0 - 1]
            ui0 = u[i0]
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - ui0 - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Augment along the path recorded in ``way``.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return [p[j] - 1 for j in range(1, m + 1)]
