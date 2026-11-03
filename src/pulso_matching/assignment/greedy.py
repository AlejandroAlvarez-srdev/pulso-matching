"""Greedy baseline: repeatedly take the globally cheapest pair whose row and column are free.

Complexity
----------
With ``K`` feasible pairs (``K <= n · m``): building the heap is O(K) with ``heapify`` and each
pop is O(log K), so the worst case is **O(n · m · log(n · m))** time and O(K) memory. In
practice it stops as soon as ``min(n, m)`` pairs are matched.

Quality
-------
Greedy is a 2-approximation for maximum-weight matching but has no constant-factor guarantee
for min-cost assignment, and it does not maximise the number of matches: taking the cheapest
pair can steal the only feasible pulser of another order. ``benchmarks/`` measures the gap
against the Hungarian optimum.
"""

from __future__ import annotations

import heapq

import numpy as np

from pulso_matching.assignment.types import CostMatrix, FloatMatrix, Pairs


def greedy_assignment(cost: CostMatrix | FloatMatrix) -> Pairs:
    """Heap-based greedy assignment; see the module docstring for Big-O.

    Accepts a ``None``-sparse matrix or a float ndarray with ``inf``/``nan`` as infeasible.
    """
    heap: list[tuple[float, int, int]]
    if isinstance(cost, np.ndarray):
        ii, jj = np.nonzero(np.isfinite(cost))
        heap = list(zip(cost[ii, jj].tolist(), ii.tolist(), jj.tolist(), strict=True))
        n, m = cost.shape if cost.ndim == 2 else (0, 0)
    else:
        heap = [(c, i, j) for i, row in enumerate(cost) for j, c in enumerate(row) if c is not None]
        n = len(cost)
        m = max((len(row) for row in cost), default=0)
    heapq.heapify(heap)

    target = min(n, m)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: Pairs = []
    while heap and len(pairs) < target:
        _, i, j = heapq.heappop(heap)
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        pairs.append((i, j))
    pairs.sort()
    return pairs
