"""Batch assignment algorithms over a (possibly sparse) cost matrix.

A cost matrix is a list of rows (orders) of columns (pulsers); ``None`` marks an infeasible pair.
Both solvers return a list of ``(row, col)`` pairs where every pair is feasible, each row and each
column appears at most once.
"""

from pulso_matching.assignment.greedy import greedy_assignment
from pulso_matching.assignment.hungarian import hungarian_assignment
from pulso_matching.assignment.types import CostMatrix, Pairs

__all__ = ["CostMatrix", "Pairs", "greedy_assignment", "hungarian_assignment"]
