"""Matching use-cases: rank candidates for one order, assign a batch of orders.

Batch pipeline and cost
-----------------------
1. Keep eligible pulsers (online, spare capacity) and index them in a :class:`GridIndex` — O(m).
2. For each order, score only pulsers within ``max_radius_km`` — O(n · k), ``k`` = neighbours.
3. Split the bipartite candidate graph into connected components with union-find — ~O(E).
   Orders in different components never compete for the same pulser, so each component is
   solved independently. Hungarian then costs Σ O(nᵢ² · mᵢ) instead of O(n² · m): a city with
   isolated neighbourhoods is many small problems, not one big one.
4. Solve each component with the selected strategy and map indices back to ids.

Each pulser takes at most one order per batch: a batch is one dispatch tick, and their load
(and therefore score) changes once they accept.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import structlog

from pulso_matching.assignment import CostMatrix, greedy_assignment, hungarian_assignment_np
from pulso_matching.models import Assignment, Candidate, Order, Pulser, Strategy
from pulso_matching.scoring import PairScore, Scorer
from pulso_matching.spatial import GridIndex

log = structlog.get_logger(__name__)

SOLVERS = {Strategy.HUNGARIAN: hungarian_assignment_np, Strategy.GREEDY: greedy_assignment}


@dataclass(frozen=True, slots=True)
class BatchResult:
    assignments: list[Assignment]
    unassigned_order_ids: list[UUID]
    total_score: float
    elapsed_ms: float


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path halving
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self._parent[self.find(a)] = self.find(b)


class MatchingService:
    def __init__(self, scorer: Scorer) -> None:
        self._scorer = scorer

    def rank(self, order: Order, pulsers: list[Pulser], limit: int) -> list[Candidate]:
        """Top-``limit`` candidates for one order. O(m log limit) with a bounded heap."""
        scored = ((s, p) for p in pulsers if (s := self._scorer.score(order, p)) is not None)
        best = heapq.nsmallest(limit, scored, key=lambda sp: (sp[0].cost, sp[1].id))
        return [
            Candidate(
                pulser_id=p.id,
                score=round(s.score, 6),
                eta_minutes=round(s.eta_minutes, 2),
                distance_km=round(s.distance_km, 3),
            )
            for s, p in best
        ]

    def candidate_edges(
        self, orders: list[Order], pulsers: list[Pulser]
    ) -> dict[tuple[int, int], PairScore]:
        """Feasible (order index, pulser index) pairs, found through the grid index."""
        radius = self._scorer.max_radius_km
        index: GridIndex[int] = GridIndex(cell_km=radius)
        for j, p in enumerate(pulsers):
            if self._scorer.is_eligible(p):
                index.insert(j, p.location.lat, p.location.lng)
        edges: dict[tuple[int, int], PairScore] = {}
        for i, order in enumerate(orders):
            for j, _ in index.query(order.pickup.lat, order.pickup.lng, radius):
                score = self._scorer.score(order, pulsers[j])
                if score is not None:
                    edges[(i, j)] = score
        return edges

    def assign_batch(
        self, orders: list[Order], pulsers: list[Pulser], strategy: Strategy
    ) -> BatchResult:
        started = time.perf_counter()
        n = len(orders)
        edges = self.candidate_edges(orders, pulsers)

        # Union-find over n order nodes followed by m pulser nodes.
        uf = _UnionFind(n + len(pulsers))
        for i, j in edges:
            uf.union(i, n + j)
        components: defaultdict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for i in range(n):
            components[uf.find(i)][0].append(i)
        for j in sorted({j for _, j in edges}):
            components[uf.find(n + j)][1].append(j)

        solve = SOLVERS[strategy]
        now = datetime.now(UTC)
        assignments: list[Assignment] = []
        for rows, cols in components.values():
            if not cols:
                continue
            matrix: CostMatrix = [
                [e.cost if (e := edges.get((i, j))) else None for j in cols] for i in rows
            ]
            for r, c in solve(matrix):
                i, j = rows[r], cols[c]
                pair = edges[(i, j)]
                assignments.append(
                    Assignment(
                        order_id=orders[i].id,
                        pulser_id=pulsers[j].id,
                        score=round(pair.score, 6),
                        eta_minutes=round(pair.eta_minutes, 2),
                        assigned_at=now,
                    )
                )

        assigned = {a.order_id for a in assignments}
        unassigned = [o.id for o in orders if o.id not in assigned]
        order_rank = {o.id: k for k, o in enumerate(orders)}
        assignments.sort(key=lambda a: order_rank[a.order_id])
        elapsed_ms = (time.perf_counter() - started) * 1000

        for a in assignments:
            log.debug("pulser.assigned", order_id=str(a.order_id), pulser_id=a.pulser_id)
        for order_id in unassigned:
            log.info("assignment.failed", order_id=str(order_id), reason="no_feasible_pulser")
        log.info(
            "batch.matched",
            strategy=strategy.value,
            orders=n,
            pulsers=len(pulsers),
            candidate_pairs=len(edges),
            components=len(components),
            assigned=len(assignments),
            unassigned=len(unassigned),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return BatchResult(
            assignments=assignments,
            unassigned_order_ids=unassigned,
            total_score=round(sum(a.score for a in assignments), 6),
            elapsed_ms=round(elapsed_ms, 3),
        )
