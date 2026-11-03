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

Capacity-aware columns
----------------------
A pulser with ``s = min(maxTasks - activeTasks, max_orders_per_pulser_per_batch)`` free slots is
expanded into ``s`` columns. Column ``k`` is "this pulser's (k+1)-th order in the batch": its load
term uses ``activeTasks + k`` and its ETA waits ``k · stacked_order_delay_minutes``. Costs are
non-decreasing in ``k`` for every order (plus a ``k · 1e-9`` tie-breaker), so an optimal solution
always fills a pulser's slots in order, and a standard one-to-one assignment solves the
many-to-one problem. The price is a wider matrix: ``m`` becomes ``Σ slots``, which Hungarian pays
linearly (O(n² · Σ slots)).

Time budget and graceful degradation
------------------------------------
Hungarian is O(n² · m); greedy is O(nm log nm) and typically 5-10x faster at dispatch sizes.
With ``hungarian_budget_ms`` set, the Hungarian solver gets a wall-clock deadline (also capped by
the caller's request deadline). Components are solved smallest first, so the cheap ones get the
optimum; if the deadline passes mid-solve, that component and every remaining one are solved
greedily **from the same cached cost matrix**, and the result is flagged ``degraded``. A slightly
worse assignment returned in time beats an optimal one the dispatcher has already given up on.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
import structlog

from pulso_matching.assignment import (
    BudgetExceededError,
    FloatMatrix,
    greedy_assignment,
    hungarian_assignment_np,
)
from pulso_matching.models import Assignment, Candidate, Order, Pulser, Strategy
from pulso_matching.scoring import PairTerms, Scorer
from pulso_matching.spatial import GridIndex
from pulso_matching.timeouts import check_deadline

log = structlog.get_logger(__name__)

SLOT_TIE_BREAK = 1e-9


@dataclass(frozen=True, slots=True)
class BatchResult:
    assignments: list[Assignment]
    unassigned_order_ids: list[UUID]
    total_score: float
    elapsed_ms: float
    degraded_components: int = 0

    @property
    def degraded(self) -> bool:
        return self.degraded_components > 0


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
    def __init__(self, scorer: Scorer, hungarian_budget_ms: float | None = None) -> None:
        self._scorer = scorer
        self._hungarian_budget_ms = hungarian_budget_ms

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
        self, orders: list[Order], pulsers: list[Pulser], deadline: float | None = None
    ) -> dict[tuple[int, int], PairTerms]:
        """Feasible (order index, pulser index) pairs with their cached slot-independent terms.

        The grid query already returns the haversine distance, so each pair's distance is
        computed exactly once per batch.
        """
        radius = self._scorer.max_radius_km
        index: GridIndex[int] = GridIndex(cell_km=radius)
        for j, p in enumerate(pulsers):
            if self._scorer.is_eligible(p):
                index.insert(j, p.location.lat, p.location.lng)
        edges: dict[tuple[int, int], PairTerms] = {}
        for i, order in enumerate(orders):
            check_deadline(deadline)
            for j, distance in index.query(order.pickup.lat, order.pickup.lng, radius):
                edges[(i, j)] = self._scorer.terms(pulsers[j], distance)
        return edges

    def _component_matrix(
        self,
        rows: list[int],
        pulser_cols: list[int],
        adjacency: dict[int, list[int]],
        edges: dict[tuple[int, int], PairTerms],
        pulsers: list[Pulser],
    ) -> tuple[FloatMatrix, list[tuple[int, int]]]:
        """Dense slot-expanded cost matrix for one component, built from the cached terms.

        One vectorised pass per slot level fills every pair's column for that slot: O(E_c · S)
        for E_c component edges and S max slots, instead of re-scoring each (pair, slot).
        """
        slots = [self._scorer.batch_slots(pulsers[j]) for j in pulser_cols]
        col_start = np.concatenate(([0], np.cumsum(slots)[:-1])).astype(np.intp)
        cols = [
            (j, k) for j, n_slots in zip(pulser_cols, slots, strict=True) for k in range(n_slots)
        ]
        pcol_of = {j: c for c, j in enumerate(pulser_cols)}

        r_idx: list[int] = []
        c_idx: list[int] = []
        for r, i in enumerate(rows):
            for j in adjacency.get(i, ()):
                r_idx.append(r)
                c_idx.append(pcol_of[j])
        r_arr = np.asarray(r_idx, dtype=np.intp)
        c_arr = np.asarray(c_idx, dtype=np.intp)
        pair_terms = [edges[(rows[r], pulser_cols[c])] for r, c in zip(r_idx, c_idx, strict=True)]
        static = np.fromiter((t.static_cost for t in pair_terms), np.float64, len(pair_terms))
        base_eta = np.fromiter(
            (t.base_eta_minutes for t in pair_terms), np.float64, len(pair_terms)
        )
        active = np.asarray([pulsers[j].active_tasks for j in pulser_cols], np.float64)[c_arr]
        max_tasks = np.asarray([pulsers[j].max_tasks for j in pulser_cols], np.float64)[c_arr]
        n_slots = np.asarray(slots, dtype=np.intp)[c_arr]

        matrix = np.full((len(rows), len(cols)), np.inf)
        for k in range(max(slots, default=0)):
            mask = n_slots > k
            costs = self._scorer.slot_costs(
                static[mask], base_eta[mask], active[mask], max_tasks[mask], k
            )
            matrix[r_arr[mask], col_start[c_arr[mask]] + k] = costs + k * SLOT_TIE_BREAK
        return matrix, cols

    def assign_batch(
        self,
        orders: list[Order],
        pulsers: list[Pulser],
        strategy: Strategy,
        deadline: float | None = None,
    ) -> BatchResult:
        """Assign ``orders`` to ``pulsers``.

        ``deadline`` is a :func:`time.perf_counter` timestamp after which nobody is waiting for
        the answer: it caps the Hungarian budget (see the module docstring), and once it passes
        the batch stops with :class:`DeadlineExceededError` instead of finishing work that would
        be thrown away.
        """
        started = time.perf_counter()
        hungarian_deadline = deadline
        if self._hungarian_budget_ms is not None:
            budget_end = started + self._hungarian_budget_ms / 1000
            hungarian_deadline = min(budget_end, deadline or budget_end)
        n = len(orders)
        edges = self.candidate_edges(orders, pulsers, deadline)
        adjacency: dict[int, list[int]] = defaultdict(list)
        for i, j in edges:
            adjacency[i].append(j)

        # Union-find over n order nodes followed by m pulser nodes.
        uf = _UnionFind(n + len(pulsers))
        for i, j in edges:
            uf.union(i, n + j)
        components: defaultdict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for i in range(n):
            components[uf.find(i)][0].append(i)
        for j in sorted({j for _, j in edges}):
            components[uf.find(n + j)][1].append(j)

        now = datetime.now(UTC)
        assignments: list[Assignment] = []
        degraded = 0
        use_greedy = strategy is Strategy.GREEDY
        for rows, pulser_cols in sorted(
            components.values(), key=lambda rc: len(rc[0]) * len(rc[1])
        ):
            if not pulser_cols:
                continue
            check_deadline(deadline)
            matrix, cols = self._component_matrix(rows, pulser_cols, adjacency, edges, pulsers)
            pairs = None
            if not use_greedy:
                try:
                    pairs = hungarian_assignment_np(matrix, deadline=hungarian_deadline)
                except BudgetExceededError:
                    check_deadline(deadline)  # the request is gone: don't even run greedy
                    use_greedy = True
            if pairs is None:
                pairs = greedy_assignment(matrix)
                degraded += strategy is Strategy.HUNGARIAN
            for r, c in pairs:
                i, (j, k) = rows[r], cols[c]
                pair = self._scorer.slot_score(edges[(i, j)], pulsers[j], k)
                assert pair is not None  # the matrix entry was finite
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
        if degraded:
            log.warning(
                "batch.degraded",
                budget_ms=self._hungarian_budget_ms,
                degraded_components=degraded,
                components=len(components),
            )
        log.info(
            "batch.matched",
            strategy=strategy.value,
            orders=n,
            pulsers=len(pulsers),
            candidate_pairs=len(edges),
            components=len(components),
            assigned=len(assignments),
            unassigned=len(unassigned),
            degraded_components=degraded,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return BatchResult(
            assignments=assignments,
            unassigned_order_ids=unassigned,
            total_score=round(sum(a.score for a in assignments), 6),
            elapsed_ms=round(elapsed_ms, 3),
            degraded_components=degraded,
        )
