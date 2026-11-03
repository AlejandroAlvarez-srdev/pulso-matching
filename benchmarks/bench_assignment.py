"""Benchmark: Hungarian vs greedy assignment, and grid pre-filter vs all-pairs scoring.

Generates synthetic instances in a square of ``--side-km`` centred on Mexico City
(``m = ratio · n`` pulsers with random rating, load and vehicle) and prints a Markdown table.

    python -m benchmarks.bench_assignment --sizes 50 100 200 400 --side-km 33 --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pulso_matching.assignment import (
    CostMatrix,
    Pairs,
    greedy_assignment,
    hungarian_assignment,
    hungarian_assignment_np,
)
from pulso_matching.config import Settings
from pulso_matching.logging_config import configure_logging
from pulso_matching.models import Location, Order, Place, Pulser, Strategy, Vehicle
from pulso_matching.scoring import Scorer
from pulso_matching.service import MatchingService

CENTER = (19.4326, -99.1332)
KM_PER_DEG = 111.195


@dataclass(frozen=True)
class Row:
    n: int
    m: int
    pairs: int
    grid_ms: float
    all_pairs_ms: float
    hungarian_ms: float
    hungarian_np_ms: float
    greedy_ms: float
    service_ms: float
    components: int
    hungarian_matched: int
    greedy_matched: int
    hungarian_cost: float
    greedy_cost: float


def make_instance(
    n: int, m: int, side_km: float, rng: random.Random
) -> tuple[list[Order], list[Pulser]]:
    half_lat = side_km / 2 / KM_PER_DEG
    half_lng = half_lat / math.cos(math.radians(CENTER[0]))

    def point() -> tuple[float, float]:
        return (
            CENTER[0] + rng.uniform(-half_lat, half_lat),
            CENTER[1] + rng.uniform(-half_lng, half_lng),
        )

    orders = []
    for _ in range(n):
        lat, lng = point()
        orders.append(Order(id=uuid.UUID(int=rng.getrandbits(128)), pickup=Place(lat=lat, lng=lng)))
    pulsers = []
    for k in range(m):
        lat, lng = point()
        max_tasks = rng.randint(1, 4)
        pulsers.append(
            Pulser(
                id=f"p{k}",
                name=f"Pulser {k}",
                location=Location(lat=lat, lng=lng),
                rating=round(rng.uniform(3.0, 5.0), 1),
                active_tasks=rng.randint(0, max_tasks - 1),
                max_tasks=max_tasks,
                vehicle=rng.choice(list(Vehicle)),
                online=True,
            )
        )
    return orders, pulsers


def timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000


def solve_cost(matrix: CostMatrix, pairs: Pairs) -> float:
    return math.fsum(c for i, j in pairs if (c := matrix[i][j]) is not None)


def _count_components(edges: Iterable[tuple[int, int]], n: int) -> int:
    """Connected components with at least one edge (orders without candidates excluded)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(n + j)
    return len({find(x) for x in list(parent)})


def run(n: int, ratio: float, side_km: float, rng: random.Random) -> Row:
    m = max(1, round(n * ratio))
    orders, pulsers = make_instance(n, m, side_km, rng)
    scorer = Scorer(Settings())
    service = MatchingService(scorer)

    edges, grid_ms = timed(lambda: service.candidate_edges(orders, pulsers))
    all_pairs, all_pairs_ms = timed(
        lambda: {
            (i, j): s
            for i, o in enumerate(orders)
            for j, p in enumerate(pulsers)
            if (s := scorer.score(o, p)) is not None
        }
    )
    assert set(edges) == set(all_pairs), "grid pre-filter must not change the candidate set"

    matrix: CostMatrix = [
        [e.cost if (e := edges.get((i, j))) else None for j in range(m)] for i in range(n)
    ]
    h_pairs, h_ms = timed(lambda: hungarian_assignment(matrix))
    np_pairs, np_ms = timed(lambda: hungarian_assignment_np(matrix))
    assert len(np_pairs) == len(h_pairs)
    assert math.isclose(solve_cost(matrix, np_pairs), solve_cost(matrix, h_pairs), abs_tol=1e-6)
    g_pairs, g_ms = timed(lambda: greedy_assignment(matrix))
    # End to end: grid + connected components + Hungarian per component.
    batch, service_ms = timed(lambda: service.assign_batch(orders, pulsers, Strategy.HUNGARIAN))
    assert len(batch.assignments) == len(h_pairs), "component split must keep the optimum"
    return Row(
        n=n,
        m=m,
        pairs=len(edges),
        grid_ms=grid_ms,
        all_pairs_ms=all_pairs_ms,
        hungarian_ms=h_ms,
        hungarian_np_ms=np_ms,
        greedy_ms=g_ms,
        service_ms=service_ms,
        components=_count_components(edges, n),
        hungarian_matched=len(h_pairs),
        greedy_matched=len(g_pairs),
        hungarian_cost=solve_cost(matrix, h_pairs),
        greedy_cost=solve_cost(matrix, g_pairs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 400])
    parser.add_argument("--ratio", type=float, default=1.5, help="pulsers per order")
    parser.add_argument("--side-km", type=float, default=33.0, help="side of the square area")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging("WARNING")
    rng = random.Random(args.seed)
    print(
        "| orders | pulsers | feasible pairs | components | grid ms | all-pairs ms "
        "| hungarian py ms | hungarian np ms | greedy ms | service ms "
        "| matched H / G | avg cost H / G | greedy gap |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|")
    for n in args.sizes:
        r = run(n, args.ratio, args.side_km, rng)
        avg_h = r.hungarian_cost / max(r.hungarian_matched, 1)
        avg_g = r.greedy_cost / max(r.greedy_matched, 1)
        gap = (avg_g - avg_h) / avg_h * 100 if avg_h else 0.0
        print(
            f"| {r.n} | {r.m} | {r.pairs} | {r.components} | {r.grid_ms:.1f} "
            f"| {r.all_pairs_ms:.1f} | {r.hungarian_ms:.1f} | {r.hungarian_np_ms:.1f} "
            f"| {r.greedy_ms:.1f} "
            f"| {r.service_ms:.1f} "
            f"| {r.hungarian_matched} / {r.greedy_matched} "
            f"| {avg_h:.4f} / {avg_g:.4f} | {gap:+.1f} % |"
        )


if __name__ == "__main__":
    main()
