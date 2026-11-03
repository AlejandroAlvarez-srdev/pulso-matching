"""Closed-loop HTTP load test for a running pulso-matching instance (asyncio + httpx).

``--concurrency`` workers each send a request, wait for the answer, and immediately send the
next one, for ``--duration`` seconds. Payloads are pre-generated so the client measures the
server, not JSON building. Prints a Markdown row with throughput and latency percentiles.

    uvicorn pulso_matching.main:app --workers 4 --no-access-log &
    python -m benchmarks.loadtest --endpoint batch --orders 50 --pulsers 75 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx

CENTER = (19.4326, -99.1332)
SPREAD_DEG = 0.15  # ≈ 33 km box, same as the dense benchmark
JSON_HEADERS = {"content-type": "application/json"}


@dataclass
class Stats:
    latencies_ms: list[float] = field(default_factory=list)
    statuses: Counter[int] = field(default_factory=Counter)
    degraded: int = 0
    transport_errors: int = 0


def _point(rng: random.Random) -> dict[str, float]:
    return {
        "lat": CENTER[0] + rng.uniform(-SPREAD_DEG, SPREAD_DEG),
        "lng": CENTER[1] + rng.uniform(-SPREAD_DEG, SPREAD_DEG),
    }


def _pulsers(rng: random.Random, m: int) -> list[dict[str, Any]]:
    out = []
    for k in range(m):
        max_tasks = rng.randint(1, 4)
        out.append(
            {
                "id": f"p{k}",
                "name": f"Pulser {k}",
                "location": _point(rng),
                "rating": round(rng.uniform(3, 5), 1),
                "activeTasks": rng.randint(0, max_tasks - 1),
                "maxTasks": max_tasks,
                "vehicle": rng.choice(["BIKE", "MOTO", "CAR"]),
                "online": True,
            }
        )
    return out


def _order(rng: random.Random) -> dict[str, Any]:
    return {"id": str(uuid.UUID(int=rng.getrandbits(128))), "pickup": _point(rng)}


def make_payloads(args: argparse.Namespace, count: int = 32) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    if args.endpoint == "match":
        return [
            {"order": _order(rng), "pulsers": _pulsers(rng, args.pulsers), "limit": 5}
            for _ in range(count)
        ]
    return [
        {
            "orders": [_order(rng) for _ in range(args.orders)],
            "pulsers": _pulsers(rng, args.pulsers),
            "strategy": args.strategy,
        }
        for _ in range(count)
    ]


async def worker(
    client: httpx.AsyncClient,
    path: str,
    payloads: list[bytes],
    stop_at: float,
    stats: Stats,
    offset: int,
) -> None:
    k = offset
    while time.perf_counter() < stop_at:
        payload = payloads[k % len(payloads)]
        k += 1
        started = time.perf_counter()
        try:
            res = await client.post(path, content=payload, headers=JSON_HEADERS)
        except httpx.TransportError:
            stats.transport_errors += 1
            continue
        stats.latencies_ms.append((time.perf_counter() - started) * 1000)
        stats.statuses[res.status_code] += 1
        if res.status_code == 200 and b'"degraded":true' in res.content:
            stats.degraded += 1


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, round(q / 100 * len(sorted_values)) - 1))
    return sorted_values[idx]


async def run_client(args: argparse.Namespace, concurrency: int) -> tuple[Stats, float]:
    path = "/match" if args.endpoint == "match" else "/match/batch"
    payloads = [json.dumps(p).encode() for p in make_payloads(args)]
    stats = Stats()
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=30, limits=limits) as client:
        (await client.get("/health")).raise_for_status()
        for p in payloads[:concurrency]:  # warm-up
            await client.post(path, content=p, headers=JSON_HEADERS)
        started = time.perf_counter()
        stop_at = started + args.duration
        await asyncio.gather(
            *(worker(client, path, payloads, stop_at, stats, offset=i) for i in range(concurrency))
        )
        return stats, time.perf_counter() - started


def client_process(args: argparse.Namespace, concurrency: int) -> tuple[Stats, float]:
    return asyncio.run(run_client(args, concurrency))


def run(args: argparse.Namespace) -> None:
    # A single Python client saturates one core around ~150 req/s of 30 KB bodies, so the
    # concurrency is spread over several client processes.
    shares = [
        args.concurrency // args.processes + (k < args.concurrency % args.processes)
        for k in range(args.processes)
    ]
    shares = [c for c in shares if c > 0]
    with ProcessPoolExecutor(max_workers=len(shares)) as pool:
        results = list(pool.map(client_process, [args] * len(shares), shares))

    stats = Stats()
    for part, _ in results:
        stats.latencies_ms.extend(part.latencies_ms)
        stats.statuses.update(part.statuses)
        stats.degraded += part.degraded
        stats.transport_errors += part.transport_errors
    elapsed = max(e for _, e in results)

    path = "/match" if args.endpoint == "match" else "/match/batch"
    lat = sorted(stats.latencies_ms)
    ok = stats.statuses.get(200, 0)
    errors = sum(stats.statuses.values()) - ok + stats.transport_errors
    size = f"1 x {args.pulsers}" if args.endpoint == "match" else f"{args.orders} x {args.pulsers}"
    print(
        f"| {path} | {size} | {args.strategy if args.endpoint == 'batch' else '-'} "
        f"| {args.concurrency} | {len(lat)} | {len(lat) / elapsed:.1f} "
        f"| {percentile(lat, 50):.1f} | {percentile(lat, 95):.1f} | {percentile(lat, 99):.1f} "
        f"| {statistics.fmean(lat) if lat else float('nan'):.1f} | {errors} | {stats.degraded} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop load test for pulso-matching.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", choices=["match", "batch"], default="batch")
    parser.add_argument("--strategy", choices=["hungarian", "greedy"], default="hungarian")
    parser.add_argument("--orders", type=int, default=50)
    parser.add_argument("--pulsers", type=int, default=75)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=15.0, help="seconds")
    parser.add_argument("--processes", type=int, default=4, help="client processes")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--header", action="store_true", help="print the table header first")
    args = parser.parse_args()
    if args.header:
        print(
            "| endpoint | size | strategy | concurrency | requests | req/s "
            "| p50 ms | p95 ms | p99 ms | mean ms | errors | degraded |"
        )
        print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    run(args)


if __name__ == "__main__":
    main()
