"""Prometheus metrics, exposed at ``GET /metrics``.

A dedicated registry (not the process-global default) keeps the exposition limited to what this
service defines and makes the app safe to construct more than once in tests. Route labels use
the path *template* so cardinality stays bounded.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()

REQUEST_LATENCY = Histogram(
    "pulso_http_request_duration_seconds",
    "HTTP request latency by route template, method and status.",
    ["method", "route", "status"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)
ASSIGNMENTS = Counter(
    "pulso_assignments",
    "Orders assigned to a pulser by /match/batch.",
    ["strategy"],
    registry=REGISTRY,
)
UNASSIGNED = Counter(
    "pulso_unassigned_orders",
    "Orders /match/batch could not assign (no feasible pulser).",
    ["strategy"],
    registry=REGISTRY,
)
DEGRADED_BATCHES = Counter(
    "pulso_degraded_batches",
    "Batches where the Hungarian budget ran out and greedy finished the job.",
    registry=REGISTRY,
)
BATCH_SIZE = Histogram(
    "pulso_batch_orders",
    "Orders per /match/batch request.",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500),
    registry=REGISTRY,
)
