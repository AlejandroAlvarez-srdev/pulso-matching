"""OpenAPI request examples shown in ``/docs`` (and reused by the API tests)."""

from __future__ import annotations

from typing import Any

from fastapi.openapi.models import Example

_PULSERS: list[dict[str, Any]] = [
    {
        "id": "p-ana",
        "name": "Ana",
        "location": {"lat": 19.4361, "lng": -99.1371},
        "rating": 4.9,
        "activeTasks": 0,
        "maxTasks": 3,
        "vehicle": "MOTO",
        "online": True,
    },
    {
        "id": "p-luis",
        "name": "Luis",
        "location": {"lat": 19.4210, "lng": -99.1630},
        "rating": 4.6,
        "activeTasks": 2,
        "maxTasks": 3,
        "vehicle": "BIKE",
        "online": True,
    },
    {
        "id": "p-sofia",
        "name": "Sofía",
        "location": {"lat": 19.4270, "lng": -99.1677},
        "rating": 4.2,
        "activeTasks": 0,
        "maxTasks": 2,
        "vehicle": "CAR",
        "online": True,
    },
]

_ORDERS: list[dict[str, Any]] = [
    {
        "id": "0b6f1d0e-6a43-4f8e-9d55-1f3f5b8c2a01",
        "pickup": {"lat": 19.4326, "lng": -99.1332, "address": "Zócalo, Centro"},
    },
    {
        "id": "0b6f1d0e-6a43-4f8e-9d55-1f3f5b8c2a02",
        "pickup": {"lat": 19.4194, "lng": -99.1617, "address": "Plaza Río de Janeiro, Roma"},
    },
    {
        "id": "0b6f1d0e-6a43-4f8e-9d55-1f3f5b8c2a03",
        "pickup": {"lat": 19.4260, "lng": -99.1700, "address": "Glorieta de Insurgentes"},
    },
]

MATCH_EXAMPLES: dict[str, Example] = {
    "centro": Example(
        summary="Rank pulsers for a pickup in Centro",
        value={"order": _ORDERS[0], "pulsers": _PULSERS, "limit": 3},
    ),
    "full_order": Example(
        summary="Full Pulso Order payload (extra fields are ignored)",
        value={
            "order": _ORDERS[0]
            | {
                "merchantId": "m-taqueria-42",
                "status": "PAID",
                "dropoff": {"lat": 19.4120, "lng": -99.1710, "address": "Condesa"},
                "items": [{"sku": "TAC-01", "name": "Tacos al pastor", "qty": 4}],
                "amountCents": 18900,
                "currency": "MXN",
                "createdAt": "2026-11-03T18:30:00Z",
            },
            "pulsers": _PULSERS,
        },
    ),
}

BATCH_EXAMPLES: dict[str, Example] = {
    "hungarian": Example(
        summary="Optimal assignment of three orders",
        value={"orders": _ORDERS, "pulsers": _PULSERS, "strategy": "hungarian"},
    ),
    "greedy": Example(
        summary="Same instance with the greedy baseline",
        value={"orders": _ORDERS, "pulsers": _PULSERS, "strategy": "greedy"},
    ),
}
