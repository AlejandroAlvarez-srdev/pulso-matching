from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from pulso_matching.config import Settings, get_settings
from pulso_matching.main import app

KM = 1 / 111.195
CDMX = (19.4326, -99.1332)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def pulser(pid: str, dlat_km: float = 0.0, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": pid,
        "name": pid.title(),
        "location": {"lat": CDMX[0] + dlat_km * KM, "lng": CDMX[1]},
        "rating": 4.5,
        "activeTasks": 0,
        "maxTasks": 3,
        "vehicle": "MOTO",
        "online": True,
    }
    body.update(overrides)
    return body


def order(dlat_km: float = 0.0) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "pickup": {"lat": CDMX[0] + dlat_km * KM, "lng": CDMX[1], "address": "Zócalo"},
    }


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_request_id_is_echoed(client: TestClient) -> None:
    res = client.get("/health", headers={"x-request-id": "abc123"})
    assert res.headers["x-request-id"] == "abc123"


def test_match_ranks_candidates(client: TestClient) -> None:
    o = order()
    res = client.post(
        "/match",
        json={
            "order": o,
            "pulsers": [pulser("far", 5), pulser("near", 1), pulser("off", 0, online=False)],
            "limit": 5,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["orderId"] == o["id"]
    assert [c["pulserId"] for c in body["candidates"]] == ["near", "far"]
    first = body["candidates"][0]
    assert set(first) == {"pulserId", "score", "etaMinutes", "distanceKm"}
    assert first["distanceKm"] == pytest.approx(1.0, abs=0.01)


def test_match_accepts_full_order_payload(client: TestClient) -> None:
    full = order() | {
        "merchantId": "m-1",
        "status": "PAID",
        "dropoff": {"lat": 19.44, "lng": -99.14, "address": "Roma"},
        "items": [{"sku": "A1", "name": "Tacos", "qty": 2}],
        "amountCents": 25000,
        "currency": "MXN",
        "createdAt": "2026-11-03T12:00:00Z",
    }
    res = client.post("/match", json={"order": full, "pulsers": [pulser("p")]})
    assert res.status_code == 200
    assert len(res.json()["candidates"]) == 1


@pytest.mark.parametrize(
    "bad",
    [
        pulser("p", rating=6),
        pulser("p", maxTasks=0),
        pulser("p", vehicle="PLANE"),
        pulser("p", location={"lat": 91, "lng": 0}),
    ],
)
def test_match_validates_pulsers(client: TestClient, bad: dict[str, Any]) -> None:
    res = client.post("/match", json={"order": order(), "pulsers": [bad]})
    assert res.status_code == 422


@pytest.mark.parametrize("strategy", ["hungarian", "greedy"])
def test_batch_assigns_each_pulser_once(client: TestClient, strategy: str) -> None:
    orders = [order(0), order(2), order(30)]
    res = client.post(
        "/match/batch",
        json={
            "orders": orders,
            "pulsers": [pulser("a", 0.5), pulser("b", 2.5)],
            "strategy": strategy,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["strategy"] == strategy
    assert {a["pulserId"] for a in body["assignments"]} == {"a", "b"}
    assert body["unassignedOrderIds"] == [orders[2]["id"]]
    assert body["totalScore"] == pytest.approx(sum(a["score"] for a in body["assignments"]))
    assert {"orderId", "pulserId", "score", "etaMinutes", "assignedAt"} <= set(
        body["assignments"][0]
    )


def test_batch_defaults_to_hungarian(client: TestClient) -> None:
    res = client.post("/match/batch", json={"orders": [order()], "pulsers": [pulser("p")]})
    assert res.json()["strategy"] == "hungarian"


def test_batch_rejects_duplicate_ids(client: TestClient) -> None:
    o = order()
    res = client.post("/match/batch", json={"orders": [o, o], "pulsers": [pulser("p")]})
    assert res.status_code == 422
    res = client.post(
        "/match/batch", json={"orders": [order()], "pulsers": [pulser("p"), pulser("p")]}
    )
    assert res.status_code == 422


def test_weights_come_from_settings(client: TestClient) -> None:
    # Rating-only weights: a far 5-star pulser beats a near 3-star one.
    app.dependency_overrides[get_settings] = lambda: Settings(
        weight_distance=0, weight_eta=0, weight_rating=1, weight_load=0
    )
    res = client.post(
        "/match",
        json={
            "order": order(),
            "pulsers": [pulser("near", 0.1, rating=3), pulser("star", 6, rating=5)],
        },
    )
    assert [c["pulserId"] for c in res.json()["candidates"]] == ["star", "near"]
