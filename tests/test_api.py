import time
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.openapi.models import Example
from fastapi.testclient import TestClient

from pulso_matching.config import Settings, get_settings
from pulso_matching.examples import BATCH_EXAMPLES, MATCH_EXAMPLES
from pulso_matching.main import app, get_service
from pulso_matching.problems import PROBLEM_JSON
from pulso_matching.scoring import Scorer
from pulso_matching.service import BatchResult, MatchingService

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


def test_validation_errors_are_problem_json(client: TestClient) -> None:
    res = client.post("/match", json={"order": order(), "pulsers": [pulser("p", rating=6)]})
    assert res.status_code == 422
    assert res.headers["content-type"] == PROBLEM_JSON
    body = res.json()
    assert body["type"].endswith("/validation")
    assert body["status"] == 422
    assert body["instance"] == "/match"
    assert body["errors"][0]["loc"] == ["body", "pulsers", 0, "rating"]


@pytest.mark.parametrize(
    ("method", "path", "status"), [("GET", "/nope", 404), ("GET", "/match", 405)]
)
def test_http_errors_are_problem_json(
    client: TestClient, method: str, path: str, status: int
) -> None:
    res = client.request(method, path)
    assert res.status_code == status
    assert res.headers["content-type"] == PROBLEM_JSON
    assert res.json()["status"] == status
    assert res.json()["type"] == "about:blank"


def test_unhandled_errors_are_problem_json() -> None:
    def broken() -> MatchingService:
        raise RuntimeError("boom")

    app.dependency_overrides[get_service] = broken
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            res = c.post("/match", json={"order": order(), "pulsers": []})
    finally:
        app.dependency_overrides.clear()
    assert res.status_code == 500
    assert res.headers["content-type"] == PROBLEM_JSON
    assert "boom" not in res.text


def test_openapi_documents_examples_and_problem_model(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    batch = spec["paths"]["/match/batch"]["post"]
    assert set(batch["requestBody"]["content"]["application/json"]["examples"]) == {
        "hungarian",
        "greedy",
    }
    assert PROBLEM_JSON in batch["responses"]["422"]["content"]
    assert "Problem" in spec["components"]["schemas"]


@pytest.mark.parametrize(
    ("path", "examples"), [("/match", MATCH_EXAMPLES), ("/match/batch", BATCH_EXAMPLES)]
)
def test_openapi_examples_are_valid_requests(
    client: TestClient, path: str, examples: dict[str, Example]
) -> None:
    for example in examples.values():
        res = client.post(path, json=example.get("value"))
        assert res.status_code == 200, res.text


def test_batch_reports_degradation_when_budget_is_exhausted(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(hungarian_budget_ms=1e-9)
    res = client.post("/match/batch", json={"orders": [order()], "pulsers": [pulser("p")]})
    body = res.json()
    assert res.status_code == 200
    assert body["degraded"] is True
    assert body["degradedComponents"] == 1
    assert len(body["assignments"]) == 1


def test_slow_request_returns_504_problem(client: TestClient) -> None:
    class SlowService(MatchingService):
        def assign_batch(self, *args: Any, **kwargs: Any) -> BatchResult:
            time.sleep(0.5)
            return super().assign_batch(*args, **kwargs)

    settings = Settings(request_timeout_ms=50)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_service] = lambda: SlowService(Scorer(settings))
    started = time.perf_counter()
    res = client.post("/match/batch", json={"orders": [order()], "pulsers": [pulser("p")]})
    assert time.perf_counter() - started < 0.4
    assert res.status_code == 504
    assert res.headers["content-type"] == PROBLEM_JSON
    assert res.json()["type"].endswith("/timeout")
