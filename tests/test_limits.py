"""Admission control: body-size limit (413) and order/pulser caps (422)."""

import json
from collections.abc import Iterator
from typing import Any

import anyio
import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from pulso_matching.config import Settings, get_settings
from pulso_matching.limits import BodySizeLimitMiddleware, PayloadTooLargeError
from pulso_matching.main import app, create_app
from pulso_matching.problems import PROBLEM_JSON
from tests.test_api import order, pulser

Headers = list[tuple[bytes, bytes]]


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def small_body_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An app whose body limit is 2 KB; the middleware reads it once, at startup."""
    monkeypatch.setenv("PULSO_MAX_BODY_BYTES", "2048")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as c:
            yield c
    finally:
        get_settings.cache_clear()


def batch_body(n_orders: int, n_pulsers: int) -> dict[str, Any]:
    return {
        "orders": [order(i * 0.1) for i in range(n_orders)],
        "pulsers": [pulser(f"p{i}", i * 0.1) for i in range(n_pulsers)],
    }


def test_body_over_limit_is_413_problem(small_body_client: TestClient) -> None:
    body = json.dumps(batch_body(5, 20)).encode()
    assert len(body) > 2048
    res = small_body_client.post(
        "/match/batch", content=body, headers={"content-type": "application/json"}
    )
    assert res.status_code == 413
    assert res.headers["content-type"] == PROBLEM_JSON
    assert res.headers["connection"] == "close"
    problem = res.json()
    assert problem["type"].endswith("/payload-too-large")
    assert problem["instance"] == "/match/batch"
    assert "2048" in problem["detail"]
    assert res.headers["x-request-id"]


def _drain(app: BodySizeLimitMiddleware, chunks: list[bytes], headers: Headers) -> int:
    """Run ``app`` over a streamed body; return how many chunks it pulled from the client."""
    pulled = 0

    async def receive() -> Message:
        nonlocal pulled
        pulled += 1
        more = pulled < len(chunks)
        return {"type": "http.request", "body": chunks[pulled - 1], "more_body": more}

    async def send(message: Message) -> None:
        raise AssertionError("the inner app should not respond")

    scope: Scope = {"type": "http", "method": "POST", "path": "/", "headers": headers}
    with pytest.raises(PayloadTooLargeError):
        anyio.run(app, scope, receive, send)
    return pulled


async def _read_everything(scope: Scope, receive: Receive, send: Send) -> None:
    while (await receive()).get("more_body", False):
        pass


def test_chunked_body_without_content_length_stops_at_the_limit() -> None:
    """No Content-Length: the limit is enforced while streaming, not trusted from a header."""
    limited = BodySizeLimitMiddleware(_read_everything, max_bytes=2048)
    chunks = [b"x" * 512] * 20
    # 2048 bytes is not over the limit; the 5th chunk (2560 bytes) is. Chunks 6-20 are never read.
    assert _drain(limited, chunks, headers=[]) == 5


def test_understated_content_length_is_still_counted() -> None:
    limited = BodySizeLimitMiddleware(_read_everything, max_bytes=2048)
    chunks = [b"x" * 512] * 20
    assert _drain(limited, chunks, headers=[(b"content-length", b"100")]) == 5


def test_overstated_content_length_is_refused_before_reading() -> None:
    limited = BodySizeLimitMiddleware(_read_everything, max_bytes=2048)
    assert _drain(limited, [b"{}"], headers=[(b"content-length", b"999999")]) == 0


def test_body_under_limit_passes(small_body_client: TestClient) -> None:
    res = small_body_client.post("/match/batch", json=batch_body(1, 2))
    assert res.status_code == 200


def test_get_requests_are_unaffected(small_body_client: TestClient) -> None:
    assert small_body_client.get("/health").status_code == 200


@pytest.mark.parametrize(
    ("limits", "n_orders", "n_pulsers", "field"),
    [
        ({"max_batch_orders": 3}, 4, 2, "orders"),
        ({"max_pulsers": 3}, 1, 4, "pulsers"),
    ],
)
def test_batch_over_count_limit_is_422_on_that_field(
    client: TestClient, limits: dict[str, int], n_orders: int, n_pulsers: int, field: str
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(**limits)  # type: ignore[arg-type]
    res = client.post("/match/batch", json=batch_body(n_orders, n_pulsers))
    assert res.status_code == 422
    assert res.headers["content-type"] == PROBLEM_JSON
    [error] = res.json()["errors"]
    assert error["loc"] == ["body", field]
    assert error["type"] == "too_long"
    assert "at most 3" in error["msg"]


def test_batch_at_count_limit_is_accepted(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_batch_orders=3, max_pulsers=3)
    res = client.post("/match/batch", json=batch_body(3, 3))
    assert res.status_code == 200
    assert len(res.json()["assignments"]) == 3


def test_match_caps_pulsers(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_pulsers=2)
    res = client.post(
        "/match", json={"order": order(), "pulsers": [pulser(f"p{i}") for i in range(3)]}
    )
    assert res.status_code == 422
    assert res.json()["errors"][0]["loc"] == ["body", "pulsers"]


def test_openapi_documents_413(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/match", "/match/batch"):
        assert "413" in paths[path]["post"]["responses"]
