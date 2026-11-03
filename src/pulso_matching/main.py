"""FastAPI application: ``POST /match``, ``POST /match/batch``, ``GET /health``."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

import structlog
from fastapi import Body, Depends, FastAPI, Request, Response

from pulso_matching import __version__
from pulso_matching.config import Settings, get_settings
from pulso_matching.examples import BATCH_EXAMPLES, MATCH_EXAMPLES
from pulso_matching.logging_config import configure_logging
from pulso_matching.models import (
    BatchMatchRequest,
    BatchMatchResponse,
    MatchRequest,
    MatchResponse,
)
from pulso_matching.problems import install_problem_handlers, problem_responses
from pulso_matching.scoring import Scorer
from pulso_matching.service import MatchingService
from pulso_matching.timeouts import deadline_in, run_with_timeout

log = structlog.get_logger(__name__)


SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_service(settings: SettingsDep) -> MatchingService:
    return MatchingService(Scorer(settings), hungarian_budget_ms=settings.hungarian_budget_ms)


ServiceDep = Annotated[MatchingService, Depends(get_service)]


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app = FastAPI(
        title="Pulso matching",
        version=__version__,
        description="Assigns Pulso orders to pulsers (fictional company).",
    )

    install_problem_handlers(app)

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["x-request-id"] = request_id
        log.info(
            "http.request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post(
        "/match",
        response_model=MatchResponse,
        response_model_by_alias=True,
        responses=problem_responses(422, 504),
    )
    async def match(
        body: Annotated[MatchRequest, Body(openapi_examples=MATCH_EXAMPLES)],
        service: ServiceDep,
        settings: SettingsDep,
    ) -> MatchResponse:
        """Rank pulsers for one order (best first)."""
        candidates = await run_with_timeout(
            lambda: service.rank(body.order, body.pulsers, body.limit), settings.request_timeout_ms
        )
        return MatchResponse(order_id=body.order.id, candidates=candidates)

    @app.post(
        "/match/batch",
        response_model=BatchMatchResponse,
        response_model_by_alias=True,
        responses=problem_responses(422, 504),
    )
    async def match_batch(
        body: Annotated[BatchMatchRequest, Body(openapi_examples=BATCH_EXAMPLES)],
        service: ServiceDep,
        settings: SettingsDep,
    ) -> BatchMatchResponse:
        """Assign N orders to M pulsers (capacity-aware) minimising total cost.

        Hungarian runs under ``PULSO_HUNGARIAN_BUDGET_MS``; past it the batch falls back to
        greedy and ``degraded`` is true. Past ``PULSO_REQUEST_TIMEOUT_MS`` the request fails
        with a 504 problem.
        """
        deadline = deadline_in(settings.request_timeout_ms)
        result = await run_with_timeout(
            lambda: service.assign_batch(body.orders, body.pulsers, body.strategy, deadline),
            settings.request_timeout_ms,
        )
        return BatchMatchResponse(
            strategy=body.strategy,
            assignments=result.assignments,
            unassigned_order_ids=result.unassigned_order_ids,
            total_score=result.total_score,
            elapsed_ms=result.elapsed_ms,
            degraded=result.degraded,
            degraded_components=result.degraded_components,
        )

    return app


app = create_app()
