"""RFC 9457 ``application/problem+json`` error responses.

Every error the service returns — validation, unknown route, wrong method, timeout, crash — has
the same machine-readable shape, so clients branch on ``type``/``status`` instead of parsing
FastAPI's default ``{"detail": ...}`` variants.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from pulso_matching.timeouts import RequestTimeoutError

PROBLEM_JSON = "application/problem+json"
PROBLEM_BASE = "https://errors.pulso.example/"

log = structlog.get_logger(__name__)


class FieldError(BaseModel):
    loc: list[str | int] = Field(description="Path to the offending field.")
    msg: str
    type: str


class Problem(BaseModel):
    """Problem Details for HTTP APIs (RFC 9457)."""

    type: str = Field(default="about:blank", description="URI identifying the problem type.")
    title: str
    status: int
    detail: str | None = None
    instance: str | None = Field(default=None, description="Request path that failed.")
    errors: list[FieldError] | None = Field(
        default=None, description="Per-field validation errors (422 only)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": PROBLEM_BASE + "validation",
                    "title": "Request validation failed",
                    "status": 422,
                    "detail": "1 invalid field",
                    "instance": "/match",
                    "errors": [
                        {
                            "loc": ["body", "pulsers", 0, "rating"],
                            "msg": "Input should be less than or equal to 5",
                            "type": "less_than_equal",
                        }
                    ],
                }
            ]
        }
    }


def problem_response(problem: Problem, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        problem.model_dump(exclude_none=True),
        status_code=problem.status,
        media_type=PROBLEM_JSON,
        headers=headers,
    )


def problem_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses=`` entries documenting problem+json bodies for ``statuses``."""
    schema = {"$ref": "#/components/schemas/Problem"}
    return {
        status: {
            "model": Problem,
            "description": HTTPStatus(status).phrase,
            "content": {PROBLEM_JSON: {"schema": schema}},
        }
        for status in statuses
    }


def install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            FieldError(loc=list(e.get("loc", ())), msg=str(e.get("msg", "")), type=e["type"])
            for e in exc.errors()
        ]
        return problem_response(
            Problem(
                type=PROBLEM_BASE + "validation",
                title="Request validation failed",
                status=422,
                detail=f"{len(errors)} invalid field{'s' if len(errors) != 1 else ''}",
                instance=request.url.path,
                errors=errors,
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem_response(
            Problem(
                title=HTTPStatus(exc.status_code).phrase,
                status=exc.status_code,
                detail=str(exc.detail) if exc.detail else None,
                instance=request.url.path,
            ),
            headers=dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(RequestTimeoutError)
    async def on_timeout(request: Request, exc: RequestTimeoutError) -> JSONResponse:
        log.warning("http.timeout", path=request.url.path, timeout_ms=exc.timeout_ms)
        return problem_response(
            Problem(
                type=PROBLEM_BASE + "timeout",
                title="Matching timed out",
                status=504,
                detail=str(exc),
                instance=request.url.path,
            )
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("http.unhandled_error", path=request.url.path)
        return problem_response(
            Problem(
                type=PROBLEM_BASE + "internal",
                title="Internal Server Error",
                status=500,
                detail="Unexpected error; see logs with this request id.",
                instance=request.url.path,
            )
        )
