"""Admission control: refuse oversized requests before they cost CPU.

``PULSO_REQUEST_TIMEOUT_MS`` starts when the handler runs, i.e. *after* the body has been read,
JSON-decoded and validated on the event loop. A huge body therefore blocks the worker for as
long as parsing takes, and no timeout covers it. Two layers close that gap:

- :class:`BodySizeLimitMiddleware` counts body bytes as they arrive and stops reading at
  ``PULSO_MAX_BODY_BYTES`` (413). A declared ``Content-Length`` above the limit is refused before
  the first chunk is read; chunked bodies without one are cut off as soon as they cross it.
- :func:`enforce_size_limits` caps the number of orders and pulsers (422 on the offending
  field), which bounds solver work: Hungarian is O(n²·M).
"""

from __future__ import annotations

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PayloadTooLargeError(HTTPException):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(status_code=413, detail=f"request body exceeds {max_bytes} bytes")
        self.max_bytes = max_bytes


class BodySizeLimitMiddleware:
    """Pure ASGI middleware, so it never buffers the body itself.

    The error is raised from ``receive``; FastAPI re-raises ``HTTPException`` from body reading,
    so it reaches the problem+json handlers like any other error.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            if declared is not None and declared > self.max_bytes:
                raise PayloadTooLargeError(self.max_bytes)
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise PayloadTooLargeError(self.max_bytes)
            return message

        await self.app(scope, limited_receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def enforce_size_limits(field: str, count: int, limit: int) -> None:
    """Reject ``body.<field>`` with a pydantic-style ``too_long`` error past ``limit`` items."""
    if count > limit:
        raise RequestValidationError(
            [
                {
                    "type": "too_long",
                    "loc": ("body", field),
                    "msg": f"List should have at most {limit} items, not {count}",
                    "input": None,
                }
            ]
        )
