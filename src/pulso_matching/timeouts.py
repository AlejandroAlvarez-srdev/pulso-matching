"""Per-request timeout for CPU-bound handlers.

The solvers are synchronous, so they run in a worker thread. ``fail_after`` bounds how long the
request waits; on expiry the thread is *abandoned* (Python threads cannot be killed) and the
client gets a 504. To stop that thread from burning CPU — and, through the GIL, from slowing
every other request, including the event loop that has to send the 504 — handlers pass the same
deadline into the service. It checks it once per order while scoring candidates, before each
component, and once per row inside Hungarian, and raises :class:`DeadlineExceededError`. Only a
single greedy pass (the cheap part) is not interruptible.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import anyio
import anyio.to_thread


class DeadlineExceededError(Exception):
    """The caller's deadline passed; any result would be thrown away, so stop working."""


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise DeadlineExceededError


class RequestTimeoutError(Exception):
    def __init__(self, timeout_ms: float) -> None:
        super().__init__(f"request exceeded {timeout_ms:g} ms")
        self.timeout_ms = timeout_ms


def deadline_in(timeout_ms: float) -> float:
    """A :func:`time.perf_counter` deadline ``timeout_ms`` from now."""
    return time.perf_counter() + timeout_ms / 1000


async def run_with_timeout[T](fn: Callable[[], T], timeout_ms: float) -> T:
    try:
        with anyio.fail_after(timeout_ms / 1000):
            return await anyio.to_thread.run_sync(fn, abandon_on_cancel=True)
    except (TimeoutError, DeadlineExceededError) as exc:
        raise RequestTimeoutError(timeout_ms) from exc
