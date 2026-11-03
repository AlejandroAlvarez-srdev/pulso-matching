"""Per-request timeout for CPU-bound handlers.

The solvers are synchronous, so they run in a worker thread. ``fail_after`` bounds how long the
request waits; on expiry the thread is *abandoned* (Python threads cannot be killed) and the
client gets a 504 immediately. To stop that thread from burning CPU, handlers also pass the same
deadline into the service, whose Hungarian solver checks it once per row and bails out. Grid
scoring and greedy are not interruptible, but they are the cheap parts.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import anyio
import anyio.to_thread


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
    except TimeoutError as exc:
        raise RequestTimeoutError(timeout_ms) from exc
