"""
Kalki Nexus - Resilience

RetryPolicy + with_retry wrap any async callable with bounded retries and
linear backoff, retrying only on exceptions marked `retryable=True` (see
core/exceptions.py). Once every attempt is exhausted, RetryExhaustedError is
raised so the graph's error node / Fallback Agent can take over deliberately
instead of the process crashing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from core.exceptions import KalkiError, RetryExhaustedError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry configuration: how many attempts, and how long to back off between them."""

    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0


def with_retry(policy: RetryPolicy, operation_name: str):
    """Decorator factory: retries an async function per `policy`, re-raising
    non-retryable KalkiErrors immediately and wrapping exhaustion in
    RetryExhaustedError."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = policy.backoff_seconds
            last_error: BaseException | None = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except KalkiError as exc:
                    last_error = exc
                    if not exc.retryable or attempt == policy.max_attempts:
                        raise
                except Exception as exc:  # noqa: BLE001 - unexpected errors are retried too, up to the limit
                    last_error = exc
                    if attempt == policy.max_attempts:
                        raise RetryExhaustedError(operation_name, attempt, last_error) from exc
                await asyncio.sleep(delay)
                delay *= policy.backoff_multiplier
            raise RetryExhaustedError(operation_name, policy.max_attempts, last_error)

        return wrapper

    return decorator
