#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function decorators: retry, timeout, background, rate-limit, singleton.

Design decisions:

- ``retrying`` is a thin, sync-only wrapper over tenacity (optional
  dependency) with exponential backoff and reraise-on-exhaustion; failures
  are logged, never silently swallowed or pushed to external services.
- ``timeout`` raises a builtin ``TimeoutError`` and always shuts down its
  executor.
- ``background_task`` returns the ``Future`` so results can be inspected;
  exceptions are logged with context.
- ``ratelimit`` is a stdlib sliding-window limiter.
- Deliberately NOT provided: process forking (not portable, unsafe to kill)
  and blocking-loop schedulers (use APScheduler or cron instead).
"""

import functools
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

try:
    from loguru import logger
except ImportError:  # pragma: no cover - pipe.py already provides the fallback
    import logging

    logger = logging.getLogger("lautpy")

try:
    from tenacity import retry, retry_if_exception, retry_if_result, stop_after_attempt, wait_exponential
except ImportError:
    retry = None  # type: ignore

F = TypeVar("F", bound=Callable)

#: Exceptions that should never be retried by default ( programming errors ).
_NON_RETRYABLE: Tuple[Type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    FileNotFoundError,
    PermissionError,
)


def retrying(
    max_retries: int = 2,
    exp_base: float = 2.0,
    max_wait: float = 60.0,
    retry_on_result: Optional[Callable[[Any], bool]] = None,
    ignored_exceptions: Optional[Tuple[Type[BaseException], ...]] = None,
) -> Callable[[F], F]:
    """Retry with exponential backoff; re-raises the last exception on exhaustion.

    Sync functions only (async support intentionally left out — use tenacity
    directly for that). Requires tenacity: ``pip install tenacity``.

    Usage::

        @retrying(max_retries=3)
        def flaky(): ...
    """
    if retry is None:
        raise ImportError("tenacity is required for retrying: pip install tenacity")

    def _should_retry(retry_state) -> bool:
        outcome = retry_state.outcome
        if outcome.failed:
            exc = outcome.exception()
            if ignored_exceptions and isinstance(exc, ignored_exceptions):
                return False
            if isinstance(exc, _NON_RETRYABLE):
                return False
            return True
        return bool(retry_on_result(outcome.result())) if retry_on_result else False

    def decorator(func: F) -> F:
        @retry(
            reraise=True,
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=1.0, exp_base=exp_base, max=max_wait),
            retry=_should_retry,
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """Run the function in a worker thread and raise ``TimeoutError`` if it
    exceeds `seconds`. The worker keeps running in the background after the
    timeout (a thread cannot be killed safely) — avoid side effects you cannot
    undo.
    """
    import concurrent.futures

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s") from None
            finally:
                executor.shutdown(wait=False)

        return wrapper

    return decorator


def background_task(func: F) -> F:
    """Fire-and-forget execution: returns immediately with the ``Future``;
    exceptions are logged with traceback when the task finishes.

    Usage::

        @background_task
        def notify(): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)

        def _log_error(fut):
            exc = fut.exception()
            if exc is not None:
                logger.error(f"background task {func.__name__} failed: {exc!r}")

        future.add_done_callback(_log_error)
        executor.shutdown(wait=False)
        return future

    return wrapper


def ratelimit(calls: int, period: float) -> Callable[[F], F]:
    """Sliding-window rate limit: allow at most `calls` invocations per
    `period` seconds (per decorated function); blocks until a slot is free.
    """
    if calls <= 0 or period <= 0:
        raise ValueError("calls and period must be positive")

    def decorator(func: F) -> F:
        lock = threading.Lock()
        timestamps: deque = deque()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                while len(timestamps) >= calls:
                    remaining = timestamps[0] + period - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    else:
                        timestamps.popleft()
                timestamps.append(time.monotonic())
            return func(*args, **kwargs)

        return wrapper

    return decorator


def singleton(cls: type) -> type:
    """Class decorator keeping a single shared instance (init args are
    ignored on subsequent instantiations).
    """
    instances = {}
    lock = threading.Lock()

    @functools.wraps(cls, updated=[])
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    get_instance.__wrapped__ = cls  # expose the original class for tests/typing
    return get_instance


def synchronized(lock: Optional[threading.Lock] = None) -> Callable[[F], F]:
    """Serialize calls to the function with a lock (a dedicated one by default)."""
    def decorator(func: F) -> F:
        fn_lock = lock if lock is not None else threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with fn_lock:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def tryer(task: str, is_trace: bool = False):
    """Context manager factory: swallow exceptions but log them.

    Usage::

        with tryer("cleanup"):
            maybe_failing()
    """
    import traceback
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        try:
            yield
        except Exception as e:  # noqa: BLE001 - this is the whole point
            logger.error(traceback.format_exc() if is_trace else f"{task}: {e!r}")

    return _ctx()
