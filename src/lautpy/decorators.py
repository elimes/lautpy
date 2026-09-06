# @Author: elimes
"""函数装饰器：重试、超时、后台执行、限流、单例。

设计决策：

- ``retrying`` 是 tenacity 的同步薄封装（可选依赖）；失败只记日志，
  绝不静默吞掉、也不推送外部服务
- ``timeout`` 超时抛内置 ``TimeoutError``，并保证关闭线程执行器
- ``background_task`` 返回 ``Future`` 供检查结果，异常带上下文记日志
- ``ratelimit`` 为纯标准库滑动窗口限流
- 有意不提供：进程 fork（不可移植、无法安全终止）、阻塞式循环调度器
  （请改用 APScheduler / cron）
"""

import functools
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar, cast

from lautpy._internal import logger

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    retry = None  # type: ignore

F = TypeVar("F", bound=Callable)

#: 默认不重试的异常（多为编程错误，重试没有意义）
_NON_RETRYABLE: tuple[type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    FileNotFoundError,
    PermissionError,
)


def retrying(
    max_retries: int = 2,
    exp_base: float = 2.0,
    max_wait: float = 60.0,
    retry_on_result: Callable[[Any], bool] | None = None,
    ignored_exceptions: tuple[type[BaseException], ...] | None = None,
) -> Callable[[F], F]:
    """指数退避重试；重试耗尽后原样抛出最后一次异常。

    仅支持同步函数（异步场景请直接使用 tenacity）。需要 tenacity：
    ``pip install "lautpy[retry]"``。

    Args:
        max_retries: 首次失败后的最大重试次数（总调用 = max_retries + 1）。
        exp_base: 退避指数基数，第 n 次重试约等待 2**n 秒（封顶 max_wait）。
        max_wait: 单次重试等待上限（秒）。
        retry_on_result: 传入谓词时，结果为 True 也触发重试。
        ignored_exceptions: 这些异常类型不重试，直接抛出；程序性错误
            （KeyboardInterrupt/FileNotFoundError 等）默认就不重试。

    Example::

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
            return not isinstance(exc, _NON_RETRYABLE)
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

        return cast("F", wrapper)

    return decorator


def timeout(seconds: float) -> Callable[[F], F]:
    """在工作线程中运行函数，超过 seconds 秒抛 ``TimeoutError``。

    注意：线程无法被安全杀死，超时后工作线程仍在后台运行——被装饰的
    函数不应有无法回滚的副作用。

    Args:
        seconds: 超时秒数。
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

        return cast("F", wrapper)

    return decorator


def background_task(func: F) -> F:
    """发射后不管（fire-and-forget）：立即返回 ``Future``，任务在后台执行。

    任务异常不会打断主流程，但会带上下文记入日志；需要结果时用返回的
    ``future.result()``。

    Example::

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

    return cast("F", wrapper)


def ratelimit(calls: int, period: float) -> Callable[[F], F]:
    """滑动窗口限流：每个被装饰函数在 period 秒内最多调用 calls 次，超限阻塞等待。

    Args:
        calls: 窗口内允许的最大调用次数，须为正。
        period: 窗口长度（秒），须为正。

    Example::

        @ratelimit(calls=5, period=1)   # 每秒最多 5 次
        def api_call(): ...
    """
    if calls <= 0 or period <= 0:
        raise ValueError("calls and period must be positive")

    def decorator(func: F) -> F:
        # Condition.wait 会释放锁：等待中的线程并行休眠、被唤醒后重新竞争，
        # 避免"持锁睡眠"导致 N 个线程的等待时间串行叠加
        condition = threading.Condition()
        timestamps: deque = deque()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with condition:
                while len(timestamps) >= calls:
                    remaining = timestamps[0] + period - time.monotonic()
                    if remaining > 0:
                        condition.wait(remaining)
                    else:
                        timestamps.popleft()
                timestamps.append(time.monotonic())
            return func(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def singleton(cls: type) -> type:
    """单例装饰器：全进程共享一个实例（再次实例化时忽略构造参数）。

    原 class 仍可通过 ``SomeClass.__wrapped__`` 访问（便于测试）。
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
    return cast("type", get_instance)


def synchronized(lock: "threading.Lock | None" = None) -> Callable[[F], F]:
    """串行化对函数的调用（默认为每个函数配一把独立锁）。

    Args:
        lock: 多个函数需要共用一把锁时显式传入。
    """
    def decorator(func: F) -> F:
        fn_lock = lock if lock is not None else threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with fn_lock:
                return func(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def tryer(task: str, is_trace: bool = False):
    """吞掉异常但记录日志的上下文管理器（容错包裹）。

    Args:
        task: 任务名，用于日志定位。
        is_trace: True 时记录完整堆栈，否则只记异常摘要。

    Example::

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
