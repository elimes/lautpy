import importlib.util
import threading
import time

import pytest

from lautpy.decorators import (
    background_task,
    ratelimit,
    retrying,
    singleton,
    synchronized,
    timeout,
    tryer,
)

has_tenacity = importlib.util.find_spec("tenacity") is not None


class TestRetrying:
    @pytest.mark.skipif(not has_tenacity, reason="tenacity not installed")
    def test_retries_until_success(self):
        calls = {"n": 0}

        @retrying(max_retries=3, max_wait=0.1)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("boom")
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    @pytest.mark.skipif(not has_tenacity, reason="tenacity not installed")
    def test_exhaustion_reraises(self):
        calls = {"n": 0}

        @retrying(max_retries=1, max_wait=0.1)
        def always_fails():
            calls["n"] += 1
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            always_fails()
        assert calls["n"] == 2  # initial attempt + 1 retry

    @pytest.mark.skipif(not has_tenacity, reason="tenacity not installed")
    def test_ignored_exceptions_not_retried(self):
        calls = {"n": 0}

        @retrying(max_retries=5, ignored_exceptions=(FileNotFoundError,))
        def missing_file():
            calls["n"] += 1
            raise FileNotFoundError("gone")

        with pytest.raises(FileNotFoundError):
            missing_file()
        assert calls["n"] == 1  # no retry for programming errors


class TestTimeout:
    def test_within_limit(self):
        @timeout(2)
        def fast():
            return 42

        assert fast() == 42

    def test_exceeds_limit(self):
        @timeout(0.1)
        def slow():
            time.sleep(1)
            return "late"

        with pytest.raises(TimeoutError):
            slow()

    def test_name_in_error(self):
        @timeout(0.05)
        def my_specific_job():
            time.sleep(0.5)

        with pytest.raises(TimeoutError, match="my_specific_job"):
            my_specific_job()


class TestBackgroundTask:
    def test_returns_future_and_runs(self):
        @background_task
        def job(value):
            return value * 2

        future = job(21)
        assert future.result(timeout=2) == 42

    def test_exception_captured_in_future(self):
        @background_task
        def bad():
            raise ValueError("oops")

        future = bad()
        with pytest.raises(ValueError, match="oops"):
            future.result(timeout=2)


class TestRatelimit:
    def test_sliding_window(self):
        @ratelimit(calls=2, period=0.3)
        def hit():
            return time.monotonic()

        start = time.monotonic()
        hit(), hit()
        third = hit()
        assert third - start >= 0.29  # third call had to wait for a slot

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            ratelimit(0, 1)


class TestMisc:
    def test_singleton(self):
        @singleton
        class Config:
            def __init__(self):
                self.name = "cfg"

        assert Config() is Config()

    def test_synchronized_thread_safety(self):
        counter = {"n": 0}

        @synchronized()
        def bump():
            n = counter["n"]
            time.sleep(0.001)  # widen the race window
            counter["n"] = n + 1

        threads = [threading.Thread(target=bump) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert counter["n"] == 10

    def test_tryer_swallows_and_returns(self):
        with tryer("cleanup"):
            raise ValueError("should not propagate")
