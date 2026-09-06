"""针对外部评审发现问题的回归测试。"""

import asyncio
import threading
import time

import pytest

from lautpy.decorators import ratelimit
from lautpy.hashing import ABTest
from lautpy.pipe import xAsyncio, xchain_, xgroup


def test_xgroup_set_no_longer_crashes():
    """set 有 __len__ 但不可切片，旧实现直接 TypeError；现在走生成器分支。"""
    result = {1, 2, 3, 4} | xgroup(2)
    flattened = result | xchain_
    assert sorted(flattened) == [1, 2, 3, 4]


def test_xgroup_dict_also_safe():
    result = {"a": 1, "b": 2} | xgroup(1)
    assert list(result) == [["a"], ["b"]]


def test_ratelimit_parallel_waiters_not_serialized():
    """Condition.wait 释放锁：4 个并发等待的线程应在约一个窗口期内放行，
    而不是每个线程串行睡满一个窗口（旧实现的缺陷）。"""
    @ratelimit(calls=2, period=0.3)
    def hit():
        return time.monotonic()

    results = []
    lock = threading.Lock()

    def worker():
        ts = hit()
        with lock:
            results.append(ts)

    start = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    elapsed = time.monotonic() - start

    assert len(results) == 4
    # 串行持锁睡眠版 ≈ 0.3s * 3 ≈ 0.9s+；并行唤醒应在 ~0.6s 内完成
    assert elapsed < 0.9, f"waiters appear serialized: {elapsed:.2f}s"


def test_abtest_ranger_is_half_open():
    """ranger 语义同 range()：(0, 1) 只覆盖 0 号桶；bins=1 时所有用户都在 0 号桶。"""
    ab = ABTest(expid="e", ranger=(0, 1), bins=1)
    assert ab.is_hit("anyone") is True

    # ranger=(1, 2) 只覆盖 1 号桶；bins=1 时无人命中（0 号桶不在区间内）
    ab_none = ABTest(expid="e", ranger=(1, 2), bins=1)
    assert ab_none.is_hit("anyone") is False


def test_xasyncio_rejects_running_loop():
    """在事件循环内调用时给出明确指引（Jupyter 高频踩坑点）。"""
    async def inner():
        with pytest.raises(RuntimeError, match="running event loop"):
            [None] | xAsyncio

    asyncio.new_event_loop().run_until_complete(inner())
    asyncio.get_event_loop_policy()
