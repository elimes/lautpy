#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : lautpy.
# @File         : pipe_utils
# @Time         : 2020/11/12 11:35 上午
# @Author       : elimes
# @Software     : PyCharm
# @Description  :
"""
Pipe-based utilities for functional-style data processing.

Usage:
    data = [1, 2, 3]
    result = data | xmap(lambda x: x * 2) | xlist
"""

import functools
import itertools
import json
import operator
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

# 可选依赖：缺失时对应管道函数不存在，其余功能不受影响（见 docs/architecture.md）
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # noqa: E731

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

try:
    import sklearn.utils
except ImportError:
    sklearn = None  # type: ignore


# === 日志 ===
# 统一走 _internal（loguru 优先，未安装则回退标准库 logging）
from lautpy._internal import logger

T = TypeVar("T")
U = TypeVar("U")

# star 导出白名单：避免 np/pd/itertools 等实现细节泄漏到 `from lautpy import *`
# 依赖第三方库的管道函数在其安装分支中追加（见下方各 `if xxx is not None` 块）
__all__ = [
    "Pipe",
    # 基础类型转换
    "xtuple", "xlist", "xset",
    # 高阶函数
    "xmap", "xmap_", "xfilter", "xfilter_", "xenumerate", "xenumerate_",
    "xchain", "xchain_", "xzip", "xreduce", "xdrop", "xdrop_",
    # 排序 & 分组
    "xsort", "xgroup",
    # 字符串 & 字典
    "xjoin", "xitemgetter", "xstartswith", "xendswith",
    "xchain_dict", "xDictValues", "xDictRemove", "xgetitem",
    # 统计
    "xCounter", "xCounterUpdate", "xUnique", "xUniquePlus", "xBloomFilter",
    "xHashBins",
    # 并发
    "xThreadPoolExecutor", "xProcessPoolExecutor", "xAsyncio",
    # 调试与输出
    "xprint", "xsse_parser", "xnext", "xtqdm",
    # 计时
    "timer",
    # 日志（README 示例依赖，保留导出）
    "logger",
]


class Pipe:
    """A decorator to enable Unix-like pipeline syntax using `|`."""

    def __init__(self, func: Callable[[T], U]):
        self.func = func
        functools.update_wrapper(self, func)

    def __ror__(self, other: T) -> U:
        return self.func(other)

    def __call__(self, *args, **kwargs) -> "Pipe":
        """Support partial application: e.g., xmap(str.upper)"""
        return Pipe(lambda x: self.func(x, *args, **kwargs))


# === 基础类型转换 ===
xtuple = Pipe(tuple)
xlist = Pipe(list)
xset = Pipe(set)


# === NumPy 支持 ===
if np is not None:
    __all__ += ["xarray", "xstack"]

    @Pipe
    def xarray(x, decimals: Optional[int] = None):
        arr = np.array(x)
        if decimals is not None:
            arr = np.round(arr, decimals)
        return arr

    @Pipe
    def xstack(iterable, axis: int = 0):
        return np.stack(iterable, axis=axis)


# === 高阶函数（返回惰性迭代器）===
# 注意：map/filter 的签名是 (func, iterable)，与管道传入的 (iterable, func) 相反，
# 因此必须交换参数顺序，否则 data | xmap(f) 会把 f 当成 iterable 报 TypeError。
xmap = Pipe(lambda iterable, func: map(func, iterable))
xfilter = Pipe(lambda iterable, func: filter(func, iterable))
xenumerate = Pipe(enumerate)
xchain = Pipe(lambda iters: itertools.chain.from_iterable(iters))
xzip = Pipe(zip)
xreduce = Pipe(lambda iterable, func: functools.reduce(func, iterable))

# 急切变体（约定：尾部下划线 = 立即求值返回 list）
xmap_ = Pipe(lambda iterable, func: list(map(func, iterable)))
xfilter_ = Pipe(lambda iterable, func: list(filter(func, iterable)))
xenumerate_ = Pipe(lambda iterable, start=0: list(enumerate(iterable, start)))
xchain_ = Pipe(lambda iters: list(itertools.chain.from_iterable(iters)))
xdrop = Pipe(lambda iterable, func: itertools.dropwhile(func, iterable))
xdrop_ = Pipe(lambda iterable, func: list(itertools.dropwhile(func, iterable)))


# === 排序 & 分组 ===
@Pipe
def xsort(iterable, reverse: bool = False):
    return sorted(iterable, reverse=reverse)


@Pipe
def xgroup(iterable, step: int = 3):
    """Group iterable into chunks of size `step`."""
    if hasattr(iterable, "__len__"):
        n = len(iterable)
        return [iterable[i : i + step] for i in range(0, n, step)]
    else:
        # For iterators without __len__
        def gen():
            it = iter(iterable)
            while True:
                chunk = list(itertools.islice(it, step))
                if not chunk:
                    break
                yield chunk

        return gen()


# === 字符串 & 字典 ===
@Pipe
def xjoin(items, sep: str = " "):
    return sep.join(map(str, items))


@Pipe
def xitemgetter(keys, d: dict):
    return operator.itemgetter(*keys)(d)


@Pipe
def xstartswith(iterable, prefix: Union[str, Tuple[str, ...]] = ("_", "__", ".")):
    if isinstance(prefix, str):
        prefix = (prefix,)
    return filter(lambda s: s.startswith(prefix), iterable)


@Pipe
def xendswith(iterable, suffix: Union[str, Tuple[str, ...]] = ("_", "__", ".")):
    if isinstance(suffix, str):
        suffix = (suffix,)
    return filter(lambda s: s.endswith(suffix), iterable)


@Pipe
def xchain_dict(dicts: List[Dict]) -> Dict:
    """Merge a list of dicts into one (later keys win)."""
    result: Dict = {}
    for d in dicts:
        result.update(d)
    return result


@Pipe
def xDictValues(d: dict, keys: Iterable, default: Any = None) -> Tuple:
    """Fetch multiple keys with a default, like dict.get for each key."""
    return tuple(d.get(k, default) for k in keys)


@Pipe
def xDictRemove(d: dict, keys: Iterable) -> dict:
    """Remove keys from a dict in place and return it."""
    for k in keys:
        d.pop(k, None)
    return d


@Pipe
def xgetitem(iterable: Iterable, index: int = 0):
    """Yield element at `index` from each item, e.g. [(0, 1), (1, 2)] | xgetitem(1)."""
    for item in iterable:
        yield operator.getitem(item, index)


# === 统计 ===
xCounter = Pipe(Counter)


@Pipe
def xCounterUpdate(iterable: Iterable[Iterable], counter: Optional[Counter] = None) -> Counter:
    """Accumulate counts from an iterable of iterables, e.g. [['w1', 'w2'], ...]."""
    counter = counter if counter is not None else Counter()
    for item in iterable:
        counter.update(item)
    return counter


@Pipe
def xUnique(iterable, keep_order: bool = True):
    if keep_order:
        return list(OrderedDict.fromkeys(iterable))
    else:
        return list(set(iterable))


@Pipe
def xUniquePlus(iterable, key_fn: Optional[Callable] = None):
    """Dedup arbitrary objects (incl. unhashable ones), keeping first occurrence.

    Hashable keys use hash(); others fall back to a pickle digest
    (note: dict key order matters).
    """
    import pickle

    seen = {}
    for element in iterable:
        key = key_fn(element) if key_fn else element
        try:
            hash(key)
        except TypeError:
            key = pickle.dumps(key, protocol=4)
        if key not in seen:
            seen[key] = element
    return list(seen.values())


@Pipe
def xBloomFilter(iterable, capacity: int = 1_000_000, error_rate: float = 0.01):
    """Build a BloomFilter (see lautpy.hashing) from an iterable of members.

    Usage::

        bloom = [i for i in range(100)] | xBloomFilter(capacity=1000)
        42 in bloom  # True
    """
    from lautpy.hashing import BloomFilter

    bloom = BloomFilter(capacity=capacity, error_rate=error_rate)
    for item in iterable:
        bloom.add(item)
    return bloom


@Pipe
def xHashBins(values: Iterable, bins: int = 3):
    """Group values into `bins` stable hash buckets (requires scikit-learn)."""
    from lautpy.hashing import hash_bins

    return hash_bins(values, bins=bins)


# === Pandas 支持 ===
if pd is not None:
    __all__ += ["xconcat_df"]

    @Pipe
    def xconcat_df(dfs, axis: int = 0, ignore_index: bool = True):
        return pd.concat(dfs, axis=axis, ignore_index=ignore_index)


# === 并发执行 ===
def _pool_map(executor_cls, iterable, func, max_workers: int, desc: str) -> list:
    """Shared implementation for the thread/process pool pipes."""
    total = len(iterable) if hasattr(iterable, "__len__") else None
    if total == 1:
        max_workers = 1

    if max_workers > 1:
        with executor_cls(max_workers=max_workers) as executor:
            if total is not None:
                return list(tqdm(executor.map(func, iterable), total=total, desc=desc))
            return list(executor.map(func, iterable))
    return list(map(func, iterable))


if joblib is not None:
    __all__ += ["xJobs"]

    @Pipe
    def xJobs(iterable, func, n_jobs: int = 3):
        """Parallel execution using joblib."""
        if n_jobs > 1:
            delayed_func = joblib.delayed(func)
            return joblib.Parallel(n_jobs=n_jobs)(delayed_func(arg) for arg in iterable)
        else:
            return list(map(func, iterable))


@Pipe
def xThreadPoolExecutor(
    iterable, func, max_workers: int = 5, desc: str = "Processing"
):
    """Thread-based parallel map with progress bar (I/O-bound tasks)."""
    return _pool_map(ThreadPoolExecutor, iterable, func, max_workers, desc)


@Pipe
def xProcessPoolExecutor(
    iterable, func, max_workers: int = 5, desc: str = "Processing"
):
    """Process-based parallel map with progress bar (CPU-bound tasks)."""
    return _pool_map(ProcessPoolExecutor, iterable, func, max_workers, desc)


# === 异步并发 ===
@Pipe
def xAsyncio(tasks, return_exceptions: bool = False):
    """Run a list of coroutines concurrently and collect results.

    Usage::

        async def job(i):
            await asyncio.sleep(0.1)
            return i

        [job(i) for i in range(10)] | xAsyncio
    """
    import asyncio

    async def _gather():
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    return asyncio.run(_gather())


# === Sklearn 支持 ===
if sklearn is not None:
    __all__ += ["xshuffle"]

    @Pipe
    def xshuffle(l, n_samples: Optional[int] = None):
        return sklearn.utils.shuffle(l, n_samples=n_samples)


# === 调试与输出 ===
@Pipe
def xprint(iterable, end: str = "\n", desc: str = "Print"):
    """Print each item with optional progress bar."""
    if desc:
        iterable = tqdm(iterable, desc=desc)
    for item in iterable:
        print(item, end=end)


# === 实用工具 ===
@Pipe
def xsse_parser(
    lines: Iterable[str],
    prefix: str = "data:",
    skip_substrings: Optional[List[str]] = None,
):
    """Parse Server-Sent Events (SSE) lines."""
    skip_substrings = skip_substrings or []
    parsed = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or not stripped.startswith(prefix):
            continue
        content = stripped[len(prefix) :]
        if any(skip in content for skip in skip_substrings):
            continue
        try:
            parsed.append(json.loads(content))
        except json.JSONDecodeError as e:
            logger.warning(f"SSE line JSON decode error: {content[:50]}... ({e})")
    return parsed


# === 进度条快捷方式 ===
@Pipe
def xtqdm(iterable, desc: Optional[str] = None):
    return tqdm(iterable, desc=desc)


@Pipe
def xnext(items: Iterable) -> Iterator:
    """Convert an iterable into a pull-based iterator: `it = xs | xnext; next(it)`."""
    return iter(items)


# === 计时 ===
@contextmanager
def timer(name: str = "timer"):
    """上下文管理器：计时代码块，结束时通过 logger 输出耗时。

    用法::

        with timer('LOG'):
            logger.info("打印一条log所花费的时间")
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[{name}] 耗时: {elapsed:.4f}s")
