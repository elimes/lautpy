#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : lautpy.
# @File         : pipe_utils
# @Time         : 2020/11/12 11:35 上午
# @Author       : liufeng
# @Email        : elimes@qq.com
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
import sys
import time
import warnings
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union
)

# 第三方依赖（请确保已安装：pip install tqdm numpy pandas joblib scikit-learn）
try:
    from tqdm.auto import tqdm
except ImportError:
    warnings.warn("tqdm not installed. Progress bars will be disabled.", ImportWarning)
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
# 优先使用 loguru（零配置、输出美观）；未安装则回退到标准库 logging。
try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("lautpy")
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(_h)
        logger.setLevel(logging.INFO)


T = TypeVar("T")
U = TypeVar("U")


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

    @Pipe
    def xarray(x, decimals: Optional[int] = None):
        arr = np.array(x)
        if decimals is not None:
            arr = np.round(arr, decimals)
        return arr


# === 高阶函数（返回惰性迭代器）===
# 注意：map/filter 的签名是 (func, iterable)，与管道传入的 (iterable, func) 相反，
# 因此必须交换参数顺序，否则 data | xmap(f) 会把 f 当成 iterable 报 TypeError。
xmap = Pipe(lambda iterable, func: map(func, iterable))
xfilter = Pipe(lambda iterable, func: filter(func, iterable))
xenumerate = Pipe(enumerate)
xchain = Pipe(lambda iters: itertools.chain.from_iterable(iters))
xzip = Pipe(zip)
xreduce = Pipe(lambda iterable, func: functools.reduce(func, iterable))


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


# === 统计 ===
xCounter = Pipe(Counter)


@Pipe
def xUnique(iterable, keep_order: bool = True):
    if keep_order:
        return list(OrderedDict.fromkeys(iterable))
    else:
        return list(set(iterable))


# === Pandas 支持 ===
if pd is not None:

    @Pipe
    def xconcat_df(dfs, axis: int = 0, ignore_index: bool = True):
        return pd.concat(dfs, axis=axis, ignore_index=ignore_index)


# === 并发执行 ===
if joblib is not None:

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
    """Thread-based parallel map with progress bar."""
    total = len(iterable) if hasattr(iterable, "__len__") else None
    if total == 1:
        max_workers = 1

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if total is not None:
                results = list(tqdm(executor.map(func, iterable), total=total, desc=desc))
            else:
                results = list(executor.map(func, iterable))
        return results
    else:
        return list(map(func, iterable))


@Pipe
def xProcessPoolExecutor(
    iterable, func, max_workers: int = 5, desc: str = "Processing"
):
    """Process-based parallel map with progress bar."""
    total = len(iterable) if hasattr(iterable, "__len__") else None
    if total == 1:
        max_workers = 1

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            if total is not None:
                results = list(tqdm(executor.map(func, iterable), total=total, desc=desc))
            else:
                results = list(executor.map(func, iterable))
        return results
    else:
        return list(map(func, iterable))


# === Sklearn 支持 ===
if sklearn is not None:

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
            print(f"JSON decode error in SSE line: {content[:50]}... ({e})", file=sys.stderr)
    return parsed


# === 进度条快捷方式 ===
@Pipe
def xtqdm(iterable, desc: Optional[str] = None):
    return tqdm(iterable, desc=desc)


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


# === 兼容性提示 ===
if np is None:
    warnings.warn("NumPy not installed. xarray is unavailable.", ImportWarning)
if pd is None:
    warnings.warn("Pandas not installed. xconcat_df is unavailable.", ImportWarning)
if joblib is None:
    warnings.warn("Joblib not installed. xJobs is unavailable.", ImportWarning)
if sklearn is None:
    warnings.warn("Scikit-learn not installed. xshuffle is unavailable.", ImportWarning)