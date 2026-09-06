# @Author: elimes
"""管道式数据处理：data | xfunc1 | xfunc2 的 Unix 风格函数族。

- 惰性函数（xmap/xfilter/xdrop）返回迭代器，用 xlist/xtuple 收口
- 急切变体以尾部下划线标记（xmap_ 等），立即求值返回 list
- 依赖 numpy/pandas/joblib/sklearn 的函数在库缺失时整个不存在

完整清单见 ``lautpy.pipe.__all__``，用法见 docs/usage.md。
"""

import functools
import itertools
import json
import operator
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, TypeVar

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
    """让普通函数支持 ``data | func`` 管道语法的装饰器。

    - ``__ror__``：实现 ``data | xfunc``，把 data 作为第一个参数调用
    - ``__call__``：部分应用，``xfunc(arg1, arg2)`` 返回绑定好参数的新 Pipe

    Example::

        @Pipe
        def xadd(data, n):
            return [v + n for v in data]

        [1, 2, 3] | xadd(10)   # [11, 12, 13]
    """

    def __init__(self, func: Callable[[T], U]):
        self.func = func
        functools.update_wrapper(self, func)

    def __ror__(self, other: T) -> U:
        return self.func(other)

    def __call__(self, *args, **kwargs) -> "Pipe":
        """Support partial application: e.g., xmap(str.upper)"""
        return Pipe(lambda x: self.func(x, *args, **kwargs))


# === 基础类型转换 ===
# === 基础类型转换（收口惰性管道用）===
xtuple = Pipe(tuple)
xtuple.__doc__ = "迭代器 → tuple。"
xlist = Pipe(list)
xlist.__doc__ = "迭代器 → list（惰性管道最常用的收口）。"
xset = Pipe(set)
xset.__doc__ = "迭代器 → set（顺便去重）。"


# === NumPy 支持 ===
if np is not None:
    __all__ += ["xarray", "xstack"]

    @Pipe
    def xarray(x, decimals: int | None = None):
        """可迭代对象 → numpy 数组；decimals 给定时按位四舍五入。"""
        arr = np.array(x)
        if decimals is not None:
            arr = np.round(arr, decimals)
        return arr

    @Pipe
    def xstack(iterable, axis: int = 0):
        """沿 axis 维度堆叠一批同形数组/序列（np.stack）。"""
        return np.stack(iterable, axis=axis)


# === 高阶函数（返回惰性迭代器）===
# 注意：map/filter 的签名是 (func, iterable)，与管道传入的 (iterable, func) 相反，
# 因此必须交换参数顺序，否则 data | xmap(f) 会把 f 当成 iterable 报 TypeError。
xmap = Pipe(lambda iterable, func: map(func, iterable))
xmap.__doc__ = "惰性 map：data | xmap(func) → 迭代器（用 xlist/xmap_ 收口）。"
xfilter = Pipe(lambda iterable, func: filter(func, iterable))
xfilter.__doc__ = "惰性 filter：保留谓词为真的元素。"
xenumerate = Pipe(enumerate)
xenumerate.__doc__ = "惰性 enumerate：带序号的迭代器。"
xchain = Pipe(lambda iters: itertools.chain.from_iterable(iters))
xchain.__doc__ = "把嵌套可迭代对象展平一层（惰性）。"
xzip = Pipe(zip)
xzip.__doc__ = "并行遍历：data | xzip(other) → zip 对象。"
xreduce = Pipe(lambda iterable, func: functools.reduce(func, iterable))
xreduce.__doc__ = "折叠：data | xreduce(func) → 聚合值。"

# 急切变体（约定：尾部下划线 = 立即求值返回 list）
xmap_ = Pipe(lambda iterable, func: list(map(func, iterable)))
xmap_.__doc__ = "急切 map，直接返回 list。"
xfilter_ = Pipe(lambda iterable, func: list(filter(func, iterable)))
xfilter_.__doc__ = "急切 filter，直接返回 list。"
xenumerate_ = Pipe(lambda iterable, start=0: list(enumerate(iterable, start)))
xenumerate_.__doc__ = "急切 enumerate，直接返回 [(序号, 元素), ...]。"
xchain_ = Pipe(lambda iters: list(itertools.chain.from_iterable(iters)))
xchain_.__doc__ = "展平一层并立即返回 list。"
xdrop = Pipe(lambda iterable, func: itertools.dropwhile(func, iterable))
xdrop.__doc__ = "惰性丢弃谓词为真的前缀（直到第一个 False）。"
xdrop_ = Pipe(lambda iterable, func: list(itertools.dropwhile(func, iterable)))
xdrop_.__doc__ = "急切版 xdrop。"


# === 排序 & 分组 ===
@Pipe
def xsort(iterable, reverse: bool = False):
    """排序返回 list（内置 sorted 的管道形式）。"""
    return sorted(iterable, reverse=reverse)


@Pipe
def xgroup(iterable, step: int = 3):
    """按固定大小分组：data | xgroup(2) → [[a, b], [c, d], ...]。

    有 __len__ 的输入返回 list 的 list；无 __len__ 的迭代器返回生成器。
    """
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
    """把元素转为 str 后用 sep 连接：["a", 1] | xjoin("-") → "a-1"。"""
    return sep.join(map(str, items))


@Pipe
def xitemgetter(keys, d: dict):
    """按 keys 从 dict 批量取值（operator.itemgetter 的管道形式）。"""
    return operator.itemgetter(*keys)(d)


@Pipe
def xstartswith(iterable, prefix: str | tuple[str, ...] = ("_", "__", ".")):
    """过滤出以 prefix 开头的元素（惰性）。"""
    if isinstance(prefix, str):
        prefix = (prefix,)
    return filter(lambda s: s.startswith(prefix), iterable)


@Pipe
def xendswith(iterable, suffix: str | tuple[str, ...] = ("_", "__", ".")):
    """过滤出以 suffix 结尾的元素（惰性）。"""
    if isinstance(suffix, str):
        suffix = (suffix,)
    return filter(lambda s: s.endswith(suffix), iterable)


@Pipe
def xchain_dict(dicts: list[dict]) -> dict:
    """合并一批 dict 为一个（后者覆盖同名键）。"""
    result: dict = {}
    for d in dicts:
        result.update(d)
    return result


@Pipe
def xDictValues(d: dict, keys: Iterable, default: Any = None) -> tuple:
    """批量取值带默认值：d | xDictValues(["a", "z"], default=0) → (值, 0)。"""
    return tuple(d.get(k, default) for k in keys)


@Pipe
def xDictRemove(d: dict, keys: Iterable) -> dict:
    """就地删除指定键并返回该 dict。"""
    for k in keys:
        d.pop(k, None)
    return d


@Pipe
def xgetitem(iterable: Iterable, index: int = 0):
    """取每个元素的第 index 位（惰性）：[(0, "a")] | xgetitem(1) → "a", ...。"""
    for item in iterable:
        yield operator.getitem(item, index)


# === 统计 ===
xCounter = Pipe(Counter)
xCounter.__doc__ = "迭代器 → Counter（词频统计）。"


@Pipe
def xCounterUpdate(iterable: Iterable[Iterable], counter: Counter | None = None) -> Counter:
    """增量计数：iterable 的每个元素须是可迭代对象，逐个并入 Counter。

    传入 counter 时在原对象上累积（常用于跨批次合并计数）。
    """
    counter = counter if counter is not None else Counter()
    for item in iterable:
        counter.update(item)
    return counter


@Pipe
def xUnique(iterable, keep_order: bool = True):
    """去重；keep_order=True 时保留首次出现顺序。"""
    if keep_order:
        return list(OrderedDict.fromkeys(iterable))
    else:
        return list(set(iterable))


@Pipe
def xUniquePlus(iterable, key_fn: Callable | None = None):
    """去重任意对象（含 dict 等不可哈希对象），保留首次出现。

    Args:
        iterable: 待去重序列。
        key_fn: 自定义去重键；如按 x["id"] 去重传 lambda x: x["id"]。

    注意：不可哈希键回退到 pickle 摘要，dict 的键顺序不同视为不同对象。
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
    """把一批元素装进布隆过滤器（见 lautpy.hashing.BloomFilter）。

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
    """按 murmurhash 稳定分组（同元素必落同组，需 scikit-learn）。"""
    from lautpy.hashing import hash_bins

    return hash_bins(values, bins=bins)


# === Pandas 支持 ===
if pd is not None:
    __all__ += ["xconcat_df"]

    @Pipe
    def xconcat_df(dfs, axis: int = 0, ignore_index: bool = True):
        """纵向/横向拼接一批 DataFrame（pd.concat 的管道形式）。"""
        return pd.concat(dfs, axis=axis, ignore_index=ignore_index)


# === 并发执行 ===
def _pool_map(executor_cls, iterable, func, max_workers: int, desc: str) -> list:
    """线程/进程池两个管道函数的公共实现。"""
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
        """joblib 并行 map（支持匿名函数）；n_jobs=1 退化为串行。"""
        if n_jobs > 1:
            delayed_func = joblib.delayed(func)
            return joblib.Parallel(n_jobs=n_jobs)(delayed_func(arg) for arg in iterable)
        else:
            return list(map(func, iterable))


@Pipe
def xThreadPoolExecutor(
    iterable, func, max_workers: int = 5, desc: str = "Processing"
):
    """线程池并行 map，自带进度条（适合 I/O 密集：爬取/调用接口）。"""
    return _pool_map(ThreadPoolExecutor, iterable, func, max_workers, desc)


@Pipe
def xProcessPoolExecutor(
    iterable, func, max_workers: int = 5, desc: str = "Processing"
):
    """进程池并行 map，自带进度条（适合 CPU 密集：批量计算）。"""
    return _pool_map(ProcessPoolExecutor, iterable, func, max_workers, desc)


# === 异步并发 ===
@Pipe
def xAsyncio(tasks, return_exceptions: bool = False):
    """并发执行一批协程并收集结果。

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
    def xshuffle(l, n_samples: int | None = None):
        """打乱顺序；n_samples 给定时随机抽取 n_samples 个（需 scikit-learn）。"""
        return sklearn.utils.shuffle(l, n_samples=n_samples)


# === 调试与输出 ===
@Pipe
def xprint(iterable, end: str = "\n", desc: str = "Print"):
    """逐个打印元素（调试用）；desc 非空时附带进度条。"""
    if desc:
        iterable = tqdm(iterable, desc=desc)
    for item in iterable:
        print(item, end=end)


# === 实用工具 ===
@Pipe
def xsse_parser(
    lines: Iterable[str],
    prefix: str = "data:",
    skip_substrings: list[str] | None = None,
):
    """解析 SSE（Server-Sent Events）行为 JSON 列表（LLM 流式输出常用）。

    Args:
        lines: 文本行序列。
        prefix: 事件前缀，默认 "data:"。
        skip_substrings: 内容含这些子串的行直接跳过（如 "[DONE]"、TRACEID）。

    解析失败的行记 warning 日志后跳过，不中断整个流。
    """
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
def xtqdm(iterable, desc: str | None = None):
    """给任意可迭代对象套上进度条。"""
    return tqdm(iterable, desc=desc)


@Pipe
def xnext(items: Iterable) -> Iterator:
    """转为拉取式迭代器（单步调试利器）：it = xs | xnext; next(it)。"""
    return iter(items)


# === 计时 ===
@contextmanager
def timer(name: str = "timer"):
    """计时代码块，结束时输出耗时日志（info 级）。

    Args:
        name: 日志中的任务标识。

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
