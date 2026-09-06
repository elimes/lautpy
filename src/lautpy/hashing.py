# -*- coding: utf-8 -*-
# @Author: elimes
"""哈希工具：md5 摘要、与 Java 口径一致的 murmurhash 分桶、AB 实验分流、布隆过滤器。

murmurhash 系功能需要 scikit-learn；BloomFilter 为纯标准库实现。
"""

import hashlib as _hashlib
import math
import pickle
from typing import Iterable, List, Optional, Tuple, Union

try:
    from sklearn.utils.murmurhash import murmurhash3_32 as _murmurhash3_32
except ImportError:
    _murmurhash3_32 = None  # type: ignore


def md5(data: Union[str, bytes]) -> str:
    """计算 md5 摘要。

    Args:
        data: str 按 utf-8 编码后计算；bytes 原样计算。

    Returns:
        str: 32 位十六进制摘要。
    """
    if isinstance(data, str):
        data = data.encode("utf8")
    return _hashlib.md5(data).hexdigest()


def murmurhash(
    key: str = "key",
    value: str = "value",
    bins: Optional[int] = None,
    str2md5: bool = True,
) -> int:
    """计算 "value:key" 的 murmurhash3_32（正值），可按 bins 取模分桶。

    与 Java Guava murmur3_32 口径一致，适合跨语言系统共用同一套分桶规则。

    Args:
        key: 键（参与哈希的字符串，与 value 以冒号拼接）。
        value: 值。
        bins: 分桶数；给定时返回 ``hash % bins``，None 返回原始哈希值。
        str2md5: 是否先对拼接串做一次 md5 再哈希（与部分历史系统对齐）。

    Requires:
        scikit-learn（未安装时抛 ImportError）。
    """
    if _murmurhash3_32 is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")
    string = f"{value}:{key}"
    if str2md5:
        string = md5(string)
    hashed = _murmurhash3_32(string, positive=True)
    return hashed % bins if bins else hashed


class ABTest:
    """基于哈希分桶的 AB 实验分流：同一用户永远落在同一桶，结果稳定可复现。

    命中规则：murmurhash(f"{user_id}:{expid}") % bins 落在 ranger 区间内。

    Example::

        ab = ABTest(expid='10001', ranger=(0, 99), bins=100)
        if ab.is_hit(user_id):   # 0~99 号桶 = 1% 流量
            ...
    """

    def __init__(self, expid: str = "10001", ranger: Tuple[int, int] = (0, 9), bins: int = 100):
        self._bins = bins
        self._ranger = set(range(*ranger))
        self._expid = expid

    def is_hit(self, value: str = "userid") -> bool:
        """判断用户是否命中实验组。

        Args:
            value: 用户标识（同一标识多次调用结果恒定）。
        """
        return murmurhash(key=self._expid, value=value, str2md5=False) % self._bins in self._ranger


def hash_bins(values: Iterable, bins: int = 3, str2md5: bool = False) -> List[List]:
    """按 murmurhash 把元素稳定分成 bins 组（同一元素必落同组）。

    适合一致性分片、按 key 分流等场景；bins 相同时重复调用分组结果不变。

    Args:
        values: 待分组的元素（内部转为 str 参与哈希）。
        bins: 组数。
        str2md5: 是否先 md5 再哈希（语义同 murmurhash）。

    Returns:
        List[List]: 分组结果（组的顺序不定），各组拼接 = 输入全集。
    """
    buckets: dict = {}
    for v in values:
        idx = murmurhash(key=str(v), value=str(v), bins=bins, str2md5=str2md5)
        buckets.setdefault(idx, []).append(v)
    return list(buckets.values())


class BloomFilter:
    """定容布隆过滤器（纯标准库实现）。

    特性：**绝不漏报**（添加过的元素一定命中）；对未添加的元素可能以
    error_rate 概率**误报**。适合黑名单、爬虫去重等可容忍少量误报的场景。

    Args（构造参数）:
        capacity: 预期元素数量上限。
        error_rate: 目标误报率，(0, 1) 区间。

    Example::

        bloom = BloomFilter(capacity=1000, error_rate=0.01)
        bloom.add("a")
        "a" in bloom    # True
    """

    def __init__(self, capacity: int = 1_000_000, error_rate: float = 0.01):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 < error_rate < 1:
            raise ValueError("error_rate must be in (0, 1)")
        m = math.ceil(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self._m = m
        self._k = max(1, round(m / capacity * math.log(2)))
        self._bits = bytearray((m + 7) // 8)
        self._count = 0

    def _hashes(self, value):
        digest = _hashlib.blake2b(pickle.dumps(value, protocol=4), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:], "big") | 1
        return ((h1 + i * h2) % self._m for i in range(self._k))

    def add(self, value) -> bool:
        """添加元素。

        Returns:
            bool: False 表示该元素可能已存在（重复添加）；True 表示首次加入。
        """
        was_present = value in self
        for idx in self._hashes(value):
            self._bits[idx >> 3] |= 1 << (idx & 7)
        if not was_present:
            self._count += 1
        return not was_present

    def __contains__(self, value) -> bool:
        return all(self._bits[idx >> 3] >> (idx & 7) & 1 for idx in self._hashes(value))

    def __len__(self) -> int:
        return self._count
