#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hash helpers: md5 digest and Java-compatible murmurhash bucketing (AB tests).

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
    """Hex md5 digest of a str (utf-8 encoded) or bytes."""
    if isinstance(data, str):
        data = data.encode("utf8")
    return _hashlib.md5(data).hexdigest()


def murmurhash(
    key: str = "key",
    value: str = "value",
    bins: Optional[int] = None,
    str2md5: bool = True,
) -> int:
    """Positive murmurhash3_32 of "value:key", optionally bucketed modulo `bins`.

    Compatible with the Java Guava murmur3_32 convention. Requires scikit-learn.
    """
    if _murmurhash3_32 is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")
    string = f"{value}:{key}"
    if str2md5:
        string = md5(string)
    hashed = _murmurhash3_32(string, positive=True)
    return hashed % bins if bins else hashed


class ABTest:
    """Bucket-based AB-test assignment.

    A user hits the experiment when murmurhash(f"{user_id}:{expid}") % bins
    falls inside `ranger`. Usage::

        if ABTest(expid='10001', ranger=(0, 99), bins=100).is_hit(user_id):
            ...
    """

    def __init__(self, expid: str = "10001", ranger: Tuple[int, int] = (0, 9), bins: int = 100):
        self._bins = bins
        self._ranger = set(range(*ranger))
        self._expid = expid

    def is_hit(self, value: str = "userid") -> bool:
        return murmurhash(key=self._expid, value=value, str2md5=False) % self._bins in self._ranger


def hash_bins(values: Iterable, bins: int = 3, str2md5: bool = False) -> List[List]:
    """Group values into `bins` stable buckets by murmurhash.

    Same value always lands in the same bucket (given the same `bins`), which
    makes this suitable for consistent sharding / hash-based AB routing.
"""
    buckets: dict = {}
    for v in values:
        idx = murmurhash(key=str(v), value=str(v), bins=bins, str2md5=str2md5)
        buckets.setdefault(idx, []).append(v)
    return list(buckets.values())


class BloomFilter:
    """Fixed-capacity Bloom filter, standard library only.

Never returns false negatives; may return false positives within
    `error_rate` for membership checks of values never added.
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
        """Add a value; return False if it was (probably) already present."""
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
