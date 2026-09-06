#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hash helpers: md5 digest and Java-compatible murmurhash bucketing (AB tests).

Adapted from meutils.hash_utils (MIT License, Copyright (c) yuanjie).
"""

import hashlib as _hashlib
from typing import Optional, Tuple, Union

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

    Results match the Java Guava murmur3_32 convention used by the original
    meutils implementation. Requires scikit-learn.
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
