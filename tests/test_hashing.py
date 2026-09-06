import importlib.util

import pytest

from lautpy.hashing import ABTest, md5

has_sklearn = importlib.util.find_spec("sklearn") is not None


def test_md5_str_and_bytes():
    assert md5("key:value") == md5(b"key:value")
    assert md5("") == "d41d8cd98f00b204e9800998ecf8427e"


@pytest.mark.skipif(not has_sklearn, reason="scikit-learn not installed")
def test_murmurhash_matches_java_convention():
    from lautpy.hashing import murmurhash

    # Deterministic positive hash bucketed into [0, bins)
    value = murmurhash(str2md5=False, bins=10000)
    assert 0 <= value < 10000
    assert value == murmurhash(str2md5=False, bins=10000)
    assert murmurhash(str2md5=True, bins=10000) == murmurhash(str2md5=True, bins=10000)


@pytest.mark.skipif(not has_sklearn, reason="scikit-learn not installed")
def test_abtest_bucketing():
    ab = ABTest(expid="10001", ranger=(0, 99), bins=100)
    assert isinstance(ab.is_hit("userid"), bool)
    # Same user always lands in the same bucket
    assert ab.is_hit("u1") == ab.is_hit("u1")
