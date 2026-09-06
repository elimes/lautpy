import importlib.util

import pytest

from lautpy.hashing import ABTest, BloomFilter, hash_bins, md5

has_sklearn = importlib.util.find_spec("sklearn") is not None


def test_md5_str_and_bytes():
    assert md5("key:value") == md5(b"key:value")
    assert md5("") == "d41d8cd98f00b204e9800998ecf8427e"


@pytest.mark.skipif(not has_sklearn, reason="scikit-learn not installed")
def test_murmurhash_deterministic():
    from lautpy.hashing import murmurhash

    value = murmurhash(str2md5=False, bins=10000)
    assert 0 <= value < 10000
    assert value == murmurhash(str2md5=False, bins=10000)
    assert murmurhash(str2md5=True, bins=10000) == murmurhash(str2md5=True, bins=10000)


@pytest.mark.skipif(not has_sklearn, reason="scikit-learn not installed")
def test_abtest_bucketing():
    ab = ABTest(expid="10001", ranger=(0, 99), bins=100)
    assert isinstance(ab.is_hit("userid"), bool)
    assert ab.is_hit("u1") == ab.is_hit("u1")  # same user, same bucket


class TestBloomFilter:
    def test_add_and_contains(self):
        bloom = BloomFilter(capacity=1000)
        for i in range(100):
            bloom.add(i)
        for i in range(100):
            assert i in bloom  # no false negatives, ever

    def test_add_returns_false_for_duplicates(self):
        bloom = BloomFilter(capacity=1000)
        assert bloom.add("a") is True
        assert bloom.add("a") is False
        assert len(bloom) == 1

    def test_unhashable_values(self):
        bloom = BloomFilter(capacity=1000)
        bloom.add({"a": 1})
        assert {"a": 1} in bloom

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            BloomFilter(capacity=0)
        with pytest.raises(ValueError):
            BloomFilter(error_rate=1.5)


@pytest.mark.skipif(not has_sklearn, reason="scikit-learn not installed")
class TestHashBins:
    def test_partition_and_determinism(self):
        values = [f"u{i}" for i in range(50)]
        bins1 = hash_bins(values, bins=4)
        bins2 = hash_bins(values, bins=4)
        assert bins1 == bins2
        flattened = sorted(v for b in bins1 for v in b)
        assert flattened == sorted(values)  # lossless partition
        assert all(len(b) > 0 for b in bins1)

    def test_bins_respected(self):
        bins = hash_bins([f"u{i}" for i in range(100)], bins=3)
        assert len(bins) <= 3
