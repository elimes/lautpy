import pytest

from lautpy.pipe import xBloomFilter, xHashBins, xUniquePlus


def test_xuniqueplus_unhashable():
    data = [{"id": 1}, {"id": 2}, {"id": 1}]
    result = data | xUniquePlus
    assert len(result) == 2 and result[0] == {"id": 1}


def test_xuniqueplus_key_fn():
    data = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 1, "v": "c"}]
    result = data | xUniquePlus(lambda x: x["id"])
    assert [x["v"] for x in result] == ["a", "b"]


def test_xbloomfilter_pipe():
    bloom = [i for i in range(100)] | xBloomFilter(capacity=1000)
    assert 42 in bloom and "nope" not in bloom


def test_xhashbins():
    pytest.importorskip("sklearn")
    bins = [f"u{i}" for i in range(30)] | xHashBins(3)
    assert len(bins) <= 3
    assert sum(len(b) for b in bins) == 30
