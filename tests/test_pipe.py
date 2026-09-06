import pytest

from collections import Counter

from lautpy.pipe import (
    Pipe, xchain_, xchain_dict, xCounterUpdate, xDictRemove, xDictValues,
    xdrop, xdrop_, xfilter_, xgetitem, xlist, xmap_, xnext, xsse_parser,
    xsort, xAsyncio,
)


def test_pipe_partial_application():
    double = Pipe(lambda x, n: x * n)  # lists repeat, ints multiply
    assert [1, 2] | double(2) == [1, 2, 1, 2]
    assert (2 | double(3)) == 6


def test_eager_variants():
    assert [1, 2, 3] | xmap_(lambda x: x * 2) == [2, 4, 6]
    assert [1, 2, 3, 4] | xfilter_(lambda x: x % 2 == 0) == [2, 4]
    assert [[1, 2], [3]] | xchain_ == [1, 2, 3]
    assert [1, 2, 3, 4] | xdrop_(lambda x: x < 3) == [3, 4]
    assert iter([1, 2, 3, 4]) | xdrop(lambda x: x < 3) | xlist == [3, 4]


def test_dict_pipes():
    assert [{"a": 1}, {"b": 2}] | xchain_dict == {"a": 1, "b": 2}
    assert {"a": 1, "b": 2, "c": 3} | xDictValues(["a", "z"], default=0) == (1, 0)
    d = {"a": 1, "b": 2}
    assert d | xDictRemove(["a"]) == {"b": 2}
    assert [(0, "x"), (1, "y")] | xgetitem(1) | xchain_ == ["x", "y"]


def test_counter_update():
    counter = [["w1", "w2"], ["w1"]] | xCounterUpdate
    assert counter == {"w1": 2, "w2": 1}
    base = Counter({"w1": 1})  # must be a Counter: dict.update(pairing) semantics differ
    [["w1"]] | xCounterUpdate(base)
    assert base["w1"] == 2  # accumulates in place


def test_xnext_and_xsort():
    it = [1, 2] | xnext  # pull-based iterator
    assert next(it) == 1 and next(it) == 2
    assert [3, 1, 2] | xsort(reverse=True) == [3, 2, 1]


def test_xasyncio():
    import asyncio

    async def job(i):
        return i * 2

    assert [job(i) for i in range(4)] | xAsyncio == [0, 2, 4, 6]


def test_sse_parser():
    lines = ['data: {"a": 1}', "data: [DONE]", "not-data", 'data: {"b": 2}']
    assert lines | xsse_parser == [{"a": 1}, {"b": 2}]


def test_no_namespace_pollution():
    import lautpy

    assert not hasattr(lautpy, "json")
    assert not hasattr(lautpy, "np")
    assert hasattr(lautpy, "xtqdm") and hasattr(lautpy, "timer") and hasattr(lautpy, "logger")
