import importlib.util
import json

import pytest

from lautpy.paths import file2json, get_config, get_resolve_path, path2list


def test_get_resolve_path(tmp_path):
    f = tmp_path / "sub" / "mod.py"
    f.parent.mkdir()
    f.write_text("")
    assert get_resolve_path("../data/x.json", f) == (tmp_path / "data" / "x.json").resolve()


def test_file2json(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"k": 1}))
    assert file2json(p) == {"k": 1}
    assert file2json(tmp_path / "missing.json") == {}


def test_file2json_yaml_optional(tmp_path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("pyyaml not installed")
    p = tmp_path / "a.yaml"
    p.write_text("k: 1")
    assert file2json(p) == {"k": 1}


def test_path2list(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")
    assert len(path2list(tmp_path, "*.txt")) == 2
    assert path2list(tmp_path / "a.txt") == [tmp_path / "a.txt"]
    assert path2list(tmp_path / "nope") == []


def test_get_config(tmp_path):
    assert get_config({"a": 1}) == {"a": 1}
    assert get_config(None) == {}
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"b": 2}))
    assert get_config(str(p)) == {"b": 2}
