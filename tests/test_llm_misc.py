import importlib.util

import pytest

import lautpy.llm as llm
from lautpy.paths import pkl_dump, pkl_load

has_openai = importlib.util.find_spec("openai") is not None


def test_pkl_roundtrip(tmp_path):
    data = {"a": [1, 2, 3], "b": {"nested": True}}
    p = pkl_dump(data, tmp_path / "obj.pkl")
    assert pkl_load(p) == data


def test_openai_missing_key(monkeypatch):
    monkeypatch.delenv("NOBODY_API_KEY", raising=False)
    if not has_openai:
        pytest.skip("openai not installed")
    with pytest.raises(RuntimeError, match="NOBODY_API_KEY"):
        llm.openai_client("nobody")


@pytest.mark.skipif(not has_openai, reason="openai not installed")
def test_openai_client_from_env(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key-123")
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
    client = llm.openai_client("moonshot")
    assert str(client.base_url).rstrip("/") == "https://api.moonshot.cn/v1"
    # cached per (service, key, base_url)
    assert llm.openai_client("moonshot") is client


def test_openai_optional_dependency(monkeypatch):
    """Without the openai package installed, a clear ImportError is raised."""
    if has_openai:
        pytest.skip("openai installed; nothing to simulate")
    with pytest.raises(ImportError, match="openai is required"):
        llm.openai_client("openai", api_key="k")
