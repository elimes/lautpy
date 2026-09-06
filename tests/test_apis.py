
import pytest

from lautpy.apis import MissingAPIKeyError, get_api_key
from lautpy.apis.client import http_session


def test_get_api_key_from_default_env(monkeypatch):
    monkeypatch.setenv("FOO_API_KEY", "  secret123  ")
    assert get_api_key("foo") == "secret123"  # stripped


def test_get_api_key_from_explicit_env(monkeypatch):
    monkeypatch.setenv("MY_CUSTOM_KEY", "xyz")
    assert get_api_key("foo", env_var="MY_CUSTOM_KEY") == "xyz"


def test_get_api_key_missing(monkeypatch):
    monkeypatch.delenv("BAR_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="BAR_API_KEY"):
        get_api_key("bar")


def test_http_session_has_retry():
    session = http_session()
    adapter = session.get_adapter("https://example.com")
    assert adapter.max_retries.total == 2
    assert adapter.max_retries.status_forcelist[0] == 429


def test_no_hardcoded_keys_in_apis_source():
    """Guard: source files under lautpy/apis must not embed long secret-like literals."""
    import re
    from pathlib import Path

    pattern = re.compile(
        r"(api_key|apikey|authorization|token|password)\s*[=:]\s*[\"'][A-Za-z0-9_\-\.]{16,}[\"']",
        re.IGNORECASE,
    )
    apis_dir = Path(__file__).resolve().parents[1] / "src" / "lautpy" / "apis"
    for f in apis_dir.rglob("*.py"):
        assert not pattern.search(f.read_text(encoding="utf-8")), f"hardcoded key in {f.name}"


def test_shared_session_identity():
    from lautpy.apis import get_shared_session

    assert get_shared_session() is get_shared_session()  # 进程级复用


def test_request_uses_shared_session(monkeypatch):
    """默认走共享 Session；显式传 session 时不落共享对象。"""
    from types import SimpleNamespace

    from lautpy.apis import client

    shared = client.get_shared_session()
    seen = {}

    def fake_request(method, url, **kwargs):
        seen["url"] = url
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(shared, "request", fake_request)
    client.request("GET", "https://example.com/a")
    assert seen["url"] == "https://example.com/a"

    own = client.http_session()
    used_own = {}
    monkeypatch.setattr(own, "request", lambda m, u, **kw: used_own.update(url=u) or
                        SimpleNamespace(raise_for_status=lambda: None))
    client.request("GET", "https://example.com/b", session=own)
    assert used_own["url"] == "https://example.com/b" and seen["url"] != used_own["url"]
