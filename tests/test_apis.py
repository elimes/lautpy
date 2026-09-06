
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
