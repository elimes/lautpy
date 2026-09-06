import pytest

import lautpy.notice as notice
from lautpy.apis import MissingAPIKeyError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse({"errcode": 0, "StatusCode": 0})

    monkeypatch.setattr(notice, "request", fake_request)
    return calls


def test_wecom_markdown_payload(capture):
    notice.wecom("body text", title="MyTitle", webhook_url="http://hook")
    call = capture[0]
    assert call["method"] == "POST"
    assert call["json"]["msgtype"] == "markdown"
    assert call["json"]["markdown"]["content"].startswith("**MyTitle**")


def test_wecom_list_content_joined(capture):
    notice.wecom(["line1", 2], webhook_url="http://hook")
    assert "line1\n2" in capture[0]["json"]["markdown"]["content"]


def test_wecom_long_content_split(capture):
    notice.wecom("x" * 9000, webhook_url="http://hook")
    assert len(capture) == 3  # 3800 + 3800 + 1400 bytes


def test_wecom_cjk_split_is_byte_safe(capture):
    text = "测" * 3000  # 9000 UTF-8 bytes; a char-based splitter would overflow 4096B
    notice.wecom(text, webhook_url="http://hook")
    assert len(capture) >= 3
    joined = "".join(c["json"]["markdown"]["content"] for c in capture)
    assert joined == text  # no characters lost or split


def test_feishu_escapes_angle_brackets(capture):
    notice.feishu("<b>bold</b>", webhook_url="http://hook")
    text = capture[0]["json"]["content"]["text"]
    assert text == "【b】bold【/b】"
    assert capture[0]["json"]["msg_type"] == "text"


def test_wecom_missing_webhook_url(monkeypatch):
    monkeypatch.delenv("WECOM_WEBHOOK_URL", raising=False)
    with pytest.raises(MissingAPIKeyError):
        notice.wecom("hi")


def test_feishu_missing_webhook_url(monkeypatch):
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)
    with pytest.raises(MissingAPIKeyError):
        notice.feishu("hi")
