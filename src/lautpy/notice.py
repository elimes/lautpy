#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Group-chat notifications via webhook robots (WeCom / Feishu).

Security model: webhook URLs are resolved from the environment at call time
and there is no default — notifications can only go where *you* configured:

- ``WECOM_WEBHOOK_URL``   WeCom (企业微信) group robot
- ``FEISHU_WEBHOOK_URL``  Feishu (飞书) custom bot
"""

from typing import Any, List, Optional, Union

from lautpy.apis.client import get_api_key, request

_WECOM_ENDPOINT = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send"
_FEISHU_ENDPOINT = "https://open.feishu.cn/open-apis/bot/v2/hook" + "/{key}"


def _wecom_url(webhook_url: Optional[str]) -> str:
    if webhook_url:
        return webhook_url
    key = get_api_key("wecom", env_var="WECOM_WEBHOOK_URL")
    return key if key.startswith("http") else f"{_WECOM_ENDPOINT}?key={key}"


def _feishu_url(webhook_url: Optional[str]) -> str:
    if webhook_url:
        return webhook_url
    key = get_api_key("feishu", env_var="FEISHU_WEBHOOK_URL")
    return key if key.startswith("http") else _FEISHU_ENDPOINT.format(key=key)


def _chunks(content: str, max_bytes: int = 3800) -> List[str]:
    """Split by UTF-8 *bytes* (WeCom/Feishu cap at ~4096 bytes, and CJK chars
    cost 3 bytes each — splitting by character count would still overflow).
    Never splits inside a multi-byte character.
    """
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [content]
    parts = []
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        while end < len(encoded) and (encoded[end] & 0xC0) == 0x80:  # inside a UTF-8 sequence
            end -= 1
        parts.append(encoded[start:end].decode("utf-8"))
        start = end
    return parts


def wecom(
    content: Union[str, List[Any]],
    title: str = "",
    mentioned_mobile_list: Optional[List[str]] = None,
    webhook_url: Optional[str] = None,
) -> dict:
    """Send a markdown message to a WeCom group robot."""
    if isinstance(content, (list, tuple)):
        content = "\n".join(map(str, content))
    content = f"**{title}**\n{content}".strip() if title else content.strip()
    payload = {
        "msgtype": "markdown",
        "markdown": {"content": content},
    }
    if mentioned_mobile_list:
        payload["markdown"]["mentioned_mobile_list"] = mentioned_mobile_list  # type: ignore[index]

    results = {}
    for i, chunk in enumerate(_chunks(payload["markdown"]["content"])):  # type: ignore[index]
        body = dict(payload)
        body["markdown"] = {"content": chunk}
        resp = request("POST", _wecom_url(webhook_url), json=body)
        results[f"part{i}"] = resp.json()
    return results


def feishu(
    content: Union[str, List[Any]],
    title: str = "",
    webhook_url: Optional[str] = None,
) -> dict:
    """Send a text message to a Feishu custom bot."""
    if isinstance(content, (list, tuple)):
        content = "\n".join(map(str, content))
    content = f"{title}\n{content}".strip()
    # Feishu renders < > as XML tags; escape them (they render as XML tags).
    text = str(content).replace("<", "【").replace(">", "】")

    results = {}
    for i, chunk in enumerate(_chunks(text)):
        body = {"msg_type": "text", "content": {"text": chunk}}
        resp = request("POST", _feishu_url(webhook_url), json=body)
        results[f"part{i}"] = resp.json()
    return results
