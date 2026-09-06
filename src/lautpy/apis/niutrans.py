# -*- coding: utf-8 -*-
# @Author: elimes
"""NiuTrans 翻译 API。

密钥在调用时从环境变量解析（NIUTRANS_API_KEY）；HTTP 走统一 client
（超时 + 重试）。接入新 API 时照此文件的写法作为模板。
"""

from typing import Optional

from lautpy.apis.client import get_api_key, request

_ENDPOINT = "https://api.niutrans.com/NiuTransServer/translation"


def translate(
    sentence: str,
    src_lan: str = "auto",
    tgt_lan: str = "en",
    api_key: Optional[str] = None,
) -> str:
    """调用 NiuTrans 翻译一段文本。

    Args:
        sentence: 原文。
        src_lan: 源语言，"auto" 自动识别。
        tgt_lan: 目标语言（如 "en" / "zh"）。
        api_key: 显式密钥；缺省读环境变量 NIUTRANS_API_KEY。

    Returns:
        str: 译文。

    Raises:
        RuntimeError: 接口返回中不含译文（如密钥无效、额度用尽）。
    """
    api_key = api_key or get_api_key("niutrans")
    resp = request(
        "GET",
        _ENDPOINT,
        params={
            "from": src_lan,
            "to": tgt_lan,
            "apikey": api_key,
            "src_text": sentence,
        },
    )
    data = resp.json()
    if "tgt_text" not in data:
        raise RuntimeError(f"NiuTrans error: {data}")
    return data["tgt_text"]
