#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NiuTrans translation API.

The key is resolved from the environment (NIUTRANS_API_KEY) at call time;
HTTP goes through the shared client with timeout and retry.
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
    """Translate text via NiuTrans; resolves the key from NIUTRANS_API_KEY."""
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
