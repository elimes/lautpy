#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""NiuTrans translation API.

Ported from meutils.apis.niutrans: the hardcoded key became an environment
variable (NIUTRANS_API_KEY), the openai-async dependency was dropped in
favor of plain requests, and a timeout was added.
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
