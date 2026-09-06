#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Keyless web utilities ported from meutils.apis.common.

Only the key-free services were ported; the ft12 variant (hardcoded key) was
dropped deliberately.
"""

from typing import Union
from urllib.parse import quote

from lautpy.apis.client import request


def shorten_url(url: str, shortener: str = "dagd") -> str:
    """Shorten a URL via a keyless service.

    shortener: 'dagd' (https://da.gd, fast) or 'tinyurl'.
    """
    if shortener == "dagd":
        resp = request("GET", "https://da.gd/shorten", params={"url": url})
        return resp.text.strip()
    if shortener == "tinyurl":
        resp = request("GET", "https://tinyurl.com/api-create.php", params={"url": url})
        return resp.text.strip()
    raise ValueError(f"Unsupported shortener: {shortener!r} (use 'dagd' or 'tinyurl')")


def data2qrcodeurl(data: Union[str, bytes]) -> str:
    """Encode data as a QR-code image URL (keyless public service)."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    return f"https://api.isoyu.com/qr/?m=1&e=L&p=20&url={quote(data)}"
