#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Keyless web utilities: URL shortening and QR-code generation."""

import socket
from pathlib import Path
from typing import Union
from urllib.parse import quote, urlparse

from lautpy.apis.client import DEFAULT_TIMEOUT, http_session, request


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


def download(url: str, filename: Union[str, Path, None] = None, chunk_size: int = 8192) -> Path:
    """Download `url` to `filename` (default: derived from the URL); returns the path."""
    with http_session() as session:
        with session.get(url, stream=True, timeout=DEFAULT_TIMEOUT) as response:
            response.raise_for_status()
            if filename is None:
                name = Path(urlparse(str(response.url)).path).name or "download.bin"
                filename = Path(name)
            p = Path(filename)
            with p.open("wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
    return p


def is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """TCP port check: True if a connection succeeds within `timeout` seconds."""
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0
