# @Author: elimes
"""无密钥 Web 工具：短链、二维码、流式下载、TCP 端口探测。"""

import socket
from pathlib import Path
from urllib.parse import quote, urlparse

from lautpy.apis.client import DEFAULT_TIMEOUT, http_session, request


def shorten_url(url: str, shortener: str = "dagd") -> str:
    """生成短链接（公共服务，无需密钥）。

    Args:
        url: 原始长链接。
        shortener: 服务选择，``'dagd'``（da.gd，速度快）或 ``'tinyurl'``。

    Returns:
        str: 短链接文本。

    Raises:
        ValueError: shortener 不受支持。
    """
    if shortener == "dagd":
        resp = request("GET", "https://da.gd/shorten", params={"url": url})
        return resp.text.strip()
    if shortener == "tinyurl":
        resp = request("GET", "https://tinyurl.com/api-create.php", params={"url": url})
        return resp.text.strip()
    raise ValueError(f"Unsupported shortener: {shortener!r} (use 'dagd' or 'tinyurl')")


def data2qrcodeurl(data: str | bytes) -> str:
    """把文本编码为二维码图片 URL（公共服务，直接塞进 <img src> 即可显示）。

    Args:
        data: 要编码的内容；bytes 按 utf-8 解码。
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    return f"https://api.isoyu.com/qr/?m=1&e=L&p=20&url={quote(data)}"


def download(url: str, filename: str | Path | None = None, chunk_size: int = 8192) -> Path:
    """流式下载文件（带超时与重试）。

    Args:
        url: 下载地址。
        filename: 保存路径；缺省时从 URL 路径推断，推断不出用 download.bin。
        chunk_size: 流式写入的分块字节数。

    Returns:
        Path: 实际写入的文件路径。
    """
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
    """TCP 端口探测：timeout 秒内能建立连接即视为开放。

    Args:
        host: 主机名或 IP。
        port: 端口号。
        timeout: 连接超时秒数。
    """
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0
