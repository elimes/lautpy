# @Author: elimes
"""第三方 API 封装的 HTTP 基建。

安全模型：密钥**绝不**写进源码，一律在调用时从环境变量解析。
所有封装的 HTTP 出口统一走 request()（默认超时 + 瞬时故障自动重试）。
"""

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 30  # seconds


class MissingAPIKeyError(RuntimeError):
    """环境中未找到所需密钥时抛出（错误信息会指明应设置的环境变量名）。"""


def get_api_key(service: str, env_var: str | None = None) -> str:
    """从环境变量解析 API 密钥。

    查找顺序：先 ``env_var``（若指定），再默认的 ``<SERVICE>_API_KEY``
    （如 service="niutrans" → NIUTRANS_API_KEY）。

    Args:
        service: 服务名（决定默认环境变量名与报错提示）。
        env_var: 显式指定的环境变量名，优先于默认规则。

    Returns:
        str: 去除首尾空白后的密钥。

    Raises:
        MissingAPIKeyError: 所有候选环境变量都未配置。
    """
    names = [env_var] if env_var else []
    default = f"{service.upper()}_API_KEY"
    if default not in names:
        names.append(default)
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    raise MissingAPIKeyError(
        f"API key for '{service}' not found. Set {' or '.join(names)} "
        f"in your environment. Never hardcode keys in source code."
    )


def http_session(retries: int = 2, backoff_factor: float = 0.5) -> requests.Session:
    """构造带重试策略的 requests.Session。

    Args:
        retries: 瞬时故障（429/5xx）的最大重试次数。
        backoff_factor: 重试退避系数（第 n 次重试等待约 backoff * 2**(n-1) 秒）。
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def request(method: str, url: str, *, timeout: float = DEFAULT_TIMEOUT,
            retries: int = 2, session: requests.Session | None = None,
            **kwargs) -> requests.Response:
    """本包统一的 HTTP 出口：强制超时 + 瞬时故障重试 + 非 2xx 抛异常。

    Args:
        method: HTTP 方法（GET/POST/...）。
        url: 请求地址。
        timeout: 超时秒数（默认 30s；**不要**为省事传 None）。
        retries: 新建 session 的重试次数；传入 session 时该参数无效。
        session: 复用调用方提供的 Session（此时由调用方负责关闭）。
        **kwargs: 透传给 requests.Session.request（params/json/headers/...）。

    Returns:
        requests.Response: 已通过 raise_for_status 校验的响应。

    Raises:
        requests.HTTPError: 响应状态码非 2xx（含重试耗尽后）。
    """
    own_session = session is None
    s = session or http_session(retries=retries)
    try:
        response = s.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response
    finally:
        if own_session:
            s.close()
