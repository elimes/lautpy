# @Author: elimes
"""OpenAI 兼容客户端工厂与服务凭证解析。

客户端按需构建并缓存（键为 service + key + base_url）；厂商列表开放——
任何暴露 OpenAI 兼容 API 的服务都能接::

    export MOONSHOT_API_KEY=...          # 可选: MOONSHOT_BASE_URL / MOONSHOT_MODEL
    export ZHIPUAI_API_KEY=...           # 可选: ZHIPUAI_BASE_URL / ZHIPUAI_MODEL

    from lautpy.llm import openai_client
    client = openai_client("moonshot")
"""

import os
from functools import lru_cache
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def resolve_credentials(
    service: str, api_key: str | None = None, base_url: str | None = None
) -> tuple[str, str | None]:
    """解析某服务的凭证（公开 API，供 notice/agent 等模块与用户代码复用）。

    Args:
        service: 服务名，决定环境变量前缀（如 "moonshot" → MOONSHOT_API_KEY）。
        api_key: 显式密钥，优先于环境变量。
        base_url: 显式接入点，优先于环境变量；缺省返回 None（官方默认端点）。

    Returns:
        tuple[str, str | None]: (api_key, base_url)。

    Raises:
        RuntimeError: 环境与入参均未提供密钥。
    """
    upper = service.upper()
    api_key = api_key or os.getenv(f"{upper}_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No API key for service '{service}'. Set {upper}_API_KEY or pass api_key=... "
            f"(keys are resolved from the environment, never hardcoded)."
        )
    base_url = base_url or os.getenv(f"{upper}_BASE_URL")
    return api_key, base_url


def resolve_model(service: str, model: str | None = None) -> str:
    """解析模型名：显式参数优先，其次 <SVC>_MODEL 环境变量。

    Raises:
        RuntimeError: 两处都未提供。
    """
    name = model or os.getenv(f"{service.upper()}_MODEL")
    if not name:
        raise RuntimeError(
            f"No model name for service '{service}'. Set {service.upper()}_MODEL "
            f"or pass model=... explicitly."
        )
    return name


def openai_client(
    service: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """获取某服务的 OpenAI 兼容客户端（同参数重复调用返回缓存实例）。

    Args:
        service: 服务名，决定读取的环境变量前缀，如 "moonshot" →
            MOONSHOT_API_KEY / MOONSHOT_BASE_URL；"openai" 走官方默认。
        api_key: 显式指定密钥（优先于环境变量）；通常留空走环境。
        base_url: 显式指定接入点（优先于环境变量）；留空时官方服务无需设置。

    Returns:
        openai.OpenAI: 可直接调用 chat.completions / embeddings 等接口的客户端。

    Raises:
        RuntimeError: 环境与入参均未提供密钥。
        ImportError: 未安装 openai（``pip install "lautpy[llm]"``）。

    Example::

        client = openai_client("moonshot")
        client.chat.completions.create(model="moonshot-v1-8k",
                                       messages=[{"role": "user", "content": "你好"}])
    """
    if OpenAI is None:
        raise ImportError("openai is required: pip install openai")
    key, url = resolve_credentials(service, api_key, base_url)
    return _cached_client(service, key, url)


@lru_cache(maxsize=32)
def _cached_client(service: str, api_key: str, base_url: str | None) -> Any:
    # dict[str, Any]：dict[str, str] 解包进 OpenAI 的多参重载会被 mypy 逐参报错
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
