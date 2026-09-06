# -*- coding: utf-8 -*-
# @Author: elimes
"""LangChain 1.x 引擎：用 lautpy 的环境变量约定与工具定义驱动 create_agent。

仅 Python >= 3.10（langchain 1.x 的要求），``pip install "lautpy[agent]"``。
"""

import os
from typing import Any, List, Optional

from lautpy.agent.tools import Tool


def _resolve_model_name(service: str, model: Optional[str]) -> str:
    name = model or os.getenv(f"{service.upper()}_MODEL")
    if not name:
        raise RuntimeError(
            f"No model name for service '{service}'. Set {service.upper()}_MODEL "
            f"or pass model=... explicitly."
        )
    return name


def build_langchain_agent(
    tools: List[Tool],
    *,
    service: str = "openai",
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **kwargs,
):
    """构建 langchain create_agent 智能体（返回 langgraph 图对象）。

    Args:
        tools: lautpy ``@tool`` 工具列表，自动转为 langchain 工具。
        service: 服务名（<SVC>_API_KEY / <SVC>_BASE_URL / <SVC>_MODEL）。
        model: 模型名，缺省读环境变量。
        system_prompt: 系统提示词。
        **kwargs: 透传给 ChatOpenAI（temperature 等）。

    环境中未配置密钥时，由 ChatOpenAI/openai 库按官方约定报错。
    """
    try:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            'langchain is required for build_agent: pip install "lautpy[agent]" '
            "(Python >= 3.10)"
        ) from e

    from lautpy import llm

    api_key, base_url = llm._resolve(service, None, None)
    model_name = _resolve_model_name(service, model)

    chat_kwargs: dict = {"model": model_name, "api_key": api_key}
    if base_url:
        chat_kwargs["base_url"] = base_url
    chat_kwargs.update(kwargs)

    return create_agent(
        ChatOpenAI(**chat_kwargs),
        tools=[t.to_langchain() for t in tools],
        system_prompt=system_prompt,
    )


def chat(agent: Any, text: str) -> str:
    """单轮对话便捷封装：调用 agent 并返回最后一条消息文本。

    Args:
        agent: build_agent 返回的 langgraph 图对象。
        text: 用户输入。
    """
    result = agent.invoke({"messages": [{"role": "user", "content": text}]})
    return result["messages"][-1].content
