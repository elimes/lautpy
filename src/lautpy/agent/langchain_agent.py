# @Author: elimes
"""LangChain 1.x 引擎：用 lautpy 的环境变量约定与工具定义驱动 create_agent。

需 langchain 1.x（``pip install "lautpy[agent]"``）。
"""

from typing import Any

from lautpy.agent.tools import Tool


def build_langchain_agent(
    tools: list[Tool],
    *,
    service: str = "openai",
    model: str | None = None,
    system_prompt: str | None = None,
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
            'langchain is required for build_agent: pip install "lautpy[agent]"'
        ) from e

    from lautpy import llm

    api_key, base_url = llm.resolve_credentials(service)
    model_name = llm.resolve_model(service, model)

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
