# @Author: elimes
"""Agent 构建工具集：统一的工具定义 + 双轨执行引擎。

- 原生引擎 ``run_agent`` / ``run_agent_stream``：仅依赖 OpenAI 兼容 API 的
  tool-calling，轻量（openai extra）；配 ``Harness`` 守卫与
  ``make_todo_tool`` / ``fs_tools`` / ``make_subagent_tool`` 等 harness 件
- LangChain 引擎 ``build_agent``：包装 langchain 1.x ``create_agent``（agent extra）
- 两侧共用同一套工具定义：用 ``@tool`` 装饰普通函数即可

模型凭证与模型名一律走环境变量（<SVC>_API_KEY / <SVC>_BASE_URL / <SVC>_MODEL），
与 lautpy.llm 的约定一致。
"""

from collections.abc import Iterator
from typing import Any

from lautpy.agent.harness import Harness, fs_tools, make_subagent_tool, make_todo_tool
from lautpy.agent.tools import Tool, tool

__all__ = [
    "Tool", "tool", "run_agent", "run_agent_stream",
    "Harness", "make_todo_tool", "fs_tools", "make_subagent_tool",
]


def build_agent(tools=None, *, service="openai", model=None, system_prompt=None, **kwargs):
    """基于 LangChain 1.x ``create_agent`` 构建智能体（需要 agent extra）。

    密钥/接入点从 <SVC>_API_KEY / <SVC>_BASE_URL 环境变量解析，
    模型名取 model 参数或 <SVC>_MODEL 环境变量。

    Args:
        tools: ``@tool`` 装饰的工具列表（自动转换为 langchain 工具）。
        service: 服务名，决定环境变量前缀（同 lautpy.llm 约定）。
        model: 模型名；缺省读 <SVC>_MODEL 环境变量。
        system_prompt: 系统提示词。
        **kwargs: 透传给 ChatOpenAI（如 temperature）。

    Returns:
        langgraph 图对象：``agent.invoke({"messages": [...]})`` 调用。
    """
    from lautpy.agent.langchain_agent import build_langchain_agent

    return build_langchain_agent(tools or [], service=service, model=model,
                                 system_prompt=system_prompt, **kwargs)


def run_agent(
    prompt: str,
    tools=None,
    *,
    service: str = "openai",
    model: str | None = None,
    system: str | None = None,
    history: list[dict] | None = None,
    max_turns: int = 8,
    client: Any | None = None,
    response_model: Any | None = None,
    harness: Harness | None = None,
) -> Any:
    """原生 Agent 主循环：LLM ⇄ 工具 轮转，直到模型给出最终回答。

    不依赖 langchain，只要一个 OpenAI 兼容端点（openai extra）即可运行。

    Args:
        prompt: 用户输入。
        tools: ``@tool`` 装饰的工具列表；None 则退化为普通对话。
        service: 服务名（决定 <SVC>_API_KEY / <SVC>_MODEL 等环境变量）。
        model: 模型名；缺省读 <SVC>_MODEL 环境变量。
        system: 系统提示词；缺省为通用助手设定。
        history: 历史消息列表（[{"role": ..., "content": ...}, ...]）。
        max_turns: 工具循环轮数上限，防止死循环。
        client: 注入自定义客户端（测试用）；缺省走 lautpy.llm.openai_client。
        response_model: pydantic 模型类；给定时最终回答按其校验并返回实例，
            校验失败自动回传错误让模型重试（需 pydantic，openai 已自带）。
        harness: ``Harness`` 守卫配置（结果截断/重复检测/审批门/字符预算/历史压缩）。

    Returns:
        str | BaseModel: 未指定 response_model 时返回回答文本；指定时返回
            校验后的 pydantic 实例。

    Raises:
        RuntimeError: 超过 max_turns 仍未得到（合法的）最终回答，或未配置模型名。

    Example::

        @tool
        def get_weather(city: str) -> str:
            '''查询城市天气。'''
            return f"{city}: 晴 25°C"

        run_agent("北京今天天气如何？", tools=[get_weather], service="moonshot")
    """
    from lautpy.agent.mini import run_agent_loop

    return run_agent_loop(prompt, tools or [], service=service, model=model,
                          system=system, history=history, max_turns=max_turns,
                          client=client, response_model=response_model,
                          harness=harness)


def run_agent_stream(
    prompt: str,
    tools=None,
    *,
    service: str = "openai",
    model: str | None = None,
    system: str | None = None,
    history: list[dict] | None = None,
    max_turns: int = 8,
    client: Any | None = None,
    harness: Harness | None = None,
) -> Iterator[str]:
    """流式版原生 Agent：逐段产出文本增量（生成器）。

    工具调用轮在内部静默执行并回填，不产出文本；最终回答的文本增量实时产出。
    参数含义同 run_agent（不支持 response_model；harness 的压缩/预算不生效）。

    Example::

        for delta in run_agent_stream("讲个故事", service="moonshot"):
            print(delta, end="", flush=True)
    """
    from lautpy.agent.mini import run_agent_stream as _stream

    return _stream(prompt, tools or [], service=service, model=model,
                   system=system, history=history, max_turns=max_turns,
                   client=client, harness=harness)
