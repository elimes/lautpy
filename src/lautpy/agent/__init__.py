# -*- coding: utf-8 -*-
# @Author: elimes
"""Agent 构建工具集：统一的工具定义 + 双轨执行引擎。

- 原生引擎 ``run_agent``：仅依赖 OpenAI 兼容 API 的 tool-calling，轻量、Python 3.8 可用
- LangChain 引擎 ``build_agent``：包装 langchain 1.x ``create_agent``（Python >= 3.10）
- 两侧共用同一套工具定义：用 ``@tool`` 装饰普通函数即可

模型密钥一律走环境变量（<SVC>_API_KEY / <SVC>_BASE_URL / <SVC>_MODEL），
与 lautpy.llm 的约定一致。
"""

from lautpy.agent.tools import Tool, tool

__all__ = ["Tool", "tool", "run_agent"]


def build_agent(tools=None, *, service="openai", model=None, system_prompt=None, **kwargs):
    """基于 LangChain 1.x ``create_agent`` 构建智能体（需要 Python >= 3.10）。

    按 ``pip install "lautpy[agent]"`` 安装 langchain 全家桶后使用。
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


def run_agent(prompt, tools=None, *, service="openai", model=None, system=None,
              history=None, max_turns: int = 8, client=None) -> str:
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

    Returns:
        str: 模型的最终回答文本。

    Raises:
        RuntimeError: 超过 max_turns 仍未得到最终回答，或未配置模型名。

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
                          client=client)
