# @Author: elimes
"""原生 Agent 主循环：OpenAI 兼容 tool-calling，零 langchain 依赖。

协议即标准 OpenAI function-calling：
1. 把 system/history/user 消息与工具 schema 一并发出
2. 模型返回 tool_calls → 逐个执行 → 以 role=tool 消息回填 → 继续下一轮
3. 模型不再调用工具时，其 content 即最终回答
"""

import json
import os
from typing import Any

from lautpy._internal import logger

DEFAULT_SYSTEM = "You are a helpful assistant. Use the provided tools when they help."


def _resolve_model(service: str, model: str | None) -> str:
    """模型名解析：显式参数优先，其次 <SVC>_MODEL 环境变量。"""
    name = model or os.getenv(f"{service.upper()}_MODEL")
    if not name:
        raise RuntimeError(
            f"No model name for service '{service}'. Set {service.upper()}_MODEL "
            f"or pass model=... explicitly."
        )
    return name


def run_agent_loop(
    prompt: str,
    tools: list[Any],
    *,
    service: str = "openai",
    model: str | None = None,
    system: str | None = None,
    history: list[dict] | None = None,
    max_turns: int = 8,
    client: Any | None = None,
) -> str:
    """执行 Agent 主循环（参数说明见 lautpy.agent.run_agent）。"""
    if client is None:
        from lautpy import llm

        client = llm.openai_client(service)

    model_name = _resolve_model(service, model)
    tool_map = {t.name: t for t in tools}
    tool_specs = [t.to_openai_spec() for t in tools]

    messages: list[dict] = [{"role": "system", "content": system or DEFAULT_SYSTEM}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": prompt})

    create_kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
    if tool_specs:
        create_kwargs["tools"] = tool_specs

    for _turn in range(max_turns):
        resp = client.chat.completions.create(**create_kwargs)
        message = resp.choices[0].message

        if not message.tool_calls:  # 没有工具调用 → 最终回答
            return message.content or ""

        # 回填助手的工具调用请求，再逐个执行工具并回填结果
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in message.tool_calls],
        })
        for tc in message.tool_calls:
            result = _execute(tool_map, tc)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    raise RuntimeError(
        f"Agent did not produce a final answer within {max_turns} turns; "
        f"increase max_turns or simplify the task."
    )


def _execute(tool_map: dict[str, Any], tool_call) -> str:
    """执行一次工具调用，把结果安全地序列化为字符串。"""
    name = tool_call.function.name
    tool_obj = tool_map.get(name)
    if tool_obj is None:
        return f"Error: unknown tool '{name}'"
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        return str(tool_obj(**args))
    except Exception as exc:  # 工具错误回传给模型自行调整，而不是中断循环
        logger.warning(f"tool {name} failed: {exc!r}")
        return f"Error executing tool {name}: {exc!r}"
