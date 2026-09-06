# @Author: elimes
"""原生 Agent 主循环：OpenAI 兼容 tool-calling，零 langchain 依赖。

协议即标准 OpenAI function-calling：
1. 把 system/history/user 消息与工具 schema 一并发出
2. 模型返回 tool_calls → 逐个执行 → 以 role=tool 消息回填 → 继续下一轮
3. 模型不再调用工具时，其 content 即最终回答

扩展能力：
- **结构化输出**：run_agent(response_model=PydanticModel) → 最终回答按模型校验，
  校验失败自动把错误回传给模型重试（需 pydantic，openai 已自带）
- **流式输出**：run_agent_stream(...) 逐段产出最终回答的文本增量
"""

import json
from collections.abc import Iterator
from typing import Any

from lautpy._internal import logger
from lautpy.llm import resolve_model

DEFAULT_SYSTEM = "You are a helpful assistant. Use the provided tools when they help."

_SCHEMA_HINT = (
    "Your final answer MUST be a single JSON object that validates against "
    "this JSON Schema:\n{schema}\nOutput raw JSON only, no markdown fences."
)


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
    response_model: Any | None = None,
) -> Any:
    """执行 Agent 主循环（参数说明见 lautpy.agent.run_agent）。

    Args:
        response_model: pydantic 模型类。给定时最终回答按其校验，返回校验后的
            实例；校验失败会把错误信息回传给模型重试（计入 max_turns）。
    """
    if client is None:
        from lautpy import llm

        client = llm.openai_client(service)

    model_name = resolve_model(service, model)
    tool_map = {t.name: t for t in tools}
    tool_specs = [t.to_openai_spec() for t in tools]

    system_text = system or DEFAULT_SYSTEM
    if response_model is not None:
        system_text = f"{system_text}\n\n{_SCHEMA_HINT.format(schema=_schema(response_model))}"

    messages: list[dict] = [{"role": "system", "content": system_text}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": prompt})

    for _turn in range(max_turns):
        create_kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
        if tool_specs:
            create_kwargs["tools"] = tool_specs
        message = client.chat.completions.create(**create_kwargs).choices[0].message

        if message.tool_calls:  # 工具调用轮：执行并回填，继续下一轮
            messages.append(_assistant_message(message))
            for tc in message.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _execute(tool_map, tc),
                })
            continue

        content = message.content or ""
        if response_model is None:
            return content
        # 结构化输出：校验失败把错误回传，让模型在下一轮修正
        try:
            return _validate_structured(content, response_model)
        except Exception as exc:
            logger.warning(f"structured output invalid (turn {_turn + 1}): {exc}")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Your output failed validation: {exc}\n"
                           f"Output raw JSON matching the schema only.",
            })

    raise RuntimeError(
        f"Agent did not produce a final answer within {max_turns} turns; "
        f"increase max_turns or simplify the task."
    )


def run_agent_stream(
    prompt: str,
    tools: list[Any],
    *,
    service: str = "openai",
    model: str | None = None,
    system: str | None = None,
    history: list[dict] | None = None,
    max_turns: int = 8,
    client: Any | None = None,
) -> Iterator[str]:
    """流式版 Agent：逐段产出文本增量（含工具轮之间的助手文本），最终回答实时流出。

    工具调用轮不产出文本（只在内部执行并回填）；若模型在发起工具调用前
    输出了文本，这些增量也会照常产出。

    Yields:
        str: 文本增量。工具循环结束后产出最终回答的逐段文本。
    """
    if client is None:
        from lautpy import llm

        client = llm.openai_client(service)

    model_name = resolve_model(service, model)
    tool_map = {t.name: t for t in tools}
    tool_specs = [t.to_openai_spec() for t in tools]

    messages: list[dict] = [{"role": "system", "content": system or DEFAULT_SYSTEM}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": prompt})

    for _turn in range(max_turns):
        create_kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
        if tool_specs:
            create_kwargs["tools"] = tool_specs

        stream = client.chat.completions.create(**create_kwargs, stream=True)
        content_parts: list[str] = []
        tool_calls: dict[int, dict] = {}  # index -> 聚合中的 tool_call
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                yield delta.content
            for tc in delta.tool_calls or []:
                slot = tool_calls.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""}
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments

        if not tool_calls:  # 本轮即最终回答，增量已实时产出
            return

        messages.append({
            "role": "assistant",
            "content": "".join(content_parts),
            "tool_calls": [{
                "id": slot["id"],
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["arguments"]},
            } for slot in tool_calls.values()],
        })
        for slot in tool_calls.values():
            messages.append({
                "role": "tool",
                "tool_call_id": slot["id"],
                "content": _execute_named(tool_map, slot["name"], slot["arguments"]),
            })

    raise RuntimeError(
        f"Agent did not produce a final answer within {max_turns} turns; "
        f"increase max_turns or simplify the task."
    )


def _assistant_message(message: Any) -> dict:
    """把 SDK 的 assistant 消息转为可回填的 dict。"""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [{
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in message.tool_calls],
    }


def _schema(response_model: Any) -> str:
    """pydantic v2 模型 → JSON Schema 文本。"""
    return json.dumps(response_model.model_json_schema(), ensure_ascii=False)


def _validate_structured(content: str, response_model: Any) -> Any:
    """把最终回答解析并按 pydantic 模型校验；容错剥离 markdown 代码栅栏。"""
    text = content.strip()
    if text.startswith("```"):  # 常见退化：模型用 ```json ... ``` 包裹
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return response_model.model_validate_json(text.strip())


def _execute(tool_map: dict[str, Any], tool_call) -> str:
    """执行一次非流式工具调用，把结果安全地序列化为字符串。"""
    return _execute_named(tool_map, tool_call.function.name, tool_call.function.arguments)


def _execute_named(tool_map: dict[str, Any], name: str, arguments: str) -> str:
    tool_obj = tool_map.get(name)
    if tool_obj is None:
        return f"Error: unknown tool '{name}'"
    try:
        args = json.loads(arguments or "{}")
        return str(tool_obj(**args))
    except Exception as exc:  # 工具错误回传给模型自行调整，而不是中断循环
        logger.warning(f"tool {name} failed: {exc!r}")
        return f"Error executing tool {name}: {exc!r}"
