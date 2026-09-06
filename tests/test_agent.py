import json
from types import SimpleNamespace

import pytest

from lautpy.agent import run_agent
from lautpy.agent.tools import Tool, tool


@tool
def get_weather(city: str, days: int = 1) -> str:
    """查询指定城市未来几天的天气。"""
    return f"{city}: 晴 25°C (next {days}d)"


def test_tool_schema_from_hints():
    assert get_weather.name == "get_weather"
    assert "天气" in get_weather.description
    params = get_weather.parameters
    assert params["properties"]["city"] == {"type": "string"}
    assert params["properties"]["days"] == {"type": "integer"}
    assert params["required"] == ["city"]  # days has a default


def test_tool_requires_docstring():
    with pytest.raises(ValueError, match="docstring"):
        tool(lambda city: city)


def test_tool_openai_spec_shape():
    spec = get_weather.to_openai_spec()
    assert spec["type"] == "function"
    fn = spec["function"]
    assert fn["name"] == "get_weather" and "parameters" in fn


def test_tool_call_passthrough():
    assert get_weather("上海", days=3) == "上海: 晴 25°C (next 3d)"


def test_tool_to_langchain_optional():
    import importlib.util

    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("langchain not installed")
    lc_tool = get_weather.to_langchain()
    assert lc_tool.name == "get_weather"


# ---------- 原生 agent 循环（mock 客户端，无网络） ----------

def _make_client(responses):
    """按脚本顺序返回响应的假 OpenAI 客户端，并记录每次收到的 messages/tools。"""
    calls = []

    class _Fake:
        class chat:  # noqa: N801 - mimic SDK attribute chain
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    calls.append(kwargs)
                    idx = len(calls) - 1
                    if idx >= len(responses):
                        raise AssertionError("script exhausted")
                    return responses[idx]

    return _Fake(), calls


def _assistant(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=content, tool_calls=tool_calls))])


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(
        name=name, arguments=json.dumps(arguments)))


def test_run_agent_tool_loop():
    responses = [
        _assistant(tool_calls=[_tool_call("c1", "get_weather", {"city": "北京"})]),
        _assistant(content="北京今天晴，25°C。"),
    ]
    fake, calls = _make_client(responses)
    answer = run_agent("北京天气？", [get_weather], model="test-model", client=fake)

    assert answer == "北京今天晴，25°C。"
    assert len(calls) == 2
    # 第一轮带了工具 schema
    assert calls[0]["tools"][0]["function"]["name"] == "get_weather"
    # 第二轮回填了工具结果
    tool_msg = calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert "北京" in tool_msg["content"]


def test_run_agent_plain_conversation():
    fake, calls = _make_client([_assistant(content="你好")])
    assert run_agent("hi", model="m", client=fake) == "你好"
    assert "tools" not in calls[0]  # 无工具时不发送 tools 字段


def test_run_agent_unknown_tool_reported_not_fatal():
    responses = [
        _assistant(tool_calls=[_tool_call("c1", "no_such_tool", {})]),
        _assistant(content="done"),
    ]
    fake, _ = _make_client(responses)
    assert run_agent("x", [get_weather], model="m", client=fake) == "done"


def test_run_agent_max_turns_exceeded():
    endless = _assistant(tool_calls=[_tool_call("c1", "get_weather", {"city": "x"})])
    fake, _ = _make_client([endless] * 5)
    with pytest.raises(RuntimeError, match="max_turns"):
        run_agent("loop", [get_weather], model="m", client=fake, max_turns=3)


def test_run_agent_requires_model(monkeypatch):
    monkeypatch.delenv("NOBODY_MODEL", raising=False)
    monkeypatch.delenv("NOBODY_API_KEY", raising=False)
    fake, _ = _make_client([])
    with pytest.raises(RuntimeError, match="NOBODY_MODEL"):
        run_agent("hi", service="nobody", client=fake)


def test_history_and_system_are_sent():
    fake, calls = _make_client([_assistant(content="ok")])
    run_agent("q", model="m", client=fake,
              system="你是猫娘", history=[{"role": "user", "content": "早"}])
    msgs = calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "你是猫娘"}
    assert {"role": "user", "content": "早"} in msgs[:-1]


def test_tool_is_a_tool():
    assert isinstance(get_weather, Tool)
