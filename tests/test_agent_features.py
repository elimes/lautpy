import json
from types import SimpleNamespace

import pytest

from lautpy.agent import run_agent, run_agent_stream
from lautpy.agent.tools import tool
from lautpy.llm import resolve_credentials, resolve_model


@tool
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b


# ---------- 测试替身 ----------

class FakeClient:
    """按脚本回放响应的假 OpenAI 客户端，记录每次调用的 kwargs。"""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        if idx >= len(self.responses):
            raise AssertionError("response script exhausted")
        return self.responses[idx]


def _final(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=content, tool_calls=None))])


def _tool_call_msg(call_id, name, arguments):
    tc = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="", tool_calls=[tc]))])


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=delta, tool_calls=None))])


def _tool_chunk(index, call_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments) if (name or arguments) else None
    tc = SimpleNamespace(index=index, id=call_id, function=fn)
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, tool_calls=[tc]))])


def _weather_model():
    """延迟加载 pydantic（openai extra 自带），构造测试用响应模型。"""
    pydantic = pytest.importorskip("pydantic")

    class Weather(pydantic.BaseModel):
        city: str
        celsius: float

    return Weather


# ---------- resolve_credentials / resolve_model ----------

def test_resolve_credentials(monkeypatch):
    monkeypatch.setenv("FOO2_API_KEY", "k1")
    monkeypatch.setenv("FOO2_BASE_URL", "https://x")
    assert resolve_credentials("foo2") == ("k1", "https://x")
    assert resolve_credentials("foo2", api_key="k2") == ("k2", "https://x")


def test_resolve_model(monkeypatch):
    monkeypatch.setenv("BAR_MODEL", "m1")
    assert resolve_model("bar") == "m1"
    assert resolve_model("bar", model="m2") == "m2"
    monkeypatch.delenv("BAZ_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="BAZ_MODEL"):
        resolve_model("baz")


# ---------- 结构化输出 ----------

def test_structured_output_success():
    Weather = _weather_model()
    fake = FakeClient([_final('{"city": "北京", "celsius": 25.0}')])
    result = run_agent("北京天气", model="m", client=fake, response_model=Weather)
    assert isinstance(result, Weather)
    assert result.city == "北京"
    # schema 提示已注入 system 消息
    assert "JSON Schema" in fake.calls[0]["messages"][0]["content"]


def test_structured_output_retries_on_invalid():
    Weather = _weather_model()
    fake = FakeClient([
        _final("这不是 JSON"),
        _final('```json\n{"city": "上海", "celsius": 30.0}\n```'),  # 带代码栅栏也能解析
    ])
    result = run_agent("上海天气", model="m", client=fake, response_model=Weather)
    assert result.celsius == 30.0
    # 第二轮的 messages 里包含校验错误回传
    assert any("failed validation" in m.get("content", "") for m in fake.calls[1]["messages"])


def test_structured_output_exhaustion():
    Weather = _weather_model()
    fake = FakeClient([_final("nope")] * 3)
    with pytest.raises(RuntimeError, match="max_turns"):
        run_agent("x", model="m", client=fake, response_model=Weather, max_turns=3)


# ---------- 流式输出 ----------

def test_stream_final_answer_yields_deltas():
    fake = FakeClient([[_chunk("北京"), _chunk("晴"), _chunk("25°C")]])
    deltas = list(run_agent_stream("北京天气", model="m", client=fake))
    assert deltas == ["北京", "晴", "25°C"]
    assert fake.calls[0].get("stream") is True


def test_stream_with_tool_loop():
    fake = FakeClient([
        # 同一轮流内，工具参数分两个增量 chunk 到达（模拟流式聚合）
        [_tool_chunk(0, call_id="c1", name="add", arguments='{"a": 1'),
         _tool_chunk(0, arguments=', "b": 2}')],
        [_chunk("结果是"), _chunk(" 3")],
    ])
    deltas = list(run_agent_stream("1+2=?", [add], model="m", client=fake))
    assert deltas == ["结果是", " 3"]  # 工具轮不产出文本
    assert "3" in fake.calls[1]["messages"][-1]["content"]  # 工具结果回填
    assert fake.calls[1]["messages"][-1]["tool_call_id"] == "c1"


def test_stream_max_turns_exceeded():
    endless = [_tool_chunk(0, call_id="c1", name="add", arguments='{"a": 1, "b": 2}')]
    fake = FakeClient([endless] * 3)
    with pytest.raises(RuntimeError, match="max_turns"):
        list(run_agent_stream("x", [add], model="m", client=fake, max_turns=2))
