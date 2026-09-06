"""harness 工程件测试（全 mock，无网络）。"""

import threading
from types import SimpleNamespace

import pytest
from test_agent_features import FakeClient, _final

from lautpy.agent import Harness, fs_tools, make_subagent_tool, make_todo_tool, run_agent
from lautpy.agent.harness import truncate_result
from lautpy.agent.tools import tool


def _tool_turn(call_id, name, arguments="{}"):
    tc = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="", tool_calls=[tc]))])


@pytest.mark.parametrize("text,limit,expect_marker", [
    ("short", 100, False),
    ("x" * 300, 100, True),
])
def test_truncate_result(text, limit, expect_marker):
    out = truncate_result(text, max_chars=limit)
    if expect_marker:
        assert len(out) < 300 and "[truncated" in out
    else:
        assert out == text


def test_harness_truncates_tool_results_in_loop():
    @tool
    def big() -> str:
        """返回一大段输出。"""
        return "y" * 5000

    fake = FakeClient([_tool_turn("c1", "big"), _final("done")])
    run_agent("go", [big], model="m", client=fake,
              harness=Harness(tool_result_max_chars=100))
    tool_msg = fake.calls[1]["messages"][-1]
    assert len(tool_msg["content"]) < 300 and "[truncated" in tool_msg["content"]


def test_harness_repeat_detection():
    calls = {"n": 0}

    @tool
    def echo(x: int) -> int:
        """原样返回。"""
        calls["n"] += 1
        return x

    loop_resp = _tool_turn("c1", "echo", '{"x": 1}')
    fake = FakeClient([loop_resp, loop_resp, loop_resp, _final("gave up")])

    answer = run_agent("loop", [echo], model="m", client=fake,
                       harness=Harness(max_repeats=2))
    assert answer == "gave up"
    assert calls["n"] == 1  # 第二次重复起不再执行
    assert "change strategy" in fake.calls[3]["messages"][-1]["content"]


def test_harness_approval_gate():
    executed = []

    @tool
    def danger(x: int) -> int:
        """危险操作。"""
        executed.append(x)
        return x

    fake = FakeClient([_tool_turn("c1", "danger", '{"x": 1}'), _final("ok")])
    run_agent("go", [danger], model="m", client=fake,
              harness=Harness(approval=lambda name, args: False))
    assert executed == []  # 被拒，未执行
    assert "denied by the approval gate" in fake.calls[1]["messages"][-1]["content"]


def test_harness_budget_forces_final():
    @tool
    def filler() -> str:
        """填充上下文。"""
        return "z" * 500

    fake = FakeClient([_tool_turn("c1", "filler"), _final("budget answer"), _final("again")])
    answer = run_agent("go", [filler], model="m", client=fake,
                       harness=Harness(max_context_chars=400))  # 工具结果回填后必超限
    assert answer == "budget answer"
    # 超预算后的一次调用不再携带 tools，并带强制收尾指令
    assert "tools" not in fake.calls[1]
    assert any("Context budget exceeded" in m.get("content", "")
               for m in fake.calls[1]["messages"])


def test_harness_compact_naive_and_summarizer():
    history = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    # 朴素截断：长 history 被裁到 system + keep_recent + user prompt
    fake = FakeClient([_final("done")])
    run_agent("q", model="m", client=fake, history=history,
              harness=Harness(compact={"keep_recent": 3}))
    # user prompt 已在压缩前的消息体里，与 history 一起参与"保留最近 3 条"
    assert len(fake.calls[0]["messages"]) == 1 + 3

    # 摘要模式：旧消息被 summarizer 替代
    fake2 = FakeClient([_final("done")])
    run_agent("q", model="m", client=fake2, history=history,
              harness=Harness(compact={"keep_recent": 3,
                                       "summarizer": lambda msgs: f"summary of {len(msgs)}"}))
    msgs = fake2.calls[0]["messages"]
    assert any("[earlier context summarized]" in m.get("content", "") for m in msgs)
    assert "summary of 18" in msgs[1]["content"]  # system 之外的全部旧消息被摘要


def test_make_todo_tools():
    todo_write, todo_mark, todo_read = make_todo_tool()
    assert todo_write(["t1", "t2", "t3"]) == "plan set: 3 tasks"
    assert todo_mark(1, "in_progress") == "task 1 -> in_progress"
    assert todo_read() == [
        {"task": "t1", "status": "pending"},
        {"task": "t2", "status": "in_progress"},
        {"task": "t3", "status": "pending"},
    ]
    assert "out of range" in todo_mark(9, "done")
    assert todo_write.parameters["properties"]["tasks"]["type"] == "array"


def test_fs_tools_sandbox(tmp_path):
    fs_read, fs_write, fs_list = fs_tools(tmp_path, max_read_chars=100)
    assert fs_write("notes/a.txt", "hello") == "written: 5 chars -> notes/a.txt"
    assert fs_read("notes/a.txt") == "hello"
    assert fs_list(".") == ["notes/"]
    assert fs_list("notes") == ["a.txt"]
    with pytest.raises(ValueError, match="sandbox"):
        fs_read("../outside.txt")
    with pytest.raises(ValueError, match="sandbox"):
        fs_write("../evil.txt", "x")
    fs_write("big.txt", "z" * 500)
    assert len(fs_read("big.txt")) < 500  # 读取也受截断保护（max_read_chars=100）


def test_fs_tools_thread_safety(tmp_path):
    _, fs_write, _ = fs_tools(tmp_path)

    def worker(i):
        fs_write(f"f{i}.txt", "x")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(list(tmp_path.glob("f*.txt"))) == 8


def test_make_subagent_tool_isolates_context():
    sub_fake = FakeClient([_final("sub answer")])

    @tool
    def noop() -> str:
        """noop。"""
        return "ok"

    sub = make_subagent_tool("researcher", "委派研究任务", [noop],
                             model="m", client=sub_fake)

    outer = FakeClient([_tool_turn("c1", "researcher", '{"task": "study"}'), _final("outer done")])
    answer = run_agent("delegate please", [sub], model="m", client=outer)

    assert answer == "outer done"
    assert sub.name == "researcher"
    # 子 agent 拿到委派任务，且与父上下文隔离（独立、最短的消息流）
    assert sub_fake.calls[0]["messages"][-1]["content"] == "study"
    assert len(sub_fake.calls[0]["messages"]) == 2
    assert len(outer.calls) == 2


def test_budget_exceeded_blocks_tool_execution():
    """回归：force_final 后模型幻觉出的 tool_calls 不得被执行。"""
    executed = []

    @tool
    def filler() -> str:
        """填充上下文。"""
        executed.append(None)
        return "z" * 500

    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="filler", arguments="{}"))
    tool_turn = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="", tool_calls=[tc]))])
    # 第 2~8 轮模型继续幻觉工具调用（脚本只回工具轮），最终以 RuntimeError 收场
    fake = FakeClient([tool_turn] * 8)
    with pytest.raises(RuntimeError, match="max_turns"):
        run_agent("go", [filler], model="m", client=fake,
                  harness=Harness(max_context_chars=400, tool_result_max_chars=None))
    assert executed == [None]  # 仅超预算前执行过一次，幻觉调用全部被拦截
    # calls 的 messages 是累积快照：最后一次调用里应含全部 7 条拦截提示
    blocked = [m["content"] for m in fake.calls[-1]["messages"]
               if m.get("role") == "tool" and "budget exceeded" in m.get("content", "")]
    assert len(blocked) == 7  # turn1..turn7 的幻觉调用全部被守卫拦截并回填提示


def test_fs_list_on_file_returns_itself(tmp_path):
    _, _, fs_list = fs_tools(tmp_path)
    fs_tools(tmp_path)[1]("doc.txt", "x")
    assert fs_list("doc.txt") == ["doc.txt"]  # 传入文件路径不再抛 NotADirectoryError
