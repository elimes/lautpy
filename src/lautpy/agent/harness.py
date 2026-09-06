# @Author: elimes
"""Agent harness 工程件：对齐 2025-2026 前沿脚手架的轻量实现。

设计原则：全部是**可组合的选项/工具构造器**，默认关闭，不把主循环改造成
图引擎（编排/MCP/可插拔后端属于 langgraph / deepagents 的领地，不在此重建）。

组成：
- ``Harness``：循环守卫配置（结果截断 / 重复检测 / 审批门 / 字符预算 / 历史压缩）
- ``make_todo_tool``：计划清单工具（任务拆解，DeepAgents planning 模式）
- ``fs_tools``：沙箱化文件读写工具（上下文卸载，DeepAgents filesystem 模式）
- ``make_subagent_tool``：把配置好的子 agent 包成工具（上下文隔离/检疫）
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from lautpy.agent.tools import Tool

DEFAULT_RESULT_MAX_CHARS = 2000


class Harness:
    """Agent 循环守卫配置（全部可选，默认关闭）。

    Args:
        tool_result_max_chars: 工具结果超过该字符数时截断并告知模型；
            None 关闭。
        max_repeats: 相同 (工具, 参数) 连续调用达到该次数后，不再执行并
            回传错误，强制模型换路；None 关闭。
        approval: 审批门 ``f(name, args) -> bool``；返回 False 时工具不执行，
            向模型回传"被拒绝"。适合危险工具的人工确认。
        max_context_chars: 对话总字符预算；超限后移除工具定义并注入
            "立即给出最终回答"指令（近似 token 守卫，不引分词器）。
        compact: 历史压缩配置 ``{"keep_recent": 8, "summarizer": f(messages) -> str}``。
            对话体超过 keep_recent*2 条时触发：summarizer 缺省为朴素截断
            （丢弃最旧消息），提供时旧消息被摘要替代。
    """

    def __init__(
        self,
        tool_result_max_chars: int | None = DEFAULT_RESULT_MAX_CHARS,
        max_repeats: int | None = 3,
        approval: Callable[[str, dict], bool] | None = None,
        max_context_chars: int | None = None,
        compact: dict[str, Any] | None = None,
    ):
        self.tool_result_max_chars = tool_result_max_chars
        self.max_repeats = max_repeats
        self.approval = approval
        self.max_context_chars = max_context_chars
        self.compact = compact

    def truncate(self, result: str) -> str:
        if not self.tool_result_max_chars or len(result) <= self.tool_result_max_chars:
            return result
        cut = self.tool_result_max_chars
        return (result[:cut]
                + f"\n[truncated: {len(result) - cut} chars omitted — "
                  f"narrow your tool query if you need the rest]")


def truncate_result(result: str, max_chars: int = DEFAULT_RESULT_MAX_CHARS) -> str:
    """截断工具输出并附提示（harness 截断逻辑的独立入口，便于单测/复用）。"""
    return Harness(tool_result_max_chars=max_chars).truncate(result)


# ---------- 计划工具（DeepAgents planning 模式） ----------

def make_todo_tool() -> list[Tool]:
    """构造一组计划清单工具（todo_write / todo_mark / todo_read）。

    状态存于闭包内（进程级、会话隔离，带锁保证并发安全），
    引导模型对多步任务先拆解再执行。
    """
    import threading

    state: list[dict] = []
    lock = threading.Lock()

    def todo_write(tasks: list[str]) -> str:
        """用新计划整体替换任务清单（每项一句短语，建议 3~7 项）。"""
        with lock:
            state.clear()
            state.extend({"task": t, "status": "pending"} for t in tasks)
        return f"plan set: {len(tasks)} tasks"

    def todo_mark(index: int, status: str) -> str:
        """更新任务状态。index 从 0 起；status 常用 pending / in_progress / done。"""
        with lock:
            if not 0 <= index < len(state):
                return f"Error: index {index} out of range (0..{len(state) - 1})"
            state[index]["status"] = status
            return f"task {index} -> {status}"

    def todo_read() -> list[dict]:
        """读取当前计划清单及各项状态。"""
        with lock:
            return [dict(item) for item in state]

    return [Tool(todo_write), Tool(todo_mark), Tool(todo_read)]


# ---------- 沙箱文件工具（DeepAgents filesystem 模式） ----------

def _within(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def fs_tools(root: str | Path, max_read_chars: int = 4000) -> list[Tool]:
    """构造沙箱化文件工具组（fs_read / fs_write / fs_list），仅限 root 目录内。

    供 agent 把中间产物卸载到文件系统（上下文卸载），root 之外的路径一律拒绝。
    """
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    def fs_read(path: str) -> str:
        """读取 root 内相对路径的文本文件内容。"""
        p = root_path / path
        if not _within(root_path, p):
            raise ValueError(f"path escapes sandbox root: {path}")
        text = p.read_text(encoding="utf-8")
        return Harness(tool_result_max_chars=max_read_chars).truncate(text)

    def fs_write(path: str, content: str) -> str:
        """把文本内容写入 root 内相对路径（自动创建父目录，覆盖同名文件）。"""
        p = root_path / path
        if not _within(root_path, p):
            raise ValueError(f"path escapes sandbox root: {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"written: {len(content)} chars -> {path}"

    def fs_list(path: str = ".") -> list[str]:
        """列出 root 内相对路径目录下的文件与子目录名（传入文件路径则返回其自身）。"""
        p = root_path / path
        if not _within(root_path, p):
            raise ValueError(f"path escapes sandbox root: {path}")
        if p.is_file():
            return [p.name]
        return sorted(item.name + ("/" if item.is_dir() else "") for item in p.iterdir())

    return [Tool(fs_read), Tool(fs_write), Tool(fs_list)]


# ---------- 子 agent 工具（DeepAgents subagents 模式） ----------

def make_subagent_tool(
    name: str,
    description: str,
    tools: list | None = None,
    *,
    service: str = "openai",
    model: str | None = None,
    system: str | None = None,
    max_turns: int = 6,
    client: Any = None,
) -> Tool:
    """把一个配置好的子 agent 包装成工具：父 agent 调用它即委派任务。

    子 agent 拥有独立消息上下文（上下文检疫），其长过程不会进入父对话。

    Args:
        name: 工具名（也是委派入口，如 "researcher"）。
        description: 工具描述，须说明何时委派、委派时传入什么。
        tools: 子 agent 可用的工具列表（可为空 = 纯推理子任务）。
        service / model / system / max_turns / client: 同 run_agent 的对应参数；
            client 可注入假客户端（测试）。
    """
    def delegate(task: str) -> str:
        from lautpy.agent import run_agent

        return run_agent(task, tools=list(tools or []), service=service, model=model,
                         system=system, max_turns=max_turns, client=client)

    fn = delegate
    fn.__name__ = name  # Tool 以函数名作为工具名
    fn.__doc__ = description
    return Tool(fn)
