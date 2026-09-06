# Agent 完整指南（lautpy.agent）

> 快速示例见 [usage.md](usage.md) 的 Agent 章节；本文是完整参考。
> 原生引擎只需 `pip install "lautpy[llm]"`；LangChain 引擎需 `"lautpy[agent]"`（Python ≥ 3.10）。

## 一、核心概念

```
@tool 普通函数 ──┬── to_openai_spec() ──→ run_agent / run_agent_stream（原生引擎）
                └── to_langchain()  ───→ build_agent（LangChain 引擎）
                                     ↑
                       Harness 守卫 + make_todo_tool / fs_tools / make_subagent_tool
```

- **统一工具定义**：一个带类型标注和 docstring 的普通函数，`@tool` 装饰后两边通用
- **环境变量三元组**：`<SVC>_API_KEY` / `<SVC>_BASE_URL` / `<SVC>_MODEL`，
  解析逻辑在 `lautpy.llm`（公开 API：`resolve_credentials` / `resolve_model`）
- **双轨引擎**：原生引擎是标准 tool-calling 循环；LangChain 引擎包装 1.x `create_agent`

## 二、工具定义（@tool）

```python
from lautpy.agent import tool

@tool
def get_weather(city: str, days: int = 1) -> str:
    """查询指定城市未来几天的天气。"""   # docstring = 工具描述，必填
    return f"{city}: 晴 25°C"
```

规则：
- **docstring 必填**（缺失时 `@tool` 直接报错）——它就是模型看到的工具说明
- JSON Schema 从类型标注自动推导；`required` = 无默认值的参数
- 支持的类型：`str/int/float/bool/list/dict` 及其泛型（如 `list[str]` → array）
- 工具对象可直接调用（`get_weather("北京")`），也可 `to_openai_spec()` / `to_langchain()`

## 三、run_agent 参数全表

```python
from lautpy.agent import run_agent

answer = run_agent(
    prompt,                      # 用户输入
    tools=[...],                 # @tool 工具列表；None = 纯对话
    service="moonshot",          # 决定环境变量前缀
    model=None,                  # 缺省读 <SVC>_MODEL
    system=None,                 # 系统提示词
    history=None,                # [{"role": ..., "content": ...}, ...]
    max_turns=8,                 # 工具循环轮数上限（防死循环）
    client=None,                 # 注入客户端（测试用）
    response_model=None,         # pydantic 模型类 → 结构化输出
    harness=None,                # Harness 守卫配置 → 见第四节
)
```

| 行为 | 说明 |
|---|---|
| 工具循环 | 模型请求工具 → 执行 → 结果回填 → 直到无工具调用 |
| 工具报错 | **回传给模型自行调整**，不中断循环（含未知工具名） |
| 超轮数 | 抛 `RuntimeError`，提示增大 max_turns |
| 返回值 | 默认 str；给了 `response_model` 返回校验后的 pydantic 实例 |

### 结构化输出（response_model）

```python
import pydantic

class Weather(pydantic.BaseModel):
    city: str
    celsius: float

w = run_agent("北京今天天气？", service="moonshot", response_model=Weather)
w.celsius
```

- JSON Schema 提示自动注入 system；最终回答校验失败 → 错误回传让模型重试
- 容错剥离 ```` ```json ```` 代码栅栏；pydantic 随 openai extra 自带

### 流式输出（run_agent_stream）

```python
from lautpy.agent import run_agent_stream

for delta in run_agent_stream("讲个故事", service="moonshot", harness=harness):
    print(delta, end="", flush=True)
```

- 工具轮静默执行回填，不产出文本；流内正确聚合跨 chunk 的 tool_calls
- harness 的截断/重复检测/审批门生效；压缩与字符预算仅非流式版支持

## 四、Harness 守卫（长任务必备）

```python
from lautpy.agent import Harness

harness = Harness(
    tool_result_max_chars=2000,   # None 关闭
    max_repeats=3,                # None 关闭
    approval=None,                # f(name, args) -> bool
    max_context_chars=None,       # None 关闭
    compact=None,                 # {"keep_recent": 8, "summarizer": f}
)
```

| 守卫 | 触发条件 | 行为 |
|---|---|---|
| 结果截断 | 工具输出 > tool_result_max_chars | 截断并附 `[truncated: N chars omitted]`，提示模型收窄查询 |
| 重复检测 | 同 (工具, 参数) 连续调用 ≥ max_repeats | 不再执行，回传"换路"错误 |
| 审批门 | 每次工具执行前 | approval 返回 False → 不执行，回传"被审批门拒绝" |
| 字符预算 | 对话总字符 > max_context_chars | **收回工具定义** + 注入"立即给出最终回答"指令 |
| 历史压缩 | 消息体 > keep_recent × 2 | 丢弃最旧消息；提供 summarizer 时旧消息被摘要替代 |

注意：
- 流式版支持截断/重复/审批；压缩与预算为非流式专属
- `PYTHONHASHSEED` 式的"跨进程预算"请配合 `max_turns` 一起设
- summarizer 签名 `f(list[dict]) -> str`，收到的是被裁掉的全部旧消息

## 五、harness 工具构造器

### 计划清单 make_todo_tool()

```python
todo = make_todo_tool()   # [todo_write, todo_mark, todo_read]
```

- `todo_write(tasks: list[str])`：整体替换计划
- `todo_mark(index: int, status: str)`：更新某项状态（pending/in_progress/done）
- `todo_read() -> list[dict]`
- 状态存于闭包（进程内会话隔离）；把三个工具加进 tools 即引导模型先拆解后执行

### 沙箱文件工具 fs_tools(root)

```python
files = fs_tools("./workspace", max_read_chars=4000)  # [fs_read, fs_write, fs_list]
```

- 所有路径限定在 root 内（`..` 逃逸直接拒绝），root 不存在自动创建
- 读取同样受 max_read_chars 截断；写自动建父目录
- 用途：agent 把中间产物卸载到文件（上下文卸载），长任务的"外部记忆"

### 子 agent 工具 make_subagent_tool()

```python
researcher = make_subagent_tool(
    "researcher", "委派调研任务：参数 task 为调研问题",
    tools=[web_search_tool], system="你是调研员，输出要点式结论",
    service="moonshot", max_turns=6,
)
answer = run_agent("调研 X 并给结论", tools=[researcher], service="moonshot")
```

- 子 agent **独立上下文**（检疫）：其内部多轮过程不进入父对话，只回传最终结论
- `client` 参数可注入假客户端（测试）

## 六、LangChain 引擎（build_agent）

```python
from lautpy.agent import build_agent, chat

agent = build_agent([get_weather], service="moonshot",
                    system_prompt="你是户外助手", temperature=0.2)
chat(agent, "北京适合爬山吗")            # 便捷单轮
agent.invoke({"messages": [...]})        # 或直接用 langgraph 原生 API
```

- 需要 `pip install "lautpy[agent]"`；凭证/模型解析与原生引擎同一套环境变量约定
- 适用场景：需要中间件、结构化输出、MCP、检查点等 langgraph 高级能力

## 七、常见配方

**长任务三件套**（防撑爆、防打转、防丢计划）：

```python
todo = make_todo_tool()
harness = Harness(tool_result_max_chars=2000, max_repeats=3,
                  max_context_chars=30000,
                  compact={"keep_recent": 8})
run_agent(task, tools=todo + my_tools, harness=harness, max_turns=24)
```

**危险工具审批**：

```python
harness = Harness(approval=lambda name, args: name not in ("fs_write", "execute_sql"))
```

**上下文检疫**：把搜索/抓取类高噪音工具放进子 agent，父 agent 只看结论。

## 八、FAQ

**Q: 工具从未被调用？** 检查 docstring 是否描述清楚（模型靠它决策）、参数类型是否可标注。

**Q: 模型反复调同一工具？** 开 `max_repeats`；同时检查工具返回内容是否足够明确。

**Q: 上下文爆了？** 组合 `tool_result_max_chars` + `compact` + `max_context_chars`。

**Q: Jupyter 里能用吗？** 原生引擎可以（不依赖 asyncio.run）；注意 `lautpy.pipe.xAsyncio` 在事件循环内不可用。

**Q: 如何接新模型服务？** 只需三个环境变量（`<SVC>_API_KEY/_BASE_URL/_MODEL`），无需改代码。
