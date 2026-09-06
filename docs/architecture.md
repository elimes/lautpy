# lautpy 架构说明

> 面向维护者/贡献者。快速上手请看 [usage.md](usage.md)。
> 当前基线：**Python ≥ 3.10**（v0.0.7.0 起）。

## 一、总体结构

```
lautpy/
├── src/lautpy/                 # 源码（src 布局，杜绝误 import 本地目录）
│   ├── __init__.py             # 入口：star 导出 pipe + 挂载子模块 + 版本检测
│   ├── _internal.py            # 内部共享件（logger 回退），下划线 = 非公开 API
│   ├── py.typed                # PEP 561 标记：下游类型检查器可见我们的类型标注
│   ├── pipe.py                 # 核心：管道式数据处理（x* 函数族、Pipe 类、timer）
│   ├── dates.py                # 日期/时间戳工具（纯标准库）
│   ├── paths.py                # 路径/配置文件（json/yaml/pickle），yaml 可选
│   ├── hashing.py              # md5 / murmurhash / ABTest / BloomFilter / hash_bins
│   ├── decorators.py           # retrying / timeout / background_task / ratelimit / ...
│   ├── notice.py               # 企微/飞书群机器人通知（webhook 走环境变量）
│   ├── llm.py                  # OpenAI 兼容客户端工厂（openai 可选）
│   ├── ml/                     # 机器学习辅助层（惰性加载，需 ml extra）
│   │   ├── seed.py             # set_seed：random/numpy/torch 一键固定
│   │   ├── metrics.py          # best_threshold / ks_stat / psi / report_lite
│   │   ├── binning.py          # 等频/等距分箱 + WOE/IV + woe_transform
│   │   ├── benchmark.py        # xbenchmark：多模型并行训练对比
│   │   ├── split.py            # xsplit（train/val/test 分层）/ xy
│   │   └── model_io.py         # model_dump/load：带元数据的模型存取
│   ├── agent/                  # Agent 构建层（双轨引擎 + 统一工具定义，见"七"）
│   │   ├── __init__.py         # run_agent（原生循环）/ build_agent（langchain）
│   │   ├── tools.py            # @tool：普通函数 → OpenAI schema / langchain 工具
│   │   ├── mini.py             # 原生 tool-calling 循环（零 langchain 依赖）
│   │   └── langchain_agent.py  # langchain 1.x create_agent 包装（需 agent extra）
│   └── apis/                   # 第三方 HTTP API 封装层
│       ├── client.py           # 基建：get_api_key（环境变量密钥）+ request（超时/重试）
│       ├── tools.py            # 无密钥工具：shorten_url / data2qrcodeurl / download / is_open
│       └── niutrans.py         # 翻译 API（示范新 API 的接入模式）
├── tests/                      # pytest 测试（117 用例，可选依赖缺失自动 skip）
├── docs/                       # architecture（本页）/ usage / agent / ml
├── CHANGELOG.md                # 版本变更记录（Unreleased 段随发版归档）
├── .data/VERSION               # 唯一版本源（pyproject 动态读取）
├── .github/workflows/
│   ├── test.yml                # CI：3.10–3.13 矩阵测试 + ruff(E/F/W/I/B/UP/SIM/C4/PERF) + mypy
│   └── publish.yml             # tag 触发 → PyPI（trusted publishing）
└── pyproject.toml              # PEP 639 元数据 + extras + ruff/pytest 配置
```

## 二、模块依赖关系（自下而上，无循环）

```
_internal (logger 回退)
   ↑
pipe (核心，仅标准库 + 可选 numpy/pandas/joblib/sklearn/tqdm)
   ↑                  ↑
dates paths hashing decorators（tenacity 可选） notice ──→ apis.client ──→ requests
                                                    llm ──→ openai（可选）
                                                     ↑
                                     agent（tools 零依赖；mini 走 llm；
                                     langchain_agent 走 langchain 1.x，需 agent extra）
                                     ml（惰性；seed 零依赖；metrics/binning 走 numpy；
                                     benchmark/split 走 sklearn+pandas）
```

- **零强制依赖**：`dependencies = []`。requests 属 `http` extra，`apis`/`notice`
  通过 PEP 562 `__getattr__` 惰性挂载（`import lautpy` 不触发 requests 导入）；
  其余第三方库同样全部 try/except 可选化
- `pipe` 里的 numpy/pandas 系管道函数在库缺失时**整个不存在**（连 `__all__` 都不注册），其余功能不受影响

## 三、设计原则（改动代码前先读）

1. **导入零副作用**：`import lautpy` 不联网、不改全局状态、不发警告（无 `warnings.filterwarnings`、无预建连接、无定时任务）。副作用只能发生在函数被调用时。
2. **可选依赖自动降级**：第三方库缺失 → 相关功能不可用并给出明确报错，其余功能照常。可选库同时登记进 pyproject 的 extras。
3. **密钥只走环境变量**：任何 API key / webhook URL 一律 `get_api_key()` 现场解析（`<SVC>_API_KEY`），源码零硬编码。`tests/test_apis.py` 有守护测试扫描，写死密钥会让 CI 失败。
4. **star 导出走 `__all__` 白名单**：`from lautpy.pipe import *` 只暴露公共 API，`np`/`itertools`/`json` 等实现细节不泄漏。新增公开函数必须同步登记 `__all__`。
5. **Python 3.10 与类型约定**：全面使用 PEP 604 联合类型（`str | None`）与内建泛型（`list[str]`），`typing` 只保留 `Any`/`TypeVar`；ruff（target py310 + UP 规则集）守护，旧写法过不了 CI。包带 `py.typed`，类型标注对下游可见。

   > ⚠️ 维护者提示：ruff 的 UP007 unsafe 修复会把 `Optional[threading.Lock]`
   > 改写成 `threading.Lock | None`——但 `threading.Lock` 是工厂函数而非类，
   > 运行时求值注解会直接 TypeError（已在 `decorators.synchronized` 用字符串
   > 注解规避）。新增注解时对"函数对象伪装的类型"（如 `threading.Lock`、
   > `re.compile`）要保持警惕。

## 四、Pipe 机制速览

```python
class Pipe:
    def __ror__(self, other):        # data | xfunc → func(data)
        return self.func(other)
    def __call__(self, *args, **kwargs):   # xfunc(2) → 部分应用，返回新 Pipe
        return Pipe(lambda x: self.func(x, *args, **kwargs))
```

- 惰性管道（`xmap`/`xfilter`/`xdrop`）返回迭代器，用 `xlist`/`xtuple` 收口
- 急切变体以尾部下划线标记（`xmap_`），直接返回 list
- 新写管道函数：简单场景直接 `@Pipe def xfoo(...)`；注意同步登记 `__all__`

## 五、Agent 层设计（`lautpy.agent`）

双轨引擎共用一套工具定义，切换引擎不改业务代码：

```
@tool 普通函数 ──┬── to_openai_spec() ──→ mini.run_agent_loop（原生引擎）
（类型标注+docstring）                       │  纯 OpenAI tool-calling 轮转，
                                            │  零 langchain 依赖
                 └── to_langchain() ────→ build_agent → langchain 1.x create_agent
                                            （中间件/结构化输出等高级能力继承）
```

- **工具契约**：`@tool` 装饰的函数必须有类型标注和 docstring（docstring 即
  工具描述，缺失直接报错）；签名自动推导 JSON Schema，required = 无默认值参数
- **模型配置三元组**：`<SVC>_API_KEY` / `<SVC>_BASE_URL` / `<SVC>_MODEL`，
  与 `lautpy.llm` 共用解析逻辑，显式参数优先于环境变量
- **循环稳健性**：工具执行异常会作为错误信息回传给模型自行调整（不中断循环）；
  未知工具名同样回传报错；`max_turns` 防死循环；`client` 可注入（测试无网络）
- **harness 守卫**（`Harness`，默认关闭）：工具结果截断、重复调用检测、
  审批门、字符预算（超限收回工具强制收尾）、历史压缩（朴素截断或
  summarizer 摘要）——对应 DeepAgents compaction 等 harness 实践
- **harness 工具构造器**：`make_todo_tool`（计划清单）、`fs_tools`（沙箱文件
  读写、上下文卸载）、`make_subagent_tool`（agent-as-tool，上下文检疫）
- **引擎选择**：只需要"模型+工具"用原生引擎（`run_agent`）；需要中间件、
  MCP 等用 `build_agent`
- **结构化输出**：`run_agent(response_model=PydanticModel)` 把 JSON Schema
  提示注入 system，最终回答校验失败自动回传错误让模型重试；容错剥离
  markdown 代码栅栏（pydantic 随 openai extra 提供）
- **流式输出**：`run_agent_stream` 逐段产出文本增量；流内聚合 tool_calls
  增量（id/name/arguments 按 index 拼接），工具轮静默执行回填
- **凭证解析**：`llm.resolve_credentials()` / `llm.resolve_model()` 为公开
  API（llm/agent/notice 共用的 <SVC>_ 三元组解析，勿再访问 `llm._resolve`）

## 六、版本与发布

1. **版本号唯一来源是 `.data/VERSION`**——pyproject 里 `dynamic = ["version"]`，setuptools 构建时自动读取，不存在第二处需要同步的版本号
2. 发版：`./release.sh <版本>` → 提交版本文件 → 打 `v*` tag → `publish.yml` 自动构建并经 trusted publishing 发布到 PyPI
3. **发版前必须经仓库所有者确认**（流程约定）
4. CI（`test.yml`）在每次 push/PR 时跑 3.10–3.13 全矩阵测试 + ruff + mypy

## 七、测试约定

- 测试位于 `tests/`，一律不依赖真实外网/真实密钥（网络类测试用 monkeypatch 替身，或用环境变量门控）
- Agent 循环测试用"脚本化假客户端"驱动：预置一串响应（工具调用→最终回答），断言消息流转，全程无网络
- 涉及可选依赖的用例用 `pytest.importorskip` / `skipif`，保证最小环境（只装 requests+pytest）也能全绿
- 本地跑全量：`uv run --with pytest --with numpy --with scikit-learn --with pyyaml --with tenacity --with openai pytest tests -q`，或 `pip install ".[dev]" && pytest`
