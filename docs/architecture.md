# lautpy 架构说明

> 面向维护者/贡献者。快速上手请看 [usage.md](usage.md)。

## 一、总体结构

```
lautpy/
├── src/lautpy/                 # 源码（src 布局，杜绝误 import 本地目录）
│   ├── __init__.py             # 入口：star 导出 pipe + 挂载子模块 + 版本检测
│   ├── _internal.py            # 内部共享件（logger 回退），下划线 = 非公开 API
│   ├── pipe.py                 # 核心：管道式数据处理（x* 函数族、Pipe 类、timer）
│   ├── dates.py                # 日期/时间戳工具（纯标准库）
│   ├── paths.py                # 路径/配置文件（json/yaml/pickle），yaml 可选
│   ├── hashing.py              # md5 / murmurhash / ABTest / BloomFilter / hash_bins
│   ├── decorators.py           # retrying / timeout / background_task / ratelimit / ...
│   ├── notice.py               # 企微/飞书群机器人通知（webhook 走环境变量）
│   ├── llm.py                  # OpenAI 兼容客户端工厂（openai 可选）
│   ├── agent/                  # Agent 构建层（双轨引擎 + 统一工具定义）
│   │   ├── __init__.py         # run_agent（原生循环）/ build_agent（langchain）
│   │   ├── tools.py            # @tool：普通函数 → OpenAI schema / langchain 工具
│   │   ├── mini.py             # 原生 tool-calling 循环（零 langchain 依赖，3.8+）
│   │   └── langchain_agent.py  # langchain 1.x create_agent 包装（3.10+）
│   └── apis/                   # 第三方 HTTP API 封装层
│       ├── client.py           # 基建：get_api_key（环境变量密钥）+ request（超时/重试）
│       ├── tools.py            # 无密钥工具：shorten_url / data2qrcodeurl / download / is_open
│       └── niutrans.py         # 翻译 API（示范新 API 的接入模式）
├── tests/                      # pytest 测试（60 用例，可选依赖缺失自动 skip）
├── docs/                       # 本文档 + 使用指南
├── .data/VERSION               # 唯一版本源（pyproject 动态读取）
├── .github/workflows/
│   ├── test.yml                # CI：3.8–3.13 矩阵测试 + ruff lint
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
                                     agent（tools 无依赖；mini 走 llm；
                                     langchain_agent 走 langchain 1.x，3.10+ 可选）
```

- **只有 `requests` 是硬依赖**；其余第三方库全部 try/except 可选化
- `pipe` 里的 numpy/pandas 系管道函数在库缺失时**整个不存在**（连 `__all__` 都不注册），其余功能不受影响

## 三、四条设计原则（改动代码前先读）

1. **导入零副作用**：`import lautpy` 不联网、不改全局状态（无 `warnings.filterwarnings`、无预建连接、无定时任务）。副作用只能发生在函数被调用时。
2. **可选依赖自动降级**：第三方库缺失 → 相关功能不可用并给出明确报错，其余功能照常。可选库同时登记进 pyproject 的 extras。
3. **密钥只走环境变量**：任何 API key / webhook URL 一律 `get_api_key()` 现场解析（`<SVC>_API_KEY`），源码零硬编码。`tests/test_apis.py` 有守护测试扫描，写死密钥会让 CI 失败。
4. **star 导出走 `__all__` 白名单**：`from lautpy.pipe import *` 只暴露公共 API，`np`/`itertools`/`json` 等实现细节不泄漏。新增公开函数必须同步登记 `__all__`。

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
- 新写管道函数：简单场景直接 `@Pipe def xfoo(...)`

## 五、版本与发布

1. **版本号唯一来源是 `.data/VERSION`**——pyproject 里 `dynamic = ["version"]`，setuptools 构建时自动读取，不存在第二处需要同步的版本号
2. 发版：`./release.sh <版本>` → 提交版本文件 → 打 `v*` tag → `publish.yml` 自动构建并经 trusted publishing 发布到 PyPI
3. **发版前必须经仓库所有者确认**（流程约定）
4. CI（`test.yml`）在每次 push/PR 时跑 3.8–3.13 全矩阵测试 + ruff

## 六、测试约定

- 测试位于 `tests/`，一律不依赖真实外网/真实密钥（网络类测试用 monkeypatch 替身，或用环境变量门控）
- 涉及可选依赖的用例用 `pytest.importorskip` / `skipif`，保证最小环境（只装 requests+pytest）也能全绿
- 本地跑全量：`uv run --with pytest --with numpy --with scikit-learn --with pyyaml --with tenacity --with openai pytest tests -q`，或 `pip install ".[dev]" && pytest`
