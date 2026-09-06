# lautpy 使用指南

> 5 分钟上手。模块内部设计见 [architecture.md](architecture.md)。

## 安装

```bash
pip install lautpy                 # 核心功能（仅依赖 requests）
pip install "lautpy[all]"          # 全部可选依赖（tenacity/openai/pyyaml/scikit-learn/numpy）

# 或按需选装
pip install "lautpy[retry]"        # 重试装饰器（tenacity）
pip install "lautpy[llm]"          # LLM 客户端（openai）
pip install "lautpy[yaml]"         # yaml 配置读取
pip install "lautpy[ml]"           # murmurhash/ABTest/shuffle（scikit-learn）
```

要求 Python ≥ 3.10。导入零副作用、无强制联网，`import lautpy` 永远是安全的。

## 1. 管道数据处理 `lautpy.pipe`

```python
from lautpy.pipe import *

# Unix 风格管道：data | func1 | func2
[1, 2, 3, 4] | xmap_(lambda x: x * 2) | xfilter_(lambda x: x > 3)   # [4, 6, 8]

# 命名约定：xmap/xfilter 返回惰性迭代器（用 xlist 收口）
#          xmap_/xfilter_ 尾部下划线 = 立即求值，直接返回 list
range(10) | xfilter(lambda x: x % 2) | xlist                        # [1, 3, 5, 7, 9]

# 分组、去重、计数
[1, 2, 3, 4, 5] | xgroup(2)                                          # [[1,2],[3,4],[5]]
["a", "b", "a"] | xUnique                                            # ["a", "b"]
[["w1", "w2"], ["w1"]] | xCounterUpdate                              # Counter(w1=2, w2=1)

# 字典工具
[{"a": 1}, {"b": 2}] | xchain_dict                                   # {"a": 1, "b": 2}
{"a": 1, "b": 2} | xDictValues(["a", "z"], default=0)                # (1, 0)
```

### 进度条 / 日志 / 计时

```python
from lautpy.pipe import *

for i in range(100) | xtqdm:                 # tqdm 进度条
    ...

with timer("处理耗时"):                       # 自动输出耗时日志
    do_something()
```

### 并发

```python
# 线程池并行（I/O 密集），自带进度条
urls | xThreadPoolExecutor(fetch, max_workers=10)

# 进程池并行（CPU 密集）
items | xProcessPoolExecutor(heavy_compute, max_workers=4)

# 协程并发
import asyncio
async def fetch(i): ...
[fetch(i) for i in range(10)] | xAsyncio
```

### SSE 流解析（LLM 流式输出常用）

```python
lines | xsse_parser                                   # 提取 data: 行并解析 JSON
sse_text.splitlines() | xsse_parser(skip_substrings=["[DONE]"])
```

## 2. 日期 `lautpy.dates`（纯标准库）

```python
from lautpy import dates

dates.date_difference(days=1)                                  # 昨天 "%Y-%m-%d %H:%M:%S"
dates.date_difference("%Y%m%d", start_date=20210222, days=1)   # "20210221"
dates.timestamp2str(1787000000.0)                              # 时间戳 → 字符串
dates.str2timestamp("2026-09-06 12:00:00")                     # 字符串 → 时间戳
dates.get_nday_list(7)                                         # 过去 7 天（ISO 日期串）
```

## 3. 哈希与 AB 实验 `lautpy.hashing`

```python
from lautpy import hashing

hashing.md5("key:value")

# murmurhash 与 Java Guava murmur3_32 口径一致（需 scikit-learn）
hashing.murmurhash("key", "value", bins=100)

# AB 实验：user 落桶是否命中实验组（同一 user 永远同桶）
ab = hashing.ABTest(expid="10001", ranger=(0, 9), bins=100)
if ab.is_hit(user_id): ...

# 稳定哈希分桶（分片/分流）
hashing.hash_bins(user_ids, bins=4)

# 布隆过滤器（纯标准库，无漏报，误报率 ≤ error_rate）
bloom = hashing.BloomFilter(capacity=1_000_000, error_rate=0.01)
bloom.add(x);  x in bloom
```

## 4. 路径与配置 `lautpy.paths`

```python
from lautpy import paths

paths.file2json("config.yaml")      # json/yaml → dict（yaml 需 extras）
paths.path2list("logs/", "*.txt")   # 文件/目录展开为 Path 列表
paths.get_config(cfg_or_path)       # dict / 配置文件路径 / None → 统一返回 dict

paths.pkl_dump(obj, "cache.pkl")    # pickle 落盘 / 读回
obj = paths.pkl_load("cache.pkl")
```

## 5. 装饰器 `lautpy.decorators`

```python
from lautpy import decorators

@decorators.retrying(max_retries=3)          # 指数退避重试（需 extras: retry）
def flaky_api(): ...

@decorators.timeout(30)                      # 超时抛 TimeoutError
def slow_call(): ...

@decorators.background_task                  # 立即返回 Future，异常记日志
def notify(): ...
future = notify()

@decorators.ratelimit(calls=5, period=1)     # 每秒最多 5 次（滑动窗口）
def api_call(): ...
```

## 6. 群机器人通知 `lautpy.notice`

```bash
# 一次性配置 webhook（支持完整 URL 或纯 key）
export WECOM_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
export FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"
```

```python
from lautpy import notice

notice.wecom("部署完成", title="CI")                 # 企微 markdown
notice.feishu(["任务1 ✓", "任务2 ✓"], title="日报")  # 飞书文本，列表自动换行
# 超长消息自动按 UTF-8 字节安全分段，不会丢字
```

## 7. LLM 客户端 `lautpy.llm`

```bash
export MOONSHOT_API_KEY="sk-..."            # 任意 OpenAI 兼容服务
# export MOONSHOT_BASE_URL="https://api.moonshot.cn/v1"   # 非官方地址时设置
```

```python
from lautpy import llm          # 需 extras: llm

client = llm.openai_client("moonshot")       # 自动读环境变量、按服务缓存
resp = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[{"role": "user", "content": "你好"}],
)
```

## 8. Agent 构建 `lautpy.agent`

双轨设计：**原生引擎**只用 OpenAI 兼容 API（`pip install "lautpy[llm]"）；；
**LangChain 引擎**包装 langchain 1.x `create_agent`（`pip install "lautpy[agent]"`，Python 3.10+）。
两侧共用同一套工具定义。

```bash
# 环境变量约定（以 moonshot 为例）
export MOONSHOT_API_KEY="sk-..."
export MOONSHOT_MODEL="moonshot-v1-8k"
# export MOONSHOT_BASE_URL="https://api.moonshot.cn/v1"
```

**第一步：定义工具**（普通函数 + 类型标注 + docstring，docstring 即工具描述，必填）

```python
from lautpy.agent import tool

@tool
def get_weather(city: str, days: int = 1) -> str:
    '''查询指定城市未来几天的天气。'''
    return f"{city}: 晴 25°C"
```

**第二步（原生引擎）**：直接跑 agent 循环

```python
from lautpy.agent import run_agent

answer = run_agent(
    "北京明天适合爬山吗？",
    tools=[get_weather],
    service="moonshot",
    system="你是户外助手，回答简洁。",
)
```

**第二步（LangChain 引擎）**：需要中间件/结构化输出等高级能力时

```python
from lautpy.agent import build_agent, chat

agent = build_agent([get_weather], service="moonshot",
                    system_prompt="你是户外助手。")
chat(agent, "北京明天适合爬山吗？")     # 便捷单轮封装
# 也可以直接用 langgraph 原生 API：agent.invoke({"messages": [...]})
```

**循环保守**：run_agent 内部是标准的 OpenAI function-calling 轮转
（模型请求工具 → 执行并回填 → 直到给出最终回答），工具抛错会把错误
信息回传给模型自行调整，而不是中断；`max_turns` 防死循环。

## 9. Web/API 工具 `lautpy.apis`

```python
from lautpy.apis import get_api_key, request, MissingAPIKeyError
from lautpy.apis.tools import shorten_url, data2qrcodeurl, download, is_open

shorten_url("https://example.com/very/long")     # 无需密钥
data2qrcodeurl("hello")                          # 二维码图片 URL
download("https://.../file.zip")                 # 流式下载（超时+重试）
is_open("127.0.0.1", 8080)                       # TCP 端口探测

# 统一的 HTTP 出口：30s 超时 + 429/5xx 自动重试
resp = request("GET", "https://api.example.com/data")

# 新接第三方 API 的标准姿势（密钥只走环境变量）
from lautpy.apis.niutrans import translate
# export NIUTRANS_API_KEY=...
translate("你好", "auto", "en")
```

## 常见问题

**Q: `xarray`/`xshuffle`/`xJobs` 提示不存在？**
对应可选依赖（numpy/scikit-learn/joblib）未安装，`pip install "lautpy[numpy]"` 等。

**Q: loguru 装了为什么日志格式没变？**
本包优先使用 loguru；确认 loguru 装在了当前解释器环境。

**Q: 如何接入一个新的第三方 API？**
在 `lautpy/apis/` 新建模块：密钥用 `get_api_key("服务名")`（读 `<服务名>_API_KEY` 环境变量），HTTP 一律走 `request()`，参照 `niutrans.py`，并补一个 mock 网络的测试。

**Q: 版本号在哪改？**
只改 `.data/VERSION`（或用 `./release.sh <版本>`），pyproject 动态读取，没有第二处。
