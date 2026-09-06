<h1 align="center">:rocket: lautpy — 常用工具类 :facepunch:</h1>

轻量的 Python 工具集：管道式数据处理 + 日期/路径/哈希等常用工具函数。核心零强制依赖，第三方库（numpy/pandas/scikit-learn 等）按需可选。

> 📖 **文档**：[快速上手 & 各模块用法](docs/usage.md) ｜ [架构说明（维护者向）](docs/architecture.md) ｜ [更新日志](CHANGELOG.md)

## Install

```bash
pip install -U lautpy
```

## Pipe 管道工具

```python
from lautpy.pipe import *

# Unix 风格管道：data | func1 | func2
[1, 2, 3, 4] | xmap_(lambda x: x * 2) | xfilter_(lambda x: x > 3)   # [4, 6, 8]

# 进度条 + 日志 + 计时
for i in range(5) | xtqdm:
    logger.info("这是一个进度条")

with timer('LOG'):
    logger.info("打印一条log所花费的时间")
```

常用管道（约定：尾部下划线 = 立即求值返回 list）：

| 类别 | 管道 |
|---|---|
| 类型转换 | `xtuple` `xlist` `xset` `xarray` |
| 高阶函数 | `xmap` `xmap_` `xfilter` `xfilter_` `xenumerate` `xchain` `xchain_` `xreduce` `xdrop` `xdrop_` |
| 字典 | `xchain_dict` `xDictValues` `xDictRemove` `xgetitem` `xitemgetter` |
| 分组/去重 | `xgroup` `xUnique` `xUniquePlus` `xHashBins` `xsort` `xCounter` `xCounterUpdate` `xBloomFilter` |
| 并发 | `xJobs` `xThreadPoolExecutor` `xProcessPoolExecutor` `xAsyncio` |
| 字符串 | `xjoin` `xstartswith` `xendswith` `xsse_parser` |

## 其他模块

```python
from lautpy import dates, hashing, paths

# 日期（纯标准库）
dates.date_difference(days=1)                    # 昨天
dates.timestamp2str(time.time())                 # 时间戳 -> 字符串
dates.get_nday_list(7)                           # 过去 7 天

# 哈希 / AB 实验（murmurhash 需 scikit-learn）
hashing.md5("key:value")
hashing.murmurhash(bins=100)                     # 与 Java Guava 口径一致
hashing.ABTest(expid='10001', ranger=(0, 99)).is_hit("userid")
hashing.hash_bins(values, bins=4)                # 稳定哈希分桶
hashing.BloomFilter(capacity=10**6)              # 纯标准库布隆过滤器

# 路径 / 配置（yaml 支持需 pyyaml）
paths.file2json("config.yaml")                   # json/yaml -> dict
paths.path2list("dir", "*.txt")                  # 文件/目录展开
```

## 装饰器 / 通知 / LLM 客户端

```python
from lautpy import decorators, notice, llm

# 装饰器（retry 需 tenacity，其余纯标准库）
@decorators.retrying(max_retries=3)       # 指数退避重试，耗尽后 reraise
@decorators.timeout(30)                   # 超时抛 TimeoutError
@decorators.background_task               # 后台执行，返回 Future，异常记日志
@decorators.ratelimit(calls=5, period=1)  # 滑动窗口限流
def flaky_api(): ...

# 群机器人通知（webhook 走环境变量：WECOM_WEBHOOK_URL / FEISHU_WEBHOOK_URL）
notice.wecom("部署完成", title="CI")
notice.feishu(["任务1 ✓", "任务2 ✓"], title="日报")

# OpenAI 兼容客户端工厂（openai 可选依赖；密钥走 <SVC>_API_KEY / <SVC>_BASE_URL）
client = llm.openai_client("moonshot")    # 任意 openai 兼容服务
```

## API 封装（`lautpy.apis`）

密钥一律走环境变量，源码中零硬编码；HTTP 统一带超时与自动重试：

```python
import os
from lautpy.apis import get_api_key, request
from lautpy.apis.tools import shorten_url, data2qrcodeurl
from lautpy.apis.niutrans import translate

os.environ["NIUTRANS_API_KEY"] = "..."   # 或在 shell / .env 中配置
translate("你好", "auto", "en")           # NiuTrans 翻译
shorten_url("https://example.com/...")   # 无需密钥的短链/二维码工具
```

## 设计约定

- `import lautpy` 零第三方强制依赖，无导入副作用
- 第三方依赖的管道函数在库缺失时自动不可用，不影响其余功能
- `from lautpy.pipe import *` 仅导出公共 API（`__all__`），不泄漏实现细节

## License

Apache-2.0
