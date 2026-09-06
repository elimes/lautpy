# MAINTAINING — lautpy 维护者手册

> 面向维护者（elimes 及未来协作者）。使用文档见 docs/usage.md，架构见 docs/architecture.md。
> 核心心法：**维护靠节奏，不靠热情**。本文把节奏写成表，照着做即可。

## 一、维护节奏表

| 周期 | 动作 | 说明 |
|---|---|---|
| 使用中 | 摩擦记录 | 用着别扭的地方（参数不顺/报错不清/缺功能）当场开 GitHub Issue——需求最真实的来源 |
| 每月（15 分钟） | 依赖与门禁体检 | CI 全绿确认；扫 extras 上游 major 版本；patch 级更新忽略 |
| 每季度（30 分钟） | 质量轮 | 覆盖率盲区复查；ruff/mypy 规则拧紧一格；architecture.md 与代码漂移检查 |
| 每半年 | meutils 上游 diff | 下载最新 wheel 与 `meutils_src/` 基线 diff，只看 pipe/decorators/纯函数（apis 不碰） |
| 每年 | 地基复审 | Python 门槛上调评估；extras 退役评估；范围治理原则修订 |

## 二、发布检查清单（每次发版逐项过）

1. [ ] CHANGELOG 的 Unreleased 段已归档为版本号
2. [ ] `.data/VERSION` 已更新为与 tag 一致的版本号（唯一版本源，pyproject 动态读取）
3. [ ] `uv build` 成功且 wheel 版本号正确
4. [ ] 全量测试通过（`uv run --with pytest --with ... pytest tests -q`）
5. [ ] 公开 API 有变更时：docstring / docs 四页 / 用户手册（manual_build 重新生成）已同步
6. [ ] **经仓库所有者确认**（流程约定，底线要求）
7. [ ] `git tag vX.Y.Z && git push origin vX.Y.Z` → Actions 绿 → PyPI 页面核对
8. [ ] 发布工作流的 tag/version 守卫失败时：先修 `.data/VERSION` 再重新打 tag，绝不手推

## 三、质量门禁（四重，别拆）

1. **pre-commit**（本地提交即拦截）：ruff + mypy，安装方式见第五节
2. **CI test.yml**：3.10–3.13 矩阵测试 + ruff + mypy + 覆盖率报告（当前 84%）
3. **CI publish.yml**：trusted publishing 发布 + tag/版本一致性守卫
4. **回归测试台账**：每个 bug = 一个永久测试（现有 12 个回归用例），免疫系统不可拆除

渐进提升的口子：覆盖率可升格为 `--cov-fail-under` 门禁；mypy 可逐模块收紧；
ruff 可增开规则组。每次质量轮只拧一格，修完再拧。

## 四、演进积压单（按状态记录，做了就划掉）

- [ ] langchain 引擎路径本地验证（目前仅在 CI 跑通）
- [ ] 覆盖率升格为 fail-under 门禁
- [ ] `xprint` tee 变体（评审记录在案）
- [ ] 共享 Session 并发 Cookie 语义的边界测试
- [ ] mypy 逐模块收紧（strict 化）
- [ ] meutils 若复活：恢复半年一次的上游 diff（台账：`meutils_src/` 为 2026.7.17 基线）

## 五、本地开发环境

```bash
# 一次性：启用版本化的提交钩子（ruff + mypy 拦在提交瞬间）
git config core.hooksPath .githooks

# 全量测试（含可选依赖）
uv run --with pytest --with requests --with numpy --with pandas --with scikit-learn \
    --with pyyaml --with tenacity --with openai pytest tests -q

# 手册重生成（改公开 API 后）
cd ../manual_build && uv run --with python-docx python main2.py
# 产物拷到桌面，文档源在 docs/
```

## 六、弃用流程

1. 旧名保留为别名 + `DeprecationWarning`（参照 `pipe.xsse_parser` → `xsse_parser_`）
2. 保持至少两个版本
3. CHANGELOG 记录迁移方式
4. 之后才允许移除

## 七、范围治理（引用 architecture.md 原则 6）

新领域进入前必须回答："为什么它属于 lautpy 而不是独立包？"——答案应包含与现有域
共享的工具或约定。已拒绝清单：langgraph 编排、MCP 客户端、imblearn 封装、
可插拔后端。拒绝清单和接受清单一样重要。

## 八、已知风险记录

- 共享 Session 高并发下 Cookie 语义可能交叉（文档已注明，极端并发传独立 session）
- `threading.Lock` 等函数伪装的类型不能直接参与 `X | None` 注解（见 decorators.synchronized）
- 已发布到 PyPI 的版本不可撤回——发版前确认流程是最后防线
