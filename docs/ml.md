# ML 辅助层完整指南（lautpy.ml）

> 需 `pip install "lautpy[ml]"`（scikit-learn / pandas / numpy）。
> 定位是"胶水层"：只补 sklearn 没有直接给的日常刚需，**不重造标准算法**。

## 模块地图

| 模块 | 能力 | 最低依赖 |
|---|---|---|
| `seed` | set_seed 全局可复现 | 无（torch 可选） |
| `metrics` | best_threshold / ks_stat / psi / report_lite | numpy |
| `binning` | 分箱 + WOE/IV + woe_transform | numpy |
| `split` | xsplit 分层切分 / xy 分离 | scikit-learn / pandas |
| `benchmark` | xbenchmark 多模型并行对比 | scikit-learn + pandas |
| `model_io` | model_dump / model_load / model_info | joblib（sklearn 自带），pickle 兜底 |

## 一、可复现性 set_seed

```python
from lautpy.ml import set_seed

seed = ml.set_seed(2026)   # 固定 random / numpy / torch(+CUDA)，返回种子便于记录
```

注意：`PYTHONHASHSEED` 对当前进程无效（解释器启动时已定）——跨进程复现需在启动前导出。

## 二、数据划分 xsplit / xy

```python
from lautpy.ml import xsplit, xy

# 两刀切分，分层标签在两刀中都保持分布；按数组分组返回
X_tr, X_val, X_te, y_tr, y_val, y_te = xsplit(
    X, y, test_size=0.2, val_size=0.1, stratify=y, random_state=42)

# 只切 train/test
X_tr, X_te = xsplit(X, y, test_size=0.2)

# DataFrame 拆 X/y
X, y = xy(df, "label")          # 多目标: xy(df, ["y1", "y2"])
```

返回顺序与 `train_test_split` 习惯一致（按数组分组）。无泄漏有测试保障。

## 三、评估指标补集 metrics

sklearn 有标准指标，但这四件日常高频、sklearn 不直接给：

```python
from lautpy.ml import best_threshold, ks_stat, psi, report_lite

# 1) 阈值搜索：按 F1 扫 200 个候选点（可传自定义谓词 f(tp, fp, fn, tn)）
thr, score = best_threshold(y_true, y_proba, metric="f1")
y_pred = (y_proba >= thr).astype(int)

# 2) KS 值：正负样本分数累计分布差的最大值（0~1）
ks = ks_stat(y_true, y_proba)          # 完全可分 = 1.0

# 3) PSI 群体稳定性：以 expected 分位数分箱比较 actual
drift = psi(train_scores, online_scores)
# 解读惯例：< 0.1 稳定 / 0.1~0.25 关注 / > 0.25 显著漂移

# 4) 一次拿全常用分类指标（有 y_score 且装了 sklearn 时含 AUC）
r = report_lite(y_true, y_score=y_proba)   # {accuracy, precision, recall, f1, auc}
```

全部纯 numpy 实现（AUC 除外）；y 必须是 0/1 二值。

## 四、分箱 + WOE/IV binning

评分卡、风控与可解释建模刚需（sklearn 无对应实现）：

```python
from lautpy.ml import woe_bins, iv_summary, woe_transform, bin_edges

# 分箱并计算每箱 WOE 与 IV（y: 0=好样本, 1=坏样本）
bins = woe_bins(income, default_flag, n_bins=5)
# 每箱: {"bin": "[3200.0, 5800.0)", "total": 200, "bad": 12,
#        "bad_rate": 0.06, "woe": -0.51, "iv": 0.09}

# 特征预测力解读惯例：IV > 0.3 强 / 0.1~0.3 中 / < 0.1 弱
iv = iv_summary(income, default_flag, n_bins=5)

# 用训练好的分箱把新数据编码为 WOE 值（喂给 LR 等可解释模型）
encoded = woe_transform(income_new, bins)

# 自定义切点 / 等距分箱
bins = woe_bins(x, y, edges=[0, 5000, 10000, float("inf")])
edges = bin_edges(x, n_bins=10, method="uniform")
```

实现细节：
- 分箱统一**左闭右开**（末箱闭），边界值只归一箱，不重不漏
- 零计数按经典做法以 0.5 平滑，WOE 保持有限值
- 常数列自动撑开一箱；重复值导致实际箱数可能少于 n_bins

## 五、多模型对比 xbenchmark

```python
from lautpy.ml import xbenchmark
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = xbenchmark(
    [("lr", LogisticRegression(max_iter=200)),
     ("rf", RandomForestClassifier(n_estimators=100))],
    X_train, y_train, X_test, y_test,
    n_jobs=3,          # joblib 并行
)
# DataFrame：index=模型名，列 fit_seconds / accuracy / f1_macro / predict_seconds，
# 按首个指标降序。自定义指标：
df = xbenchmark(models, ..., scoring={"auc": lambda est, X, y: roc_auc_score(y, est.predict_proba(X)[:, 1])})
```

## 六、模型存取 model_io

```python
from lautpy.ml import model_dump, model_load, model_info

model_dump(pipeline, "model.pkl",
           meta={"features": cols, "train_rows": len(y_train), "version": "v3"})

model = model_load("model.pkl")
model, meta = model_load("model.pkl", with_meta=True)
meta = model_info("model.pkl")       # 只看元数据
```

- joblib 优先（大数组友好），未安装时回退 pickle
- payload 带格式标记：裸 pickle 的旧模型载入会明确报错，不会静默错配

## 七、设计边界（有意不做的）

- **不重造 sklearn**：标准化/编码器/模型本体直接用原生
- **不做 imblearn 封装**：重采样 `pip install imblearn` 即可
- **不做训练循环/早停**：那是 PyTorch/LightGBM 框架的领地
- **不新增依赖**：以上能力全部由 `ml` extra（sklearn/pandas/numpy）覆盖
