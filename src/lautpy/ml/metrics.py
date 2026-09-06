# @Author: elimes
"""评估指标补集：sklearn 不直接给、但日常高频手写的那些。

- best_threshold：二分类阈值搜索（F1/准确率/自定义谓词）
- ks_stat：评分卡/风控标配的 KS 值
- psi：群体稳定性指数（跨期特征/分数漂移监控）
- report_lite：一次输出常用分类指标 dict

全部纯 numpy 实现；report_lite 在安装 scikit-learn 时额外给出 AUC。
"""

from collections.abc import Callable

import numpy as np

_EPS = 1e-12


def _to_array(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """返回 (tp, fp, fn, tn)，正类约定为 1。"""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def _metric_value(metric: str | Callable[[int, int, int, int], float],
                  tp: int, fp: int, fn: int, tn: int) -> float:
    if callable(metric):
        return metric(tp, fp, fn, tn)
    if metric == "f1":
        return 2 * tp / (2 * tp + fp + fn + _EPS)
    if metric == "accuracy":
        return (tp + tn) / (tp + fp + fn + tn + _EPS)
    if metric == "precision":
        return tp / (tp + fp + _EPS)
    if metric == "recall":
        return tp / (tp + fn + _EPS)
    raise ValueError(f"unknown metric {metric!r}: use f1/accuracy/precision/recall or a callable")


def best_threshold(
    y_true,
    y_score,
    metric: str | Callable[[int, int, int, int], float] = "f1",
    bins: int = 200,
) -> tuple[float, float]:
    """二分类阈值搜索：扫描候选阈值，返回 (最优阈值, 最优指标值)。

    Args:
        y_true: 真实标签（0/1）。
        y_score: 正类概率或连续分数。
        metric: "f1" / "accuracy" / "precision" / "recall"，或
            自定义谓词 ``f(tp, fp, fn, tn) -> float``。
        bins: 候选阈值数量（在分数区间内均匀取点）。

    Returns:
        tuple[float, float]: (best_threshold, best_score)。

    Example::

        thr, score = best_threshold(y_true, y_proba, metric="f1")
        y_pred = (y_proba >= thr).astype(int)
    """
    y_true = _to_array(y_true, "y_true")
    y_score = _to_array(y_score, "y_score")
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if set(np.unique(y_true)) - {0.0, 1.0}:
        raise ValueError("y_true must be binary (0/1)")

    lo, hi = float(y_score.min()), float(y_score.max())
    candidates = np.linspace(lo, hi + _EPS, max(int(bins), 2))

    # 排序一次 + 后缀计数：O(n log n) 内得到全部候选阈值的混淆矩阵
    order = np.argsort(y_score, kind="stable")
    s_sorted = y_score[order]
    y_sorted = y_true[order]
    n = y_sorted.size
    total_pos = float(np.sum(y_sorted == 1))
    cum_pos = np.cumsum(y_sorted)  # 升序前 i 个样本中的正样本数
    idx = np.searchsorted(s_sorted, candidates, side="left")  # score >= thr 的起始下标
    pos_pred = n - idx
    tp = total_pos - np.where(idx > 0, cum_pos[np.maximum(idx - 1, 0)], 0)
    fp = pos_pred - tp
    fn = total_pos - tp
    tn = n - pos_pred - fn

    if callable(metric):
        best_t, best_v = lo, -np.inf
        for thr, tp_, fp_, fn_, tn_ in zip(candidates, tp, fp, fn, tn, strict=True):
            value = metric(int(tp_), int(fp_), int(fn_), int(tn_))
            if value > best_v:
                best_t, best_v = float(thr), value
        return best_t, best_v

    if metric == "f1":
        values = 2 * tp / (2 * tp + fp + fn + _EPS)
    elif metric == "accuracy":
        values = (tp + tn) / (tp + fp + fn + tn + _EPS)
    elif metric == "precision":
        values = tp / (tp + fp + _EPS)
    elif metric == "recall":
        values = tp / (tp + fn + _EPS)
    else:
        raise ValueError(f"unknown metric {metric!r}: use f1/accuracy/precision/recall or a callable")
    i = int(np.argmax(values))
    return float(candidates[i]), float(values[i])


def ks_stat(y_true, y_score) -> float:
    """计算 KS 统计量：正负样本分数累计分布差的最大值（0~1，越大区分度越好）。

    Args:
        y_true: 真实标签（0/1）。
        y_score: 正类概率或连续分数。
    """
    y_true = _to_array(y_true, "y_true")
    y_score = _to_array(y_score, "y_score")
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if set(np.unique(y_true)) - {0.0, 1.0}:
        raise ValueError("y_true must be binary (0/1)")

    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_pos = float(np.sum(y_sorted == 1))
    n_neg = float(np.sum(y_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("KS needs both classes present in y_true")
    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def psi(expected, actual, bins: int = 10) -> float:
    """群体稳定性指数（PSI）：分布漂移监控，>0.25 通常视为显著漂移。

    以 expected 的分位数分箱，比较 actual 落入各箱的占比。

    Args:
        expected: 基准分布（如训练集分数/特征值）。
        actual: 待比较分布（如线上分数）。
        bins: 分箱数量。

    Returns:
        float: PSI 值（>= 0；两分布完全一致时为 0）。
    """
    exp_arr = _to_array(expected, "expected")
    act_arr = _to_array(actual, "actual")
    if exp_arr.size < 2:
        raise ValueError("expected needs at least 2 samples")
    edges = np.unique(np.quantile(exp_arr, np.linspace(0, 1, max(int(bins), 2) + 1)))
    if edges.size < 3:
        # 离散分数：分位数边界去重后不足以形成多箱（如 0/1 档位），坍缩会让漂移
        # 完全不可见——回退等距分箱；常数基准再人为撑开两箱
        edges = np.unique(np.linspace(exp_arr.min(), exp_arr.max(), max(int(bins), 2) + 1))
    if edges.size < 2:
        c = float(edges[0])
        edges = np.array([c - _EPS, c + _EPS])
    edges[0], edges[-1] = -np.inf, np.inf  # 开区间兜底，覆盖 actual 超出基准范围的部分

    exp_ratio = np.histogram(exp_arr, bins=edges)[0] / exp_arr.size
    act_ratio = np.histogram(act_arr, bins=edges)[0] / act_arr.size
    exp_ratio = np.clip(exp_ratio, _EPS, None)
    act_ratio = np.clip(act_ratio, _EPS, None)
    return float(np.sum((act_ratio - exp_ratio) * np.log(act_ratio / exp_ratio)))


def report_lite(
    y_true,
    y_pred: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """一次输出常用分类指标。

    Args:
        y_true: 真实标签（0/1）。
        y_pred: 预测标签（0/1）；未提供且给了 y_score 时按 threshold 二值化。
        y_score: 正类概率/分数（可选；提供时额外计算 AUC，需 scikit-learn）。
        threshold: y_score → y_pred 的判定阈值。

    Returns:
        dict[str, float]: accuracy / precision / recall / f1（+ auc）。
    """
    y_true = _to_array(y_true, "y_true")
    if y_pred is None and y_score is None:
        raise ValueError("provide either y_pred or y_score")
    if y_pred is None:
        y_pred = (_to_array(y_score, "y_score") >= threshold).astype(int)
    y_pred = _to_array(y_pred, "y_pred")

    result = {}
    for name in ("accuracy", "precision", "recall", "f1"):
        result[name] = round(_metric_value(name, *_binary_counts(y_true, y_pred)), 6)
    if y_score is not None:
        try:
            from sklearn.metrics import roc_auc_score

            result["auc"] = round(float(roc_auc_score(y_true, y_score)), 6)
        except ImportError:
            result["auc"] = float("nan")  # type: ignore[assignment]
    return result
