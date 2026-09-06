# @Author: elimes
"""特征分箱与 WOE/IV：评分卡、风控与可解释建模的刚需（sklearn 无对应实现）。

约定：y 为 0/1 二值标签（0=好样本，1=坏样本）；WOE = ln(坏样本占比 / 好样本占比)，
IV = Σ (坏样本占比 - 好样本占比) × WOE。零计数按经典做法以 0.5 平滑。
"""

from typing import Any, Literal

import numpy as np

_EPS = 1e-12
_ADJUST = 0.5  # 零计数平滑量


def _validate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("x/y are empty")
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("y must be binary (0/1)")
    return x, y


def bin_edges(
    x,
    n_bins: int = 10,
    method: Literal["quantile", "uniform"] = "quantile",
) -> np.ndarray:
    """计算分箱边界（去重后的单调递增边界数组）。

    Args:
        x: 连续特征。
        n_bins: 目标箱数（实际箱数可能因重复值减少）。
        method: "quantile" 等频 / "uniform" 等距。

    Returns:
        np.ndarray: 边界数组（长度 = 实际箱数 + 1），首尾为 -inf/+inf。
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("x is empty")
    if method == "quantile":
        edges = np.quantile(arr, np.linspace(0, 1, max(int(n_bins), 2) + 1))
    elif method == "uniform":
        edges = np.linspace(arr.min(), arr.max(), max(int(n_bins), 2) + 1)
    else:
        raise ValueError(f"unknown method {method!r}: use 'quantile' or 'uniform'")
    edges = np.unique(edges)
    if edges.size < 2:  # 常数列：人为撑开一箱
        edges = np.array([edges[0], edges[0] + _EPS])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def woe_bins(
    x,
    y,
    n_bins: int = 10,
    method: Literal["quantile", "uniform"] = "quantile",
    edges: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """对单个特征做分箱并计算每箱的 WOE 与 IV 贡献。

    NaN 处理遵循评分卡惯例：缺失样本单独成箱（bin="missing"，排在结果末尾），
    不与任何数值箱混淆——因此各箱 total 之和恒等于样本数。

    Args:
        x: 连续特征。
        y: 0/1 二值标签。
        n_bins: 目标箱数（edges 给定时忽略）。
        method: 分箱方式。
        edges: 自定义切点（含 -inf/+inf 兜底可省略）。

    Returns:
        list[dict]: 每箱一个 dict，键为 bin（区间描述）、lo/hi（数值边界，供
            woe_transform 与下游使用）、total、bad、bad_rate、woe、iv，按箱升序；
            输入含 NaN 时末尾附加 missing 箱。

    Example::

        bins = woe_bins(income, default_flag, n_bins=5)
        sum(b["iv"] for b in bins)   # 该特征的 IV
    """
    x, y = _validate(x, y)
    nan_mask = np.isnan(x)
    if edges is None:
        edges = bin_edges(x[~nan_mask], n_bins=n_bins, method=method)
    edges = np.asarray(edges, dtype=float)
    if edges[0] != -np.inf:
        edges = np.concatenate([[-np.inf], edges])
    if edges[-1] != np.inf:
        edges = np.concatenate([edges, [np.inf]])

    n_bad = float(np.sum(y == 1)) or _EPS
    n_good = float(np.sum(y == 0)) or _EPS

    def _stat(total: int, bad: float) -> dict[str, Any]:
        good = total - bad
        bad_adj = bad if bad else _ADJUST
        good_adj = good if good else _ADJUST
        woe = float(np.log((bad_adj / n_bad) / (good_adj / n_good)))
        iv = float((bad_adj / n_bad - good_adj / n_good) * woe)
        return {
            "total": total,
            "bad": int(bad),
            "bad_rate": round(bad / total, 6) if total else 0.0,
            "woe": round(woe, 6),
            "iv": round(iv, 6),
        }

    result = []
    x_num = x[~nan_mask]
    y_num = y[~nan_mask]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = ((x_num >= lo) & (x_num <= hi) if hi == np.inf else (x_num >= lo) & (x_num < hi))
        # 统一左闭右开：边界值只归一箱，不重不漏
        stats = _stat(int(np.sum(mask)), float(np.sum(y_num[mask] == 1)))
        label = f"[{lo}, {hi})" if hi != np.inf else f"[{lo}, +inf]"
        result.append({"bin": label, "lo": float(lo), "hi": float(hi), **stats})

    if nan_mask.any():  # 评分卡惯例：缺失样本单独成箱
        stats = _stat(int(nan_mask.sum()), float(np.sum(y[nan_mask] == 1)))
        result.append({"bin": "missing", "lo": np.nan, "hi": np.nan, **stats})
    return result


def iv_summary(x, y, n_bins: int = 10, method: Literal["quantile", "uniform"] = "quantile") -> float:
    """计算单特征的信息价值 IV（特征预测力：>0.3 强 / 0.1~0.3 中 / <0.1 弱）。"""
    return round(float(sum(b["iv"] for b in woe_bins(x, y, n_bins=n_bins, method=method))), 6)


def woe_transform(x, bins: list[dict[str, Any]]) -> np.ndarray:
    """按 woe_bins 的结果把特征值替换为所在箱的 WOE 值（编码连续特征）。

    向量化实现（np.searchsorted）。NaN 的行为：训练分箱存在 missing 箱
    （即 woe_bins 输入含 NaN）时映射到 missing 箱的 WOE；否则原样保留 NaN。

    Args:
        x: 原始特征值。
        bins: woe_bins 的返回结果。

    Returns:
        np.ndarray: 与 x 等长的 WOE 编码数组。
    """
    los = np.array([b["lo"] for b in bins], dtype=float)
    woes = np.array([b["woe"] for b in bins], dtype=float)
    has_missing = bins[-1]["bin"] == "missing"

    arr = np.asarray(x, dtype=float).ravel()
    out = np.full(arr.shape, np.nan)
    valid = ~np.isnan(arr)
    # 统一左闭右开、末箱闭：searchsorted(right)-1 与分箱规则一致
    idx = np.searchsorted(los, arr[valid], side="right") - 1
    idx = np.clip(idx, 0, len(bins) - 1)
    out[valid] = woes[idx]
    if has_missing and np.isnan(arr).any():
        out[np.isnan(arr)] = woes[-1]
    return out
