"""lautpy.ml 各模块测试（numpy/sklearn/pandas 缺失时自动 skip）。"""

import pytest

np = pytest.importorskip("numpy")  # metrics / binning 的硬前置

from lautpy.ml.seed import set_seed  # noqa: E402


def test_set_seed_reproducible_random():
    import random

    set_seed(2026)
    a = [random.random() for _ in range(5)]
    set_seed(2026)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_reproducible_numpy():
    set_seed(7)
    a = np.random.rand(5)
    set_seed(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_seed_returns_seed():
    assert set_seed(99) == 99


# ---------- metrics ----------

from lautpy.ml.metrics import best_threshold, ks_stat, psi, report_lite  # noqa: E402


def test_best_threshold_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    thr, score = best_threshold(y_true, y_score, metric="f1")
    assert score == pytest.approx(1.0) and 0.2 < thr <= 0.8


def test_best_threshold_custom_callable():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.3, 0.4, 0.6, 0.7])
    thr, _ = best_threshold(y_true, y_score, metric=lambda tp, fp, fn, tn: tp - fp)
    y_pred = (y_score >= thr).astype(int)
    assert int(((y_pred == 1) & (y_true == 1)).sum()) >= 1


def test_ks_stat_bounds():
    y = np.array([0] * 5 + [1] * 5)
    perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert ks_stat(y, perfect) == 1.0  # 完全可分
    noise = np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.5, 0.4, 0.5, 0.6, 0.5])
    assert 0 <= ks_stat(y, noise) < 0.6
    with pytest.raises(ValueError, match="both classes"):
        ks_stat(np.ones(5), np.random.rand(5))


def test_psi_identical_vs_shifted():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 5000)
    assert psi(base, base.copy()) < 0.01  # 同分布 ≈ 0
    drifted = rng.normal(1.5, 1, 5000)
    assert psi(base, drifted) > 0.5  # 显著漂移


def test_report_lite_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    r = report_lite(y_true, y_pred=y_pred)
    assert r["accuracy"] == 0.75 and r["recall"] == 1.0
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(0.8, abs=1e-4)


def test_report_lite_with_score_adds_auc():
    pytest.importorskip("sklearn")
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    r = report_lite(y_true, y_score=y_score)
    assert r["auc"] == 1.0


# ---------- binning ----------

from lautpy.ml.binning import bin_edges, iv_summary, woe_bins, woe_transform  # noqa: E402


def _separable_xy():
    x = np.arange(100, dtype=float)
    y = (x >= 50).astype(int)  # 完全可分：x 越大越坏
    return x, y


def test_bin_edges_shapes():
    x, _ = _separable_xy()
    edges = bin_edges(x, n_bins=5, method="quantile")
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)  # 单调递增且去重


def test_woe_bins_perfect_separation_iv():
    x, y = _separable_xy()
    bins = woe_bins(x, y, n_bins=4)
    assert sum(b["total"] for b in bins) == 100  # 无损
    # 完全可分：好样本箱 WOE 为负、坏样本箱 WOE 为正，IV 很大
    assert bins[0]["woe"] < 0 < bins[-1]["woe"]
    assert iv_summary(x, y, n_bins=4) > 3


def test_woe_bins_zero_count_smoothed():
    x = np.array([1.0, 1.0, 2.0, 2.0])
    y = np.array([0, 0, 0, 0])  # 无坏样本：零计数走 0.5 平滑，不抛异常
    bins = woe_bins(x, y, n_bins=2)
    assert all(np.isfinite(b["woe"]) for b in bins)


def test_woe_transform_roundtrip():
    x, y = _separable_xy()
    bins = woe_bins(x, y, n_bins=4)
    encoded = woe_transform(x, bins)
    assert encoded.shape == x.shape
    # 单调特征：后半段（坏样本区）编码值应高于前半段
    assert np.mean(encoded[50:]) > np.mean(encoded[:50])


# ---------- benchmark / split / model_io ----------

def test_xbenchmark_table():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from lautpy.ml.benchmark import benchmark_table, xbenchmark

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.1, size=200) > 0).astype(int)
    from sklearn.dummy import DummyClassifier
    from sklearn.tree import DecisionTreeClassifier

    models = [("dummy", DummyClassifier(strategy="most_frequent")),
              ("tree", DecisionTreeClassifier(random_state=0))]
    df = benchmark_table(models, X[:150], y[:150], X[150:], y[150:], n_jobs=1)
    assert isinstance(df, pd.DataFrame) and len(df) == 2
    assert {"fit_seconds", "accuracy", "f1_macro"} <= set(df.columns)
    # 管道形式（estimators 为 list，无 __or__，Pipe.__ror__ 正常接管）
    df2 = models | xbenchmark(X[:150], y[:150], X[150:], y[150:], n_jobs=1)
    assert len(df2) == 2


def test_xsplit_shapes_and_no_leakage():
    pytest.importorskip("sklearn")
    from lautpy.ml.split import xsplit

    X = np.arange(100).reshape(-1, 1)
    y = np.array([0, 1] * 50)
    X_tr, X_val, X_te, y_tr, y_val, y_te = xsplit(
        X, y, test_size=0.2, val_size=0.1, stratify=y, random_state=0)
    assert len(X_tr) + len(X_val) + len(X_te) == 100
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)
    assert set(X_tr.ravel()) & set(X_te.ravel()) == set()  # 无泄漏


def test_xsplit_two_way():
    pytest.importorskip("sklearn")
    from lautpy.ml.split import xsplit

    a, b = xsplit(list(range(50)), test_size=0.2, random_state=1)
    assert len(a) + len(b) == 50


def test_xy_split():
    pd = pytest.importorskip("pandas")
    from lautpy.ml.split import xy

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]})
    X, y = xy(df, "label")
    assert list(X.columns) == ["a", "b"] and list(y) == [0, 1]


def test_model_dump_load_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.dummy import DummyClassifier

    from lautpy.ml.model_io import model_dump, model_info, model_load

    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0], [1]], [0, 1])
    p = model_dump(model, tmp_path / "m.pkl", meta={"features": ["f1"]})

    loaded = model_load(p)
    assert loaded.predict([[0]]).tolist() == [1]
    assert model_info(p)["features"] == ["f1"]
    model2, meta = model_load(p, with_meta=True)
    assert meta["saved_at"] and model2 is not loaded


def test_model_load_rejects_bare_pickle(tmp_path):
    from lautpy.ml.model_io import model_load

    p = tmp_path / "bare.pkl"
    p.write_bytes(__import__("pickle").dumps({"just": "data"}))
    with pytest.raises(ValueError, match="not a lautpy model_dump"):
        model_load(p)


# ---------- 评审回归（2026-09 外部评审） ----------

def test_psi_discrete_scores_not_collapsed():
    """回归：离散分数下 PSI 曾坍缩为 0，漂移完全不可见。"""
    from lautpy.ml.metrics import psi

    base = np.array([0.0] * 99 + [1.0])
    online = np.array([1.0] * 100)
    value = psi(base, online)
    assert value > 1.0  # 极端漂移必须可见


def test_xy_direct_call():
    pd = pytest.importorskip("pandas")
    from lautpy.ml.split import xy

    df = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
    X, y = xy(df, "label")
    assert list(X.columns) == ["a"] and list(y) == [0, 1]
    # 已知限制（见 xy docstring）：DataFrame 在管道左侧时 | 被 pandas
    # 元素级 OR 接管，无法触发 Pipe.__ror__ —— xy 只提供直接调用


def test_woe_bins_nan_missing_bin():
    """回归：NaN 样本此前静默掉出所有箱，现在单独成 missing 箱。"""
    from lautpy.ml.binning import woe_bins, woe_transform

    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    y = np.array([0, 0, 1, 1, 1])
    bins = woe_bins(x, y, n_bins=2)
    assert sum(b["total"] for b in bins) == 5  # 不再丢样本
    assert bins[-1]["bin"] == "missing" and bins[-1]["total"] == 1
    assert all("lo" in b and "hi" in b for b in bins[:-1])  # 数值边界字段

    encoded = woe_transform(np.array([np.nan, 2.0]), bins)
    assert not np.isnan(encoded[0])  # NaN 映射到 missing 箱 WOE
    assert encoded[1] == bins[0]["woe"] or encoded[1] == bins[1]["woe"]
