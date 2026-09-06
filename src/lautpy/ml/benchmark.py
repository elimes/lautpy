# @Author: elimes
"""多模型一键横向对比：并行 fit + 指标汇总为 DataFrame。

需要 ``pip install "lautpy[ml]"``（scikit-learn + pandas；sklearn 自带 joblib）。
"""

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from lautpy._internal import logger
from lautpy.pipe import Pipe


def benchmark_table(
    estimators: Sequence[tuple[str, Any]],
    X_train,
    y_train,
    X_test,
    y_test,
    scoring: dict[str, Callable[[Any, Any, Any], float]] | None = None,
    n_jobs: int = 3,
) -> "pd.DataFrame":
    """并行训练一组模型并输出指标汇总表（每行一个模型）。

    默认指标：fit_seconds（训练耗时）、predict_seconds（预测 + 全部指标打分的总耗时）、
    accuracy、f1_macro。自定义指标传 ``scoring={"名称": f(estimator, X, y) -> float}``。

    注意：joblib 并行时每个进程持有 estimator 的拷贝——**原对象不会被 fit**，
    结果一律以返回的行为准；需要复用训练好的模型时改用 n_jobs=1。

    Args:
        estimators: ``(名称, 已配置的 estimator)`` 序列，如
            ``[("lr", LogisticRegression()), ("rf", RandomForestClassifier())]``。
        X_train / y_train / X_test / y_test: 训练与测试数据。
        scoring: 自定义指标（覆盖默认 accuracy/f1_macro 时全部替换）。
        n_jobs: joblib 并行度（未安装 joblib 时退化为串行）。

    Returns:
        pd.DataFrame: 索引为模型名，按 f1_macro（或首个自定义指标）降序。

    Example::

        # 管道形式（xbenchmark 为 Pipe 包装）
        df = [("lr", LogisticRegression(max_iter=200)),
              ("rf", RandomForestClassifier())] | xbenchmark(X_train, y_train, X_test, y_test)
        # 直接调用
        df = benchmark_table(models, X_train, y_train, X_test, y_test)
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError('pandas is required for xbenchmark: pip install "lautpy[ml]"') from e

    scoring = scoring or _default_scoring()
    runner = _make_runner(X_train, y_train, X_test, y_test, scoring)

    try:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs)(delayed(runner)(name, est) for name, est in estimators)
    except ImportError:
        logger.warning("joblib not installed, running serially")
        rows = [runner(name, est) for name, est in estimators]

    df = pd.DataFrame(rows).set_index("model")
    first_metric = next(iter(scoring))
    return df.sort_values(first_metric, ascending=False)


xbenchmark = Pipe(benchmark_table)
"""benchmark_table 的管道形式：``models | xbenchmark(X_train, ...)``。"""


def _default_scoring() -> dict[str, Callable[[Any, Any, Any], float]]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": lambda est, X, y: float(accuracy_score(y, est.predict(X))),
        "f1_macro": lambda est, X, y: float(f1_score(y, est.predict(X), average="macro")),
    }


def _make_runner(X_train, y_train, X_test, y_test,
                 scoring: dict[str, Callable[[Any, Any, Any], float]]):
    def runner(name: str, estimator: Any) -> dict[str, Any]:
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = round(time.perf_counter() - start, 4)

        start = time.perf_counter()
        row: dict[str, Any] = {"model": name, "fit_seconds": fit_seconds}
        for metric_name, fn in scoring.items():
            row[metric_name] = round(fn(estimator, X_test, y_test), 6)
        row["predict_seconds"] = round(time.perf_counter() - start, 4)
        return row

    return runner
