# @Author: elimes
"""数据划分：管道友好的 train/val/test 切分与 X/y 分离。

需要 ``pip install "lautpy[ml]"``（分层抽样依赖 scikit-learn）。
"""

from collections.abc import Hashable

import pandas as pd
from sklearn.model_selection import train_test_split


def xsplit(
    *arrays,
    test_size: float = 0.2,
    val_size: float | None = None,
    stratify=None,
    random_state: int = 42,
) -> tuple:
    """把一个或多个同长数组切成 train/val/test（或 train/test）。

    Args:
        *arrays: 同长度的数组/列表/DataFrame（如 X, y）。
        test_size: 测试集比例（相对全量）。
        val_size: 验证集比例（相对全量）；None 时只切 train/test 两份。
        stratify: 分层依据（分类标签数组）；两刀切分都保持分布。
        random_state: 随机种子。

    Returns:
        tuple: 按数组分组返回（与 train_test_split 习惯一致），例如
            ``xsplit(X, y, val_size=0.1)`` → ``(X_train, X_val, X_test, y_train, y_val, y_test)``。

    Example::

        X_tr, X_val, X_te, y_tr, y_val, y_te = xsplit(
            X, y, test_size=0.2, val_size=0.1, stratify=y)
    """
    if not arrays:
        raise ValueError("provide at least one array")
    if val_size is not None and (val_size <= 0 or test_size + val_size >= 1):
        raise ValueError("require val_size > 0 and test_size + val_size < 1")

    work = list(arrays)
    if stratify is not None:
        work.append(stratify)  # 分层标签作为最后一个数组随切分走，天然对齐

    if val_size is None:
        out = train_test_split(*work, test_size=test_size, random_state=random_state,
                               stratify=stratify)
        trains, tests = out[0::2], out[1::2]
        has_strata = stratify is not None
        return tuple(t for pair in zip(_drop_strata(trains, has_strata),
                                       _drop_strata(tests, has_strata), strict=True)
                     for t in pair)

    holdout = test_size / (1 - val_size)  # 第二刀相对剩余量的比例
    first = train_test_split(*work, test_size=val_size, random_state=random_state,
                             stratify=stratify)
    train_parts, _val_parts = first[0::2], first[1::2]
    strata_train = train_parts[-1] if stratify is not None else None
    if stratify is not None:
        train_parts = train_parts[:-1]

    second = train_test_split(*train_parts, strata_train, test_size=holdout,
                              random_state=random_state, stratify=strata_train)
    has_strata = stratify is not None
    trains = _drop_strata(second[0::2], has_strata)
    vals = _drop_strata(second[1::2], has_strata)
    tests = _drop_strata(_val_parts, has_strata)
    # 按数组分组返回：(X_train, X_val, X_test, y_train, y_val, y_test)
    return tuple(t for group in zip(trains, vals, tests, strict=True) for t in group)


def _drop_strata(parts: list, has_strata: bool) -> list:
    """剔除每组末尾的分层标签数组（未使用分层时不裁剪）。"""
    return parts[:-1] if has_strata else parts


def xy(df: pd.DataFrame, target: Hashable | list[Hashable]) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    """把 DataFrame 拆成特征矩阵 X 与目标 y。

    Args:
        df: 原始数据。
        target: 目标列名（单个或列表，支持多输出）。

    Returns:
        tuple[pd.DataFrame, pd.Series | pd.DataFrame]: (X, y)。

    Example::

        X, y = xy(df, "label")

    注意：不支持 ``df | xy(...)`` 管道写法——pandas DataFrame 重载了 ``|``
    （元素级 OR），会抢先接管运算符，Pipe 的 __ror__ 不会被触发。
    """
    y = df[target]  # list 与标量列名在 df[...] 下行为一致
    return df.drop(columns=target), y


# 注意：xy 不提供 Pipe 包装——DataFrame 在管道左侧时，pandas 的 __or__ 会
# 接管 | 运算符（元素级 OR），Pipe.__ror__ 不会被调用。

