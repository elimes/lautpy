# @Author: elimes
"""模型存取：带元数据的 joblib/pickle 序列化。

joblib 可用时优先（对 numpy 大数组更友好），否则回退 pickle。

⚠️ 安全提示：反序列化（pickle/joblib）等价于执行任意代码——**只加载
可信来源的模型文件**，不要 load 来路不明的 .pkl。
"""

import pickle
import time
from pathlib import Path
from typing import Any

from lautpy._internal import logger


def model_dump(model: Any, file: str | Path, meta: dict | None = None) -> Path:
    """保存模型及元数据（训练时间、库版本、调用方自定义字段）。

    Args:
        model: 任意可序列化的模型对象（sklearn pipeline / xgboost / 自定义类）。
        file: 目标文件路径。
        meta: 附加元数据（如特征列清单、训练参数、数据集版本）。

    Returns:
        Path: 写入的文件路径。

    Example::

        model_dump(pipeline, "model.pkl",
                   meta={"features": cols, "train_rows": len(y_train)})
    """
    try:
        import joblib

        dump = joblib.dump
    except ImportError:
        logger.debug("joblib not installed, falling back to pickle")
        dump = None

    payload = {
        "model": model,
        "meta": {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lautpy_version": _lautpy_version(),
            **(meta or {}),
        },
    }
    p = Path(file)
    if dump is not None:
        dump(payload, p)
    else:
        p.write_bytes(pickle.dumps(payload, protocol=4))
    return p


def model_load(file: str | Path, with_meta: bool = False) -> Any:
    """读取 model_dump 保存的模型。

    Args:
        file: 模型文件路径。
        with_meta: True 时返回 ``(model, meta)``，否则仅返回模型。

    Raises:
        ValueError: 文件不是本模块的 payload 格式（如裸 pickle 的模型）。

    ⚠️ 反序列化等于执行任意代码，只加载可信来源的文件。
    """
    p = Path(file)
    try:
        import joblib

        payload = joblib.load(p)
    except ImportError:
        payload = pickle.loads(p.read_bytes())

    if isinstance(payload, dict) and "model" in payload and "meta" in payload:
        return (payload["model"], payload["meta"]) if with_meta else payload["model"]
    raise ValueError(
        f"{p} is not a lautpy model_dump payload; load bare pickles yourself "
        f"(or re-save with model_dump)."
    )


def model_info(file: str | Path) -> dict:
    """读取模型的元数据 dict。

    注意：元数据与模型存于同一文件，本函数同样需要完整反序列化 payload——
    它只是省去你手动解包，**没有性能捷径**。
    """
    _, meta = model_load(file, with_meta=True)
    return meta


def _lautpy_version() -> str:
    from lautpy import __version__

    return __version__
