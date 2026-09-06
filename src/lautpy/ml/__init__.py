# @Author: elimes
"""机器学习辅助层：可复现性、指标补集、分箱 WOE/IV、模型对比、数据划分、模型存取。

定位是"胶水层"：只补 sklearn 没有直接给的日常刚需，不重造标准算法。

子模块按需惰性加载（PEP 562）：``import lautpy.ml`` 本身不触发 numpy/sklearn 导入；
首次访问 ``lautpy.ml.metrics`` 等属性时才导入，依赖缺失时抛出带安装指引的
ImportError。set_seed 为零依赖，随包直接可用。
"""

from lautpy.ml.seed import set_seed

__all__ = ["set_seed"]

_LAZY_SUBMODULES = frozenset({"metrics", "binning", "split", "benchmark", "model_io", "seed"})


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        try:
            module = importlib.import_module(f".{name}", __name__)
        except ImportError as e:
            raise ImportError(
                f"lautpy.ml.{name} 需要可选依赖（numpy/pandas/scikit-learn）："
                f'pip install "lautpy[ml]"（原始错误：{e}）'
            ) from e
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)
