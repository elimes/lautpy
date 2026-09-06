# @Author: elimes
"""lautpy —— 轻量 Python 工具集（import lautpy 零强制第三方依赖）。

导入即获得全部管道函数（``from lautpy.pipe import *`` 的效果），
其余模块按需访问：``agent`` / ``dates`` / ``paths`` / ``hashing`` /
``decorators`` / ``llm`` 直接可用；``apis`` / ``notice`` 惰性加载
（首次访问时导入，需要 requests，即 ``pip install "lautpy[http]"``）。

用法见 docs/usage.md；`__version__` 来自包元数据。
"""

from . import agent, dates, decorators, hashing, llm, paths  # noqa: F401
from .pipe import *  # noqa: F401,F403

# 惰性子模块（PEP 562）：import lautpy 时不加载 requests，首次访问才导入
_LAZY_SUBMODULES = frozenset({"apis", "notice"})


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


def _get_version() -> str:
    """从包元数据读取版本号；源码目录直接运行时回退读取仓库根的 .data/VERSION。"""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("lautpy")
    except PackageNotFoundError:
        pass
    # 未安装：回退到仓库根的权威版本文件（src/lautpy/__init__.py -> 上两级即仓库根）
    from pathlib import Path

    version_file = Path(__file__).resolve().parents[2] / ".data" / "VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip()
    return "0.0.0+local"


__version__ = _get_version()
del _get_version
