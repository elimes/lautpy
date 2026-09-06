#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : X.
# @File         : __init__.py
# @Time         : 2020/11/12 10:54 上午
# @Author       : liufeng
# @Email        : elimes@qq.com
# @Software     : PyCharm
# @Description  :

from .pipe import *  # noqa: F401,F403


def _get_version() -> str:
    """从包元数据读取版本号；源码目录直接运行时回退读取仓库根的 .data/VERSION。"""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # Python 3.7 需借助 backport
        try:
            from importlib_metadata import PackageNotFoundError, version
        except ImportError:
            PackageNotFoundError = None
    if PackageNotFoundError is not None:
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
