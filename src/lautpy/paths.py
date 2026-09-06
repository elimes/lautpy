# -*- coding: utf-8 -*-
# @Author: elimes
"""路径与配置文件工具（json/yaml/pickle），yaml 支持需安装 pyyaml。"""

import json
import pickle
from pathlib import Path
from typing import Any, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def get_resolve_path(path: Union[str, Path], file: Union[str, Path]) -> Path:
    """以 file 所在目录为基准解析相对路径（常配合 __file__ 定位包内资源）。

    Args:
        path: 相对路径。
        file: 基准文件（通常是调用处的 ``__file__``）。

    Returns:
        Path: 解析后的绝对路径。

    Example::

        get_resolve_path('../data/config.json', __file__)
    """
    return (Path(file).parent / Path(path)).resolve()


def _load_structured(text_or_bytes) -> dict:
    if yaml is not None:
        return yaml.safe_load(text_or_bytes)
    raise ImportError("pyyaml is required for yaml support: pip install pyyaml")


def file2json(path: Union[str, Path]) -> dict:
    """读取 .json / .yml / .yaml 配置文件为 dict。

    Args:
        path: 配置文件路径。

    Returns:
        dict: 解析结果；路径无效或扩展名不支持时返回空 dict（不抛异常）。
    """
    p = Path(path)
    if not p.is_file():
        return {}
    if p.name.endswith(".json"):
        return json.loads(p.read_bytes())
    if p.name.endswith((".yml", ".yaml")):
        return _load_structured(p.read_bytes()) or {}
    return {}


def path2list(path: Union[str, Path], pattern: str = "*") -> List[Path]:
    """把文件或目录展开为 Path 列表。

    Args:
        path: 文件（返回单元素列表）或目录（按 pattern glob）。
        pattern: 目录展开时的通配模式，默认全部文件。

    Returns:
        List[Path]: 路径无效时返回空列表。
    """
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return list(p.glob(pattern))
    return []


def get_config(config_init: Optional[Union[str, dict]]) -> dict:
    """归一化配置来源：dict、json/yaml 文件路径、None 均可，统一返回 dict。

    Args:
        config_init: dict 原样返回；字符串且指向存在的文件则读文件；
            其余情况返回空 dict。
    """
    if isinstance(config_init, str) and Path(config_init).is_file():
        return file2json(config_init)
    if isinstance(config_init, dict):
        return config_init
    return {}


def pkl_dump(obj: Any, file: Union[str, Path]) -> Path:
    """把对象序列化落盘（pickle 二进制，protocol=4）。

    Args:
        obj: 任意可 pickle 的对象。
        file: 目标文件路径。

    Returns:
        Path: 写入的路径，便于链式使用。
    """
    p = Path(file)
    p.write_bytes(pickle.dumps(obj, protocol=4))
    return p


def pkl_load(file: Union[str, Path]) -> Any:
    """读回 pkl_dump 落盘的对象。

    Args:
        file: pkl_dump 生成的文件路径。
    """
    return pickle.loads(Path(file).read_bytes())
